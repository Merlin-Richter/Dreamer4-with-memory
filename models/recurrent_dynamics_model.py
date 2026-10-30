"""Frozen state-history backbone with learned, bounded recurrent memory.

Each layer writes only its state tokens into a fixed memory bank at block
boundaries. Within a block, the original causal attention stays intact.
Training backpropagates through every memory update; no history is detached.
"""
import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.long_context_dynamics_model import LongContextDynamicsModel, softcap, visibility


class StateWriter(nn.Module):
    """Old slots attend to old memory and this block's before/after state tokens."""
    def __init__(self, width, slots):
        super().__init__()
        self.width = width
        self.initial = nn.Parameter(torch.randn(slots, width) * .02)
        self.types = nn.Parameter(torch.randn(3, width) * .02)
        self.norm = nn.RMSNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.gate = nn.Linear(2 * width, width)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, old, raw, processed):
        if old is None:
            old = self.initial.expand(raw.shape[0], -1, -1)
        values = torch.cat((old, raw, processed), 1)
        typed = torch.cat((old + self.types[0], raw + self.types[1], processed + self.types[2]), 1)
        pos = torch.arange(raw.shape[1], device=raw.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, self.width, 2, device=raw.device) * (-math.log(10000.) / self.width))
        pe = torch.zeros_like(raw[0])
        pe[:, 0::2], pe[:, 1::2] = (pos * freq).sin(), (pos * freq).cos()
        typed = typed + torch.cat((torch.zeros_like(old[0]), pe, pe), 0)[None]
        q, k, v = self.q(self.norm(old)), self.k(self.norm(typed)), self.v(values)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            weights = (q.float() @ k.float().transpose(-2, -1) / math.sqrt(self.width)).softmax(-1)
            read = weights @ v.float()
        joined = torch.cat((old, read), -1)
        out = old + torch.sigmoid(self.gate(joined)) * self.update(joined)
        return self.norm(out).float()


class MemoryReader(nn.Module):
    """Content-only memory projections, initialized from the parent's attention."""
    def __init__(self, attention):
        super().__init__()
        for name in ('q', 'kv', 'q_norm', 'k_norm'):
            setattr(self, name, copy.deepcopy(getattr(attention, name)))
        self.mass_bias = nn.Parameter(torch.zeros(attention.heads))
        self.requires_grad_(True)

    def forward(self, x, memory, attention, past):
        q = self.q_norm(attention._heads(self.q(x)))
        k, v = self.kv(memory).chunk(2, -1)
        k, v = self.k_norm(attention._heads(k)), attention._heads(v)
        # A fixed bank stands for a growing history. Preserve the research
        # reader's historical mass correction, without inventing slot timestamps.
        mass = self.mass_bias[None, :, None, None] + math.log(max(past, 1) / memory.shape[1])
        return q, k, v, mass


@dataclass
class RecurrentState:
    """Persistent memory plus fewer than block_size clean, uncompressed frames."""
    memory: list | None
    latents: torch.Tensor
    actions: torch.Tensor
    start: int = 0


class RecurrentDynamicsModel(nn.Module):
    persistent_memory = True

    def __init__(self, config, *, block_size=16, memory_slots=16, recompute=True):
        super().__init__()
        if not 1 <= block_size <= config.max_temporal_length or memory_slots < 1:
            raise ValueError('Need positive memory slots and a block within the parent context')
        self.config, self.block_size, self.memory_slots = config, block_size, memory_slots
        self.recompute = recompute
        self.n_latents, self.bottleneck_dim, self.n_actions = config.n_latents, config.latent_dim, config.n_actions
        self.backbone = LongContextDynamicsModel(config, recompute=False)
        self.backbone.requires_grad_(False)
        self.writers = nn.ModuleList([StateWriter(config.width, memory_slots) for _ in range(config.depth)])
        self.readers = nn.ModuleList([MemoryReader(b.attention) for b in self.backbone.blocks])

    def initialize(self, parent_weights):
        self.backbone.load_state_dict(parent_weights)
        for reader, block in zip(self.readers, self.backbone.blocks):
            for name in ('q', 'kv', 'q_norm', 'k_norm'):
                getattr(reader, name).load_state_dict(getattr(block.attention, name).state_dict())

    def chunk(self, latents, tau, actions, memory, start, write_memory=True,
              teacher_cache=None, lse_weight=1.):
        """Return predictions, memory, attention loss and its two unweighted terms."""
        batch, length = latents.shape[:2]
        if not 1 <= length <= self.block_size or tau.shape != (batch, length) or actions.shape != tau.shape:
            raise ValueError('Need one aligned signal/action per frame within a single block')
        model = self.backbone
        x = model._tokens(latents, tau, actions)
        tokens = x.shape[2]
        positions = torch.arange(start + 1, start + length + 1, device=x.device)[None].expand(batch, -1)
        qpos = positions.repeat_interleave(tokens, 1)
        kpos = torch.cat((positions, positions.repeat_interleave(tokens - 1, 1)), 1)
        index = torch.arange(length * tokens, device=x.device)
        allowed = visibility(length, tokens)(None, None, index[:, None], index[None])
        updated = []
        value_loss = x.new_zeros((), dtype=torch.float32)
        lse_loss = x.new_zeros((), dtype=torch.float32)
        for i, block in enumerate(model.blocks):
            norm, raw, att = block.norm1(x), x[:, :, -1], block.attention
            flat = norm.flatten(1, 2)
            q = att._rope(att.q_norm(att._heads(att.q(flat))), qpos)
            keys = torch.cat((norm[:, :, -1], norm[:, :, :-1].flatten(1, 2)), 1)
            k, v = att.kv(keys).chunk(2, -1)
            k, v = att._rope(att.k_norm(att._heads(k)), kpos), att._heads(v)
            scale = att.logit_scale.float().clamp(max=math.log(100.)).exp()[None]
            q = (q.float() * scale).to(q.dtype)
            old = None if memory is None else memory[i]
            if old is not None:
                mq, mk, mv, mass = self.readers[i](flat, old, att, start)
                mq = (mq.float() * scale).to(mq.dtype)
            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = softcap(q.float() @ k.float().transpose(-2, -1) / math.sqrt(att.head_dim),
                                 None, None, None, None).masked_fill(~allowed, -torch.inf)
                values = v.float()
                if old is not None:
                    historical = softcap(mq.float() @ mk.float().transpose(-2, -1) / math.sqrt(att.head_dim),
                                         None, None, None, None) + mass
                    if teacher_cache is not None:
                        # The target uses the student's backbone query, not its reader query.
                        # Detach the entire target, including this query's dependence on memory.
                        with torch.no_grad():
                            old_k, old_v = teacher_cache[i]
                            old_scores = softcap(q.float() @ old_k[:, :, :start].float().transpose(-2, -1)
                                                 / math.sqrt(att.head_dim), None, None, None, None)
                            old_lse = old_scores.logsumexp(-1)
                            old_read = old_scores.softmax(-1) @ old_v[:, :, :start].float()
                        memory_read = historical.softmax(-1) @ mv.float()
                        value_loss = value_loss + (memory_read - old_read).square().sum(-1).mean()
                        lse_loss = lse_loss + (historical.logsumexp(-1) - old_lse).square().mean()
                    scores = torch.cat((historical, scores), -1)
                    values = torch.cat((mv.float(), values), 2)
                read = scores.softmax(-1) @ values
            x = x + att.out(read.to(flat.dtype).transpose(1, 2).flatten(2)).reshape_as(x)
            x = x + block.mlp(block.norm2(x))
            if write_memory:
                updated.append(self.writers[i](old, raw, x[:, :, -1]))
        prediction = model.latent_out(model.out_norm(x[:, :, :self.n_latents]))
        loss = value_loss + lse_weight * lse_loss
        return prediction, updated if write_memory else memory, loss, torch.stack((value_loss, lse_loss))

    def forward(self, latents, tau, actions):
        memory, predictions = None, []
        if latents.shape[1] < 1 or tau.shape != latents.shape[:2] or actions.shape != tau.shape:
            raise ValueError('Need a nonempty sequence with aligned signal/action ids')
        for start in range(0, latents.shape[1], self.block_size):
            end = start + self.block_size
            args = (latents[:, start:end], tau[:, start:end], actions[:, start:end], memory, start)
            if self.training and self.recompute:
                prediction, memory, _, _ = checkpoint(self.chunk, *args, use_reentrant=False)
            else:
                prediction, memory, _, _ = self.chunk(*args)
            predictions.append(prediction)
        return torch.cat(predictions, 1)

    def distill_loss(self, clean, actions, generator=None, *, lse_weight=1.):
        """Match old-history attention; sum layers and average blocks with memory.

        The frozen backbone supplies one full-sequence teacher pass. Prediction
        MSEs are diagnostics only; all memory writes remain in the gradient graph.
        """
        shape, device = clean.shape[:2], clean.device
        if not self.block_size < shape[1] <= self.config.max_temporal_length or actions.shape != shape:
            raise ValueError('Need aligned actions and more than one block within the teacher context')
        mixture = torch.rand(shape, device=device, generator=generator)
        middle = torch.randint(1, 127, shape, device=device, generator=generator)
        tau = torch.where(mixture < .5, 127, torch.where(mixture < .75, 0, middle))
        noise = torch.randn(clean.shape, device=device, generator=generator)
        amount = tau.float()[..., None, None] / 127
        noisy = amount * clean + (1 - amount) * noise
        with torch.no_grad():
            teacher, cache = self.backbone(noisy, tau, actions, return_cache=True)
        memory, predictions, losses, terms, metrics = None, [], [], [], {}
        for b, start in enumerate(range(0, shape[1], self.block_size)):
            end = start + self.block_size
            args = (noisy[:, start:end], tau[:, start:end], actions[:, start:end],
                    memory, start, True, cache, lse_weight)
            if self.training and self.recompute:
                prediction, memory, loss, errors = checkpoint(self.chunk, *args, use_reentrant=False)
            else:
                prediction, memory, loss, errors = self.chunk(*args)
            predictions.append(prediction.detach())
            if start > 0:
                losses.append(loss)
                terms.append(errors.detach())
                metrics[f'block_{b}_loss'] = loss.detach()
        loss = torch.stack(losses).mean()
        value_loss, lse_loss = torch.stack(terms).mean(0)
        with torch.no_grad():
            prediction = torch.cat(predictions, 1).float()
            error = (prediction - clean.float()).square().mean((-1, -2))
            corrupted = tau < 127
            metrics.update(distillation_loss=loss.detach(), value_loss=value_loss, lse_loss=lse_loss,
                teacher_mse=(prediction - teacher.float()).square().mean(),
                prediction_mse=(error * corrupted).sum() / corrupted.sum().clamp_min(1))
        return loss, metrics

    @torch.no_grad()
    def observe(self, latents, actions, state=None):
        """Commit clean frames, compressing each full block exactly once.

        Returns a new state without mutating the input, so branches can share it.
        No old images survive a block boundary. Reset state at episode boundaries.
        """
        if actions.shape != latents.shape[:2]:
            raise ValueError('Supply one action per observed frame')
        if state is None:
            state = RecurrentState(None, latents[:, :0], actions[:, :0])
        memory, pending, pending_actions, start = state.memory, state.latents, state.actions, state.start
        offset = 0
        while offset < latents.shape[1]:
            take = min(self.block_size - pending.shape[1], latents.shape[1] - offset)
            pending = torch.cat((pending, latents[:, offset:offset + take]), 1)
            pending_actions = torch.cat((pending_actions, actions[:, offset:offset + take]), 1)
            offset += take
            if pending.shape[1] == self.block_size:
                _, memory, _, _ = self.chunk(pending, torch.full_like(pending_actions, 127), pending_actions, memory, start)
                start += self.block_size
                pending, pending_actions = pending[:, :0], pending_actions[:, :0]
        return RecurrentState(memory, pending, pending_actions, start)

    @torch.no_grad()
    def imagine(self, action, state, K=None):
        """One generated frame and updated state; noisy queries never commit writes."""
        K = self.config.inference_steps if K is None else K
        if not isinstance(K, int) or not 1 <= K <= 127 or action.shape != (state.latents.shape[0], 1):
            raise ValueError('Need 1..127 denoising steps and one action per batch element')
        z = torch.randn((action.shape[0], 1, self.n_latents, self.bottleneck_dim),
                        device=state.latents.device, dtype=state.latents.dtype)
        actions = torch.cat((state.actions, action), 1)
        tau = torch.full_like(actions, 127)
        schedule = [round(k * 127 / K) for k in range(K + 1)]
        for begin, end in zip(schedule[:-1], schedule[1:]):
            tau[:, -1] = begin
            prediction, _, _, _ = self.chunk(torch.cat((state.latents, z), 1), tau, actions,
                                       state.memory, state.start, write_memory=False)
            z = z + (prediction[:, -1:] - z) * ((end - begin) / (127 - begin))
        return z, self.observe(z, action, state)

    @torch.no_grad()
    def generate(self, context, n_generate, K=None, action_idx=None):
        """Convenience wrapper; use observe/imagine to retain state across calls."""
        batch, prefix = context.shape[:2]
        if n_generate < 1 or action_idx is None or action_idx.shape != (batch, prefix + n_generate):
            raise ValueError('Supply actions for the full prefix and positive generation length')
        state = self.observe(context, action_idx[:, :prefix])
        generated = []
        for i in range(n_generate):
            z, state = self.imagine(action_idx[:, prefix + i:prefix + i + 1], state, K)
            generated.append(z)
        return torch.cat(generated, 1)
