"""Growing state-history dynamics, ported from the original combined_state.py.

Every layer reads current-frame tokens and causal state tokens. Older image,
register, action and signal tokens are inaccessible. CUDA uses block-sparse
FlexAttention; CPU/MPS use the same mask with ordinary attention. Per-layer
activation recomputation preserves gradients through the entire episode.

This is the uncompressed teacher. Memory still grows with context length;
recurrent compression is a later stage. action[t] produced latent[t].
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class LongContextDynamicsConfig:
    n_latents: int = 4
    latent_dim: int = 64
    n_actions: int = 17
    width: int = 128
    heads: int = 4
    depth: int = 8
    n_registers: int = 4
    mlp_ratio: float = 3.0
    max_temporal_length: int = 256
    signal_levels: int = 128
    inference_steps: int = 16


def visibility(length, tokens):
    """Queries are time-major; keys are all states, then time-major local tokens."""
    def allowed(batch, head, query, key):
        time = query // tokens
        past_state = (key < length) & (key <= time)
        same_frame = (key >= length) & ((key - length) // (tokens - 1) == time)
        return past_state | same_frame
    return allowed


def softcap(score, batch, head, query, key):
    return 50. * torch.tanh(score / 50.)


_flex_attention = None


class StateHistoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.q = nn.Linear(config.width, config.width)
        self.kv = nn.Linear(config.width, 2 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.logit_scale = nn.Parameter(torch.full((config.heads, 1, 1), math.log(4.)))

    def _heads(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    def _rope(self, x, positions):
        with torch.autocast(device_type=x.device.type, enabled=False):
            frequency = 10_000. ** (-2 * torch.arange(
                self.head_dim // 2, device=x.device, dtype=torch.float32) / self.head_dim)
            angles = positions.float()[..., None] * frequency
            cos, sin = angles.cos().unsqueeze(-3), angles.sin().unsqueeze(-3)
            first, second = x.float().chunk(2, dim=-1)
            rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
        return rotated.to(x.dtype)

    def forward(self, x, positions, mask, backend):
        batch, length, tokens, width = x.shape
        flat = x.flatten(1, 2)
        # Packing all state keys together lets the sparse kernel skip blocks of
        # invisible local tokens. It does not pool, truncate or detach history.
        keys = torch.cat((x[:, :, -1], x[:, :, :-1].flatten(1, 2)), 1)
        qpos = positions.repeat_interleave(tokens, 1)
        kpos = torch.cat((positions, positions.repeat_interleave(tokens - 1, 1)), 1)
        q = self._rope(self.q_norm(self._heads(self.q(flat))), qpos)
        k, v = self.kv(keys).chunk(2, -1)
        k = self._rope(self.k_norm(self._heads(k)), kpos)
        v = self._heads(v)
        scale = self.logit_scale.float().clamp(max=math.log(100.)).exp()[None]
        q = (q.float() * scale).to(q.dtype)
        if backend == 'flex':
            global _flex_attention
            if _flex_attention is None:
                from torch.nn.attention.flex_attention import flex_attention
                _flex_attention = torch.compile(flex_attention, dynamic=False)
            output = _flex_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                     score_mod=softcap, block_mask=mask)
        else:
            # Bound temporary score storage for local inspection. Splitting
            # queries changes neither their keys nor the attention distribution.
            parts = []
            with torch.autocast(device_type=x.device.type, enabled=False):
                for start in range(0, q.shape[2], 256):
                    scores = q[:, :, start:start + 256].float() @ k.float().transpose(-2, -1) / math.sqrt(self.head_dim)
                    allowed = mask[start:start + 256]
                    weights = softcap(scores, None, None, None, None).masked_fill(~allowed, -torch.inf).softmax(-1)
                    parts.append((weights @ v.float()).to(v.dtype))
                output = torch.cat(parts, 2)
        return self.out(output.transpose(1, 2).flatten(2)).reshape(batch, length, tokens, width)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(config.width * config.mlp_ratio)
        self.value = nn.Linear(config.width, hidden)
        self.gate = nn.Linear(config.width, hidden)
        self.out = nn.Linear(hidden, config.width)

    def forward(self, x):
        return self.out(self.value(x) * torch.nn.functional.silu(self.gate(x)))


class StateHistoryBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.width)
        self.attention = StateHistoryAttention(config)
        self.norm2 = nn.RMSNorm(config.width)
        self.mlp = SwiGLU(config)

    def forward(self, x, positions, mask, backend):
        x = x + self.attention(self.norm1(x), positions, mask, backend)
        return x + self.mlp(self.norm2(x))


class LongContextDynamicsModel(nn.Module):
    def __init__(self, config, *, backend='auto', recompute=True):
        super().__init__()
        if min(config.n_latents, config.latent_dim, config.n_actions, config.heads, config.depth) < 1:
            raise ValueError('Latent dimensions, actions, heads and depth must be positive')
        if config.width < 1 or config.width % (2 * config.heads):
            raise ValueError('Width must divide into heads with even head dimension')
        if config.max_temporal_length < 2 or config.n_registers < 0 or config.mlp_ratio <= 0:
            raise ValueError('Need context >= 2, nonnegative registers and positive MLP ratio')
        if config.signal_levels != 128 or not 1 <= config.inference_steps <= 127:
            raise ValueError('The research recipe uses 128 signal levels and 1..127 sampling steps')
        if backend not in ('auto', 'dense', 'flex'):
            raise ValueError('Attention backend must be auto, dense or flex')
        self.config, self.backend, self.recompute = config, backend, recompute
        self.n_latents, self.bottleneck_dim, self.n_actions = config.n_latents, config.latent_dim, config.n_actions
        self._mask_key, self._mask = None, None
        self.latent_in = nn.Linear(config.latent_dim, config.width)
        self.latent_slots = nn.Parameter(torch.randn(config.n_latents, config.width) * .02)
        self.registers = nn.Parameter(torch.randn(config.n_registers, config.width) * .02)
        self.action_embedding = nn.Embedding(config.n_actions, config.width)
        self.tau_embedding = nn.Embedding(config.signal_levels, config.width // 2)
        # Preserve the original backbone/checkpoint layout. Only the last row is
        # used: the step condition is constant, with no shortcut training loss.
        self.step_embedding = nn.Embedding(config.signal_levels.bit_length(), config.width // 2)
        self.state_token = nn.Parameter(torch.randn(1, config.width) * .02)
        self.blocks = nn.ModuleList([StateHistoryBlock(config) for _ in range(config.depth)])
        self.out_norm = nn.RMSNorm(config.width)
        self.latent_out = nn.Linear(config.width, config.latent_dim)

    def _tokens(self, latents, tau, actions):
        batch, length = latents.shape[:2]
        step = torch.full_like(tau, self.step_embedding.num_embeddings - 1)
        signal = torch.cat((self.tau_embedding(tau), self.step_embedding(step)), -1)
        return torch.cat((self.latent_in(latents) + self.latent_slots,
                          self.registers.expand(batch, length, -1, -1),
                          self.action_embedding(actions)[:, :, None], signal[:, :, None],
                          self.state_token.expand(batch, length, -1, -1)), 2)

    def forward(self, latents, tau, actions, *, return_cache=False):
        """(B,T,N,D) noisy latents + (B,T) signal/action ids -> clean latents."""
        batch, length = latents.shape[:2]
        if not 1 <= length <= self.config.max_temporal_length:
            raise ValueError('Input length must be within the configured context')
        if tau.shape != (batch, length) or actions.shape != (batch, length):
            raise ValueError('Supply one signal level and aligned action per frame')
        x = self._tokens(latents, tau, actions)
        tokens = x.shape[2]
        positions = torch.arange(1, length + 1, device=x.device)[None].expand(batch, -1)
        backend = ('flex' if x.is_cuda else 'dense') if self.backend == 'auto' else self.backend
        key = (length, tokens, str(x.device), backend)
        if key != self._mask_key:
            allowed = visibility(length, tokens)
            if backend == 'flex':
                if not x.is_cuda:
                    raise ValueError('FlexAttention requires CUDA here; use dense for CPU/MPS')
                from torch.nn.attention.flex_attention import create_block_mask
                self._mask = create_block_mask(allowed, None, None, length * tokens,
                                               length * tokens, device=x.device, _compile=True)
            else:
                q = torch.arange(length * tokens, device=x.device)[:, None]
                k = torch.arange(length * tokens, device=x.device)[None, :]
                self._mask = allowed(None, None, q, k)
            self._mask_key = key
        # Keep the mask local: a later forward must not change recomputation.
        mask = self._mask
        cache = []
        for block in self.blocks:
            if return_cache:
                att = block.attention
                k, v = att.kv(block.norm1(x[:, :, -1])).chunk(2, -1)
                cache.append((att._rope(att.k_norm(att._heads(k)), positions), att._heads(v)))
            if self.training and self.recompute:
                x = checkpoint(block, x, positions, mask, backend, use_reentrant=False)
            else:
                x = block(x, positions, mask, backend)
        prediction = self.latent_out(self.out_norm(x[:, :, :self.n_latents]))
        return (prediction, cache) if return_cache else prediction

    def _cached_frame(self, latent, tau, action, cache):
        """Read past state K/V and this frame; return a candidate updated cache.

        Denoising queries discard the candidate. Only a separate clean-frame
        pass is committed, so noisy queries never contaminate future history.
        """
        x = self._tokens(latent, tau, action)
        batch, _, tokens, width = x.shape
        past = cache[0][0].shape[2] if cache else 0
        positions = torch.full((batch, tokens), past + 1, device=x.device)
        updated = []
        for i, block in enumerate(self.blocks):
            att = block.attention
            norm = block.norm1(x[:, 0])
            q = att._rope(att.q_norm(att._heads(att.q(norm))), positions)
            k, v = att.kv(torch.cat((norm[:, -1:], norm[:, :-1]), 1)).chunk(2, -1)
            k = att._rope(att.k_norm(att._heads(k)), positions)
            v = att._heads(v)
            state_k, state_v = k[:, :, :1], v[:, :, :1]
            if cache:
                old_k, old_v = cache[i]
                state_k, state_v = torch.cat((old_k, state_k), 2), torch.cat((old_v, state_v), 2)
                k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
            updated.append((state_k, state_v))
            scale = att.logit_scale.float().clamp(max=math.log(100.)).exp()[None]
            q = (q.float() * scale).to(q.dtype)
            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(att.head_dim)
                weights = softcap(scores, None, None, None, None).softmax(-1)
                read = (weights @ v.float()).to(v.dtype)
            x = x + att.out(read.transpose(1, 2).flatten(2)).reshape(batch, 1, tokens, width)
            x = x + block.mlp(block.norm2(x))
        return self.latent_out(self.out_norm(x[:, :, :self.n_latents])), updated

    def loss(self, clean, actions, generator=None):
        """Research recipe: 50% clean, 25% noise, 25% intermediate per frame.

        Only corrupted frames contribute to MSE; clean frames provide context.
        There is one loss, no auxiliary or shortcut objective. FP32 loss even
        when the forward runs under BF16 autocast.
        """
        shape, device = clean.shape[:2], clean.device
        mixture = torch.rand(shape, device=device, generator=generator)
        middle = torch.randint(1, 127, shape, device=device, generator=generator)
        tau = torch.where(mixture < .5, 127, torch.where(mixture < .75, 0, middle))
        noise = torch.randn(clean.shape, device=device, generator=generator)
        amount = tau.float()[..., None, None] / 127
        prediction = self(amount * clean + (1 - amount) * noise, tau, actions)
        errors = (prediction.float() - clean.float()).square().mean((-1, -2))
        corrupted = tau < 127
        # Empty corruption is possible for a tiny smoke batch; return zero with
        # a valid gradient rather than NaN. Ordinary long batches have ~50% scored frames.
        return (errors * corrupted).sum() / corrupted.sum().clamp_min(1)

    @torch.no_grad()
    def generate(self, context, n_generate, K=None, action_idx=None):
        """Euler sampling with clean history, bounded by the trained context.

        K=1 reproduces the research noise-to-clean query; K=16 is the repo's
        inspection default. No hidden state survives outside the retained history.
        Clean-history state K/V are reused across denoising steps. The cache is
        rebuilt from retained frames whenever the window slides.
        """
        K = self.config.inference_steps if K is None else K
        if not isinstance(K, int) or not 1 <= K <= 127 or n_generate < 1:
            raise ValueError('Need 1..127 denoising steps and positive generation length')
        batch, prefix = context.shape[:2]
        if action_idx is None or action_idx.shape != (batch, prefix + n_generate):
            raise ValueError('Supply actions for the full prefix and generated sequence')
        schedule = [round(k * 127 / K) for k in range(K + 1)]
        sequence, generated, cache = context, [], []
        for i in range(n_generate):
            history = sequence[:, -(self.config.max_temporal_length - 1):]
            end = prefix + i
            actions = action_idx[:, end - history.shape[1]:end + 1]
            cached_length = cache[0][0].shape[2] if cache else 0
            if cached_length != history.shape[1]:
                # Rebuild after sliding the window. Merely dropping old K/V
                # would relay information from frames outside the allowed history.
                _, cache = self(history, torch.full_like(actions[:, :-1], 127),
                                actions[:, :-1], return_cache=True)
            action = actions[:, -1:]
            tau = torch.zeros_like(action)
            z = torch.randn((batch, 1, self.n_latents, self.bottleneck_dim),
                            device=context.device, dtype=context.dtype)
            for start, stop in zip(schedule[:-1], schedule[1:]):
                tau[:, -1] = start
                clean, _ = self._cached_frame(z, tau, action, cache)
                z = z + (clean - z) * ((stop - start) / (127 - start))
            generated.append(z)
            sequence = torch.cat((sequence, z), 1)
            if i + 1 < n_generate:
                _, cache = self._cached_frame(z, torch.full_like(action, 127), action, cache)
        return torch.cat(generated, 1)
