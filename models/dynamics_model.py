"""
Dreamer 4 interactive dynamics model (Section 3.2).

The dynamics model operates on the sequence of continuous representations produced by the
frozen tokenizer. For every frame it consumes
``n_latents`` latent tokens of size ``bottleneck_dim`` and predicts the *clean* latents of
that frame from a noised version of it, using its causal history as context. It is trained
with a **shortcut forcing** objective: diffusion forcing (a per-frame signal level) combined
with shortcut models (conditioning on the requested step size ``d``) so that frames can be
sampled with as few as ``K = 4`` forward passes at inference.

Key choices from the paper:
  * x-prediction: the network predicts the clean latents ``z1`` (not the velocity), rather than predicting velocity.
  * x-space flow loss at the finest step ``d_min = 1/max_sampling_steps``; bootstrap loss that
    distills two ``d/2`` steps for the larger step sizes (Eq. 7).
  * ramp loss weight ``w(tau) = 0.9*tau + 0.1`` to focus capacity on the high-signal levels.
  * signal level ``tau`` and step size ``d`` are discrete, encoded with embedding lookups whose
    channels are concatenated into a single shortcut token per frame.

For unlabeled video the action modality is just a learned embedding (the "unlabeled video"
case in the paper). For labeled data, set ``n_actions`` > 0: an embedding table maps each
discrete action id to a per-frame feature added to that learned embedding. ``loss`` and
``generate`` accept per-frame ``action_idx`` (the occluded-bouncing env emits curtain-state
ids); ``forward`` takes the pre-computed action features.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DynamicsModelConfig():

    dtype: torch.dtype = torch.bfloat16

    # Must match the frozen tokenizer that produces the representations.
    bottleneck_dim: int = 64
    n_latents: int = 4

    embedding_dim: int = 256
    max_temporal_length: int = 16

    n_heads: int = 16
    mlp_ratio: float = 3.0
    depth: int = 8

    drop_rate: float = 0.1
    att_drop_rate: float = 0.1
    att_logit_soft_cap: float = 50

    n_action_tokens: int = 1
    n_registers: int = 4

    # Discrete action conditioning. 0 => unlabeled video (only the learned action embedding is
    # used). >0 => a lookup table maps each discrete action id to a per-frame action feature.
    n_actions: int = 0

    # Shortcut forcing schedule.
    max_sampling_steps: int = 128   # K_max; finest step d_min = 1/K_max. Must be a power of two.
    inference_steps: int = 4        # K used per frame at generation time (d = 1/K).
    context_signal: float = 0.9     # Context signal level during rollout: 1 = clean, 0 = noise.
    ramp_min: float = 0.1           # w(tau) = (1 - ramp_min) * tau + ramp_min.


class Attention(nn.Module):
    """Block-causal attention. Space layers attend fully within a frame; temporal layers
    attend causally across time and use RoPE on the time axis."""

    def __init__(self, config: DynamicsModelConfig, is_temporal):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim / self.n_heads

        assert int(self.head_dim) == self.head_dim
        self.head_dim = int(self.head_dim)

        self.scale = 1 / (self.head_dim ** 0.5)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.soft_cap_act = nn.Tanh()
        self.soft_cap = config.att_logit_soft_cap

        self.is_temporal = is_temporal
        self.dropout = nn.Dropout(config.drop_rate)
        self.att_droput = nn.Dropout(config.att_drop_rate)
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)

        # RoPE tables over the temporal axis.
        d_half = self.head_dim // 2
        freqs = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / self.head_dim)
        angles = torch.outer(torch.arange(config.max_temporal_length, dtype=torch.float32), freqs)
        self.register_buffer('cos', torch.cos(angles))
        self.register_buffer('sin', torch.sin(angles))

    def forward(self, x: torch.Tensor):
        # x: (B, T, N, C)
        B, T, N, C = x.shape

        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.reshape((B, T, N, 3, self.n_heads, -1))
        if not self.is_temporal:
            qkv = qkv.permute((3, 4, 0, 1, 2, 5))  # (3, heads, B, T, N, head_dim)
        else:
            qkv = qkv.permute((3, 4, 0, 2, 1, 5))  # (3, heads, B, N, T, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        if not self.is_temporal:
            # Full self-attention within a frame: latents, registers, actions and the
            # shortcut token all exchange information.
            mask = None
        else:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

            # RoPE on the time axis.
            q_first, q_second = q[..., :self.head_dim // 2], q[..., self.head_dim // 2:]
            k_first, k_second = k[..., :self.head_dim // 2], k[..., self.head_dim // 2:]
            cos, sin = self.cos[:T], self.sin[:T]
            q = torch.concat((q_first * cos + -q_second * sin, q_second * cos + q_first * sin), dim=-1)
            k = torch.concat((k_first * cos + -k_second * sin, k_second * cos + k_first * sin), dim=-1)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.att_droput(attn)

        x = attn @ v
        if not self.is_temporal:
            x = x.permute(1, 2, 3, 0, 4).reshape(B, T, N, C)
        else:
            x = x.permute(1, 3, 2, 0, 4).reshape(B, T, N, C)

        return self.dropout(self.proj(x))


class SwiGLU(nn.Module):
    def __init__(self, config: DynamicsModelConfig) -> None:
        super().__init__()
        self.ll1_1 = nn.Linear(config.embedding_dim, int(config.embedding_dim * config.mlp_ratio))
        self.ll1_2 = nn.Linear(config.embedding_dim, int(config.embedding_dim * config.mlp_ratio))
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.drop_rate)
        self.ll2 = nn.Linear(int(config.embedding_dim * config.mlp_ratio), config.embedding_dim)

    def forward(self, x):
        x = self.ll1_1(x) * self.act(self.ll1_2(x))
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: DynamicsModelConfig, is_temporal) -> None:
        super().__init__()
        self.att = Attention(config, is_temporal)
        self.ln1 = nn.RMSNorm(config.embedding_dim)
        self.mlp = SwiGLU(config)
        self.ln2 = nn.RMSNorm(config.embedding_dim)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DynamicsModel(nn.Module):
    """
    Block-causal transformer that denoises per-frame latents via shortcut forcing.

    Token layout per frame (along the spatial axis): ``[action tokens | latent tokens |
    register tokens | shortcut token]``. Only the latent-token outputs are read out as the
    x-prediction of the clean representation.
    """

    def __init__(self, config: DynamicsModelConfig) -> None:
        super().__init__()
        self.config = config
        E = config.embedding_dim
        self.n_latents = config.n_latents
        self.bottleneck_dim = config.bottleneck_dim
        self.n_action_tokens = config.n_action_tokens
        self.n_registers = config.n_registers

        self.K_max = config.max_sampling_steps
        assert (self.K_max & (self.K_max - 1)) == 0, "max_sampling_steps must be a power of two"
        self.n_d = self.K_max.bit_length()  # number of distinct step sizes: K in {1,2,...,K_max}

        # Latent <-> model-dim projections. Latents are read out for the x-prediction.
        self.in_proj = nn.Linear(config.bottleneck_dim, E)
        self.out_norm = nn.RMSNorm(E)
        self.out_proj = nn.Linear(E, config.bottleneck_dim)

        # Learned tokens. Kept in float32 (default) so the model also runs without autocast.
        self.action_embedding = nn.Parameter(0.05 * torch.rand((config.n_action_tokens, E)))
        self.register_tokens = nn.Parameter(0.05 * torch.rand((config.n_registers, E)))

        # Discrete action conditioning: map each action id to a per-frame feature that is added
        # to the learned action embedding. Absent (n_actions == 0) => unlabeled video.
        self.n_actions = config.n_actions
        if config.n_actions > 0:
            self.action_table = nn.Embedding(config.n_actions, config.n_action_tokens * E)

        # Discrete shortcut conditioning: tau and d each get half the channels, concatenated.
        assert E % 2 == 0
        self.tau_embedding = nn.Embedding(self.K_max, E // 2)
        self.d_embedding = nn.Embedding(self.n_d, E // 2)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, is_temporal=((i + 1) % 4 == 0))
            for i in range(config.depth)
        ])

    # ------------------------------------------------------------------ helpers
    def _d_value(self, d_idx: torch.Tensor) -> torch.Tensor:
        # d = 1 / K, with K = 2 ** d_idx.
        return torch.pow(2.0, -d_idx.float())

    def _tau_value(self, tau_idx: torch.Tensor) -> torch.Tensor:
        # tau grid points are integer multiples of d_min = 1 / K_max.
        return tau_idx.float() / self.K_max

    def sample_tau_d(self, B: int, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-frame (tau_idx, d_idx) following Eq. 4.

        d ~ 1/U({1, 2, ..., K_max})  ->  d_idx = log2(K) ~ U({0, ..., n_d-1}).
        tau ~ U({0, d, 2d, ..., 1-d}) on the grid implied by the sampled d.
        """
        d_idx = torch.randint(0, self.n_d, (B, T), device=device)
        K = torch.pow(2, d_idx)                               # steps for this frame
        step = (torch.rand((B, T), device=device) * K).long()  # uniform in {0, ..., K-1}
        step = torch.minimum(step, K - 1)
        # tau_idx = step * (K_max / K), i.e. snap to the d_min grid.
        tau_idx = step * torch.pow(2, self.n_d - 1 - d_idx)
        return tau_idx, d_idx

    def action_features(self, action_idx: torch.Tensor) -> torch.Tensor:
        """(B, T) long action ids -> (B, T, n_action_tokens, E) feature tokens.

        Returns ``None`` when the model is unlabeled or no ids are given, so callers can pass the
        result straight through to ``forward`` (which then uses only the learned embedding).
        """
        if action_idx is None or self.n_actions == 0:
            return None
        B, T = action_idx.shape
        return self.action_table(action_idx).reshape(B, T, self.n_action_tokens, -1)

    # ------------------------------------------------------------------ forward
    def forward(self, z_tilde: torch.Tensor, tau_idx: torch.Tensor, d_idx: torch.Tensor,
                actions: torch.Tensor = None) -> torch.Tensor:
        """Predict clean latents (x-prediction) from noised latents.

        z_tilde:  (B, T, n_latents, bottleneck_dim) noised representations.
        tau_idx:  (B, T) long, discrete signal level per frame.
        d_idx:    (B, T) long, discrete step size per frame.
        actions:  optional (B, T, n_action_tokens, E) action features to add to the learned
                  action embedding. ``None`` => unlabeled video.
        returns:  (B, T, n_latents, bottleneck_dim) predicted clean latents.
        """
        B, T, L, _ = z_tilde.shape

        lat = self.in_proj(z_tilde)  # (B, T, L, E)

        shortcut = torch.concat(
            (self.tau_embedding(tau_idx), self.d_embedding(d_idx)), dim=-1
        ).unsqueeze(2)  # (B, T, 1, E)

        action = self.action_embedding.expand(B, T, -1, -1)
        if actions is not None:
            action = action + actions
        register = self.register_tokens.expand(B, T, -1, -1)

        x = torch.concat((action, lat, register, shortcut), dim=2)  # (B, T, N, E)

        for block in self.blocks:
            x = block(x)

        lat_start = self.n_action_tokens
        x = x[:, :, lat_start:lat_start + L, :]
        x = self.out_norm(x)
        return self.out_proj(x)

    # ------------------------------------------------------------------ loss
    def loss(self, z1: torch.Tensor, action_idx: torch.Tensor = None) -> torch.Tensor:
        """Shortcut forcing loss over a clip of clean representations.

        z1:         (B, T, n_latents, bottleneck_dim) clean tokenizer latents.
        action_idx: optional (B, T) long discrete action ids, aligned per frame.
        """
        B, T, L, _ = z1.shape
        device = z1.device
        actions = self.action_features(action_idx)

        tau_idx, d_idx = self.sample_tau_d(B, T, device)
        tau = self._tau_value(tau_idx)[..., None, None]  # (B, T, 1, 1)
        d = self._d_value(d_idx)[..., None, None]

        z0 = torch.randn_like(z1)
        z_tilde = (1 - tau) * z0 + tau * z1

        z_hat1 = self(z_tilde, tau_idx, d_idx, actions)  # x-prediction (with gradient)

        is_min = (d_idx == self.n_d - 1)[..., None, None]  # finest step => pure flow loss

        # --- bootstrap targets: distill two d/2 steps (Eq. 7), stop-gradient. ---
        with torch.no_grad():
            half_d_idx = (d_idx + 1).clamp(max=self.n_d - 1)
            half_d = self._d_value(half_d_idx)[..., None, None]
            # tau + d/2 on the d_min grid; clamp keeps masked (is_min) entries in range.
            tau_inc = torch.pow(2, (self.n_d - 2 - d_idx).clamp(min=0))
            tau2_idx = (tau_idx + tau_inc).clamp(max=self.K_max - 1)
            tau2 = self._tau_value(tau2_idx)[..., None, None]

            y1 = self(z_tilde, tau_idx, half_d_idx, actions)
            b1 = (y1 - z_tilde) / (1 - tau)
            z_prime = z_tilde + b1 * half_d

            y2 = self(z_prime, tau2_idx, half_d_idx, actions)
            b2 = (y2 - z_prime) / (1 - tau2)

            v_target = (b1 + b2) / 2

        flow_loss = (z_hat1 - z1) ** 2
        v_pred = (z_hat1 - z_tilde) / (1 - tau)
        boot_loss = (1 - tau) ** 2 * (v_pred - v_target) ** 2

        per_token = torch.where(is_min, flow_loss, boot_loss)

        w = (1 - self.config.ramp_min) * tau + self.config.ramp_min  # ramp weight, Eq. 8
        return (w * per_token).mean()

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def _denoise_next(self, context: torch.Tensor, K: int,
                      actions: torch.Tensor = None) -> torch.Tensor:
        """Generate one clean frame conditioned on ``context`` (clean latents), using K
        shortcut steps. ``context`` is held at signal level tau_ctx (=context_signal,
        near 1.0 = near-clean).

        actions: optional (B, T_ctx + 1, n_action_tokens, E) action features covering the
                 context frames and the new frame being generated.
        """
        B, T_ctx, L, _ = context.shape
        device = context.device
        d_val = 1.0 / K
        d_idx_val = (K).bit_length() - 1  # log2(K)

        tau_ctx = self.config.context_signal
        tau_ctx_idx = round(tau_ctx * self.K_max)
        # Hold the (clean) context at signal level tau_ctx for this frame's sampling:
        # ctx = tau_ctx * context + (1 - tau_ctx) * noise. High tau_ctx => near-clean.
        ctx_noised = (1 - tau_ctx) * torch.randn_like(context) + tau_ctx * context

        d_col = torch.full((B, T_ctx + 1), d_idx_val, device=device, dtype=torch.long)
        tau_col = torch.full((B, T_ctx + 1), tau_ctx_idx, device=device, dtype=torch.long)

        z = torch.randn((B, 1, L, self.bottleneck_dim), device=device)  # pure noise, tau = 0
        for k in range(K):
            tau = k / K
            tau_col[:, -1] = round(tau * self.K_max)
            inp = torch.concat((ctx_noised, z), dim=1)
            z_hat1 = self(inp, tau_col, d_col, actions)[:, -1:]  # x-prediction for the new frame
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * d_val
        return z  # (B, 1, L, bottleneck_dim)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, n_generate: int, K: int = None,
                 action_idx: torch.Tensor = None) -> torch.Tensor:
        """Autoregressively roll out ``n_generate`` frames after ``context``.

        context:    (B, T_ctx, n_latents, bottleneck_dim) clean latents from the tokenizer.
        action_idx: optional (B, T_ctx + n_generate) long action ids for every frame in the
                    context and the frames to generate (required when the model is action
                    conditioned, so the curtain state of each generated frame is known).
        returns:    (B, n_generate, n_latents, bottleneck_dim) generated clean latents.
        """
        K = K or self.config.inference_steps
        max_ctx = self.config.max_temporal_length - 1
        T_ctx = context.shape[1]
        act_feat = self.action_features(action_idx)  # (B, T_total, n_act, E) or None
        seq = context
        generated = []
        for i in range(n_generate):
            window = seq[:, -max_ctx:]
            w = window.shape[1]
            new_idx = T_ctx + i               # absolute index of the frame being generated
            act_window = None
            if act_feat is not None:
                # action features for [window frames ... new frame], matching _denoise_next input
                act_window = act_feat[:, new_idx - w : new_idx + 1]
            nxt = self._denoise_next(window, K, act_window)
            generated.append(nxt)
            seq = torch.concat((seq, nxt), dim=1)
        return torch.concat(generated, dim=1)
