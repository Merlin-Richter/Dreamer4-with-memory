"""Vanilla causal dynamics on frozen tokenizer latents.

Dreamer 4-style spatial/temporal transformer, trained with ordinary flow matching
in clean-latent (x-prediction) space. Each frame gets its own signal level
(diffusion forcing). There are no shortcut targets or step-size embeddings.

Inputs use the saved dataset convention: action[t] produced latent[t]. Temporal
attention is causal; each prediction sees only its current noisy frame and past.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DynamicsModelConfig():

    dtype: torch.dtype = torch.float32

    # Must match the frozen tokenizer that produces the representations.
    bottleneck_dim: int = 64
    n_latents: int = 4

    embedding_dim: int = 256
    max_temporal_length: int = 16

    n_heads: int = 16
    mlp_ratio: float = 3.0
    depth: int = 8

    drop_rate: float = 0.0
    att_drop_rate: float = 0.0
    att_logit_soft_cap: float = 50

    n_action_tokens: int = 1
    n_registers: int = 4

    # Discrete action conditioning. 0 => unlabeled video (only the learned action embedding is
    # used). >0 => a lookup table maps each discrete action id to a per-frame action feature.
    n_actions: int = 0

    # Signal grid: tau in {0, 1/N, ..., (N-1)/N}. Sampling uses this same grid.
    signal_levels: int = 128
    inference_steps: int = 64
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
            # signal-level token all exchange information.
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
    Block-causal transformer that predicts clean latents from independently noised frames.

    Token layout per frame (along the spatial axis): ``[action tokens | latent tokens |
    register tokens | signal-level token]``. Only the latent-token outputs are read out as the
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

        if config.depth < 4 or config.max_temporal_length < 2:
            raise ValueError('Need depth >= 4 for temporal attention and context length >= 2')
        if E < 1 or config.n_heads < 1 or E % (2 * config.n_heads):
            raise ValueError('Width must be divisible by twice the head count for RoPE')
        if config.signal_levels < 2 or config.inference_steps < 1 or config.signal_levels % config.inference_steps:
            raise ValueError('Inference steps must be a positive divisor of signal_levels')
        if not 0 <= config.context_signal < 1 or not 0 < config.ramp_min <= 1:
            raise ValueError('Need context_signal in [0,1) and ramp_min in (0,1]')

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

        self.tau_embedding = nn.Embedding(config.signal_levels, E)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, is_temporal=((i + 1) % 4 == 0))
            for i in range(config.depth)
        ])

    def sample_tau(self, B, T, device, generator=None):
        """Uniform, independent signal levels for each frame in each clip."""
        return torch.randint(self.config.signal_levels, (B, T), device=device, generator=generator)

    def _tau_value(self, tau_idx):
        return tau_idx.float() / self.config.signal_levels

    def action_features(self, action_idx: torch.Tensor) -> torch.Tensor:
        """(B, T) long action ids -> (B, T, n_action_tokens, E) feature tokens.

        Returns None for an unlabeled model. Action-conditioned models require ids.
        """
        if self.n_actions == 0:
            return None
        if action_idx is None:
            raise ValueError('This model requires aligned action ids')
        B, T = action_idx.shape
        return self.action_table(action_idx).reshape(B, T, self.n_action_tokens, -1)

    # ------------------------------------------------------------------ forward
    def forward(self, z_tilde: torch.Tensor, tau_idx: torch.Tensor,
                actions: torch.Tensor = None) -> torch.Tensor:
        """Predict clean latents (x-prediction) from noised latents.

        z_tilde:  (B, T, n_latents, bottleneck_dim) noised representations.
        tau_idx:  (B, T) long, discrete signal level per frame.
        actions:  optional (B, T, n_action_tokens, E) action features to add to the learned
                  action embedding. ``None`` => unlabeled video.
        returns:  (B, T, n_latents, bottleneck_dim) predicted clean latents.
        """
        B, T, L, _ = z_tilde.shape
        if T > self.config.max_temporal_length:
            raise ValueError('Input exceeds the model context length')
        if self.n_actions and actions is None:
            raise ValueError('This model requires action features')

        lat = self.in_proj(z_tilde)  # (B, T, L, E)

        signal = self.tau_embedding(tau_idx).unsqueeze(2)  # (B, T, 1, E)

        action = self.action_embedding.expand(B, T, -1, -1)
        if actions is not None:
            action = action + actions
        register = self.register_tokens.expand(B, T, -1, -1)

        x = torch.concat((action, lat, register, signal), dim=2)  # (B, T, N, E)

        for block in self.blocks:
            x = block(x)

        lat_start = self.n_action_tokens
        x = x[:, :, lat_start:lat_start + L, :]
        x = self.out_norm(x)
        return self.out_proj(x)

    def loss(self, z1: torch.Tensor, action_idx: torch.Tensor = None, generator=None):
        """Ramp-weighted clean-latent MSE; one forward pass per update.

        z1: (B,T,N,D) clean latents; action_idx: (B,T) aligned action ids.
        An optional generator makes validation noise repeatable without consuming
        the training RNG. This is a denoising objective, not rollout accuracy.
        """
        B, T, _, _ = z1.shape
        tau_idx = self.sample_tau(B, T, z1.device, generator)
        tau = self._tau_value(tau_idx)[..., None, None]
        noise = torch.randn(z1.shape, device=z1.device, dtype=z1.dtype, generator=generator)
        noisy = (1 - tau) * noise + tau * z1
        prediction = self(noisy, tau_idx, self.action_features(action_idx))
        weight = (1 - self.config.ramp_min) * tau + self.config.ramp_min
        return (weight * (prediction - z1).square()).mean()

    @torch.no_grad()
    def _denoise_next(self, context, K, actions=None):
        """Euler integration from noise to a new frame, holding noisy history fixed."""
        B, T, L, _ = context.shape
        levels = self.config.signal_levels
        ctx_idx = min(round(self.config.context_signal * levels), levels - 1)
        ctx_tau = ctx_idx / levels
        history = ctx_tau * context + (1 - ctx_tau) * torch.randn_like(context)
        tau_idx = torch.full((B, T + 1), ctx_idx, device=context.device, dtype=torch.long)
        z = torch.randn((B, 1, L, self.bottleneck_dim), device=context.device, dtype=context.dtype)
        for k in range(K):
            tau_idx[:, -1] = k * (levels // K)
            tau = k / K
            clean = self(torch.cat((history, z), dim=1), tau_idx, actions)[:, -1:]
            velocity = (clean - z) / (1 - tau)
            z = z + velocity / K
        return z

    @torch.no_grad()
    def generate(self, context, n_generate, K=None, action_idx=None):
        """Predict future latents from a prefix and the supplied future actions.

        context: (B,T,N,D); action_idx: (B,T+n_generate), including prefix actions.
        K is the number of Euler steps per frame; it must divide signal_levels.
        At most max_temporal_length-1 past frames condition each new frame.
        """
        K = self.config.inference_steps if K is None else K
        if not isinstance(K, int) or K < 1 or self.config.signal_levels % K:
            raise ValueError('K must be a positive divisor of signal_levels')
        if n_generate < 1:
            raise ValueError('n_generate must be positive')
        B, T = context.shape[:2]
        if self.n_actions and (action_idx is None or action_idx.shape != (B, T + n_generate)):
            raise ValueError('Supply actions for the full prefix and generated sequence')
        features = self.action_features(action_idx)
        sequence = context
        generated = []
        for i in range(n_generate):
            window = sequence[:, -(self.config.max_temporal_length - 1):]
            end = T + i
            actions = None if features is None else features[:, end - window.shape[1]:end + 1]
            next_frame = self._denoise_next(window, K, actions)
            generated.append(next_frame)
            sequence = torch.cat((sequence, next_frame), dim=1)
        return torch.cat(generated, dim=1)
