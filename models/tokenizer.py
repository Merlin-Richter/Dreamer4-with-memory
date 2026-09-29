import math

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AutoEncoderConfig():
    
    dtype:torch.dtype = torch.bfloat16

    embedding_dim: int = 256
    max_temporal_length: int = 16

    n_heads: int = 16
    mlp_ratio: float = 3.0
    depth: int = 3
    temporal_layout: str = 'sts'  # 'sts': spatial/temporal/spatial; 'ssst': three spatial blocks then temporal

    drop_rate: int = 0.1
    att_drop_rate: int = 0.1
    att_logit_soft_cap: float = 50

    patch_size: int = 8
    img_input_H: int = 64
    img_input_W: int = 64

    n_latents: int = 4
    bottleneck_dim: int = 64
    output_sigmoid: bool = True

    # MAE patch-dropout range (per-image p ~ U[min, max]); applied only in train mode.
    mae_min_mask: float = 0.0
    mae_max_mask: float = 0.9
    


class Attention(nn.Module):
    def __init__(self, config: AutoEncoderConfig, is_encoder, is_temporal):
        super().__init__()
        # input is of shape (Batch, n_tokens, Token_dim)
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim / self.n_heads
        self.n_latents = config.n_latents

        assert int(self.head_dim) == self.head_dim
        self.head_dim = int(self.head_dim)

        self.base_scale = 1/(self.head_dim ** 0.5)
        self.logit_scale = nn.Parameter(torch.full((self.n_heads, 1, 1, 1, 1), math.log(4.0)))
        self.max_logit_scale = math.log(100.0)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.soft_cap_act = nn.Tanh()
        self.soft_cap = config.att_logit_soft_cap

        self.is_encoder = is_encoder
        self.is_temporal = is_temporal
        self.dropout = nn.Dropout(config.drop_rate)
        self.att_droput = nn.Dropout(config.att_drop_rate)
        self.qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)

        # RoPE
        d_half = self.head_dim // 2
        
        cos = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / self.head_dim)
        sin = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / self.head_dim)

        cos = torch.outer(torch.arange(config.max_temporal_length, dtype=torch.float32), cos)
        sin = torch.outer(torch.arange(config.max_temporal_length, dtype=torch.float32), sin)
        cos = torch.cos(cos)
        sin = torch.sin(sin)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)


    
    def forward(self, x: torch.Tensor):
        # input is of shape (Batch, n_tokens, Token_dim)
        B, T, N, C = x.shape

        qkv: torch.Tensor = self.qkv(x)
        # shape (Batch, Time, n_tokens, 3*Token_dim)
        # we want it in (Batch, Time, n_tokens, 3, heads, rest)
        qkv = qkv.reshape((B, T, N, 3, self.n_heads, -1))
        if not self.is_temporal:
            qkv = qkv.permute((3, 4, 0, 1, 2, 5))
            # shape1: (heads, B, T, N, head_dim)
        else:
            qkv = qkv.permute((3, 4, 0, 2, 1, 5))
            # shape2: (heads, B, N, T, head_dim)

        
        q, k, v = qkv[0], qkv[1], qkv[2]
        

        q, k = self.q_norm(q), self.k_norm(k)

        if not self.is_temporal:
            mask = torch.zeros((N, N), device=x.device).bool()

            if self.is_encoder: mask[:-self.n_latents, -self.n_latents:] = True
            else:               mask[-self.n_latents:, :-self.n_latents] = True

        else:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

            # RoPE
            q_first, q_second = q[:, :, :, :, :self.head_dim // 2], q[:, :, :, :, self.head_dim // 2:]
            k_first, k_second = k[:, :, :, :, :self.head_dim // 2], k[:, :, :, :, self.head_dim // 2:]

            q = torch.concat((q_first * self.cos[:T] + -q_second * self.sin[:T], q_second * self.cos[:T] + q_first * self.sin[:T]), dim=-1)
            k = torch.concat((k_first * self.cos[:T] + -k_second * self.sin[:T], k_second * self.cos[:T] + k_first * self.sin[:T]), dim=-1)

        scale = self.base_scale * torch.exp(self.logit_scale.clamp(max=self.max_logit_scale))
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)

        attn = self.att_droput(attn)
        # shape1: (heads, B, T, N, N)
        # shape2: (heads, B, N, T, T)

        x = attn @ v
        # shape1: (heads, B, T, N, head_dim)
        # shape2: (heads, B, N, T, head_dim)

        # zurück zu (B, T, N, C)
        if not self.is_temporal:
            x = x.permute(1, 2, 3, 0, 4).reshape(B, T, N, C)
        else:
            x = x.permute(1, 3, 2, 0, 4).reshape(B, T, N, C)

        return self.dropout(self.proj(x))
    


class SwiGLU(nn.Module):
    def __init__(self, config: AutoEncoderConfig) -> None:
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
    def __init__(self, config: AutoEncoderConfig, is_encoder, is_temporal) -> None:
        super().__init__()
        self.att = Attention(config, is_encoder, is_temporal)
        self.ln1 = nn.RMSNorm(config.embedding_dim)
        self.mlp = SwiGLU(config)
        self.ln2 = nn.RMSNorm(config.embedding_dim)


    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig) -> None:
        super().__init__()

        assert not (config.img_input_H % config.patch_size) and not (config.img_input_W % config.patch_size)

        self.patch_size = config.patch_size
        self.n_patches = (config.img_input_H // config.patch_size) * (config.img_input_W // config.patch_size)

        self.mae_min_mask = config.mae_min_mask
        self.mae_max_mask = config.mae_max_mask

        self.patch_proj = nn.Linear(config.patch_size * config.patch_size * 3, config.embedding_dim)
        self.learned_position_embedding = nn.Parameter(0.05*torch.rand((self.n_patches, config.embedding_dim), dtype=config.dtype))
        self.learned_dropped_patch_replacement_token = nn.Parameter(0.05*torch.rand((config.embedding_dim,), dtype=config.dtype))

        self.learned_latents = nn.Parameter(0.05*torch.rand((config.n_latents, config.embedding_dim), dtype=config.dtype))
        self.n_latents = config.n_latents

        self.blocks = nn.ModuleList([TransformerBlock(config, is_encoder=True,
            is_temporal=(i % 3 == 1) if config.temporal_layout == 'sts' else ((i + 1) % 4 == 0))
            for i in range(config.depth)])

        self.norm = nn.RMSNorm(config.embedding_dim)
        self.bottleneck_proj = nn.Linear(config.embedding_dim, config.bottleneck_dim)
        self.act = nn.Tanh()
    
    
    def patchify(self, x):
        B, T, H, W, C = x.shape

        # (B, T, H, W, C) → (B, T, C, H, W) for easier patching
        x = x.permute((0, 1, 4, 2, 3))

        # (B, T, C, H, W) → (B, T, C, n_h, patch_size, n_w, patch_size)
        x = x.reshape((B, T, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size))

        # → (B, T, n_h, n_w, patch_size, patch_size, C)
        x = x.permute((0, 1, 3, 5, 4, 6, 2))

        # → (B, T, n_patches, patch_size*patch_size*C)
        return x.reshape((B, T, self.n_patches, self.patch_size * self.patch_size * C))


    def forward(self, x):
        B, T, H, W, C = x.shape
        x = self.patchify(x)

        x = self.patch_proj(x)
        if self.training:
            mask_chances = torch.rand((B, T, 1), device=x.device) * (self.mae_max_mask - self.mae_min_mask) + self.mae_min_mask
            mask = torch.rand((B, T, self.n_patches), device=x.device) < mask_chances
            x[mask] = self.learned_dropped_patch_replacement_token.to(x.dtype)
        x = x + self.learned_position_embedding


        latents = self.learned_latents.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        x = torch.concat((x, latents), dim=2)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x[:, :, -self.n_latents:, :])

        return self.act(self.bottleneck_proj(x))



class Decoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig) -> None:
        super().__init__()
        assert not (config.img_input_H % config.patch_size) and not (config.img_input_W % config.patch_size)

        self.patch_size = config.patch_size
        self.h_patches = config.img_input_H // config.patch_size
        self.w_patches = config.img_input_W // config.patch_size
        self.n_patches = (config.img_input_H // config.patch_size) * (config.img_input_W // config.patch_size)

        self.output_sigmoid = config.output_sigmoid
        self.from_bottleneck_proj = nn.Linear(config.bottleneck_dim, config.embedding_dim)
        self.learned_patch_tokens = nn.Parameter(0.05*torch.rand((self.n_patches, config.embedding_dim), dtype=config.dtype))
        
        self.blocks = nn.ModuleList([TransformerBlock(config, is_encoder=False,
            is_temporal=(i % 3 == 1) if config.temporal_layout == 'sts' else ((i + 1) % 4 == 0))
            for i in range(config.depth)])

        self.norm = nn.RMSNorm(config.embedding_dim)
        self.patch_token_to_img_patch_proj = nn.Linear(config.embedding_dim, config.patch_size*config.patch_size*3)


    def imagify_patches(self, x):
        # x.shape = (B, self.n_patches, self.patch_size * self.patch_size * C)
        B, T, N, d = x.shape

        # (B, n_H, n_W, 16, 16, 3)
        x = x.reshape((B, T, self.h_patches, self.w_patches, self.patch_size, self.patch_size, 3))

        # (B, n_H, 16, n_W, 16, 3)
        x = x.permute((0, 1, 2, 4, 3, 5, 6))
        
        x = x.reshape((B, T, self.h_patches*self.patch_size, self.w_patches*self.patch_size, 3))

        return x


    def forward(self, x):
        # Input are just the low dimensional latent tokens
        B, T, N, d = x.shape

        x = self.from_bottleneck_proj(x)

        learned_patch_tokens = self.learned_patch_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        x = torch.concat((learned_patch_tokens, x), dim=2)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x[:, :, :self.n_patches, :])

        x = self.patch_token_to_img_patch_proj(x)
        if self.output_sigmoid:
            x = torch.sigmoid(x)
        x = self.imagify_patches(x)

        return x



class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

