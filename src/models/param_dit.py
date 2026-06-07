"""DiT-style transformer denoiser for flat INR parameter vectors."""

import math

import torch
import torch.nn as nn

from src.models.hyperdiff_tokenizer import (
    HyperDiffDetokenizer,
    HyperDiffTokenizer,
)
from src.models.param_tokenizer import (
    ParamDetokenizer,
    ParamTokenizer,
)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: (B,) tensor of timestep indices
        dim: embedding dimension
    Returns:
        (B, dim) sinusoidal embedding tensor
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1))
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ParamDiTBlock(nn.Module):
    """Transformer block with AdaLN-style timestep conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)
        attn_input = _modulate(self.norm1(x), shift_attn, scale_attn)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_output

        mlp_input = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class ParamDiT(nn.Module):
    """Denoise flat INR parameters using weight-column tokens."""

    def __init__(
        self,
        param_shapes,
        hidden_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        time_dim=None,
        tokenizer="column",
        tokens_per_tensor=1,
        chunk_size=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim or hidden_dim
        tokenizer = str(tokenizer).lower()
        if tokenizer in {"column", "param", "columns"}:
            self.tokenizer = ParamTokenizer(param_shapes, hidden_dim=hidden_dim)
            self.detokenizer = ParamDetokenizer(
                self.tokenizer.specs,
                self.tokenizer.num_params,
                hidden_dim=hidden_dim,
            )
        elif tokenizer in {"hyperdiff", "hyperdiffusion", "chunk", "layer"}:
            self.tokenizer = HyperDiffTokenizer(
                param_shapes,
                hidden_dim=hidden_dim,
                tokens_per_tensor=tokens_per_tensor,
                chunk_size=chunk_size,
            )
            self.detokenizer = HyperDiffDetokenizer(
                self.tokenizer.specs,
                self.tokenizer.num_params,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown ParamDiT tokenizer '{tokenizer}'. " "Expected one of: column, hyperdiff.")
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            ParamDiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, params, timesteps):
        timesteps = timesteps.squeeze(-1)  # handle (B, 1) -> (B,)
        tokens = self.tokenizer(params)
        cond = self.time_embed(timestep_embedding(timesteps, self.time_dim).to(params.dtype))
        for block in self.blocks:
            tokens = block(tokens, cond)
        tokens = self.final_norm(tokens)
        return self.detokenizer(tokens)
