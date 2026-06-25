import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.models.helper_modules import SinusoidalLearnableTimeEmbedding


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.time_proj = nn.Linear(t_embed_dim, out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # ReZero: block starts as identity
        self.res_scale = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)

        # Norm before residual add
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)

        # ReZero keeps this at identity until network learns to use it
        return self.shortcut(x) + self.res_scale * F.silu(h)


class LatentUNetNoisePredictor(nn.Module):
    """
    MPS-Optimized UNet noise predictor for latent token sequences.
    Replaces GroupNorm with LayerNorm to prevent Apple Silicon gradient explosions.
    """

    def __init__(
        self,
        n_patches: int,  # e.g., 64
        latent_dim: int,  # e.g., 16
        hidden_dim: int = 64,  # Base channel width for convolutions
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.latent_dim = latent_dim
        self.spatial_dim = int(math.sqrt(n_patches))
        assert self.spatial_dim * self.spatial_dim == n_patches, "n_patches must be a perfect square"

        # --- Time embedding ---
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(t_embed_dim * 4, t_embed_dim),
            nn.SiLU(),
        )

        # --- Input mapping ---
        self.input_conv = nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, padding=1)

        # --- UNet blocks ---
        self.down1 = UNetBlock(hidden_dim, hidden_dim, t_embed_dim)
        self.pool1 = nn.MaxPool2d(2)  # 8x8 -> 4x4

        self.down2 = UNetBlock(hidden_dim, hidden_dim * 2, t_embed_dim)
        self.pool2 = nn.MaxPool2d(2)  # 4x4 -> 2x2

        self.mid1 = UNetBlock(hidden_dim * 2, hidden_dim * 2, t_embed_dim)
        self.mid2 = UNetBlock(hidden_dim * 2, hidden_dim * 2, t_embed_dim)

        self.up1 = UNetBlock(hidden_dim * 4, hidden_dim, t_embed_dim)
        self.up2 = UNetBlock(hidden_dim * 2, hidden_dim, t_embed_dim)

        # --- Output mapping ---
        self.output_conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z is already (B, C, H, W)
        t_emb = self.time_mlp(self.time_embed(t))

        x1 = self.input_conv(z)

        x2 = self.down1(x1, t_emb)
        x3 = self.pool1(x2)

        x4 = self.down2(x3, t_emb)
        x5 = self.pool2(x4)

        # Bottleneck
        x5 = self.mid1(x5, t_emb)
        x5 = self.mid2(x5, t_emb)

        # Decode with Skip connections
        x_up1 = F.interpolate(x5, size=x4.shape[-2:], mode="nearest")
        x_up1 = torch.cat([x_up1, x4], dim=1)
        x6 = self.up1(x_up1, t_emb)

        x_up2 = F.interpolate(x6, size=x2.shape[-2:], mode="nearest")
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x7 = self.up2(x_up2, t_emb)

        out_spatial = self.output_conv(x7)

        return out_spatial


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Creates a standard static 2D Sin-Cos positional embedding.
    grid_size: int or tuple (H, W)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    # Crucial Fix: Split the total embed_dim across the 2 axes (X and Y)
    # so that their concatenation equals the total embed_dim.
    assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"
    axis_dim = embed_dim // 2

    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="ij"), dim=0)
    grid = grid.reshape(2, 1, grid_size[0], grid_size[1])

    # Pass axis_dim instead of full embed_dim
    pos_embed = get_1d_sincos_pos_embed_from_grid(axis_dim, grid)
    return torch.from_numpy(pos_embed).float().unsqueeze(0)  # Now perfectly outputs (1, H*W, embed_dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim, grid):
    import numpy as np

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    grid = grid.numpy()
    out = []
    for g in grid:
        out_sin = np.sin(np.outer(g, omega))
        out_cos = np.cos(np.outer(g, omega))
        out.append(np.concatenate([out_sin, out_cos], axis=1))
    return np.concatenate(out, axis=1)


class DiTBlock(nn.Module):
    """Transformer block that uses Adaptive Layer Norm (adaLN) for time conditioning."""

    def __init__(self, d_model, n_heads, d_ff, dropout, t_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

        # AdaLN parameter projection: splits into scale (gamma) and shift (beta) multipliers
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 6 * d_model))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Generate scale and shift modulations from time embedding
        mods = self.adaLN_modulation(t_emb).unsqueeze(1)  # (B, 1, 6*d_model)

        # 1. Attention Block with adaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h_attn, _ = self.attn(h, h, h)
        x = x + gate_msa * h_attn  # gate added

        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


class LatentTransformerNoisePredictor(nn.Module):
    """
    Optimized Diffusion Transformer (DiT) Noise Predictor for Latent Tokens.

    Replaces time-token prepending with Adaptive Layer Norm (adaLN) modulation
    and utilizes explicit 2D spatial sin-cos positional embeddings.
    """

    def __init__(
        self,
        latent_size: tuple[int, int],  # Pass explicit tuple (H', W'), e.g., (4, 4)
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)
        self.n_patches = self.latent_size[0] * self.latent_size[1]
        self.latent_dim = latent_dim

        # --- Time embedding ---
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(t_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # --- Input Projection ---
        self.token_embed = nn.Linear(latent_dim, d_model)

        # --- Fixed Static 2D Positional Embeddings ---
        # No longer needs +1 padding for a time token
        pos_embed = get_2d_sincos_pos_embed(d_model, self.latent_size)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        # --- Transformer Block Stack (DiT Layers) ---
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, d_ff, dropout, t_dim=d_model) for _ in range(n_layers)])

        # --- Readout Projection ---
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        self.token_readout = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.token_readout.weight)
        nn.init.zeros_(self.token_readout.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C, H, W) noisy latent
            t: (B, 1) normalized time sequence
        """
        B, C, H, W = z.shape  # noqa: N806

        # Convert Spatial Layout to Sequence Matrix -> (B, N, C)
        x = z.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project features and add fixed geographic spatial coordinates
        print(f"Positional embedding shape: {self.pos_embed.shape}, x shape: {x.shape}")
        x = self.token_embed(x) + self.pos_embed

        # Compute global continuous time vectors
        t_emb = self.time_proj(self.time_embed(t))  # (B, d_model)

        # Pass through the specialized adaLN transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final mapping reconstruction
        shift, scale = self.final_modulation(t_emb).unsqueeze(1).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.token_readout(x)  # (B, N, latent_dim)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
