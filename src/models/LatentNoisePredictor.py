import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.models.helper_modules import SinusoidalLearnableTimeEmbedding


class MPSStableUNetBlock(nn.Module):
    """An MPS-stable ResNet-style block using LayerNorm and Scale-Shift conditioning."""

    def __init__(self, in_channels: int, out_channels: int, t_embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # LayerNorm over channels, height, and width to avoid GroupNorm MPS bugs
        self.norm1 = nn.LayerNorm(out_channels)

        # Predicts both scale (gamma) and shift (beta) from time embedding
        self.time_proj = nn.Linear(t_embed_dim, out_channels * 2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # First Conv
        h = self.conv1(x)

        # LayerNorm expects (B, H, W, C), so we permute temporarily
        h = h.permute(0, 2, 3, 1)
        h = self.norm1(h)
        h = h.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        h = F.silu(h)

        # Scale and Shift (AdaLN style) calculation from time embedding
        t_params = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, 2*C, 1, 1)
        scale, shift = torch.chunk(t_params, 2, dim=1)
        h = h * (1 + scale) + shift

        # Second Conv
        h = self.conv2(h)
        h = h.permute(0, 2, 3, 1)
        h = self.norm2(h)
        h = h.permute(0, 3, 1, 2)

        return F.silu(h + self.shortcut(x))


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
        self.time_mlp = nn.Sequential(nn.Linear(t_embed_dim, t_embed_dim), nn.SiLU())

        # --- Input mapping ---
        self.input_conv = nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, padding=1)

        # --- UNet blocks ---
        self.down1 = MPSStableUNetBlock(hidden_dim, hidden_dim, t_embed_dim)
        self.pool1 = nn.MaxPool2d(2)  # 8x8 -> 4x4

        self.down2 = MPSStableUNetBlock(hidden_dim, hidden_dim * 2, t_embed_dim)
        self.pool2 = nn.MaxPool2d(2)  # 4x4 -> 2x2

        self.mid1 = MPSStableUNetBlock(hidden_dim * 2, hidden_dim * 2, t_embed_dim)
        self.mid2 = MPSStableUNetBlock(hidden_dim * 2, hidden_dim * 2, t_embed_dim)

        self.up1 = MPSStableUNetBlock(hidden_dim * 4, hidden_dim, t_embed_dim)
        self.up2 = MPSStableUNetBlock(hidden_dim * 2, hidden_dim, t_embed_dim)

        # --- Output mapping ---
        self.output_conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)

        # Zero-init output layer to ensure it outputs a safe distribution at step 0
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, D = z.shape  # noqa: N806
        H = W = self.spatial_dim  # noqa: N806

        # 1. Sequence to 2D image layout: (B, N, D) -> (B, D, H, W)
        x = z.transpose(1, 2).view(B, D, H, W)

        # 2. Time embedding
        t_emb = self.time_mlp(self.time_embed(t))

        # 3. UNet Forward Pass
        x1 = self.input_conv(x)

        x2 = self.down1(x1, t_emb)
        x3 = self.pool1(x2)

        x4 = self.down2(x3, t_emb)
        x5 = self.pool2(x4)

        # Bottleneck
        x5 = self.mid1(x5, t_emb)
        x5 = self.mid2(x5, t_emb)

        # Decode with Skip connections
        x_up1 = F.interpolate(x5, size=(4, 4), mode="nearest")
        x_up1 = torch.cat([x_up1, x4], dim=1)
        x6 = self.up1(x_up1, t_emb)

        x_up2 = F.interpolate(x6, size=(8, 8), mode="nearest")
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x7 = self.up2(x_up2, t_emb)

        out_spatial = self.output_conv(x7)

        # 4. Flatten back to tokens: (B, D, H, W) -> (B, N, D)
        eps_hat = out_spatial.view(B, D, N).transpose(1, 2)
        return eps_hat


class LatentTransformerNoisePredictor(nn.Module):
    """
    Transformer noise predictor for latent token sequences.

    Operates natively on (B, n_patches, D) — no chunking needed.
    Time is injected as a prepended token, dropped at readout.

    Args:
        n_patches   : number of latent spatial tokens
        latent_dim  : token feature dimension D
        d_model     : transformer internal dimension
        n_heads     : number of attention heads
        n_layers    : number of transformer encoder layers
        d_ff        : feedforward dimension
        dropout     : dropout rate
        t_embed_dim : sinusoidal time embedding dimension
    """

    def __init__(
        self,
        n_patches: int,
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.latent_dim = latent_dim
        self.d_model = d_model

        # --- Time embedding ---
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(t_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # --- Per-token input projection: latent_dim → d_model ---
        self.token_embed = nn.Linear(latent_dim, d_model)

        # --- Positional embedding: n_patches + 1 time token ---
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Readout: d_model → latent_dim per token ---
        self.final_norm = nn.LayerNorm(d_model)
        self.token_readout = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.token_readout.weight)
        nn.init.zeros_(self.token_readout.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, n_patches, D) noisy latent tokens
            t: (B,) normalised time in [0, 1]
        Returns:
            eps_hat: (B, n_patches, D) predicted noise
        """
        # Project tokens to d_model
        x = self.token_embed(z)  # (B, n_patches, d_model)

        # Build and prepend time token
        t_emb = self.time_embed(t)  # (B, t_embed_dim)
        t_tok = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([t_tok, x], dim=1)  # (B, n_patches+1, d_model)
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Drop time token, project back to latent_dim
        x = x[:, 1:, :]  # (B, n_patches, d_model)
        x = self.final_norm(x)
        return self.token_readout(x)  # (B, n_patches, D)
