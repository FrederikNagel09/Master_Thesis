import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# =============================================================================
# Time embedding module
# =============================================================================


class SinusoidalLearnableTimeEmbedding(nn.Module):
    """
    Maps scalar t in [0,1] to a rich sinusoidal embedding, then projects it.
    Identical in spirit to the positional encoding used in DDPM / transformer
    models, but operating on continuous normalised time rather than integer steps.
    """

    def __init__(self, embed_dim: int = 128, T: int = 1000):  # noqa: N803
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.T = T
        # Learnable projection on top of the fixed sinusoidal features
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (batch, 1)  normalised time in [0, 1]
        returns: (batch, embed_dim)
        """
        # gets hafl of the embedding dim in sin and half in cos, with frequencies from 1 to self.T
        half = self.embed_dim // 2

        # compute frequencies on the same device and dtype as t to avoid issues
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)  # (half,)

        # Scale t to [0, T] range and compute sinusoidal features
        args = t * freqs.unsqueeze(0) * self.T

        # Gets Sinusodal time embedding by concatenating sin and cos features
        time_embedding = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch, embed_dim)

        # Applies learnable projection to the sinusoidal embedding to get the final time embedding
        time_embedding = self.proj(time_embedding)  # (batch, embed_dim)

        return time_embedding


# =============================================================================
# Residual blocks with time conditioning for Unet architecture
# =============================================================================


class TimeConditionedResBlock(nn.Module):
    """
    Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm,
    with time embedding injected as a scale+shift into the first GroupNorm
    (AdaGN style, as used in ADM/Dhariwal & Nichol 2021).
    Includes a residual connection with a 1x1 conv if channel dims differ.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)

        # Projects time embedding to scale and shift for AdaGN on gn1
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_ch * 2),  # split into scale + shift
        )

        # Residual projection if channel count changes
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (batch, t_dim)
        h = self.conv1(x)
        h = self.gn1(h)

        # AdaGN: modulate normalised features with time-derived scale/shift
        t_out = self.t_proj(t_emb)  # (batch, out_ch*2)
        scale, shift = t_out.chunk(2, dim=1)  # each (batch, out_ch)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)

        return h + self.res_conv(x)
