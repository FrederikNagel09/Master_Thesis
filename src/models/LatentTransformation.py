import torch
import torch.nn as nn

from src.models.helper_modules import SinusoidalLearnableTimeEmbedding


class LatentResBlock(nn.Module):
    """
    Stride-1 residual block for latent-space feature maps.
    Time conditioning is injected via channel-wise scale-shift after GroupNorm.
    """

    def __init__(self, channels: int, t_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        # Projects time embedding to scale + shift for each channel
        self.t_proj = nn.Linear(t_embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W) — latent feature map
            t_emb: (B, t_embed_dim) — time embedding
        Returns:
            (B, C, H, W) — residual-updated feature map
        """
        scale, shift = self.t_proj(t_emb).chunk(2, dim=-1)  # each (B, C)
        # Reshape for broadcasting over spatial dims
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm1(x)
        h = h * (1 + scale) + shift  # AdaGN: condition on time
        h = self.act(self.conv1(h))
        h = self.norm2(h)
        h = self.act(self.conv2(h))
        return x + h


class LatentTransformation(nn.Module):
    """
    Residual conv-based transformation F_phi(z, t) operating on latent feature maps.
    Architecture: stack of stride-1 ResBlocks with time conditioning via AdaGN.
    Returns (1 - t) * z + t * f_bar to interpolate between identity and learned transform.
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 2,
        t_embed_dim: int = 32,
    ):
        super().__init__()
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        # Project to hidden dim, process, project back
        self.proj_in = nn.Conv2d(latent_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([LatentResBlock(hidden_channels, t_embed_dim) for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C, H, W) — latent sample from encoder
            t: (B, 1)        — normalized time in [0, 1]
        Returns:
            (B, C, H, W) — transformed latent, same shape as input
        """
        t_emb = self.time_embed(t)  # (B, t_embed_dim)
        h = self.proj_in(z)
        for block in self.blocks:
            h = block(h, t_emb)
        f_bar = self.proj_out(h)
        # Reshape t for broadcasting over (C, H, W)
        t_bc = t[:, :, None, None]
        return (1 - t_bc) * z + t_bc * f_bar
