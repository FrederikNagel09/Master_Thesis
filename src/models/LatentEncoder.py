import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.models.trans_inr_helpers import TransformerEncoder


class LatentEncoder(nn.Module):
    """
    Encodes an image to a latent feature map consumable by LatentTokenizer.

    Two modes controlled by transformer_depth:
        depth=0 : CNN only          (fast, easy to train)
        depth>0 : CNN + Transformer (better global context)

    Args:
        in_channels     : input image channels (1 for MNIST)
        latent_dim      : output channel depth C_latent; must match
                          LatentTokenizer's latent_dim
        latent_size     : (H', W') spatial size of output feature map; must
                          match LatentTokenizer's latent_size
        transformer_depth: number of TransformerEncoder layers (0 = CNN only)
        n_head          : attention heads (only used if transformer_depth > 0)
        head_dim        : head dimension  (only used if transformer_depth > 0)
        ff_dim          : feedforward dim (only used if transformer_depth > 0)
        dropout         : dropout rate    (only used if transformer_depth > 0)
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        latent_size: tuple[int, int],
        transformer_depth: int = 0,
        n_head: int = 4,
        head_dim: int = 32,
        ff_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)

        hidden_dim = latent_dim * 2

        bottleneck_dim = hidden_dim * 2

        self.cnn = nn.Sequential(
            # Encoder: preserve → halve → halve
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, bottleneck_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # Decoder: upsample → upsample → project to latent_dim
            nn.ConvTranspose2d(bottleneck_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=1),
        )

        # --- Optional transformer ---
        # operates on spatial tokens (B, H'*W', latent_dim), then reshapes back
        self.use_transformer = transformer_depth > 0
        if self.use_transformer:
            self.transformer = TransformerEncoder(
                dim=latent_dim,
                depth=transformer_depth,
                n_head=n_head,
                head_dim=head_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            n_tokens = self.latent_size[0] * self.latent_size[1]
            self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent feature map.

        Args:
            x: (B, in_channels, H, W)
        Returns:
            z: (B, latent_dim, H', W') — matches LatentTokenizer input contract
        """
        z = self.cnn(x)  # (B, latent_dim, H', W')
        if z.shape[-2:] != self.latent_size:
            z = F.interpolate(z, size=self.latent_size, mode="bilinear", align_corners=False)

        if self.use_transformer:
            B, C, H, W = z.shape  # noqa: N806
            # Flatten spatial dims to token sequence
            tokens = z.flatten(2).transpose(1, 2)  # (B, H'*W', latent_dim)
            tokens = tokens + self.pos_emb
            tokens = self.transformer(tokens)  # (B, H'*W', latent_dim)
            # Reshape back to feature map
            z = tokens.transpose(1, 2).reshape(B, C, H, W)

        return z
