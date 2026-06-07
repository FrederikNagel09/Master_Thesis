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
        hidden_dim: int = 512,
        ff_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)

        bottleneck_dim = hidden_dim * 4

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


class ResNetBasicBlock(nn.Module):
    """Standard ResNet Basic Block with two 3x3 convolutions and a residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetLatentEncoder(nn.Module):
    """
    Encodes an image to a latent feature map using a scalable ResNet-18 backbone.

    Replaces static interpolation with a learnable ConvTranspose2d layer to map
    directly to the requested latent dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        latent_size: tuple[int, int],
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)

        ch1 = hidden_dim  # default: 64
        ch2 = hidden_dim * 2  # default: 128
        ch3 = hidden_dim * 4  # default: 256
        ch4 = hidden_dim * 8  # default: 512

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(ch1), nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_stage(ch1, ch1, num_blocks=2, stride=1)
        self.layer2 = self._make_stage(ch1, ch2, num_blocks=2, stride=2)
        self.layer3 = self._make_stage(ch2, ch3, num_blocks=2, stride=2)
        self.layer4 = self._make_stage(ch3, ch4, num_blocks=2, stride=2)

        # Separate learnable heads for the distribution parameters
        self.upsample_mu = nn.ConvTranspose2d(in_channels=ch4, out_channels=latent_dim, kernel_size=4, stride=2, padding=1)
        self.upsample_logvar = nn.ConvTranspose2d(in_channels=ch4, out_channels=latent_dim, kernel_size=4, stride=2, padding=1)

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBasicBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) — Input image batch
        Returns:
            z: (B, latent_dim, H', W') — Matches LatentTokenizer contract
        """
        # Feature extraction
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        mu = self.upsample_mu(out)
        logvar = self.upsample_logvar(out)

        if mu.shape[-2:] != self.latent_size:
            mu = nn.functional.interpolate(mu, size=self.latent_size, mode="bilinear", align_corners=False)
            logvar = nn.functional.interpolate(logvar, size=self.latent_size, mode="bilinear", align_corners=False)

        logvar = logvar.clamp(min=-12, max=4.0)
        
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
