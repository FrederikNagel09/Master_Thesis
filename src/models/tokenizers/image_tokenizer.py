import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.models.utils.attention import Attention, LocalAttention


class ImageTokenizer(nn.Module):
    """
    Tokenise a raw image into patch tokens.

    Args:
        in_channels  : number of image channels (1 for MNIST grayscale)
        image_size   : (H, W) or a single int when H == W
        patch_size   : (ph, pw) or a single int
        dim          : transformer embedding dimension
        n_head       : number of attention heads
        head_dim     : dimension per head
        padding      : optional symmetric padding applied before unfolding
        dropout      : dropout probability (currently unused, kept for API compat)
    """

    def __init__(
        self,
        in_channels,
        image_size,
        patch_size,
        dim,
        n_head,
        head_dim,
        padding=0,
        dropout=0.0,
    ):  # noqa: ARG002
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.patch_size = patch_size
        self.padding = padding

        # Each patch is flattened to (in_channels * ph * pw) and projected to dim
        self.prefc = nn.Linear(in_channels * patch_size[0] * patch_size[1], dim)

        padded_h = image_size[0] + padding[0] * 2
        padded_w = image_size[1] + padding[1] * 2
        n_patches = (padded_h // patch_size[0]) * (padded_w // patch_size[1])

        # Learned positional embeddings — one per patch
        self.posemb = nn.Parameter(torch.randn(1, n_patches, dim))

        # window_size = patches per spatial row so N_patches % window_size == 0
        local_window = padded_h // patch_size[0]
        self.local_attn = LocalAttention(
            dim, window_size=local_window, n_head=n_head, head_dim=head_dim
        )
        self.global_attn = Attention(dim, n_head=n_head, head_dim=head_dim)

    def forward(self, x, *args, **kwargs):  # noqa: ARG002
        """
        Args:
            x : (B, C, H, W)  raw image tensor
        Returns:
            tokens : (B, N, dim)
        """
        p = self.patch_size

        #print(f"patch size: {p}, padding: {self.padding}")
        # F.unfold → (B, C*ph*pw, L)
        x = F.unfold(x, p, stride=p, padding=self.padding)
        x = x.permute(0, 2, 1).contiguous()  # (B, N, C*ph*pw)

        x = self.prefc(x)  # (B, N, dim)
        x = x + self.posemb

        x = self.local_attn(x)
        x = self.global_attn(x)

        return x  # (B, N, dim)
