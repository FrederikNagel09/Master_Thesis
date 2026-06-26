import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.models.utils.attention import Attention, LocalAttention


class VolumeTokenizer(nn.Module):
    """
    Tokenise a 3D volume into patch tokens.
    Treats input (B, C, D, H, W) as a 2D grid by flattening the D dimension
    into the spatial H dimension for the unfolding operation.
    """

    def __init__(
        self, in_channels, vol_size, patch_size, dim, n_head, head_dim, padding=0
    ):
        super().__init__()
        # Ensure patch_size is a 3-tuple
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)

        # ADD THIS: Ensure padding is a 3-tuple
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.patch_size = patch_size
        self.padding = padding

        # Each patch is (in_channels * pd * ph * pw)
        patch_volume = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.prefc = nn.Linear(patch_volume, dim)

        # Calculate number of patches
        D, H, W = vol_size  # noqa: N806
        pd, ph, pw = patch_size
        pad_d, pad_h, pad_w = padding

        n_patches = (
            ((D + 2 * pad_d) // pd) * ((H + 2 * pad_h) // ph) * ((W + 2 * pad_w) // pw)
        )
        self.posemb = nn.Parameter(torch.randn(1, n_patches, dim))

        # Local/Global attention configuration
        # For 3D, window_size represents the spatial extent of a slice or sub-volume
        local_window = (H + 2 * pad_h) // ph
        self.local_attn = LocalAttention(
            dim, window_size=local_window, n_head=n_head, head_dim=head_dim
        )
        self.global_attn = Attention(dim, n_head=n_head, head_dim=head_dim)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape  # noqa: N806
        pd, ph, pw = self.patch_size

        # 3D unfolding: We fold the D dimension into the height dimension
        # (B, C, D, H, W) -> (B, C, D*H, W)
        x = x.view(B, C, D * H, W)

        # Apply 2D unfold (kernel size covers depth and spatial)
        # Note: We treat pd*ph as the 'height' of the kernel
        x = F.unfold(
            x,
            kernel_size=(pd * ph, pw),
            stride=(pd * ph, pw),
            padding=(self.padding[0] * self.padding[1], self.padding[2]),
        )

        # (B, C*pd*ph*pw, N) -> (B, N, C*pd*ph*pw)
        x = x.permute(0, 2, 1).contiguous()

        x = self.prefc(x)
        x = x + self.posemb
        x = self.local_attn(x)
        x = self.global_attn(x)
        return x
