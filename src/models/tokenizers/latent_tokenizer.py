import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.models.utils.attention import Attention, LocalAttention


class LatentTokenizer(nn.Module):
    def __init__(
        self,
        latent_dim,
        latent_size,
        patch_size,
        dim,
        n_head,
        head_dim,
        padding=0,
        dropout=0.0,
    ):  # noqa: ARG002
        super().__init__()
        if isinstance(latent_size, int):
            latent_size = (latent_size, latent_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding

        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * latent_dim, dim)

        n_patches = ((latent_size[0] + padding[0] * 2) // patch_size[0]) * (
            (latent_size[1] + padding[1] * 2) // patch_size[1]
        )
        self.posemb = nn.Parameter(torch.randn(1, n_patches, dim))

        self.local_attn = LocalAttention(
            dim, window_size=patch_size[0], n_head=n_head, head_dim=head_dim
        )
        self.global_attn = Attention(dim, n_head=n_head, head_dim=head_dim)

    def forward(self, x, *args, **kwargs):  # noqa: ARG002
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding)
        x = x.permute(0, 2, 1).contiguous()
        x = self.prefc(x)
        x = x + self.posemb
        x = self.local_attn(x)
        x = self.global_attn(x)
        return x
