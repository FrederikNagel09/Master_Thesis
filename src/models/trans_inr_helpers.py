import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.configs.general_config import GLOBAL_DEBUG_BOOL

# ---------------------------------------------------------------------------
# Image Tokenizer  (replaces LatentTokenizer)
# ---------------------------------------------------------------------------
# Instead of receiving a pre-computed latent, this module accepts a raw image
# tensor of shape (B, C, H, W) and converts it into a sequence of patch tokens
# that can be consumed by the transformer encoder.
#
# The architecture mirrors the original LatentTokenizer but the first linear
# projection now maps from (C * patch_h * patch_w) rather than from
# (latent_dim * patch_h * patch_w).  Local + global attention are kept.
# ---------------------------------------------------------------------------


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

    def __init__(self, in_channels, image_size, patch_size, dim, n_head, head_dim, padding=0, dropout=0.0):  # noqa: ARG002
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
        self.local_attn = LocalAttention(dim, window_size=local_window, n_head=n_head, head_dim=head_dim)
        self.global_attn = Attention(dim, n_head=n_head, head_dim=head_dim)

    def forward(self, x, *args, **kwargs):  # noqa: ARG002
        """
        Args:
            x : (B, C, H, W)  raw image tensor
        Returns:
            tokens : (B, N, dim)
        """
        p = self.patch_size
        # F.unfold → (B, C*ph*pw, L)
        x = F.unfold(x, p, stride=p, padding=self.padding)
        x = x.permute(0, 2, 1).contiguous()  # (B, N, C*ph*pw)

        x = self.prefc(x)  # (B, N, dim)
        x = x + self.posemb

        x = self.local_attn(x)
        x = self.global_attn(x)

        return x  # (B, N, dim)


# ---------------------------------------------------------------------------
# Keep the original LatentTokenizer for backward-compatibility
# ---------------------------------------------------------------------------


class LatentTokenizer(nn.Module):
    def __init__(self, latent_dim, latent_size, patch_size, dim, n_head, head_dim, padding=0, dropout=0.0):  # noqa: ARG002
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

        n_patches = ((latent_size[0] + padding[0] * 2) // patch_size[0]) * ((latent_size[1] + padding[1] * 2) // patch_size[1])
        self.posemb = nn.Parameter(torch.randn(1, n_patches, dim))

        self.local_attn = LocalAttention(dim, window_size=patch_size[0], n_head=n_head, head_dim=head_dim)
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


# ---------------------------------------------------------------------------
# Batched linear layer helper (used by SIREN)
# ---------------------------------------------------------------------------


def batched_linear_mm(x, wb):
    """
    x  : (B, N, D1)
    wb : (B, D1+1, D2)  — last row is the bias
    """
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb)


# ---------------------------------------------------------------------------
# SIREN  (unchanged from original)
# ---------------------------------------------------------------------------


class SIREN(nn.Module):
    def __init__(self, depth, in_dim, out_dim, hidden_dim, out_bias=0, omega=30.0):
        super().__init__()
        self.omega = omega
        self.depth = depth
        self.param_shapes = dict()  # noqa: C408

        last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f"wb{i}"] = (last_dim + 1, cur_dim)
            last_dim = cur_dim

        self.params = None
        self.out_bias = out_bias

        if GLOBAL_DEBUG_BOOL:
            print("==================== DEBUG: SIREN init ====================")
            print(f"  depth: {depth}, in_dim: {in_dim}, out_dim: {out_dim}, hidden_dim: {hidden_dim}")
            print(f"  param_shapes: {self.param_shapes}")
            print("================================================================")

    def siren_activation(self, x):
        return torch.sin(self.omega * x)

    def init_wb(self, shape, name):
        if name == "wb0":
            num_input = shape[0] - 1
            bound = 1 / num_input
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.uniform_(weight, -bound, bound)
            bias = torch.zeros(shape[1], 1)
            return torch.cat([weight, bias], dim=1).t().detach()
        else:
            num_input = shape[0] - 1
            bound = np.sqrt(6 / num_input) / self.omega
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.uniform_(weight, -bound, bound)
            bias = torch.zeros(shape[1], 1)
            return torch.cat([weight, bias], dim=1).t().detach()

    def set_params(self, params):
        self.params = params

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1:-1]  # noqa: N806
        x = x.view(B, -1, x.shape[-1])

        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f"wb{i}"])
            x = self.siren_activation(x) if i < self.depth - 1 else torch.tanh(x)

        x = x.view(B, *query_shape, -1)
        return x

    def get_last_layer(self):
        return self.params[f"wb{self.depth - 1}"]

class MLP_INR(nn.Module):  # noqa: N801
    def __init__(self, depth, in_dim, out_dim, hidden_dim, out_bias=0):
        """omega kept in signature for drop-in compatibility, not used."""
        super().__init__()
        self.depth = depth
        self.param_shapes = dict()  # noqa: C408
        self.out_bias = out_bias

        last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f"wb{i}"] = (last_dim + 1, cur_dim)
            last_dim = cur_dim

        self.params = None

    def init_wb(self, shape: tuple, name: str) -> torch.Tensor:  # noqa: ARG002
        """
        Initialise a weight+bias matrix using Kaiming uniform (He init).

        Args:
            shape: (fan_in + 1, fan_out) — rows are [weights | bias]
            name:  parameter name e.g. 'wb0' (unused, kept for API parity)
        Returns:
            Tensor of shape `shape`, detached.
        """
        fan_in = shape[0] - 1
        weight = torch.empty(shape[1], fan_in)
        nn.init.kaiming_uniform_(weight, a=0, mode="fan_in", nonlinearity="relu")
        bias = torch.zeros(shape[1], 1)
        return torch.cat([weight, bias], dim=1).t().detach()

    def set_params(self, params: dict) -> None:
        """Set externally generated parameters (e.g. from a hypernetwork)."""
        self.params = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP INR.

        Args:
            x: (B, *query_shape, in_dim)
        Returns:
            Tensor of shape (B, *query_shape, out_dim)
        """
        B, query_shape = x.shape[0], x.shape[1:-1]  # noqa: N806
        x = x.view(B, -1, x.shape[-1])

        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f"wb{i}"])
            x = F.relu(x) if i < self.depth - 1 else torch.tanh(x)

        x = x.view(B, *query_shape, -1)
        return x

    def get_last_layer(self) -> torch.Tensor:
        """Returns the last layer's weight+bias matrix."""
        return self.params[f"wb{self.depth - 1}"]
# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim, n_head, head_dim, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim**-0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head), [q, k, v])  # noqa: C417
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalAttention(nn.Module):
    def __init__(self, dim, window_size=2, n_head=4, head_dim=32):  # noqa: ARG002
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads=n_head, batch_first=True)

    def forward(self, x):
        _B, N, _D = x.shape  # noqa: N806
        W = self.window_size  # noqa: N806
        G = N // W  # noqa: N806
        assert N % W == 0, f"window_size={W} does not divide N={N} evenly!"

        x = einops.rearrange(x, "b (g w) d -> (b g) w d", g=G, w=W)
        x, _ = self.attn(x, x, x)
        x = einops.rearrange(x, "(b g) w d -> b (g w) d", g=G, w=W)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


# ---------------------------------------------------------------------------
# Transformer (encoder + decoder)
# ---------------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                        PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, memory):
        for norm_self_attn, norm_cross_attn, norm_ff in self.layers:
            x = x + norm_self_attn(x)
            x = x + norm_cross_attn(x, to=memory)
            x = x + norm_ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, encoder_depth, decoder_depth, n_head, head_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(dim, encoder_depth, n_head, head_dim, ff_dim, dropout)
        self.decoder = TransformerDecoder(dim, decoder_depth, n_head, head_dim, ff_dim, dropout)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

    def get_last_layer(self):
        return self.decoder.layers[-1][-1].fn.net[-2].weight
