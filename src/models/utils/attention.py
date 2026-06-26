import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

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
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head),
            [q, k, v],
        )  # noqa: C417
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
