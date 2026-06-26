import torch.nn as nn

from src.models.utils.attention import Attention, FeedForward, PreNorm

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
    def __init__(
        self, dim, encoder_depth, decoder_depth, n_head, head_dim, ff_dim, dropout=0.0
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            dim, encoder_depth, n_head, head_dim, ff_dim, dropout
        )
        self.decoder = TransformerDecoder(
            dim, decoder_depth, n_head, head_dim, ff_dim, dropout
        )

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

    def get_last_layer(self):
        return self.decoder.layers[-1][-1].fn.net[-2].weight
