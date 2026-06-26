import math
import sys

# ---------------------------------------------------------------------------
# Re-use helpers from trans_inr_helpers
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

sys.path.append(".")


from src.models.utils.helper_modules import SinusoidalLearnableTimeEmbedding
from src.models.utils.transformer import TransformerEncoder


class TransInrNoisePredictor(nn.Module):
    """ """

    def __init__(
        self,
        weight_dim: int,
        dim: int,
        depth: int,
        n_head: int,
        head_dim: int,
        ff_dim: int,
        chunk_size: int = 128,
        t_embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.weight_dim = weight_dim
        self.chunk_size = chunk_size
        self.dim = dim

        # 1. Chunking logic
        self.padded_dim = math.ceil(weight_dim / chunk_size) * chunk_size
        self.n_tokens = self.padded_dim // chunk_size

        # 2. Time Embedding (MLP for richer signal)
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(t_embed_dim, dim), nn.SiLU())

        # 3. Input Projection
        self.token_embed = nn.Linear(chunk_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Backbone: Reusing your TransformerEncoder
        self.transformer = TransformerEncoder(
            dim=dim,
            depth=depth,
            n_head=n_head,
            head_dim=head_dim,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # 5. Output Head: Direct projection to noise
        self.noise_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, chunk_size))

        # Initialize head to zero or small values to help initial stability
        # nn.init.zeros_(self.noise_head[1].weight)
        # nn.init.zeros_(self.noise_head[1].bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, weight_dim)  noisy weight vector
            t : (B,)             timesteps
        """
        B = z.shape[0]  # noqa: N806

        # --- Step 1: Tokenize ---
        if self.padded_dim > self.weight_dim:
            pad = z.new_zeros(B, self.padded_dim - self.weight_dim)
            z_pad = torch.cat([z, pad], dim=-1)
        else:
            z_pad = z

        # Reshape to (B, N_tokens, Chunk_size)
        tokens = z_pad.view(B, self.n_tokens, self.chunk_size)
        x = self.token_embed(tokens)  # (B, N, dim)

        # ==============================
        # --- Step 2: Dense Conditioning (Option B) ---
        # Get time vector
        t_emb = self.time_mlp(self.time_embed(t))

        # Inject time and position into EVERY token
        x = x + self.pos_embed + t_emb.unsqueeze(1)

        # --- Step 3: Transformer Backbone ---
        x = self.transformer(x)  # (B, N, dim)

        # --- Step 4: Predict Noise ---
        out_tokens = self.noise_head(x)  # (B, N, chunk_size)

        # Flatten and unpad
        eps_hat = out_tokens.reshape(B, self.padded_dim)[:, : self.weight_dim]

        return eps_hat
