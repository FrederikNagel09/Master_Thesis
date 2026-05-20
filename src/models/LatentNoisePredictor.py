import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.models.helper_modules import SinusoidalLearnableTimeEmbedding


class LatentMLPNoisePredictor(nn.Module):
    """
    MLP noise predictor for latent token sequences.

    Flattens (B, n_patches, D) → (B, n_patches*D), runs MLP, unflattens back.

    Args:
        n_patches   : number of latent spatial tokens
        latent_dim  : token feature dimension D
        hidden_dim  : MLP hidden width
        n_blocks    : number of residual blocks
        t_embed_dim : sinusoidal time embedding dimension
    """

    def __init__(
        self,
        n_patches: int,
        latent_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.latent_dim = latent_dim
        flat_dim = n_patches * latent_dim

        # --- Time embedding ---
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_proj = nn.Linear(t_embed_dim, hidden_dim)

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- Residual blocks ---
        self.blocks = nn.ModuleList()
        self.t_projs = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                    ]
                )
            )
            self.t_projs.append(nn.Linear(t_embed_dim, hidden_dim))

        # --- Output projection: back to flat latent dim ---
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, flat_dim),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, n_patches, D) noisy latent tokens
            t: (B,) normalised time in [0, 1]
        Returns:
            eps_hat: (B, n_patches, D) predicted noise
        """
        B = z.shape[0]  # noqa: N806
        z_flat = z.reshape(B, -1)  # (B, n_patches*D)

        t_emb = self.time_embed(t)  # (B, t_embed_dim)
        t_global = self.time_proj(t_emb)  # (B, hidden_dim)

        h = self.input_proj(z_flat)  # (B, hidden_dim)
        h = h + t_global

        for (norm, lin1, lin2), t_proj in zip(self.blocks, self.t_projs):  # noqa: B905
            residual = h
            h = norm(h)
            h = lin1(h)
            h = h + t_proj(t_emb)
            h = F.silu(h)
            h = lin2(h)
            h = h + residual

        eps_flat = self.output_proj(h)  # (B, n_patches*D)
        return eps_flat.reshape(B, self.n_patches, self.latent_dim)


class LatentTransformerNoisePredictor(nn.Module):
    """
    Transformer noise predictor for latent token sequences.

    Operates natively on (B, n_patches, D) — no chunking needed.
    Time is injected as a prepended token, dropped at readout.

    Args:
        n_patches   : number of latent spatial tokens
        latent_dim  : token feature dimension D
        d_model     : transformer internal dimension
        n_heads     : number of attention heads
        n_layers    : number of transformer encoder layers
        d_ff        : feedforward dimension
        dropout     : dropout rate
        t_embed_dim : sinusoidal time embedding dimension
    """

    def __init__(
        self,
        n_patches: int,
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.latent_dim = latent_dim
        self.d_model = d_model

        # --- Time embedding ---
        self.time_embed = SinusoidalLearnableTimeEmbedding(t_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(t_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # --- Per-token input projection: latent_dim → d_model ---
        self.token_embed = nn.Linear(latent_dim, d_model)

        # --- Positional embedding: n_patches + 1 time token ---
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Readout: d_model → latent_dim per token ---
        self.final_norm = nn.LayerNorm(d_model)
        self.token_readout = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.token_readout.weight)
        nn.init.zeros_(self.token_readout.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, n_patches, D) noisy latent tokens
            t: (B,) normalised time in [0, 1]
        Returns:
            eps_hat: (B, n_patches, D) predicted noise
        """
        # Project tokens to d_model
        x = self.token_embed(z)  # (B, n_patches, d_model)

        # Build and prepend time token
        t_emb = self.time_embed(t)  # (B, t_embed_dim)
        t_tok = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([t_tok, x], dim=1)  # (B, n_patches+1, d_model)
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Drop time token, project back to latent_dim
        x = x[:, 1:, :]  # (B, n_patches, d_model)
        x = self.final_norm(x)
        return self.token_readout(x)  # (B, n_patches, D)
