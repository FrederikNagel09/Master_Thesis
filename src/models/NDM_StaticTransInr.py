"""
NDM_StaticTransInr.py
NDMStaticINR wired to TransInrEncoder as W(x).

Inherits everything from NDMStaticINR unchanged:
    _sample_zt, _l_diff, _l_prior, sample_weight

Only two things are overridden:
    _inr_decode  — inflate flat weights → param dict → SIREN (TransInr style)
    _l_rec       — normalise target to [0, 1] to match SIREN output space

The key contract satisfied:
    self.W = TransInrEncoder
    self.W(x)         → (B, weight_dim)   x can be flat or spatial
    self.W.weight_dim → int
    self.W.inr        → SIREN shared for decoding
    self.W.inflate()  → flat → param dict
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.configs.general_config import GLOBAL_DEBUG_BOOL
from src.models.NDM_INR import NDMStaticINR

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class NDMStaticTransInr(NDMStaticINR):
    """
    NDMStaticINR with TransInrEncoder as W(x) and the TransInr SIREN
    for decoding.

    Parameters
    ----------
    network          : noise predictor  ε_θ(z_t, t)
    encoder          : TransInrEncoder
    coord_grid       : (H, W, 2) coordinate grid for SIREN queries
    beta_1, beta_T, T, sigma_tilde_factor, data_dim, img_size
                     : forwarded to NeuralDiffusionModelINR unchanged
    """

    def __init__(
        self,
        network: nn.Module,
        encoder,  # TransInrEncoder
        coord_grid: torch.Tensor,  # (H, W, 2)
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 1000,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
        data_dim: int = 784,
        img_size: int = 28,
    ):
        super().__init__(
            network=network,
            W=encoder,
            inr=encoder.inr,  # share the SIREN
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            sigma_tilde_factor=sigma_tilde_factor,
            data_dim=data_dim,
            img_size=img_size,
            use_modulation=False,  # modulation lives inside the encoder
        )

        # Register coord grid as buffer so it moves with the model
        self.register_buffer("trans_coord", coord_grid, persistent=False)

    # -------------------------------------------------------------------------
    # INR decode  —  inflate → set_params → SIREN
    # -------------------------------------------------------------------------

    def _inr_decode(
        self,
        flat_weights: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode flat weight vectors to pixel values using the TransInr SIREN.

        Parameters
        ----------
        flat_weights : (B, weight_dim)
        coords       : optional; uses trans_coord if None

        Returns
        -------
        pixels : (B, H*W)
        """
        B = flat_weights.shape[0]  # noqa: N806

        # Inflate flat vector → structured param dict
        param_dict = self.W.inflate(flat_weights)

        # Hand params to the shared SIREN
        self.inr.set_params(param_dict)

        # Coordinate grid
        if coords is None:  # noqa: SIM108
            coord = self.trans_coord.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        else:
            coord = coords

        # SIREN forward: (B, H, W, 2) → (B, H, W, C_out)
        pixels = self.inr(coord)
        if GLOBAL_DEBUG_BOOL and random.random() < 0.1:
            print("==================== DEBUG: _inr_decode.py ====================")
            print(f"Decoded pixels shape: {pixels.shape}")
            print(f"Pixel value range: {pixels.min().item():.4f} to {pixels.max().item():.4f}")
            print("================================================================")

        # Flatten and squeeze channel dim → (B, H*W) for C_out=1
        return pixels.reshape(B, -1)

    # -------------------------------------------------------------------------
    # _l_rec override — fix range mismatch between [-1,1] input and [0,1] SIREN
    # -------------------------------------------------------------------------

    def _l_rec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss at t = 0.
        x : (B, data_dim) flat, in [-1, 1]
        """
        weights = self.W(x)
        x_recon = self._inr_decode(weights)  # now outputs [-1, 1] via tanh
        if GLOBAL_DEBUG_BOOL and random.random() < 0.1:
            print("==================== DEBUG: _l_rec.py 1====================")
            print(f"x_recon (reconstructed images): min {x_recon.min().item():.4f}, max {x_recon.max().item():.4f}")
            print(f"shape x_recon: {x_recon.shape}")
            print("================================================================")

        x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)
        if x_recon.shape != x_flat.shape:
            x_recon = x_recon.view_as(x_flat)

        return 0.5 * ((x_flat - x_recon) ** 2).sum(dim=-1)
