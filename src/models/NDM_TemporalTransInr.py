"""
NDM_TemporalTransInr.py
NDMTemporalINR wired to TransInrTemporalEncoder as F_phi.

Inherits everything from NDMTemporalINR unchanged:
    _sample_zt, _l_diff, _l_prior, _l_rec, negative_elbo, loss

Only two things are overridden:
    _inr_decode  — inflate flat weights → param dict → SIREN (TransInr style)
    sample_weight — uses self.F_phi.weight_dim and spatial reshape

The key contract satisfied:
    self.F_phi = TransInrTemporalEncoder
    self.F_phi(x, t_norm) → (B, weight_dim)   x can be flat or spatial
    self.F_phi.weight_dim  → int
    self.F_phi.inr         → SIREN shared for decoding
    self.F_phi.inflate()   → flat → param dict
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.NDM_INR import NDMTemporalINR


class NDMTemporalTransInr(NDMTemporalINR):
    """
    NDMTemporalINR with TransInrTemporalEncoder as F_phi and
    the TransInr SIREN for decoding.

    Parameters
    ----------
    network          : noise predictor  ε_θ(z_t, t)
    encoder          : TransInrTemporalEncoder
    coord_grid       : (H, W, 2) coordinate grid for SIREN queries
    beta_1, beta_T, T, sigma_tilde_factor, data_dim, img_size
                     : forwarded to NeuralDiffusionModelINR unchanged
    """

    def __init__(
        self,
        network: nn.Module,
        encoder,  # TransInrTemporalEncoder
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
            F_phi=encoder,
            inr=encoder.inr,  # share the SIREN
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            sigma_tilde_factor=sigma_tilde_factor,
            data_dim=data_dim,
            img_size=img_size,
            use_modulation=False,  # modulation is inside the encoder
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
        param_dict = self.F_phi.inflate(flat_weights)

        # Hand params to the shared SIREN
        self.inr.set_params(param_dict)

        # Coordinate grid
        if coords is None:  # noqa: SIM108
            coord = self.trans_coord.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        else:
            coord = coords

        # SIREN forward: (B, H, W, 2) → (B, H, W, C_out)
        pixels = self.inr(coord)

        # Flatten and squeeze channel dim
        return pixels.reshape(B, -1)  # (B, H*W) for C_out=1

    # -------------------------------------------------------------------------
    # _l_rec override — NDMTemporalINR._l_rec compares flat x against x_recon
    # but does not normalise to [0,1]. Fix range here.
    # -------------------------------------------------------------------------

    def _l_rec(self, x: torch.Tensor, z_t: torch.Tensor, t_idx: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss — decodes through the noise-predicting network.
        x : (B, data_dim) flat, in [-1, 1]
        """
        eps_hat = self.network(z_t, t_norm.unsqueeze(1))
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)

        # Recover predicted clean weights via network
        x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        x_recon = self._inr_decode(x_hat)  # (B, H*W)

        x_flat = x.reshape(x.shape[0], -1)  # keep in [-1, 1], fix normalisation separately later
        
        if x_recon.shape != x_flat.shape:
            x_recon = x_recon.view_as(x_flat)

        return 0.5 * ((x_flat - x_recon) ** 2).sum(dim=-1)


    # -------------------------------------------------------------------------
    # Shared ELBO structure
    # -------------------------------------------------------------------------
    def negative_elbo(self, x: torch.Tensor):
        """
        Estimates the negative ELBO:
            L = E[ l_diff ] + prior_mask * l_prior + l_rec
        Parameters
        ----------
        x : (batch, data_dim)
        Returns
        -------
        (scalar mean loss, l_diff mean, l_prior mean, l_rec mean)
        """
        batch_size = x.shape[0]

        # Sample random time step  t ~ Uniform{1, ..., T}
        t_idx = torch.randint(1, self.T + 1, (batch_size,), device=x.device) - 1
        t_norm = t_idx.float() / (self.T - 1)

        # Forward: z_t ~ q(z_t | x)
        z_t, _, Wx = self._sample_zt(x, t_idx, t_norm.unsqueeze(1))  # noqa: N806

        # Three loss terms
        l_diff = self._l_diff(x, z_t, t_idx, t_norm, Wx)  # (batch,)
        l_prior = self._l_prior(x)  # (batch,)
        l_rec = self._l_rec(x, z_t, t_idx, t_norm)  # (batch,)

        # apply prior mask and scaling:
        prior_mask = (t_idx == self.T - 1).float()
        l_prior = prior_mask * l_prior

        # Combine to get ELBO (mean over batch)
        elbo = l_diff + l_prior + l_rec

        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()



    # -------------------------------------------------------------------------
    # Sampling — mirrors NDMTemporalINR.sample_weight but uses TransInr SIREN
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample_weight(self, n_samples: int = 1) -> torch.Tensor:
        weight_dim = self.F_phi.weight_dim
        device = self.sqrt_alpha_cumprod.device
        theta_t = torch.randn(n_samples, weight_dim, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDMTemporalTransInr Sampling", total=self.T):
            t_idx = torch.full((n_samples,), t, dtype=torch.long, device=device)
            t_norm = torch.full((n_samples, 1), t / max(self.T - 1, 1), device=device)

            eps_hat = self.network(theta_t, t_norm)
            alpha_t = self.sqrt_alpha_cumprod[t].unsqueeze(0)
            sigma_t = self.sigma[t].unsqueeze(0)
            theta_t_hat = (theta_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                return theta_t_hat

            s = t - 1
            s_idx = torch.full((n_samples,), s, dtype=torch.long, device=device)
            s_norm = torch.full((n_samples, 1), s / max(self.T - 1, 1), device=device)

            # Decode predicted weights → pixel space → re-encode at t and s
            x_hat_pixels = self._inr_decode(theta_t_hat)  # (B, H*W)

            Fx_hat_t = self.F_phi(x_hat_pixels, t_norm)  # noqa: N806
            Fx_hat_s = self.F_phi(x_hat_pixels, s_norm)  # noqa: N806

            alpha_s = self.sqrt_alpha_cumprod[s].view(1, 1)
            sigma_s_sq = self.sigma_sq[s].view(1, 1)
            sigma_t_val = self.sigma[t].view(1, 1)
            alpha_t_val = self.sqrt_alpha_cumprod[t].view(1, 1)
            sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx)[0].view(1, 1)

            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t_val.clamp(min=1e-6)
            mu = alpha_s * Fx_hat_s + coeff * (theta_t - alpha_t_val * Fx_hat_t)
            noise = torch.randn_like(theta_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(theta_t)
            theta_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        return theta_t_hat  # safety fallback
