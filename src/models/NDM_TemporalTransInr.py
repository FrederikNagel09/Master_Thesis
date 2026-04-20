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


class WeightScaler(nn.Module):
    def __init__(self, dim, momentum=0.1):
        super().__init__()
        self.dim = dim
        self.momentum = momentum

        # register_buffer ensures these stay with the model but are NOT trainable parameters
        self.register_buffer("running_mean", torch.zeros(1, dim))
        self.register_buffer("running_std", torch.ones(1, dim))

    def forward(self, x, reverse=False, training=True):
        """
        x: (batch_size, dim)
        reverse: False for encoding (to N(0,1)), True for decoding (back to INR scale)
        training: If True, updates the running stats.
        """
        if not reverse:
            if training:
                # Calculate current batch stats
                # Using keepdim=True to ensure broadcasting works smoothly
                batch_mean = x.mean(dim=0, keepdim=True)
                batch_std = x.std(dim=0, keepdim=True) + 1e-6

                # Update running statistics (Exponential Moving Average)
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std

                # Use current batch stats for standardization during training
                return (x - batch_mean) / batch_std
            else:
                # Use remembered stats for standardization during inference/validation
                return (x - self.running_mean) / self.running_std

        else:
            # Re-scaling for INR (Reverse process)
            return (x * self.running_std) + self.running_mean


class NDMTemporalTransInr(nn.Module):
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

    # -------------------------------------------------------------------------
    # Initialize Model
    # -------------------------------------------------------------------------
    def __init__(
        self,
        NoisePredictor: nn.Module,  # noqa: N803
        WeightEncoder: nn.Module,  # noqa: N803
        coord_grid: torch.Tensor,  # (H, W, 2)
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 1000,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
        data_dim: int = 784,
        img_size: int = 28,
    ):
        super().__init__()
        # Initialize model components and noise schedule buffers
        self.data_dim = data_dim
        self.img_size = img_size
        self.noise_predictor = NoisePredictor
        self.weight_encoder = WeightEncoder
        self.inr = WeightEncoder.inr

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.sigma_tilde_factor = sigma_tilde_factor

        # --- NEW: Learnable Scaler ---
        self.scaler = WeightScaler(WeightEncoder.weight_dim)

        # --- Noise schedule ---
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)

        self.i = 0

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq", 1.0 - alpha_cumprod)
        self.register_buffer("sigma", (1.0 - alpha_cumprod).sqrt())
        # --- Pre-build pixel coordinate grid for MNIST (28x28) ---
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (784, 2)
        self.register_buffer("coords", coords.unsqueeze(0))  # (1, 784, 2)
        # Register coord grid as buffer so it moves with the model
        self.register_buffer("trans_coord", coord_grid, persistent=False)

    # -------------------------------------------------------------------------
    # Main callable functions:
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n_samples: int = 1, coords: torch.Tensor | None = None) -> torch.Tensor:
        """
        Sample from the model by sampling weights and decoding to pixel space.
        """
        theta = self.sample_weight(n_samples)
        return self.decode_weights(theta, coords)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the negative ELBO for a batch of input images x.
        """
        return self.negative_elbo(x)

    # -------------------------------------------------------------------------
    # Negative ELBO Computation:
    # -------------------------------------------------------------------------
    def negative_elbo(self, x: torch.Tensor) -> torch.Tensor:
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

        # --- Sample random time step ---
        t_idx = torch.randint(1, self.T + 1, (batch_size,), device=x.device) - 1  # 0-indexed
        t_norm = t_idx.float() / (self.T - 1)

        # --- Forward process: z_t ~ q_phi(z_t | x) ---
        z_t, _, Fx_t = self._construct_theta_t(x, t_idx, t_norm)  # noqa: N806

        # --- Three terms of the objective ---
        l_diff = self._l_diff(x, z_t, t_idx, t_norm, Fx_t)  # (batch,)

        l_prior = self._l_prior(x)  # (batch,)

        l_rec = self._l_rec(x, z_t, t_idx)  # scalar

        prior_mask = 1.0 * (t_idx == self.T - 1).float()
        elbo = l_diff + prior_mask * l_prior + l_rec

        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Loss term Helpers:
    # -------------------------------------------------------------------------
    def _l_rec(self, x, z_t, t_idx) -> torch.Tensor:  # noqa: ARG002
        """
        Reconstruction is done my taking Theta Prime, decoding it to pixel space, and comparing to the original image x.

        The idea is for this loss term to push the Weight Encoder to produce Theta prime weights that create good reconstructed images.
        Essentially we want the weight encoder to procuse good weights that the diffusion process, then can learn to recreate.
        """
        B = x.shape[0]  # noqa: N806
        t_zero = torch.zeros(B, device=x.device)  # t_norm = 0 for all
        theta_0 = self.weight_encoder(x, t_zero)  # F_phi(x, 0)

        x_recon = self._inr_decode(theta_0)
        x_flat = x.reshape(B, -1).clamp(-1, 1)
        if x_recon.shape != x_flat.shape:
            x_recon = x_recon.view_as(x_flat)

        return 0.5 * ((x_flat - x_recon) ** 2).sum(dim=-1)  # (B,)

    def _l_diff(self, x, theta_t, t_idx, t_norm, theta_prime_t):
        """
        Computes L_diff for time-independent W(x).

        Predicts the clean weight vector theta_prime_hat from the noisy theta_t at
        timestep t, and computes the ELBO-weighted squared error against the true
        clean weight vector theta_prime:

            l_diff = ||theta_prime - theta_prime_hat||^2 / (2 * sigma_tilde^2(s, t))

        where theta_prime_hat = (theta_t - sigma_t * eps_hat) / alpha_t is the
        noise-free estimate recovered from the predicted noise eps_hat.
        """
        # Predict noise at time step t
        eps_hat = self.noise_predictor(theta_t, t_norm.unsqueeze(1))  # (batch, weight_dim)

        # Recover the noise-free estimate of theta_prime from the noisy theta_t and predicted noise eps_hat
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        theta_prime_t_hat = (theta_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        # Define next time step s = t-1
        s_idx = (t_idx - 1).clamp(min=0)
        s_norm = s_idx.float() / (self.T - 1)

        x_hat = self._inr_decode(theta_prime_t_hat)  # (batch, data_dim)

        Fx_hat_t = self.weight_encoder(x_hat, t_norm)  # noqa: N806
        Fx_hat_s = self.weight_encoder(x_hat, s_norm)  # noqa: N806
        Fx_s = self.weight_encoder(x, s_norm)  # noqa: N806

        alpha_s = self.sqrt_alpha_cumprod[s_idx].unsqueeze(1)
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).unsqueeze(1)  # compute once
        coeff = (self.sigma_sq[s_idx].unsqueeze(1) - sigma_tilde_sq).clamp(min=0).sqrt()
        coeff = coeff / self.sigma[t_idx].unsqueeze(1).clamp(min=1e-6)

        diff = alpha_s * (Fx_s - Fx_hat_s) + coeff * alpha_t * (Fx_hat_t - theta_prime_t)
        l_diff = (diff**2).sum(dim=-1) / (2.0 * sigma_tilde_sq.squeeze(1).clamp(min=1e-8))
        return l_diff

    def _l_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes the KL divergence between the initial noise distribution at time step T and the distribution of Theta Prime.
        """
        T_idx = self.T - 1  # noqa: N806
        t_norm_T = torch.ones(x.shape[0], device=x.device)  # noqa: N806
        Fx_T = self.weight_encoder(x, t_norm_T)  # (batch, 784)  # noqa: N806 #################################################

        sigma_T_sq = self.sigma_sq[T_idx]  # scalar  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # scalar  # noqa: N806
        d = self.weight_encoder.weight_dim  # dimensionality of the weight space

        # Eq. 20:  0.5 * [ d*(sigma_T^2 - log(sigma_T^2) - 1) + alpha_T^2 * ||F||^2 ]
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (Fx_T**2).sum(dim=-1))
        return kl  # (batch,)

    # -------------------------------------------------------------------------
    # Sampling Helpers:
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def decode_weights(self, weights: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        return self._inr_decode(weights, coords)

    @torch.no_grad()
    def sample_weight(self, n_samples: int = 1) -> torch.Tensor:
        shape = (n_samples, self.weight_encoder.weight_dim)  # (n_samples, weight_dim)
        device = self.sqrt_alpha_cumprod.device
        z_t = torch.randn(shape, device=device)

        # Pre-compute all scalar coefficients — avoids repeated indexing inside loop
        alpha = self.sqrt_alpha_cumprod  # (T,)
        sigma = self.sigma  # (T,)
        sigma_sq = self.sigma_sq  # (T,)
        T_minus_1 = max(self.T - 1, 1)  # noqa: N806

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM Sampling", total=self.T, leave=False):
            t_norm = torch.full((n_samples, 1), t / T_minus_1, device=device)  # (n, 1) for noise_predictor
            t_norm_flat = torch.full((n_samples,), t / T_minus_1, device=device)  # (n,) for weight_encoder

            # --- Predict x_hat ---
            eps_hat = self.noise_predictor(z_t, t_norm)
            alpha_t = alpha[t]
            sigma_t = sigma[t]
            theta_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)
            x_hat = self._inr_decode(theta_hat)  # (n_samples, data_dim)

            if t == 0:
                z_t = theta_hat
                break

            s = t - 1
            s_norm_flat = torch.full((n_samples,), s / T_minus_1, device=device)  # (n,) for weight_encoder

            # --- Batch both F_phi calls into one forward pass ---
            x_hat_2x = torch.cat([x_hat, x_hat], dim=0)  # (2*n, data_dim)
            t_norm_2x = torch.cat([s_norm_flat, t_norm_flat], dim=0)  # (2*n,)
            Fx_2x = self.weight_encoder(x_hat_2x, t_norm_2x)  # (2*n, weight_dim)  # noqa: N806
            Fx_hat_s, Fx_hat_t = Fx_2x.chunk(2, dim=0)  # noqa: N806

            # --- Pre-looked-up scalars, no .view() reshaping needed ---
            alpha_s = alpha[s]
            sigma_s_sq = sigma_sq[s]
            sigma_tilde_sq = self._sigma_tilde_sq(torch.tensor([s], device=device), torch.tensor([t], device=device))[0]

            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t.clamp(min=1e-6)
            mu = alpha_s * Fx_hat_s + coeff * (z_t - alpha_t * Fx_hat_t)

            z_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * torch.randn_like(z_t) if sigma_tilde_sq.item() > 0 else mu

        return z_t

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
        param_dict = self.weight_encoder.inflate(flat_weights)

        # Hand params to the shared SIREN
        self.inr.set_params(param_dict)

        # Coordinate grid
        if coords is None:  # noqa: SIM108
            coord = self.trans_coord.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        else:
            coord = coords

        # SIREN forward: (B, H, W, 2) → (B, H, W, C_out)
        pixels = self.inr(coord)

        # Flatten and squeeze channel dim → (B, H*W) for C_out=1
        return pixels.reshape(B, -1)

    # -------------------------------------------------------------------------
    # Basic Helpers:
    # -------------------------------------------------------------------------
    def _sigma_tilde_sq(self, s_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        sigma_s_sq = self.sigma_sq[s_idx]
        sigma_t_sq = self.sigma_sq[t_idx]
        alpha_t_sq = self.alpha_cumprod[t_idx]
        alpha_s_sq = self.alpha_cumprod[s_idx]

        base = (sigma_t_sq - alpha_t_sq / alpha_s_sq * sigma_s_sq) * sigma_s_sq / sigma_t_sq
        return self.sigma_tilde_factor * base

    def _construct_theta_t(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given Theta Prime, we construct the noise variant theta_t at time step t_idx using the noise schedule parameters.

        Returns:
        - theta_t: The noisy version of theta_prime at time step t_idx.
        - epsilon: The noise added to theta_prime to get theta_t.
        - Fx: The filtered version of the input x.
        """
        # Encoder image into Theta Prime
        Theta_prime_t = self.weight_encoder(x, t_norm)  # noqa: N806

        # Construct Theta_t using the noise schedule parameters for time step t_idx
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(Theta_prime_t)

        thata_t = alpha_t * Theta_prime_t + sigma_t * epsilon

        return thata_t, epsilon, Theta_prime_t
