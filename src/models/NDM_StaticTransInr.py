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

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from src.configs.general_config import GLOBAL_DEBUG_BOOL, probability_threshold


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

                if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
                    print(f"DEBUG WeightScaler Batch Mean: {batch_mean.mean().item():.4f}, Batch Std: {batch_std.mean().item():.4f}")

                # Update running statistics (Exponential Moving Average)
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std

                # Use current batch stats for standardization during training
                print(f"DEBUG WeightScaler Forward: batch mean {batch_mean.mean().item():.4f} and std {batch_mean.std().item():.4f}")
                print(f"DEBUG WeightScaler Forward: batch std {batch_std.mean().item():.4f} and std {batch_std.std().item():.4f}")
                return (x - batch_mean) / batch_std
            else:
                # Use remembered stats for standardization during inference/validation
                return (x - self.running_mean) / self.running_std

        else:
            # Re-scaling for INR (Reverse process)
            print(
                f"DEBUG WeightScaler Reverse: running mean {self.running_mean.mean().item():.4f} and {self.running_mean.std().item():.4f}"
            )
            print(
                f"DEBUG WeightScaler Reverse: running std {self.running_std.mean().item():.4f} and std {self.running_std.std().item():.4f}"
            )
            return (x * self.running_std) + self.running_mean


class NDMStaticTransInr(nn.Module):
    """
    NDMStaticINR with TransInrEncoder as W(x) and INRModulator for decoding.

    Pipeline:
        Train : encoder → scaler → [diffusion] → modulator → SIREN → pixels
        Sample: randn   → denoise              → modulator → SIREN → pixels

    Parameters
    ----------
    NoisePredictor   : noise predictor ε_θ(z_t, t)
    WeightEncoder    : TransInrEncoder  (image → flat trans_out)
    WeightModulator  : INRModulator     (flat trans_out → flat INR weights → SIREN)
    coord_grid       : (H, W, 2) coordinate grid for SIREN queries
    beta_1, beta_T, T, sigma_tilde_factor, data_dim, img_size
                     : forwarded to noise schedule unchanged
    """

    def __init__(
        self,
        NoisePredictor: nn.Module,  # noqa: N803
        WeightEncoder: nn.Module,  # noqa: N803
        WeightModulator: nn.Module,  # noqa: N803
        coord_grid: torch.Tensor,  # (H, W, 2)
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 1000,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
        data_dim: int = 784,
        img_size: int = 28,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.img_size = img_size
        self.noise_predictor = NoisePredictor
        self.weight_encoder = WeightEncoder
        self.weight_modulator = WeightModulator

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.sigma_tilde_factor = sigma_tilde_factor

        # Scaler now operates on trans_out_dim (not weight_dim)
        self.scaler = WeightScaler(WeightEncoder.trans_out_dim)

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

        # Pixel coordinate grid
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (784, 2)
        self.register_buffer("coords", coords.unsqueeze(0))  # (1, 784, 2)
        self.register_buffer("trans_coord", coord_grid, persistent=False)

    # -------------------------------------------------------------------------
    # Main callable functions
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n_samples: int = 1, coords: torch.Tensor | None = None) -> torch.Tensor:
        """Sample from the model by denoising → modulating → decoding."""
        flat_weights = self.sample_weight(n_samples)
        return self.decode_weights(flat_weights, coords)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the negative ELBO for a batch of input images x."""
        return self.negative_elbo(x)

    # -------------------------------------------------------------------------
    # Negative ELBO
    # -------------------------------------------------------------------------
    def negative_elbo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates the negative ELBO: L = E[l_diff] + prior_mask * l_prior + l_rec
        Args
        ----
        x : (B, data_dim)
        Returns
        -------
        (scalar mean loss, l_diff mean, l_prior mean, l_rec mean)
        """
        batch_size = x.shape[0]
        t_idx = torch.randint(1, self.T + 1, (batch_size,), device=x.device) - 1
        t_norm = t_idx.float() / (self.T - 1)

        # Encode image → flat trans_out, then scale to ~N(0,1)
        trans_out_raw = self.weight_encoder(x)  # (B, trans_out_dim)
        trans_out_scaled = self.scaler(trans_out_raw, reverse=False)  # (B, trans_out_dim)

        if GLOBAL_DEBUG_BOOL:
            print(
                f"DEBUG raw encoder: mean={trans_out_raw.mean():.4f}, "
                f"std={trans_out_raw.std():.4f}, "
                f"min={trans_out_raw.min():.4f}, "
                f"max={trans_out_raw.max():.4f}"
            )
            print(
                f"DEBUG scaled: mean={trans_out_scaled.mean():.4f}, "
                f"std={trans_out_scaled.std():.4f}, "
                f"min={trans_out_scaled.min():.4f}, "
                f"max={trans_out_scaled.max():.4f}"
            )

        # Diffusion loss operates on scaled trans_out
        theta_t, epsilon = self._construct_theta_t(trans_out_scaled, t_idx)
        l_diff = self._l_diff(theta_t, t_norm, epsilon)  # (B,)
        l_prior = self._l_prior(theta_prime=trans_out_scaled)  # (B,)

        # Reconstruction: scaled trans_out → modulator → SIREN → pixels
        l_rec = self._l_rec(x, trans_out_scaled)  # (B,)

        prior_mask = (t_idx == self.T - 1).float()
        l_prior = prior_mask * l_prior

        elbo = l_diff + l_prior + l_rec
        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Loss term helpers
    # -------------------------------------------------------------------------
    def _l_rec(self, x: torch.Tensor, trans_out_scaled: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: modulate scaled trans_out → decode → MSE vs x.
        Args
        ----
        x                : (B, data_dim)
        trans_out_scaled : (B, trans_out_dim)
        Returns
        -------
        loss : (B,)
        """
        x_recon = self._inr_decode(trans_out_scaled)
        x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)
        if x_recon.shape != x_flat.shape:
            x_recon = x_recon.view_as(x_flat)
        return 0.5 * ((x_flat - x_recon) ** 2).sum(dim=-1)

    def _l_diff(self, theta_t: torch.Tensor, t_norm: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Diffusion loss: MSE between predicted and true noise.
        Args
        ----
        theta_t : (B, trans_out_dim)  noisy scaled trans_out
        t_norm  : (B,)                normalised time step
        epsilon : (B, trans_out_dim)  true noise
        Returns
        -------
        loss : (B,)
        """
        eps_hat = self.noise_predictor(theta_t, t_norm.unsqueeze(1))
        return F.mse_loss(eps_hat, epsilon, reduction="none").mean(dim=-1)

    def _l_prior(self, theta_prime: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between q(z_T | x) and N(0, I).
        Args
        ----
        theta_prime : (B, trans_out_dim)
        Returns
        -------
        kl : (B,)
        """
        T_idx = self.T - 1  # noqa: N806
        sigma_T_sq = self.sigma_sq[T_idx]  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # noqa: N806
        d = theta_prime.shape[-1]
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (theta_prime**2).sum(dim=-1))
        return kl

    # -------------------------------------------------------------------------
    # Sampling helpers
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def decode_weights(self, flat_weights: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """Decode flat INR weights to pixels via SIREN."""
        return self._inr_decode(flat_weights, coords)

    @torch.no_grad()
    def sample_weight(self, n_samples: int = 1) -> torch.Tensor:
        """
        Denoise from pure Gaussian noise → modulate → return flat INR weights.
        Args
        ----
        n_samples : number of samples
        Returns
        -------
        flat_weights : (B, weight_dim)  ready for SIREN decoding
        """
        trans_out_dim = self.weight_encoder.trans_out_dim
        device = self.sqrt_alpha_cumprod.device
        clip_value = 3

        curr_theta = torch.randn(n_samples, trans_out_dim, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM Sampling", total=self.T):
            t_norm = torch.full((n_samples, 1), t / (self.T - 1), device=device)
            eps_hat = self.noise_predictor(curr_theta, t_norm)

            alpha_bar = self.alpha_cumprod[t]
            alpha = self.alpha[t]
            beta = self.beta[t]

            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
            theta_0 = (curr_theta - sqrt_one_minus_alpha_bar * eps_hat) / torch.sqrt(alpha_bar)
            theta_0_clipped = torch.clamp(theta_0, -clip_value, clip_value)

            if t > 0:
                alpha_bar_prev = self.alpha_cumprod[t - 1]
                coeff_x0 = (torch.sqrt(alpha_bar_prev) * beta) / (1.0 - alpha_bar)
                coeff_xt = (torch.sqrt(alpha) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)
                mean = coeff_x0 * theta_0_clipped + coeff_xt * curr_theta
                sigma = torch.sqrt(beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                curr_theta = mean + sigma * torch.randn_like(curr_theta)
            else:
                curr_theta = theta_0_clipped

            if GLOBAL_DEBUG_BOOL and t % 100 == 0:
                print(f"DEBUG SAMPLE t={t}: mean={curr_theta.mean():.4f}, std={curr_theta.std():.4f}")

        # Denoised scaled trans_out → modulator → flat INR weights
        return self.weight_modulator(curr_theta)  # (B, weight_dim)

    # -------------------------------------------------------------------------
    # INR decode
    # -------------------------------------------------------------------------
    def _inr_decode(
        self,
        trans_out_scaled: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Modulate scaled trans_out → flat INR weights → SIREN → pixels.
        Args
        ----
        trans_out_scaled : (B, trans_out_dim)
        coords           : optional coord grid; uses trans_coord if None
        Returns
        -------
        pixels : (B, H*W)
        """
        B = trans_out_scaled.shape[0]  # noqa: N806

        # Modulate → flat INR weights
        flat_weights = self.weight_modulator(trans_out_scaled)  # (B, weight_dim)

        # Inflate flat weights → param dict → set on SIREN
        param_dict = self.weight_modulator.inflate_weights(flat_weights)
        self.weight_modulator.inr.set_params(param_dict)

        if coords is None:  # noqa: SIM108
            coord = self.trans_coord.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        else:
            coord = coords

        pixels = self.weight_modulator.inr(coord)

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("==================== DEBUG: _inr_decode ====================")
            print(f"Pixel value range: {pixels.min().item():.4f} to {pixels.max().item():.4f}")
            print("=============================================================")

        return pixels.reshape(B, -1)

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------
    def _sigma_tilde_sq(self, s_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        sigma_s_sq = self.sigma_sq[s_idx]
        sigma_t_sq = self.sigma_sq[t_idx]
        alpha_t_sq = self.alpha_cumprod[t_idx]
        alpha_s_sq = self.alpha_cumprod[s_idx]
        base = (sigma_t_sq - alpha_t_sq / alpha_s_sq * sigma_s_sq) * sigma_s_sq / sigma_t_sq
        return self.sigma_tilde_factor * base

    def _construct_theta_t(self, theta_prime: torch.Tensor, t_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to theta_prime at time step t_idx.
        Args
        ----
        theta_prime : (B, trans_out_dim)
        t_idx       : (B,)
        Returns
        -------
        theta_t : (B, trans_out_dim), epsilon : (B, trans_out_dim)
        """
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(theta_prime)
        theta_t = alpha_t * theta_prime + sigma_t * epsilon
        return theta_t, epsilon
