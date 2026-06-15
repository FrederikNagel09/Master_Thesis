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

import math
import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from src.configs.general_config import GLOBAL_DEBUG_BOOL, probability_threshold

if TYPE_CHECKING:
    import numpy as np


class ParamNormalizer(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-6):
        """
        Streamlined per-dimension EMA normalizer.
        Functionally identical to the reference code running in 'per_dim' mode.
        """
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        # Running statistics tracks every coordinate independently
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_std", torch.ones(dim))

        # Reference trick: Fast-forward initialization on the first batch
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor, reverse: bool = False, training: bool = True) -> torch.Tensor:
        """
        Args:
            x:        Flat parameter/modulation tensor of shape (B, dim)
            reverse:  False = Normalize, True = Denormalize
            training: If True, updates running tracking buffers
        """
        if not reverse:
            # --- FORWARD: NORMALIZE ---
            if training:
                # Detach params just like the reference does in self.update(params.detach())
                detached_x = x.detach()
                batch_mean = detached_x.mean(dim=0)
                batch_std = detached_x.std(dim=0, unbiased=False).clamp(min=self.eps)

                with torch.no_grad():
                    if not bool(self.initialized):
                        # First batch snapshot: Hard copy stats to avoid slow burn-in
                        self.running_mean.copy_(batch_mean)
                        self.running_std.copy_(batch_std)
                        self.initialized.fill_(True)
                    else:
                        # Subsequent batches: Smoothly interpolate
                        self.running_mean.lerp_(batch_mean, self.momentum)
                        self.running_std.lerp_(batch_std, self.momentum)

            # Always normalize using the running buffers
            return (x - self.running_mean) / self.running_std.clamp(min=self.eps)

        else:
            # --- REVERSE: DENORMALIZE ---
            return x * self.running_std.clamp(min=self.eps) + self.running_mean


class WeightDiffusion(nn.Module):
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
        normalize: bool = True,
        lambda_kl: float = 5e-3,
        probablistic: bool = False,
        stop_gradient_flow: bool = True,
    ):
        super().__init__()
        # Initialize model components and noise schedule buffers
        self.data_dim = data_dim
        self.img_size = img_size
        self.denoiser = NoisePredictor
        self.weight_encoder = WeightEncoder
        self.inr = WeightEncoder.inr

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.sigma_tilde_factor = sigma_tilde_factor

        self.normalize = normalize
        self.probablistic = probablistic
        self.stop_gradient_flow = stop_gradient_flow

        if self.normalize:
            self.scaler = ParamNormalizer(WeightEncoder.modulation_dim, momentum=0.01, eps=1e-6)

        self.lambda_kl = lambda_kl

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
    def sample(
        self, n_samples: int = 1, coords: torch.Tensor | None = None, collect_snapshots: bool = False, debug: bool = False
    ) -> torch.Tensor:
        """
        Sample from the model by sampling weights and decoding to pixel space.
        """
        if collect_snapshots:
            theta, snapshots = self.sample_weight(n_samples, collect_snapshots=True, debug=debug)
            theta = self.weight_encoder.decode_modulations(theta)
            images = self.decode_weights(theta, coords)
            return images, snapshots

        theta = self.sample_weight(n_samples)
        theta = self.weight_encoder.decode_modulations(theta)
        return self.decode_weights(theta, coords)

    def loss(self, x: torch.Tensor, lambda_kl: float = 5e-3) -> torch.Tensor:
        """
        Computes the negative ELBO for a batch of input images x.
        """
        return self.negative_elbo(x, lambda_kl=lambda_kl)

    def compute_rec_loss(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Computes average reconstruction loss over the full validation set.

        Args:
            val_loader: Validation DataLoader yielding (x, _) batches. Shape (B, data_dim).
        Returns:
            Mean reconstruction loss (scalar float) across all batches.
        """
        self.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(next(self.parameters()).device)

                if self.probablistic:
                    mean, logvar = self.weight_encoder(x)
                    theta_prime_raw = self.weight_encoder._reparameterize(mean, logvar)
                else:
                    theta_prime_raw = self.weight_encoder(x)

                theta = self.weight_encoder.decode_modulations(theta_prime_raw)
                total_loss += self._l_rec(x, theta).mean().item()
                n_batches += 1

        return total_loss / n_batches

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes input images x into the latent space of the weight encoder.

        Args:
            x: (B, data_dim) input images, can be flat or spatial
        Returns:
            theta_prime_raw: (B, weight_dim) raw latent vectors before normalization
            mean: (B, weight_dim) mean of the latent distribution (only if probabilistic)
            logvar: (B, weight_dim) log variance of the latent distribution (only if probabilistic)
        """
        if self.probablistic:
            mean, logvar = self.weight_encoder(x)
            theta_prime_raw = self.weight_encoder._reparameterize(mean, logvar)

            if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
                std = torch.exp(0.5 * logvar)
                print(f"==================== Probablistic Components {self.i}: ====================")
                print(f"[Encoder] mu:     mean={mean.mean():.3f}, std={mean.std():.3f}, min={mean.min():.3f}, max={mean.max():.3f}")
                print(f"[Encoder] logvar: mean={logvar.mean():.3f}, std={logvar.std():.3f}, min={logvar.min():.3f}, max={logvar.max():.3f}")
                print(f"[Encoder] std:    mean={std.mean():.3f}, std={std.std():.3f}, min={std.min():.3f}, max={std.max():.3f}")
                print(
                    f"theta mean={theta_prime_raw.mean():.4f},",
                    f"std={theta_prime_raw.std():.4f}, min={theta_prime_raw.min():.4f}, max={theta_prime_raw.max():.4f}",
                )
                print("================================================================\n")
        else:
            theta_prime_raw = self.weight_encoder(x)
            mean = logvar = None  # type: ignore[assignment]
            # Split long debug print into two lines to satisfy line length limits
            if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
                print(f"==================== Deterministic Components {self.i}: ====================")
                print(f"theta mean={theta_prime_raw.mean():.4f}, std={theta_prime_raw.std():.4f}")
                print(f"theta min={theta_prime_raw.min():.4f}, max={theta_prime_raw.max():.4f}")

        return theta_prime_raw, mean, logvar

    def get_reconstructions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given input images x, returns the reconstructed images from the weight encoder and SIREN decoder.

        Args:
            x: (B, data_dim) input images, can be flat or spatial
        Returns:
            x_recon: (B, data_dim) reconstructed images, flattened to match input shape
        """
        theta_prime_raw, _, _ = self.encode(x)

        theta = self.weight_encoder.decode_modulations(theta_prime_raw)

        x_recon = self._inr_decode(theta)

        return x_recon

    # -------------------------------------------------------------------------
    # Negative ELBO Computation:
    # -------------------------------------------------------------------------
    def negative_elbo(self, x: torch.Tensor, lambda_kl: float = 1.0) -> torch.Tensor:
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

        # Sample random time step  t ~ Uniform{1, ..., T} - range [1, T]
        t_idx = torch.randint(1, self.T, (batch_size,), device=x.device)
        t_norm = t_idx.float() / (self.T - 1)

        theta_prime_raw, _, logvar = self.encode(x)

        theta_prime = self.scaler(theta_prime_raw, reverse=False) if self.normalize else theta_prime_raw

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("==================== DEBUG: Normalization ====================")
            print(
                f"DEBUG raw encoder: mean={theta_prime_raw.mean():.4f}, "
                f"std={theta_prime_raw.std():.4f}, "
                f"min={theta_prime_raw.min():.4f}, "
                f"max={theta_prime_raw.max():.4f}"
            )
            print(
                f"DEBUG normalized: mean={theta_prime.mean():.4f}, "
                f"std={theta_prime.std():.4f}, "
                f"min={theta_prime.min():.4f}, "
                f"max={theta_prime.max():.4f}"
            )
            print("==============================================================\n")

            # Prints forwars process statistics for the first batch only, at specific time steps
            if self.i % 10 == 0:
                print("\n######### Forward Process Statistics: #########")
                # 1. Define the steps we want to see
                t_steps = [
                    self.T - 1,
                    self.T * 0.9,
                    self.T * 0.8,
                    self.T * 0.7,
                    self.T * 0.6,
                    self.T * 0.5,
                    self.T * 0.4,
                    self.T * 0.3,
                    self.T * 0.2,
                    self.T * 0.1,
                    0,
                ]

                # 2. Convert to a long tensor on the correct device
                t_idx_debug = torch.tensor(t_steps, dtype=torch.long, device=theta_prime.device)

                for t in t_idx_debug:
                    # Use .item() for the index but keep the tensor for schedule lookup
                    idx = t.item()

                    # 3. Retrieve schedule parameters for this specific step
                    # We use [idx] to get the scalar, then unsqueeze to handle broadcasting
                    alpha_t = self.sqrt_alpha_cumprod[idx]
                    sigma_t = self.sigma[idx]

                    # 4. Generate the noisy sample (Forward Process)
                    epsilon_t = torch.randn_like(theta_prime)
                    # Note: theta_prime is (Batch, Dim), alpha_t is scalar
                    theta_t = alpha_t * theta_prime + sigma_t * epsilon_t

                    print(f"DEBUG SAMPLE t={idx:3d}: mean={theta_t.mean():.4f}, std={theta_t.std():.4f}")

                print("###############################################\n")
                print(f"DEBUG SCALED THETA: mean={theta_prime.mean():.4e}, std={theta_prime.std():.4e}")
                print(f"Debug range of scaled theta_prime: min={theta_prime.min().item():.4f}, max={theta_prime.max().item():.4f}")
        self.i += 1
        # Construct theta_t by adding noise to theta_prime according to the noise schedule at time step t_idx
        theta_prime = theta_prime.detach() if self.stop_gradient_flow else theta_prime

        theta_t, epsilon = self._forward_process(theta_prime, t_idx)

        # Given theta_t, and theta_prime we compute the three loss terms:
        l_diff = self._l_diff(theta_t, t_norm, epsilon, theta_prime, t_idx)

        theta = self.weight_encoder.decode_modulations(theta_prime_raw)
        l_rec = self._l_rec(x, theta)

        if self.probablistic:
            l_prior = self._l_entropy(logvar)
            elbo = (self.T - 1) * l_diff + l_rec - lambda_kl * l_prior  # minus, not plus
        else:
            l_prior = torch.zeros_like(l_diff)
            elbo = (self.T - 1) * l_diff + l_rec

        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    @torch.no_grad()
    def compute_full_elbo(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Computes the exact negative ELBO over the full validation set by looping
        over all T timesteps per batch. Matches training behavior exactly.

        Args:
            val_loader: DataLoader yielding (x, label) batches where x can be
                        flattened or spatial image tensors.
        Returns:
            mean Negative ELBO scalar (lower is better)
        """
        self.eval()
        device = self.sqrt_alpha_cumprod.device
        total_loss = 0.0
        n_batches = 0
        for x, _ in val_loader:
            x = x.to(device)
            batch_size = x.shape[0]

            # --------- 1. Input Shape Flattening ---------
            x_flat = x.reshape(batch_size, -1) if x.dim() > 2 else x

            # --------- 2. Encode Once Per Batch ---------
            if self.probablistic:
                mean, logvar = self.weight_encoder(x_flat)
                theta_prime_raw = self.weight_encoder._reparameterize(mean, logvar)
                l_prior = self._l_entropy(logvar)  # (B,)
            else:
                theta_prime_raw = self.weight_encoder(x_flat)
                l_prior = torch.zeros(batch_size, device=device)  # (B,)

            # --------- 3. Scale / Normalize Weights ---------
            theta_prime = self.scaler(theta_prime_raw, reverse=False) if self.normalize else theta_prime_raw

            # Apply gradient routing behavior identical to training flow
            theta_prime = theta_prime.detach() if self.stop_gradient_flow else theta_prime

            # --------- 4. Compute Non-Diffusion Core Losses ---------
            theta = self.weight_encoder.decode_modulations(theta_prime_raw)
            l_rec = self._l_rec(x_flat, theta, debug=False)  # (B,)

            # --------- 5. Integrate Diffusion Loss Over All T ---------
            l_diff_sum = torch.zeros(batch_size, device=device)  # (B,)

            for t in tqdm(range(self.T), desc="Sampling", total=self.T):
                # Construct uniform timestep configurations
                t_idx = torch.full((batch_size,), t, dtype=torch.long, device=device)
                t_norm = torch.full((batch_size,), t / (self.T - 1), device=device)

                # Generate the specific noisy weight vectors for timestep t
                theta_t, epsilon = self._forward_process(theta_prime, t_idx)

                # Accumulate the unscaled, step-specific weighted MSE loss
                # Fixed: Added theta_prime (x0) to the argument match list
                l_diff_sum += self._l_diff(theta_t, t_norm, epsilon, theta_prime, t_idx, debug=False)  # (B,)

            # --------- 6. Aggregate Total Negative ELBO Per Sample ---------
            # Scaled exactly to match your negative_elbo calculation configuration
            elbo = l_diff_sum + l_rec - self.lambda_kl * l_prior if self.probablistic else l_diff_sum + l_rec

            total_loss += elbo.mean().item()
            n_batches += 1

        self.train()
        return total_loss / n_batches

    # -------------------------------------------------------------------------
    # Loss term Helpers:
    # -------------------------------------------------------------------------
    def _l_rec(self, x, theta_prime_raw, debug=True) -> torch.Tensor:
        """
        Reconstruction is done my taking Theta Prime, decoding it to pixel space, and comparing to the original image x.

        The idea is for this loss term to push the Weight Encoder to produce Theta prime weights that create good reconstructed images.
        Essentially we want the weight encoder to procuse good weights that the diffusion process, then can learn to recreate.
        """
        # Send theta_prime through the shared SIREN decoder to get reconstructed images.
        x_recon = self._inr_decode(theta_prime_raw)
        # Flatten original images and to make comparison easier.
        x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)
        if x_recon.shape != x_flat.shape:
            x_recon = x_recon.view_as(x_flat)

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold and debug:
            print("############# Reconstruction Loss: #################")
            print("x_flat shape:", x_flat.shape)
            print(
                "x_flat.min():",
                x_flat.min(),
                "\nx_flat.max():",
                x_flat.max(),
                "\nx_flat.mean():",
                x_flat.mean(),
                "\nx_flat.std():",
                x_flat.std(),
            )
            print("x_recon shape:", x_recon.shape)
            print(
                "x_recon.min():",
                x_recon.min(),
                "\nx_recon.max():",
                x_recon.max(),
                "\nx_recon.mean():",
                x_recon.mean(),
                "\nx_recon.std():",
                x_recon.std(),
            )
            print("###############################################\n")

        return 0.5 * ((x_flat - x_recon) ** 2).sum(dim=-1)

    def _l_diff(self, theta_t, t_norm, epsilon, x0, t_idx, debug=True) -> torch.Tensor:
        """
        V-prediction diffusion loss: network predicts v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0.
        Args:
            theta_t:  (B, weight_dim) noisy weights at timestep t
            t_norm:   (B,) timestep normalised to [0, 1]
            epsilon:  (B, weight_dim) noise sample used to corrupt x0
            x0:       (B, weight_dim) clean weights
            t_idx:    (B,) integer timestep indices into the schedule buffers
        Returns:
            (B,) per-sample MSE loss between predicted and target v
        """
        v_hat = self.denoiser(theta_t, t_norm.unsqueeze(1))

        sqrt_ab = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sqrt_1mab = self.sigma[t_idx].unsqueeze(1)

        v_target = sqrt_ab * epsilon - sqrt_1mab * x0

        mse = F.mse_loss(v_hat, v_target, reduction="none")

        # Bin MSE by timestep to see where the model fails
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold and debug:
            t_flat = t_norm.flatten()
            low_t_mask = t_flat < 0.2
            high_t_mask = t_flat > 0.8
            if low_t_mask.any():
                print(f"MSE @ low  t (<0.2): {mse[low_t_mask].mean():.4f}")
            if high_t_mask.any():
                print(f"MSE @ high t (>0.8): {mse[high_t_mask].mean():.4f}")

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold and debug:
            print("############# Diffusion Loss: #################")
            print("v_target shape:", v_target.shape)
            print(
                "v_target.min():",
                v_target.min(),
                "\nv_target.max():",
                v_target.max(),
                "\nv_target.mean():",
                v_target.mean(),
                "\nv_target.std():",
                v_target.std(),
            )
            print("v_hat shape:", v_hat.shape)
            print(
                "v_hat.min():",
                v_hat.min(),
                "\nv_hat.max():",
                v_hat.max(),
                "\nv_hat.mean():",
                v_hat.mean(),
                "\nv_hat.std():",
                v_hat.std(),
            )
            print("MSE shape:", mse.shape)
            print(
                "MSE.min():",
                mse.min(),
                "\nMSE.max():",
                mse.max(),
                "\nMSE.mean():",
                mse.mean(),
                "\nMSE.std():",
                mse.std(),
            )
            print("###############################################\n")

        return mse.mean(dim=-1)  # (B,)

    def _l_entropy(self, logvar: torch.Tensor) -> torch.Tensor:
        """
        Negative entropy of q(z|x) = N(mu, exp(logvar)), including constants.

        Args:
            logvar: (B, latent_dim, H', W')
        Returns:
            (B,) per-sample negative entropy
        """
        # Use .mean(dim=-1) to average across all latent dimensions cleanly.
        # Invert the sign to negative so that minimizing this term maximizes true entropy.
        entropy_per_dim = 0.5 * (1.0 + torch.log(torch.as_tensor(2.0 * math.pi, device=logvar.device)) + logvar)
        return entropy_per_dim.mean(dim=-1)  # (B,) — sum over latent dims, return positive entropy

    # -------------------------------------------------------------------------
    # Sampling Helpers:
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def decode_weights(self, weights: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        return self._inr_decode(weights, coords)

    @torch.no_grad()
    def sample_weight(
        self,
        n_samples: int = 1,
        collect_snapshots: bool = False,
        debug: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, np.ndarray]]:
        """
        Sample weight vectors via reverse diffusion with v-prediction.
        Args:
            n_samples:         Number of weight samples to generate.
            collect_snapshots: If True, also return weight distributions at T_VALUES.
        Returns:
            curr_theta: (n_samples, weight_dim) sampled weights.
            snapshots:  {t_value: flat np.ndarray} — only returned if collect_snapshots=True.
        """
        weight_dim = self.weight_encoder.modulation_dim
        device = self.sqrt_alpha_cumprod.device
        T_values = {self.T - 1, 3 * self.T // 4, self.T // 2, self.T // 4, 0}  # noqa: N806
        snapshots: dict[int, np.ndarray] = {}

        curr_theta = torch.randn(n_samples, weight_dim, device=device)

        if GLOBAL_DEBUG_BOOL and debug:
            fixed_theta = torch.randn(1, weight_dim, device=device)
            t_high = torch.full((1, 1), 999 / (self.T - 1), device=device)
            t_low = torch.full((1, 1), 0 / (self.T - 1), device=device)
            v_high = self.denoiser(fixed_theta, t_high)
            v_low = self.denoiser(fixed_theta, t_low)
            print("===== TIME SENSITIVITY CHECK =====")
            print(f"v_hat @ t=999: mean={v_high.mean():.4f}, std={v_high.std():.4f}")
            print(f"v_hat @ t=0  : mean={v_low.mean():.4f},  std={v_low.std():.4f}")
            print(f"max abs diff : {(v_high - v_low).abs().max():.4f}")
            print("==================================")

        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling", total=self.T):
            t_norm = torch.full((n_samples,), t / (self.T - 1), device=device).unsqueeze(-1)

            v_hat = self.denoiser(curr_theta, t_norm)  # (n_samples, weight_dim)

            sqrt_ab = self.sqrt_alpha_cumprod[t]  # scalar
            sqrt_1mab = self.sigma[t]  # scalar

            # Recover x0 and eps from v, then compute DDPM posterior mean
            x0_hat = sqrt_ab * curr_theta - sqrt_1mab * v_hat  # (n_samples, weight_dim)
            eps_hat = sqrt_1mab * curr_theta + sqrt_ab * v_hat  # (n_samples, weight_dim)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]

            # Standard DDPM posterior mean, now using x0_hat recovered from v
            coeff1 = 1.0 / torch.sqrt(alpha_t)
            coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = coeff1 * (curr_theta - coeff2 * eps_hat)

            curr_theta = mean + torch.sqrt(beta_t) * torch.randn_like(curr_theta) if t > 0 else mean

            if collect_snapshots and t in T_values:
                snapshots[t] = curr_theta.detach().cpu().numpy().flatten()

            if (t % 100 == 0 and GLOBAL_DEBUG_BOOL and debug) or (t == 0 and GLOBAL_DEBUG_BOOL and debug):
                print("################## Sampling: ##############################")
                print(f"Sampling step {t}/{self.T}:")
                print(
                    f"v_hat stats      : mean={v_hat.mean():.4f},      std={v_hat.std():.4f}",
                    f"min={v_hat.min():.4f}, max={v_hat.max():.4f}",
                )
                print(
                    f"x0_hat stats     : mean={x0_hat.mean():.4f},     std={x0_hat.std():.4f}",
                    f"min={x0_hat.min():.4f}, max={x0_hat.max():.4f}",
                )
                print(
                    f"curr_theta stats : mean={curr_theta.mean():.4f}, std={curr_theta.std():.4f}",
                    f"min={curr_theta.min():.4f}, max={curr_theta.max():.4f}",
                )
                print("###########################################################\n")

        if self.normalize:
            curr_theta = self.scaler(curr_theta, reverse=True, training=False)

        if collect_snapshots:
            return curr_theta, snapshots
        return curr_theta

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
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("==================== DEBUG: _inr_decode.py ====================")
            print(f"Decoded pixels shape: {pixels.shape}")
            print(f"Pixel value range: {pixels.min().item():.4f} to {pixels.max().item():.4f}")
            print("================================================================")

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

    def _forward_process(self, theta_prime, t_idx):
        """
        Given Theta Prime, we construct the noise variant theta_t at time step t_idx using the noise schedule parameters.

        Returns:
        - theta_t: The noisy version of theta_prime at time step t_idx.
        - epsilon: The noise added to theta_prime to get theta_t.
        """
        # Initialize time step parameters
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)

        # Randomly sample noise epsilon from standard normal distribution
        epsilon = torch.randn_like(theta_prime)

        # Construct theta_t using the noise schedule formula
        theta_t = alpha_t * theta_prime + sigma_t * epsilon

        return theta_t, epsilon
