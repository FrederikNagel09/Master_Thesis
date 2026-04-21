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


class NDMStaticTransInr(nn.Module):
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

        # Sample random time step  t ~ Uniform{1, ..., T} - range [1, T]
        t_idx = torch.randint(1, self.T + 1, (batch_size,), device=x.device) - 1
        # Normalize time step to [0, 1] for network input
        t_norm = t_idx.float() / (self.T - 1)

        # Send image through Weight Encoder to get Theta_prime
        theta_prime = self.weight_encoder(x)  # (batch, weight_dim)

        if GLOBAL_DEBUG_BOOL:
            print(
                f"DEBUG normalized: mean={theta_prime.mean():.4f}, "
                f"std={theta_prime.std():.4f}, "
                f"min={theta_prime.min():.4f}, "
                f"max={theta_prime.max():.4f}"
            )

            # Prints forwars process statistics for the first batch only, at specific time steps
            if self.i == 1:
                print("\n######### Forward Process Statistics: #########")
                # 1. Define the steps we want to see
                t_steps = [999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

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
                self.i += 1

                print(f"DEBUG SCALED THETA: mean={theta_prime.mean():.4e}, std={theta_prime.std():.4e}")
                print(f"Debug range of scaled theta_prime: min={theta_prime.min().item():.4f}, max={theta_prime.max().item():.4f}")

        # Construct theta_t by adding noise to theta_prime according to the noise schedule at time step t_idx
        theta_t, epsilon = self._construct_theta_t(theta_prime, t_idx)

        # Given theta_t, and theta_prime we compute the three loss terms:
        l_diff = self._l_diff(theta_t, t_norm, epsilon)  # (batch,)
        l_prior = self._l_prior(theta_prime=theta_prime)  # (batch,)

        l_rec = self._l_rec(x, theta_prime)

        # apply prior mask and scaling:
        prior_mask = (t_idx == self.T - 1).float()
        l_prior = prior_mask * l_prior

        elbo = l_diff + l_prior + l_rec

        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Loss term Helpers:
    # -------------------------------------------------------------------------
    def _l_rec(self, x, theta_prime_raw) -> torch.Tensor:
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

        return 0.5 * ((x_flat - x_recon) ** 2).sum(dim=-1)

    def _l_diff(self, theta_t, t_norm, epsilon):
        """
        Computes L_diff for time-independent W(x).

        Predicts the clean weight vector theta_prime_hat from the noisy theta_t at
        timestep t, and computes the ELBO-weighted squared error against the true
        clean weight vector theta_prime:

            l_diff = ||theta_prime - theta_prime_hat||^2 / (2 * sigma_tilde^2(s, t))

        where theta_prime_hat = (theta_t - sigma_t * eps_hat) / alpha_t is the
        noise-free estimate recovered from the predicted noise eps_hat.
        """
        # Predict noise at time step t_idx using the noise predictor network
        eps_hat = self.noise_predictor(theta_t, t_norm.unsqueeze(1))  # (batch, weight_dim)

        # 5. SIMPLE MSE LOSS
        return F.mse_loss(eps_hat, epsilon, reduction="none").sum(dim=-1)

    def _l_prior(self, theta_prime: torch.Tensor) -> torch.Tensor:
        """
        computes the KL divergence between the initial noise distribution at time step T and the distribution of Theta Prime.
        """
        T_idx = self.T - 1  # noqa: N806
        sigma_T_sq = self.sigma_sq[T_idx]  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # noqa: N806
        d = theta_prime.shape[-1]

        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (theta_prime**2).sum(dim=-1))
        return kl

    # -------------------------------------------------------------------------
    # Sampling Helpers:
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def decode_weights(self, weights: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        return self._inr_decode(weights, coords)

    @torch.no_grad()
    def sample_weight(self, n_samples: int = 1) -> torch.Tensor:
        weight_dim = self.weight_encoder.weight_dim
        device = self.sqrt_alpha_cumprod.device
        clip_value = 3
        # 1. Start from pure Gaussian noise
        curr_theta = torch.randn(n_samples, weight_dim, device=device)

        # The loop runs from T-1 down to 0
        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM Sampling", total=self.T):
            # Prepare normalized time input (Batch,) or (Batch, 1) based on your predictor
            t_norm = torch.full((n_samples, 1), t / (self.T - 1), device=device)

            # 2. Predict the noise added to the current state
            eps_hat = self.noise_predictor(curr_theta, t_norm)

            # 3. Retrieve schedule parameters
            alpha_bar = self.alpha_cumprod[t]
            alpha = self.alpha[t]
            beta = self.beta[t]

            # 4. Predict the clean weights (x0)
            # This formula projects the current noisy sample back to the "signal"
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

            theta_0 = (curr_theta - sqrt_one_minus_alpha_bar * eps_hat) / torch.sqrt(alpha_bar)

            # 5. Clip predicted weights to the training distribution
            theta_0_clipped = torch.clamp(theta_0, -clip_value, clip_value)

            # 6. Step back to z_{t-1}
            if t > 0:
                alpha_bar_prev = self.alpha_cumprod[t - 1]

                # Coefficients to calculate the mean of the distribution p(z_{t-1} | z_t, x_0)
                coeff_x0 = (torch.sqrt(alpha_bar_prev) * beta) / (1.0 - alpha_bar)
                coeff_xt = (torch.sqrt(alpha) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)

                mean = coeff_x0 * theta_0_clipped + coeff_xt * curr_theta

                # Use the posterior variance (more stable than raw beta)
                sigma = torch.sqrt(beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))

                noise = torch.randn_like(curr_theta)
                curr_theta = mean + sigma * noise
            else:
                # At t=0, the denoised sample is just our predicted x0
                curr_theta = theta_0_clipped

            if GLOBAL_DEBUG_BOOL:
                print(
                    "DEBUG theta_0 before clipping: "
                    f"mean={theta_0.mean():.4f}, std={theta_0.std():.4f}, "
                    f"min={theta_0.min():.4f}, max={theta_0.max():.4f}"
                )
                print(
                    "DEBUG theta_0 after clipping: "
                    f"mean={theta_0_clipped.mean():.4f}, std={theta_0_clipped.std():.4f}, "
                    f"min={theta_0_clipped.min():.4f}, max={theta_0_clipped.max():.4f}"
                )
                # Debug prints to verify the explosion is gone
                if t % 100 == 0:
                    print(f"DEBUG SAMPLE t={t}: mean={curr_theta.mean():.4f}, std={curr_theta.std():.4f}")

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

    def _construct_theta_t(self, theta_prime, t_idx):
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
