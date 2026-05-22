from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from src.configs.general_config import GLOBAL_DEBUG_BOOL, probability_threshold

if TYPE_CHECKING:
    import numpy as np


class LatentScaler(nn.Module):
    def __init__(self, latent_dim: int, momentum: float = 0.1):
        """
        EMA-based normalizer for latent feature maps.
        Args:
            latent_dim: channel depth of latent (C)
            momentum:   EMA update rate
        Returns: None
        """
        super().__init__()
        self.momentum = momentum
        # Shape (1, C, 1, 1) for broadcasting over (B, C, H', W')
        self.register_buffer("running_mean", torch.zeros(1, latent_dim, 1, 1))
        self.register_buffer("running_std", torch.ones(1, latent_dim, 1, 1))

    def forward(self, z: torch.Tensor, reverse: bool = False, training: bool = True) -> torch.Tensor:
        """
        Normalize or denormalize a latent feature map.
        Args:
            z:        (B, latent_dim, H', W')
            reverse:  False = normalize to N(0,1), True = denormalize
            training: if True, updates EMA stats (only relevant when reverse=False)
        Returns:
            z_scaled: (B, latent_dim, H', W')
        """
        if not reverse:
            if training:
                # Per-channel stats, reducing over B, H', W'
                batch_mean = z.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
                batch_std = z.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std

                return (z - batch_mean) / batch_std
            else:
                return (z - self.running_mean) / self.running_std
        else:
            return z * self.running_std + self.running_mean


class NDMLatentDiffusion(nn.Module):
    """
    Latent-space diffusion model using noise prediction (ε-prediction).

    Pipeline:
        x → LatentEncoder → z (B, latent_dim, H', W')
        z → forward diffusion → z_t
        z_t + t → NoisePredictor → ε_hat                (l_diff)
        z → TransInr decoder → x_hat                    (l_rec)
        z flattened at T → KL prior                     (l_prior)

    Parameters
    ----------
    noise_predictor : ε_θ(z_t, t), operates on (B, latent_dim, H', W')
    latent_encoder  : x (B, C, H, W) → z (B, latent_dim, H', W')
    decoder         : TransInr, takes (B, latent_dim, H', W') → (B, C, H, W)
    coord_grid      : (H, W, 2) coordinate grid for SIREN queries
    latent_dim      : channel depth of latent feature map
    latent_size     : (H', W') spatial size of latent feature map
    beta_1, beta_T, T : noise schedule parameters
    data_dim        : flattened image size (H*W)
    img_size        : spatial image size
    """

    def __init__(
        self,
        noise_predictor: nn.Module,
        latent_encoder: nn.Module,
        decoder: nn.Module,
        coord_grid: torch.Tensor,  # (H, W, 2)
        latent_dim: int,
        latent_size: tuple[int, int],
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 1000,  # noqa: N803
        data_dim: int = 784,
        img_size: int = 28,
        normalize: bool = True,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)
        self.n_patches = self.latent_size[0] * self.latent_size[1]

        self.noise_predictor = noise_predictor
        self.latent_encoder = latent_encoder
        self.decoder = decoder

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self._normalize = normalize

        self.i = 0
        self.latent_scaler = LatentScaler(latent_dim)
        self.lambda_kl = 1e-3

        # --- Noise schedule ---
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)

        # Calculate the exact ELBO loss weight coefficient for noise prediction
        # w(t) = beta_t / (2 * alpha_t * (1 - alpha_cumprod_t))
        l_diff_weights = beta / (2 * alpha * (1.0 - alpha_cumprod))

        # Register it alongside your other buffers
        self.register_buffer("l_diff_weights", l_diff_weights)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq", 1.0 - alpha_cumprod)
        self.register_buffer("sigma", (1.0 - alpha_cumprod).sqrt())
        self.register_buffer("coord_grid", coord_grid, persistent=False)

        self.register_buffer("latent_mean", torch.zeros(1, latent_dim, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, latent_dim, 1, 1))
        self._latent_stats_set = False

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Compute the negative ELBO for a batch of images.

        Args:
            x: (B, C, H, W)
        Returns:
            (total_loss, l_diff, l_prior, l_rec) — scalar means
        """
        return self._negative_elbo(x)

    @torch.no_grad()
    def sample(self, n_samples: int = 1, collect_snapshots: bool = False) -> torch.Tensor:
        """
        Generate images by reverse diffusion in latent space.

        Args:
            n_samples: number of images to generate
            collect_snapshots: whether to collect snapshots during sampling
        Returns:
            images: (n_samples, data_dim)
        """
        # Sample latents:
        if collect_snapshots:
            z, snapshots = self._sample_latent(n_samples, collect_snapshots=True)
            if self._normalize:
                z = self._denormalize_z(z)
            return self._decode_latent(z), snapshots
        else:
            z = self._sample_latent(n_samples, collect_snapshots=collect_snapshots)  # (B, latent_dim, H', W')
            if self._normalize:
                z = self._denormalize_z(z)
            return self._decode_latent(z)  # (B, data_dim)

    # -------------------------------------------------------------------------
    # Normalization stuff
    # -------------------------------------------------------------------------

    def _normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """Standardize latents to approx N(0, I). Args: z (B, C, H', W'). Returns: same shape."""
        return self.latent_scaler(z, reverse=False, training=self.training)

    def _denormalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """Invert _normalize_z. Args: z (B, C, H', W'). Returns: same shape."""
        return self.latent_scaler(z, reverse=True)

    # -------------------------------------------------------------------------
    # ELBO
    # -------------------------------------------------------------------------
    def _negative_elbo(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Estimates L = l_diff + l_rec + l_prior.

        Args:
            x: (B, C, H, W)
        Returns:
            (total_loss, l_diff, l_prior, l_rec) — scalar means
        """
        B = x.shape[0]  # noqa: N806

        if x.dim() == 2:
            channels = self.data_dim // (self.img_size * self.img_size)
            x = x.view(B, channels, self.img_size, self.img_size)

        mu, logvar = self.latent_encoder(x)
        z_raw = self.latent_encoder.reparameterize(mu, logvar)
        z = self._normalize_z(z_raw) if self._normalize else z_raw

        t_idx = torch.randint(0, self.T, (B,), device=x.device)
        t_norm = t_idx.float().unsqueeze(-1) / (self.T - 1)  # (B, 1)

        z_t, epsilon = self._forward_process(z.detach(), t_idx)

        l_diff = self._l_diff(z_t, t_norm, t_idx, epsilon)
        l_prior = self._l_prior(mu, logvar)
        l_rec = self._l_rec(x, z_raw)

        total = l_diff + self.lambda_kl * l_prior + l_rec

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("############# Negative ELBO: #################")
            print("data x shape:", x.shape)
            print("time indices (t_idx):", t_idx.shape, "min/max:", t_idx.min(), t_idx.max())
            print("normed time (t_norm):", t_norm.shape, "min/max:", t_norm.min(), t_norm.max())
            print("z shape:", z.shape)
            print("z.min():", z.min(), "\nz.max():", z.max(), "\nz.mean():", z.mean(), "\nz.std():", z.std())
            print("z_t shape:", z_t.shape)
            print("z_t.min():", z_t.min(), "\nz_t.max():", z_t.max(), "\nz_t.mean():", z_t.mean(), "\nz_t.std():", z_t.std())
            print("epsilon shape:", epsilon.shape)
            print(
                "epsilon.min():",
                epsilon.min(),
                "\nepsilon.max():",
                epsilon.max(),
                "\nepsilon.mean():",
                epsilon.mean(),
                "\nepsilon.std():",
                epsilon.std(),
            )
            print("###############################################\n")

            # Prints forwars process statistics for the first batch only, at specific time steps
            if self.i % 100 == 0:
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
                t_idx_debug = torch.tensor(t_steps, dtype=torch.long, device=z.device)

                for t in t_idx_debug:
                    # Use .item() for the index but keep the tensor for schedule lookup
                    idx = t.item()

                    # 3. Retrieve schedule parameters for this specific step
                    # We use [idx] to get the scalar, then unsqueeze to handle broadcasting
                    alpha_t = self.sqrt_alpha_cumprod[idx]
                    sigma_t = self.sigma[idx]

                    # 4. Generate the noisy sample (Forward Process)
                    epsilon_t = torch.randn_like(z)
                    # Note: z is (Batch, Dim), alpha_t is scalar
                    z_t = alpha_t * z + sigma_t * epsilon_t

                    print(f"t={idx:3d}/{self.T}: mean={z_t.mean():.4f}, std={z_t.std():.4f}")

                print("###############################################\n")
                self.i += 1

        return total.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Loss terms
    # -------------------------------------------------------------------------
    def _l_diff(
        self,
        z_t: torch.Tensor,
        t_norm: torch.Tensor,
        t_idx: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Noise-prediction MSE loss.

        Args:
            z_t:     (B, latent_dim, H', W') noisy latent
            t_norm:  (B,) timestep normalised to [0, 1]
            epsilon: (B, latent_dim, H', W') ground-truth noise
        Returns:
            (B,) per-sample loss
        """

        # 2. Get the token prediction from the model
        eps_hat = self.noise_predictor(z_t, t_norm)  # (B, H*W, C)

        # 2. Compute MSE over the channel, height, and width dimensions
        mse = F.mse_loss(eps_hat, epsilon, reduction="none")  # (B, C, H', W')

        unscaled_loss = mse.sum(dim=(-3, -2, -1))  # Sum over C, H, W to get (B,)

        # Pull weights for the batch: shape (B,)
        w_t = self.l_diff_weights[t_idx]

        # 4. Scale the per-sample loss
        l_diff_loss = w_t * unscaled_loss  # (B,)

        # Debug logs updated to match the spatial reality
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("############# Diffusion Loss: #################")
            print("epsilon shape:", epsilon.shape)
            print(
                "epsilon.min():",
                epsilon.min(),
                "\nepsilon.max():",
                epsilon.max(),
                "\nepsilon.mean():",
                epsilon.mean(),
                "\nepsilon.std():",
                epsilon.std(),
            )
            print("eps_hat shape:", eps_hat.shape)
            print(
                "eps_hat.min():",
                eps_hat.min(),
                "\neps_hat.max():",
                eps_hat.max(),
                "\neps_hat.mean():",
                eps_hat.mean(),
                "\neps_hat.std():",
                eps_hat.std(),
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
            print("l_diff_loss shape:", l_diff_loss.shape)
            print(
                "l_diff_loss.min():",
                l_diff_loss.min(),
                "\nl_diff_loss.max():",
                l_diff_loss.max(),
                "\nl_diff_loss.mean():",
                l_diff_loss.mean(),
                "\nl_diff_loss.std():",
                l_diff_loss.std(),
            )
            print("###############################################\n")

        return l_diff_loss

    def _l_prior(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between q(z|x) = N(mu, exp(logvar)) and N(0, I).

        Args:
            mu:     (B, latent_dim, H', W')
            logvar: (B, latent_dim, H', W')
        Returns:
            (B,) per-sample KL, scaled by lambda_kl
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, C, H', W')
        return kl.sum(dim=(-3, -2, -1))

    def _l_rec(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Pixel-space reconstruction loss.

        Args:
            x: (B, C, H, W) original images
            z: (B, latent_dim, H', W') clean latent (attached to encoder graph)
        Returns:
            (B,) per-sample MSE
        """
        x_hat = self.decoder(z, self.coord_grid)
        x_hat = x_hat.reshape(x_hat.shape[0], -1)
        x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
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
            print("x_hat shape:", x_hat.shape)
            print(
                "x_hat.min():", x_hat.min(), "\nx_hat.max():", x_hat.max(), "\nx_hat.mean():", x_hat.mean(), "\nx_hat.std():", x_hat.std()
            )
            print("###############################################\n")

        return 0.5 * ((x_flat - x_hat) ** 2).sum(dim=-1)

    # -------------------------------------------------------------------------
    # Diffusion helpers
    # -------------------------------------------------------------------------
    def _forward_process(
        self,
        z: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples z_t = √ᾱ_t · z + sigma_t · ε.

        Args:
            z:     (B, latent_dim, H', W')
            t_idx: (B,) integer timestep indices
        Returns:
            z_t:     (B, latent_dim, H', W')
            epsilon: (B, latent_dim, H', W')
        """
        # (B, 1, 1, 1) for broadcasting over (latent_dim, H', W')
        alpha_t = self.sqrt_alpha_cumprod[t_idx].view(-1, 1, 1, 1)
        sigma_t = self.sigma[t_idx].view(-1, 1, 1, 1)

        epsilon = torch.randn_like(z)
        z_t = alpha_t * z + sigma_t * epsilon
        return z_t, epsilon

    # -------------------------------------------------------------------------
    # Sampling helpers
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _sample_latent(self, n_samples: int, collect_snapshots: bool = False) -> torch.Tensor:
        """
        Reverse diffusion in latent space (DDPM, ε-prediction).

        Args:
            n_samples: number of samples
            collect_snapshots: whether to collect snapshots at specific timesteps for debugging
        Returns:
            z: (n_samples, latent_dim, H', W')
        """
        device = self.sqrt_alpha_cumprod.device
        H, W = self.latent_size  # noqa: N806
        z = torch.randn(n_samples, self.latent_dim, H, W, device=device)
        T_values = {self.T - 1, 3 * self.T // 4, self.T // 2, self.T // 4, 0}  # noqa: N806
        snapshots: dict[int, np.ndarray] = {}

        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling", total=self.T):
            t_norm = torch.full((n_samples,), t / (self.T - 1), device=device).unsqueeze(-1)

            # 2. Predict noise tokens
            eps_hat = self.noise_predictor(z, t_norm)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]

            # Standard DDPM formulation for the posterior mean
            coeff1 = 1.0 / torch.sqrt(alpha_t)
            coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

            mean = coeff1 * (z - coeff2 * eps_hat)

            if t > 0:
                # Standard simplified variance
                sigma = torch.sqrt(beta_t)
                z = mean + sigma * torch.randn_like(z)
            else:
                z = mean

            if collect_snapshots and t in T_values:
                snapshots[t] = z.detach().cpu().numpy().flatten()

            # Print statistics every 100 steps for debugging
            if (t % 100 == 0 and GLOBAL_DEBUG_BOOL) or (t == 0 and GLOBAL_DEBUG_BOOL):
                print("################## Sampling: ##############################")
                print(f"Sampling step {t}/{self.T}:")
                print(
                    f"predicted noise (eps_hat) stats: mean={eps_hat.mean():.4f}, std={eps_hat.std():.4f}",
                    f"min={eps_hat.min():.4f}, max={eps_hat.max():.4f}",
                )
                print(f"z stats: mean={z.mean():.4f}, std={z.std():.4f}", f"min={z.min():.4f}, max={z.max():.4f}")
                print("###########################################################\n")

        if collect_snapshots:
            return z, snapshots
        return z

    @torch.no_grad()
    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent feature map to pixel space via TransInr.

        Args:
            z: (B, latent_dim, H', W')
        Returns:
            pixels: (B, data_dim)
        """
        # TransInr expects (B, C, H, W) and returns (B, C_out, H, W)
        x_hat = self.decoder(z, self.coord_grid)  # (B, C_out, H, W)
        return x_hat.reshape(x_hat.shape[0], -1)  # (B, data_dim)
