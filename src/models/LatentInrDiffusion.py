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


class LatentDiffusion(nn.Module):
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

        self.i = 0

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
    def sample(self, n_samples: int = 1, collect_snapshots: bool = False, debug: bool = True) -> torch.Tensor:
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
            z, snapshots = self._sample_latent(n_samples, collect_snapshots=True, debug=debug)

            return self._decode_latent(z), snapshots
        else:
            z = self._sample_latent(n_samples, collect_snapshots=collect_snapshots, debug=debug)  # (B, latent_dim, H', W')

            return self._decode_latent(z)  # (B, data_dim)

    # -------------------------------------------------------------------------
    # Metric Computations
    # -------------------------------------------------------------------------

    def compute_rec_loss(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Computes average reconstruction loss over the full validation set.

        Args:
            val_loader: Validation DataLoader yielding (x, _) batches. Shape (B, C, H, W).
        Returns:
            Mean reconstruction loss (scalar float) across all batches.
        """
        self.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, _ in val_loader:
                B = x.shape[0]  # noqa: N806
                if x.dim() == 2:
                    channels = self.data_dim // (self.img_size * self.img_size)
                    x = x.view(B, channels, self.img_size, self.img_size)

                x = x.to(next(self.parameters()).device)

                z_raw = self.encode(x)

                total_loss += self._l_rec(x, z_raw).mean().item()
                n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def compute_full_elbo(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Computes the exact ELBO over the full validation set.
        Loops over all T timesteps per batch — expensive, only call at checkpoints.

        Args:
            val_loader: DataLoader yielding (x, label) batches
        Returns:
            mean ELBO scalar (higher is better)
        """
        self.eval()
        device = self.sqrt_alpha_cumprod.device
        total_elbo = 0.0
        n_batches = 0

        total_steps = len(val_loader) * self.T
        pbar = tqdm(total=total_steps, desc="Validation")

        for x, _ in val_loader:
            x = x.to(device)
            B = x.shape[0]  # noqa: N806

            if x.dim() == 2:
                channels = self.data_dim // (self.img_size * self.img_size)
                x = x.view(B, channels, self.img_size, self.img_size)

            # Encode once per batch
            mu, logvar = self.latent_encoder(x)
            z = self.latent_encoder.reparameterize(mu, logvar)

            # l_rec and l_entropy computed once per batch
            l_rec = self._l_rec(x, z, debug=False)  # (B,)
            l_entropy = self._l_entropy(logvar)

            # Sum l_diff over all t
            l_diff_sum = torch.zeros(B, device=device)  # (B,)
            for t in range(self.T):
                t_idx = torch.full((B,), t, dtype=torch.long, device=device)
                t_norm = torch.full((B, 1), t / (self.T - 1), device=device)
                z_t, epsilon = self._forward_process(z, t_idx)

                if t == 0:
                    l_diff_sum += self._l_latent_rec(z_t, t_norm, z)  # (B,)
                else:
                    l_diff_sum += self._l_diff(z_t, t_norm, epsilon, t_idx, debug=False)  # (B,)

                pbar.update(1)

            # Full ELBO per sample (negative, so lower is better during training)
            elbo = l_diff_sum - l_entropy + l_rec  # (B,)
            total_elbo += elbo.mean().item()
            n_batches += 1

        pbar.close()
        self.train()
        return total_elbo / n_batches

    # -------------------------------------------------------------------------
    # Helper Functions
    # ------------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.latent_encoder(x)

        z_raw = self.latent_encoder.reparameterize(mu, logvar)

        self.print_encoded_stats(logvar, mu, z_raw)

        return z_raw, mu, logvar

    def get_reconstructions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get reconstructed images from the model for a batch of inputs.

        Args:
            x: (B, C, H, W) input images
        Returns:
            x_hat: (B, data_dim) reconstructed images in flattened form
        """
        z_raw, _, _ = self.encode(x)

        x_recon = self._decode_latent(z_raw)

        return x_recon

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
        ######### Input shape check ##########
        B = x.shape[0]  # noqa: N806
        if x.dim() == 2:
            channels = self.data_dim // (self.img_size * self.img_size)
            x = x.view(B, channels, self.img_size, self.img_size)

        ######### Encode Image ##########
        z, _, logvar = self.encode(x)

        ######### Sample Time Steps ##########
        t_idx = torch.randint(0, self.T, (B,), device=x.device)
        t_norm = t_idx.float().unsqueeze(-1) / (self.T - 1)  # (B, 1)

        ######### Forward Process ##########
        z_t, epsilon = self._forward_process(z, t_idx)

        ######### Compute diffusion loss terms ##########
        mask_t0 = t_idx == 0
        mask_tdiff = ~mask_t0

        # --- Diffusion loss: only t>0 ---
        l_diff = torch.zeros(B, device=x.device)
        if mask_tdiff.any():
            l_diff[mask_tdiff] = self._l_diff(z_t[mask_tdiff], t_norm[mask_tdiff], epsilon[mask_tdiff], t_idx[mask_tdiff])
        # --- Latent reconstruction loss: only t=0 ---
        l_latent_rec = torch.zeros(B, device=x.device)
        if mask_t0.any():
            l_latent_rec[mask_t0] = self._l_latent_rec(z_t[mask_t0], t_norm[mask_t0], z[mask_t0])

        ######### Compute Reconstruction loss terms ##########
        l_rec = self._l_rec(x, z)

        ######### Compute entropy loss ##########
        l_entropy = self._l_entropy(logvar)

        ########## Total Loss ##########
        scale = self.T - 2
        total = l_rec - l_entropy + scale * l_diff + l_latent_rec

        if GLOBAL_DEBUG_BOOL:
            self.print_masking_debug(t_idx, l_diff, l_latent_rec, mask_t0, mask_tdiff)

        self.print_final_elbo_stats(x, t_idx, t_norm, z, z_t, epsilon)

        return total.mean(), l_diff.mean(), l_entropy.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Loss terms
    # -------------------------------------------------------------------------
    def _l_diff(
        self,
        z_t: torch.Tensor,
        t_norm: torch.Tensor,
        epsilon: torch.Tensor,
        t_idx: torch.Tensor,
        debug: bool = True,
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

        alpha_t = self.alpha[t_idx]
        alpha_bar_t = self.alpha_cumprod[t_idx]
        beta_t = self.beta[t_idx]

        scaling = beta_t / (2 * alpha_t * (1.0 - alpha_bar_t))

        unscaled_loss = mse.mean(dim=(-3, -2, -1))  # Sum over C, H, W to get (B,)

        self.print_mse_low_and_high(t_norm, unscaled_loss, debug=debug)

        l_diff_loss = scaling * unscaled_loss

        self.print_debug_info(epsilon, eps_hat, mse, l_diff_loss)

        return l_diff_loss

    def _l_entropy(self, logvar: torch.Tensor) -> torch.Tensor:
        """
        Negative entropy of q(z|x) = N(mu, exp(logvar)), including constants.

        Args:
            logvar: (B, latent_dim, H', W')
        Returns:
            (B,) per-sample negative entropy
        """

        entropy_per_dim = 0.5 * (1.0 + torch.log(torch.as_tensor(2.0 * math.pi, device=logvar.device)) + logvar)

        return entropy_per_dim.mean(dim=(-3, -2, -1))

    def _l_rec(self, x: torch.Tensor, z: torch.Tensor, debug: bool = True) -> torch.Tensor:
        """
        Pixel-space reconstruction loss.

        Args:
            x: (B, C, H, W) original images
            z: (B, latent_dim, H', W') clean latent (attached to encoder graph)
        Returns:
            (B,) per-sample MSE
        """
        x_hat = self._decode_latent(z)

        x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)

        self.print_l_rec_stats(x_flat, x_hat, debug=debug)

        return 0.5 * ((x_flat - x_hat) ** 2).sum(dim=-1)

    def _l_latent_rec(
        self,
        z_t0: torch.Tensor,
        t_norm_t0: torch.Tensor,
        z_clean_t0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the t=0 reconstruction term: -log p_θ(z_0 | z_1).
        Predicts epsilon at t=0, recovers denoiser mean, returns MSE / (2*beta_0).

        Args:
            z_t0:       (B0, C, H, W) — noisy latents where t=0
            t_norm_t0:  (B0, 1)       — normalised time (all zeros)
            z_clean_t0: (B0, C, H, W) — corresponding clean latents
        Returns:
            (B0,) per-sample losses
        """
        eps_pred = self.noise_predictor(z_t0, t_norm_t0)

        alpha_0 = self.sqrt_alpha_cumprod[0]
        sigma_0 = self.sigma[0]
        beta_0 = self.beta[0]

        mu_theta = (z_t0 - sigma_0 * eps_pred) / alpha_0

        return F.mse_loss(mu_theta, z_clean_t0, reduction="none").mean(dim=[1, 2, 3]) / (2.0 * beta_0)

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
    def _sample_latent(self, n_samples: int, collect_snapshots: bool = False, debug: bool = True) -> torch.Tensor:
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

            self.print_sampling_stats(t, eps_hat, z, debug=debug)

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

        x_hat = self.decoder(z, self.coord_grid)  # (B, C_out, H, W)
        return x_hat.reshape(x_hat.shape[0], -1)  # (B, data_dim)

    # ------------------------------------------------------------------------
    # Debugging and Statistics Printing Functions
    # ------------------------------------------------------------------------

    def print_encoded_stats(self, logvar: torch.Tensor, mu: torch.Tensor, z_raw: torch.Tensor) -> None:
        """
        Print statistics of the encoded latent representation for debugging.

        Args:
            logvar: (B, latent_dim) log variance of the encoded latent variables
            mu: (B, latent_dim) mean of the encoded latent variables
            z_raw: (B, latent_dim) raw encoded latent variables
        """
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            std = torch.exp(0.5 * logvar)
            print(f"==================== Probablistic Components {self.i}: ====================")
            print(f"[Encoder] mu:     mean={mu.mean():.3f}, std={mu.std():.3f}, min={mu.min():.3f}, max={mu.max():.3f}")
            print(f"[Encoder] logvar: mean={logvar.mean():.3f}, std={logvar.std():.3f}, min={logvar.min():.3f}, max={logvar.max():.3f}")
            print(f"[Encoder] std:    mean={std.mean():.3f}, std={std.std():.3f}, min={std.min():.3f}, max={std.max():.3f}")
            print(
                f"theta mean={z_raw.mean():.4f}," f"std={z_raw.std():.4f}, min={z_raw.min():.4f}, max={z_raw.max():.4f}",
            )
            print("================================================================\n")

    def print_final_elbo_stats(
        self, x: torch.Tensor, t_idx: torch.Tensor, t_norm: torch.Tensor, z: torch.Tensor, z_t: torch.Tensor, epsilon: torch.Tensor
    ) -> None:
        """
        Print statistics of the final ELBO computation for debugging.
        """

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

    def print_mse_low_and_high(self, t_norm, unscaled_loss, debug: bool = True) -> None:
        # Bin MSE by timestep to see where the model fails
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold and debug:
            t_flat = t_norm.flatten()
            low_t_mask = t_flat < 0.2  # t in [0, 0.2]
            high_t_mask = t_flat > 0.8  # t in [0.8, 1.0]
            if low_t_mask.any():
                print(f"MSE @ low  t (<0.2): {unscaled_loss[low_t_mask].mean():.4f}")
            if high_t_mask.any():
                print(f"MSE @ high t (>0.8): {unscaled_loss[high_t_mask].mean():.4f}")

    def print_debug_info(self, epsilon, eps_hat, mse, l_diff_loss, debug: bool = True) -> None:
        # Debug logs updated to match the spatial reality
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold and debug:
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

    def print_l_rec_stats(self, x_flat, x_hat, debug: bool = True) -> None:
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
            print("x_hat shape:", x_hat.shape)
            print(
                "x_hat.min():", x_hat.min(), "\nx_hat.max():", x_hat.max(), "\nx_hat.mean():", x_hat.mean(), "\nx_hat.std():", x_hat.std()
            )
            print("###############################################\n")

    def print_sampling_stats(self, t, eps_hat, z, debug: bool = True) -> None:
        # Print statistics every 100 steps for debugging
        if debug and GLOBAL_DEBUG_BOOL and (t % 100 == 0 or t == 0):
            print("################## Sampling: ##############################")
            print(f"Sampling step {t}/{self.T}:")
            print(
                f"predicted noise (eps_hat) stats: mean={eps_hat.mean():.4f}, std={eps_hat.std():.4f}",
                f"min={eps_hat.min():.4f}, max={eps_hat.max():.4f}",
            )
            print(f"z stats: mean={z.mean():.4f}, std={z.std():.4f}", f"min={z.min():.4f}, max={z.max():.4f}")
            print("###########################################################\n")

    def print_masking_debug(
        self,
        t_idx: torch.Tensor,
        l_diff: torch.Tensor,
        l_latent_rec: torch.Tensor,
        mask_t0: torch.Tensor,
        mask_tdiff: torch.Tensor,
    ) -> None:
        """
        Verify t-sampling coverage and masking correctness.

        Args:
            t_idx:       (B,) sampled timestep indices
            l_diff:      (B,) diffusion loss per sample
            l_latent_rec:(B,) latent-rec loss per sample
            mask_t0:     (B,) bool — True where t==0
            mask_tdiff:  (B,) bool — True where t>0
        Returns:
            None — prints only
        """
        n_t0 = mask_t0.sum().item()
        n_tdiff = mask_tdiff.sum().item()
        B = t_idx.shape[0]  # noqa: N806

        print("\n========== Masking Debug ==========")
        print(f"Batch size:        {B}")
        print(f"t=0  samples:      {n_t0}  ({100*n_t0/B:.1f}%,  expected ~{100/self.T:.1f}%)")
        print(f"t>0  samples:      {n_tdiff} ({100*n_tdiff/B:.1f}%, expected ~{100*(self.T-1)/self.T:.1f}%)")
        print(f"t_idx min/max:     {t_idx.min().item()} / {t_idx.max().item()}")
        print(f"t_idx covers 0:    {(t_idx == 0).any().item()}")
        print(f"mask_t0 | mask_tdiff covers all: {(mask_t0 | mask_tdiff).all().item()}")
        print(f"masks overlap (should be False):  {(mask_t0 & mask_tdiff).any().item()}")

        # Verify loss routing — each term should be active only on its mask
        l_diff_on_t0 = l_diff[mask_t0].abs().sum().item() if n_t0 else 0.0
        l_lrec_on_tdiff = l_latent_rec[mask_tdiff].abs().sum().item() if n_tdiff else 0.0

        print(f"\nl_diff sum on t=0 slots (should be 0):   {l_diff_on_t0:.6f}")
        print(f"l_latent_rec sum on t>0 slots (should be 0): {l_lrec_on_tdiff:.6f}")

        if n_t0 > 0:
            print(f"\nl_latent_rec on t=0: mean={l_latent_rec[mask_t0].mean():.4f}")
        if n_tdiff > 0:
            print(f"l_diff      on t>0: mean={l_diff[mask_tdiff].mean():.4f}")
        print("===================================")
