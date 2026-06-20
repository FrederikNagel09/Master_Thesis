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
        normalize: bool = True,
        lambda_kl: float = 5e-3,
        scaling: bool = True,
        latent_recon: bool = True,
        probabilistic: bool = True,
        stop_gradient_flow: bool = True,
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
        self.__do_scaling = scaling
        self._do_latent_recon = latent_recon
        self._probabilistic = probabilistic
        self.stop_gradient_flow = stop_gradient_flow

        self.i = 0
        self.latent_scaler = LatentScaler(latent_dim)
        self.lambda_kl = lambda_kl

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
    def loss(self, x: torch.Tensor, lambda_kl: float) -> tuple[torch.Tensor, ...]:
        """
        Compute the negative ELBO for a batch of images.

        Args:
            x: (B, C, H, W)
        Returns:
            (total_loss, l_diff, l_prior, l_rec) — scalar means
        """
        return self._negative_elbo(x, lambda_kl)

    def loss_vae(self, x: torch.Tensor, beta: float) -> tuple[torch.Tensor, ...]:
        """
        Compute the negative ELBO for a batch of images.
        """
        return self._negative_elbo_vae(x, beta)

    def loss_ddpm(self, x: torch.Tensor, lambda_kl: float | None = None) -> tuple[torch.Tensor, ...]:  # noqa: ARG002
        """
        Plain epsilon-prediction MSE loss for stage-2 (frozen-encoder) DDPM training.

        Encoder/decoder are run under no_grad since they're frozen during this stage.
        l_rec and l_prior are computed for logging/diagnostics only (e.g. confirming
        the frozen encoder's reconstruction quality holds steady) — they contribute
        zero gradient and are NOT included in total, so the logged total loss stays
        a clean read on DDPM optimization progress rather than a constant offset.

        Args:
            x: (B, C, H, W) input images
            lambda_kl: unused, accepted only to match the (model, x, lambda_kl) call
                    signature shared with loss() in the training loop
        Returns:
            (total_loss, l_diff, l_prior, l_rec) — scalar means; l_prior and l_rec
            are real (non-zero) diagnostic values but do not affect total or gradients
        """
        B = x.shape[0]  # noqa: N806
        if x.dim() == 2:
            channels = self.data_dim // (self.img_size * self.img_size)
            x = x.view(B, channels, self.img_size, self.img_size)

        ######### Encode (frozen, no grad) ##########
        with torch.no_grad():
            z_raw, mu, logvar = self.encode(x)
            l_rec = self._l_rec(x, z_raw, debug=False)
            l_prior = self._l_kl(mu, logvar)
        z = z_raw  # normalization disabled (self._normalize is always False)

        ######### Sample timesteps, apply forward noising ##########
        t_idx = torch.randint(0, self.T, (B,), device=x.device)
        t_norm = t_idx.float().unsqueeze(-1) / (self.T - 1)
        z_t, epsilon = self._forward_process(z, t_idx)

        ######### Plain epsilon-MSE (L_simple, unweighted) ##########
        eps_hat = self.noise_predictor(z_t, t_norm)
        l_diff = F.mse_loss(eps_hat, epsilon, reduction="none").mean(dim=(-3, -2, -1))  # (B,)

        total = l_diff  # l_rec / l_prior excluded — logging only, see docstring

        return total.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    def _negative_elbo_vae(self, x: torch.Tensor, beta: float) -> tuple[torch.Tensor, ...]:
        """
        Negative ELBO for the VAE-only stage (no diffusion loss).

        Args:
            x: (B, C, H, W) input images
        Returns:
            (total_loss, l_diff, l_prior, l_rec) — scalar means
        """
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
        z, mu, logvar = self.encode(x)

        ######### Compute diffusion loss terms ##########
        l_rec = self._l_rec(x, z)
        l_kl = self._l_kl(mu, logvar)

        l_diff = torch.zeros_like(l_rec)
        l_entropy = l_kl

        total = l_rec + beta * l_kl

        return total.mean(), l_diff.mean(), l_entropy.mean(), l_rec.mean()

    def _l_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q(z|x) || N(0,I)), closed form
        # Returns (B,) per-sample
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=(-3, -2, -1))

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
            if self._normalize:
                z = self._denormalize_z(z)
            return self._decode_latent(z), snapshots
        else:
            z = self._sample_latent(n_samples, collect_snapshots=collect_snapshots, debug=debug)  # (B, latent_dim, H', W')
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

                mu, logvar = self.latent_encoder(x)
                z_raw = mu if not self._probabilistic else self.latent_encoder.reparameterize(mu, logvar)

                total_loss += self._l_rec(x, z_raw).mean().item()
                n_batches += 1

        return total_loss / n_batches

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.latent_encoder(x)
        if self._probabilistic:
            z_raw = self.latent_encoder.reparameterize(mu, logvar)
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
        else:
            z_raw = mu  # Use mean directly for deterministic latents (ablation)
            if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
                print(f"theta mean={z_raw.mean():.4f}, std={z_raw.std():.4f}")
                print(f"theta min={z_raw.min():.4f}, max={z_raw.max():.4f}")
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
    def _negative_elbo(self, x: torch.Tensor, lambda_kl: float) -> tuple[torch.Tensor, ...]:
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
        z_raw, _, logvar = self.encode(x)

        ######### Normalize Latents ##########
        z = self._normalize_z(z_raw) if self._normalize else z_raw

        ######### Sample Time Steps ##########
        t_idx = torch.randint(1, self.T, (B,), device=x.device)
        t_norm = t_idx.float().unsqueeze(-1) / (self.T - 1)  # (B, 1)

        ######### Apply noise ##########
        z = z.detach() if self.stop_gradient_flow else z

        z_t, epsilon = self._forward_process(z, t_idx)

        ######### Compute diffusion loss terms ##########
        if self._do_latent_recon:
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
        else:
            l_diff = self._l_diff(z_t, t_norm, epsilon, t_idx)
            l_latent_rec = torch.zeros_like(l_diff)

        ######### Compute image reconstruction and entropy loss ##########
        l_entropy = self._l_entropy(logvar) if self._probabilistic else torch.zeros_like(l_diff)

        l_rec = self._l_rec(x, z_raw)

        if self.__do_scaling:
            total = (self.T - 1) * (l_diff + l_latent_rec) + lambda_kl * l_entropy + l_rec
        else:
            scaling = self.T - 1
            # ramp scaling up from 0 to 1 over the first 50000 steps determined by self.i
            if self.i < 50000:
                scaling *= self.i / 50000
            total = scaling * l_diff - lambda_kl * l_entropy + l_rec

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

        return total.mean(), l_diff.mean(), l_entropy.mean(), l_rec.mean()

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
            z_raw = self.latent_encoder.reparameterize(mu, logvar) if self._probabilistic else mu
            z = self._normalize_z(z_raw) if self._normalize else z_raw

            # l_rec and l_entropy computed once per batch
            l_rec = self._l_rec(x, z_raw, debug=False)  # (B,)
            l_entropy = self._l_entropy(logvar) if self._probabilistic else torch.zeros(B, device=device)  # (B,)

            # Sum l_diff over all t
            l_diff_sum = torch.zeros(B, device=device)  # (B,)
            for t in range(self.T):
                t_idx = torch.full((B,), t, dtype=torch.long, device=device)
                t_norm = torch.full((B, 1), t / (self.T - 1), device=device)
                z_t, epsilon = self._forward_process(z, t_idx)

                if t == 0 and self._do_latent_recon:
                    l_diff_sum += self._l_latent_rec(z_t, t_norm, z)  # (B,)
                else:
                    l_diff_sum += self._l_diff(z_t, t_norm, epsilon, t_idx, debug=False)  # (B,)

                pbar.update(1)

            # Full ELBO per sample (negative, so lower is better during training)
            elbo = l_diff_sum + l_entropy + l_rec  # (B,)
            total_elbo += elbo.mean().item()
            n_batches += 1

        pbar.close()
        self.train()
        return total_elbo / n_batches

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

        # Bin MSE by timestep to see where the model fails
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold and debug:
            t_flat = t_norm.flatten()
            low_t_mask = t_flat < 0.2  # t in [0, 0.2]
            high_t_mask = t_flat > 0.8  # t in [0.8, 1.0]
            if low_t_mask.any():
                print(f"MSE @ low  t (<0.2): {unscaled_loss[low_t_mask].mean():.4f}")
            if high_t_mask.any():
                print(f"MSE @ high t (>0.8): {unscaled_loss[high_t_mask].mean():.4f}")

        if self.__do_scaling:  # noqa: SIM108
            # 4. Scale the per-sample loss
            l_diff_loss = scaling * unscaled_loss
        else:
            l_diff_loss = unscaled_loss

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

        return l_diff_loss

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
        return entropy_per_dim.mean(dim=(-3, -2, -1))  # (B,) — sum over latent dims, return positive entropy

    def _l_rec(self, x: torch.Tensor, z: torch.Tensor, debug: bool = True) -> torch.Tensor:
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

        mu_theta = (z_t0 - (beta_0 / sigma_0) * eps_pred) / alpha_0

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
