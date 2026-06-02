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


class LatentNDMDiffusion(nn.Module):
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
        latent_transformer: nn.Module,
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
        sigma_tilde_factor: float = 1.0,
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
        self.latent_transformer = latent_transformer

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self._normalize = normalize

        self.i = 0
        self.latent_scaler = LatentScaler(latent_dim)
        self.lambda_kl = lambda_kl

        # --- Noise schedule ---
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)

        self.sigma_tilde_factor = sigma_tilde_factor

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

    def _sigma_tilde_sq(self, s_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        sigma_s_sq = self.sigma_sq[s_idx]
        sigma_t_sq = self.sigma_sq[t_idx]
        alpha_t_sq = self.alpha_cumprod[t_idx]
        alpha_s_sq = self.alpha_cumprod[s_idx]

        base = (sigma_t_sq - alpha_t_sq / alpha_s_sq * sigma_s_sq) * sigma_s_sq / sigma_t_sq
        return self.sigma_tilde_factor * base
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
        mu, logvar = self.latent_encoder(x)
        z_raw = self.latent_encoder.reparameterize(mu, logvar)

        ######### Normalize Latents ##########
        z = self._normalize_z(z_raw) if self._normalize else z_raw

        ######### Sample Time Steps ##########
        t_idx = torch.randint(0, self.T, (B,), device=x.device)
        t_norm = t_idx.float().unsqueeze(-1) / (self.T - 1)  # (B, 1)

        ######### latent transformer ##########
        Fz = self.latent_transformer(z.detach(), t_norm)

        ######### Apply noise ##########
        z_t, epsilon = self._forward_process(Fz, t_idx)

        l_diff = self._l_diff(z.detach(), z_t, t_norm, t_idx, Fz)

        ######### Compute image reconstruction and entropy loss ##########
        l_prior = self._l_prior(mu, logvar)
        l_rec = self._l_rec(x, z_raw)

        total = (self.T-1) * l_diff + lambda_kl * l_prior + l_rec

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("############# Negative ELBO: #################")
            print("data x shape:", x.shape)
            print("time indices (t_idx):", t_idx.shape, "min/max:", t_idx.min().item(), t_idx.max().item())
            print("normed time (t_norm):", t_norm.shape, "min/max:", t_norm.min().item(), t_norm.max().item())
            print("z shape:", z.shape)
            print("z.min():", z.min().item(), "\nz.max():", z.max().item(), "\nz.mean():", z.mean().item(), "\nz.std():", z.std().item())
            print("z_trans shape:", Fz.shape)
            print(
                "z_trans.min():",
                Fz.min().item(),
                "\nz_trans.max():",
                Fz.max().item(),
                "\nz_trans.mean():",
                Fz.mean().item(),
                "\nz_trans.std():",
                Fz.std().item(),
            )
            print("z_t shape:", z_t.shape)
            print("z_t.min():", z_t.min().item(), "\nz_t.max():", z_t.max().item(), "\nz_t.mean():", z_t.mean().item(), "\nz_t.std():", z_t.std().item())
            print("epsilon shape:", epsilon.shape)
            print(
                "epsilon.min():",
                epsilon.min().item(),
                "\nepsilon.max():",
                epsilon.max().item(),
                "\nepsilon.mean():"
                epsilon.mean().item(),
                "\nepsilon.std():",
                epsilon.std().item(),
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

        return total.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Loss terms
    # -------------------------------------------------------------------------
    def _l_diff(
        self,
        z: torch.Tensor,
        z_t: torch.Tensor,
        t_norm: torch.Tensor,
        t_idx: torch.Tensor,
        Fz_t: torch.Tensor,
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

        alpha_t = self.sqrt_alpha_cumprod[t_idx].view(-1, 1, 1, 1)
        sigma_t = self.sigma[t_idx].view(-1, 1, 1, 1)
        z_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)
        
        s_idx = (t_idx - 1).clamp(min=0)
        s_norm = s_idx.float() / (self.T - 1)

        B = z.shape[0]

        Fz_hat_t = self.latent_transformer(z_hat, t_norm)  # noqa: N806
        Fz_hat_s = self.latent_transformer(z_hat, s_norm.unsqueeze(1))  # noqa: N806
        Fz_s     = self.latent_transformer(z, s_norm.unsqueeze(1))      # noqa: N806

        alpha_s = self.sqrt_alpha_cumprod[s_idx].view(-1, 1, 1, 1)
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).view(-1, 1, 1, 1)
        coeff = (self.sigma_sq[s_idx].view(-1, 1, 1, 1) - sigma_tilde_sq).clamp(min=0).sqrt()
        coeff = coeff / self.sigma[t_idx].view(-1, 1, 1, 1).clamp(min=1e-6)

        diff = alpha_s * (Fz_s - Fz_hat_s) + coeff * alpha_t * (Fz_hat_t - Fz_t)
        mse = (diff**2).view(B, -1).sum(dim=-1) / (2.0 * sigma_tilde_sq.view(B).clamp(min=1e-8))

        alpha_t = self.alpha[t_idx]
        alpha_bar_t = self.alpha_cumprod[t_idx]
        beta_t = self.beta[t_idx]

        scaling = beta_t / (2 * alpha_t * (1.0 - alpha_bar_t))

        # Bin MSE by timestep to see where the model fails
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            t_flat = t_norm.flatten()
            low_t_mask  = t_flat < 0.2   # t in [0, 0.2]
            high_t_mask = t_flat > 0.8   # t in [0.8, 1.0]
            if low_t_mask.any():
                print(f"MSE @ low  t (<0.2): {mse[low_t_mask].mean():.4f}")
            if high_t_mask.any():
                print(f"MSE @ high t (>0.8): {mse[high_t_mask].mean():.4f}")


        # Debug logs updated to match the spatial reality
        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("############# Diffusion Loss: #################")
            print("Fz_t shape:", Fz_t.shape)
            print("Fz_t mean:", Fz_t.mean().item(), "std:", Fz_t.std().item(), "min:", Fz_t.min().item(), "max:", Fz_t.max().item())
            print("Fz_hat_t shape:", Fz_hat_t.shape)
            print("Fz_hat_t mean:", Fz_hat_t.mean().item(), "std:", Fz_hat_t.std().item(), "min:", Fz_hat_t.min().item(), "max:", Fz_hat_t.max().item())
            print("___________________________________________________")
            print("Fz_s shape:", Fz_s.shape)
            print("Fz_s mean:", Fz_s.mean().item(), "std:", Fz_s.std().item(), "min:", Fz_s.min().item(), "max:", Fz_s.max().item())
            print("Fz_hat_s shape:", Fz_hat_s.shape)
            print("Fz_hat_s mean:", Fz_hat_s.mean().item(), "std:", Fz_hat_s.std().item(), "min:", Fz_hat_s.min().item(), "max:", Fz_hat_s.max().item())
            print("___________________________________________________")
            print("z shape:", z.shape)
            print("z mean:", z.mean().item(), "std:", z.std().item(), "min:", z.min().item(), "max:", z.max().item())
            print("z_hat shape:", z_hat.shape)
            print("z_hat mean:", z_hat.mean().item(), "std:", z_hat.std().item(), "min:", z_hat.min().item(), "max:", z_hat.max().item())
            print("################################################################\n")


        return scaling * mse
            


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
        z_t = torch.randn(n_samples, self.latent_dim, H, W, device=device)
        T_values = {self.T - 1, 3 * self.T // 4, self.T // 2, self.T // 4, 0}  # noqa: N806
        snapshots: dict[int, np.ndarray] = {}

        # Pre-compute all scalar coefficients — avoids repeated indexing inside loop
        alpha = self.sqrt_alpha_cumprod  # (T,)
        sigma = self.sigma  # (T,)
        sigma_sq = self.sigma_sq  # (T,)
        T_minus_1 = max(self.T - 1, 1)  # noqa: N806

        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling", total=self.T):
            t_norm = torch.full((n_samples,), t / (self.T - 1), device=device).unsqueeze(-1)

            # 2. Predict noise tokens
            eps_hat = self.noise_predictor(z_t, t_norm)
            
            alpha_t = alpha[t]
            sigma_t = sigma[t]
            
            z_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)
            
            if t == 0:
                z_t = z_hat
                if collect_snapshots and t in T_values: 
                    snapshots[t] = z_t.detach().cpu().numpy().flatten()
                break

            s = t - 1
            s_norm = torch.full((n_samples, 1), s / T_minus_1, device=device)

            # --- Batch both F_phi calls into one forward pass ---
            Fz_hat_s = self.latent_transformer(z_hat, s_norm)  # noqa: N806
            Fz_hat_t = self.latent_transformer(z_hat, t_norm)  # noqa: N806
            

            alpha_s = alpha[s]
            sigma_s_sq = sigma_sq[s]

            sigma_tilde_sq = self._sigma_tilde_sq(torch.tensor([s], device=device), torch.tensor([t], device=device))[0]
            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t.clamp(min=1e-6)
            mu = alpha_s * Fz_hat_s + coeff * (z_t - alpha_t * Fz_hat_t)
            z_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * torch.randn_like(z_t) if sigma_tilde_sq.item() > 0 else mu
            if collect_snapshots and t in T_values: 
                snapshots[t] = z_t.detach().cpu().numpy().flatten()

            # Print statistics every 100 steps for debugging
            if (t % 100 == 0 and GLOBAL_DEBUG_BOOL) or (t == 0 and GLOBAL_DEBUG_BOOL):
                print("################## Sampling: ##############################")
                print(f"Sampling step {t}/{self.T}:")
                print(
                    f"predicted noise (eps_hat) stats: mean={eps_hat.mean():.4f}, std={eps_hat.std():.4f}",
                    f"min={eps_hat.min():.4f}, max={eps_hat.max():.4f}",
                )
                print(f"z stats: mean={z_t.mean():.4f}, std={z_t.std():.4f}", f"min={z_t.min():.4f}, max={z_t.max():.4f}")
                print("###########################################################\n")

        if collect_snapshots:
            return z_t, snapshots
        return z_t

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
