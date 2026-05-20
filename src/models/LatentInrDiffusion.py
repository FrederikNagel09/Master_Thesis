from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm


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

        # --- Noise schedule ---
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq", 1.0 - alpha_cumprod)
        self.register_buffer("sigma", (1.0 - alpha_cumprod).sqrt())
        self.register_buffer("coord_grid", coord_grid, persistent=False)

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
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Generate images by reverse diffusion in latent space.

        Args:
            n_samples: number of images to generate
        Returns:
            images: (n_samples, data_dim)
        """
        z = self._sample_latent(n_samples)  # (B, latent_dim, H', W')
        return self._decode_latent(z)  # (B, data_dim)

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

        # Reshape to (B, C, H, W) if input is flattened (B, data_dim)
        if x.dim() == 2:
            channels = self.data_dim // (self.img_size * self.img_size)
            x = x.view(B, channels, self.img_size, self.img_size)

        z = self.latent_encoder(x)  # (B, latent_dim, H', W')

        t_idx = torch.randint(0, self.T, (B,), device=x.device)
        t_norm = t_idx.float().unsqueeze(-1) / (self.T - 1)  # (B, 1)

        # Detach only for the diffusion path — l_rec needs gradients through z
        z_t, epsilon = self._forward_process(z, t_idx)

        l_diff = self._l_diff(z_t, t_norm, epsilon)
        l_prior = self._l_prior(z)  # analytical, no encoder gradient needed
        l_rec = self._l_rec(x, z)  # z kept attached — trains encoder

        total = (self.T - 2) * l_diff + l_prior + l_rec
        return total.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Loss terms
    # -------------------------------------------------------------------------
    def _l_diff(
        self,
        z_t: torch.Tensor,
        t_norm: torch.Tensor,
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
        H, W = self.latent_size  # noqa: N806
        # (B, latent_dim, H', W') → (B, n_patches, latent_dim)
        z_t_tokens = z_t.permute(0, 2, 3, 1).reshape(z_t.shape[0], H * W, self.latent_dim)

        eps_hat_tokens = self.noise_predictor(z_t_tokens, t_norm)  # (B, H'*W', latent_dim)

        # (B, H'*W', latent_dim) → (B, latent_dim, H', W')
        eps_hat = eps_hat_tokens.reshape(z_t.shape[0], H, W, self.latent_dim).permute(0, 3, 1, 2)

        return F.mse_loss(eps_hat, epsilon, reduction="none").sum(dim=(-3, -2, -1))

    def _l_prior(self, z: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between q(z_T | z) and N(0, I), computed analytically.

        Args:
            z: (B, latent_dim, H', W') clean latent
        Returns:
            (B,) per-sample KL
        """
        T_idx = self.T - 1  # noqa: N806
        sigma_T_sq = self.sigma_sq[T_idx]  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # noqa: N806

        z_flat = z.reshape(z.shape[0], -1)  # (B, latent_dim * H' * W')
        d = z_flat.shape[-1]

        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (z_flat**2).sum(dim=-1))
        return kl

    def _l_rec(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Pixel-space reconstruction loss.

        Args:
            x: (B, C, H, W) original images
            z: (B, latent_dim, H', W') clean latent (attached to encoder graph)
        Returns:
            (B,) per-sample MSE
        """
        x_hat = self._decode_latent(z)  # (B, data_dim)
        x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)
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
    def _sample_latent(self, n_samples: int) -> torch.Tensor:
        """
        Reverse diffusion in latent space (DDPM, ε-prediction).

        Args:
            n_samples: number of samples
        Returns:
            z: (n_samples, latent_dim, H', W')
        """
        device = self.sqrt_alpha_cumprod.device
        H, W = self.latent_size  # noqa: N806
        z = torch.randn(n_samples, self.latent_dim, H, W, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling", total=self.T):
            t_norm = torch.full((n_samples,), t / (self.T - 1), device=device).unsqueeze(-1)

            # (B, latent_dim, H', W') → (B, H'*W', latent_dim)
            z_tokens = z.permute(0, 2, 3, 1).reshape(n_samples, H * W, self.latent_dim)
            eps_hat_tokens = self.noise_predictor(z_tokens, t_norm)
            # (B, H'*W', latent_dim) → (B, latent_dim, H', W')
            eps_hat = eps_hat_tokens.reshape(n_samples, H, W, self.latent_dim).permute(0, 3, 1, 2)

            alpha_bar = self.alpha_cumprod[t]
            beta_t = self.beta[t]

            # Recover clean latent estimate from ε-prediction
            sqrt_recip_abar = (1.0 / alpha_bar).sqrt()
            sqrt_recip_abar_m1 = (1.0 / alpha_bar - 1.0).sqrt()
            z0_hat = sqrt_recip_abar * z - sqrt_recip_abar_m1 * eps_hat

            if t > 0:
                alpha_bar_prev = self.alpha_cumprod[t - 1]
                alpha_t = self.alpha[t]

                coeff_z0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar)
                coeff_zt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                mean = coeff_z0 * z0_hat + coeff_zt * z

                sigma = torch.sqrt(beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                z = mean + sigma * torch.randn_like(z)
            else:
                z = z0_hat

        return z  # (n_samples, latent_dim, H', W')

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
