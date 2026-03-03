"""
Neural Diffusion Model (NDM) - Model Architecture
Based on: "Neural Diffusion Models" (Bartosh et al., 2024)

Two networks:
  - F_phi(x, t): learned data transformer (applied before noise injection)
  - x_hat_theta(z_t, t): denoiser that predicts x from noisy z_t

Forward marginal:  q_phi(z_t | x) = N(z_t; alpha_t * F_phi(x,t), sigma_t^2 * I)
vs DDPM:           q(z_t | x)     = N(z_t; alpha_t * x,           sigma_t^2 * I)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm.asyncio import tqdm


class EMA:
    """Exponential moving average of model parameters for cleaner samples."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply(self, model: nn.Module) -> dict:
        """Load EMA weights into model, return original weights to restore later."""
        original = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()})
        return original

    def restore(self, model: nn.Module, original: dict):
        model.load_state_dict(original)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (shared by both networks)
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float tensor of timesteps in [0, 1] or [0, T]
        returns: (B, dim) sinusoidal embedding
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Simple residual block with time conditioning."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# F_phi: Learned data transformer
# ---------------------------------------------------------------------------
# Takes x (clean image) + t and outputs a transformed image of the same shape.
# This is what replaces the identity in DDPM's forward process.
# Architecture is intentionally lighter than the denoiser — it only needs to
# learn a smooth warp of the data manifold, not undo stochastic noise.


class FPhi(nn.Module):
    """
    F_phi(x, t): time-dependent learned transformation of clean data.

    Input:  x in R^(B, C, H, W),  t in R^(B,)
    Output: F_phi(x, t) in R^(B, C, H, W)   (same shape as x)

    Uses a lightweight UNet so it can learn spatial transformations.
    Skip connection from input ensures F_phi(x,t) ≈ x at init (via zero-init
    on the final conv), matching the DDPM baseline at the start of training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        c = base_channels
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Encoder
        self.enc_in = nn.Conv2d(in_channels, c, 3, padding=1)
        self.enc_r1 = ResBlock(c, time_emb_dim)
        self.down1 = Downsample(c)
        self.enc_r2 = ResBlock(c, time_emb_dim)
        self.down2 = Downsample(c)

        # Bottleneck
        self.mid_r1 = ResBlock(c, time_emb_dim)
        self.mid_r2 = ResBlock(c, time_emb_dim)

        # Decoder
        self.up2 = Upsample(c)
        self.dec_r2 = ResBlock(c, time_emb_dim)
        self.up1 = Upsample(c)
        self.dec_r1 = ResBlock(c, time_emb_dim)

        # Zero-init final conv → F_phi starts as identity
        self.out_conv = nn.Conv2d(c, in_channels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))  # (B, time_emb_dim)

        h = self.enc_in(x)
        h = self.enc_r1(h, t_emb)
        skip1 = h
        h = self.down1(h)
        h = self.enc_r2(h, t_emb)
        skip2 = h
        h = self.down2(h)

        h = self.mid_r1(h, t_emb)
        h = self.mid_r2(h, t_emb)

        h = self.up2(h) + skip2
        h = self.dec_r2(h, t_emb)
        h = self.up1(h) + skip1
        h = self.dec_r1(h, t_emb)

        delta = self.out_conv(h)
        return x + delta  # residual: starts as identity, learns a warp


# ---------------------------------------------------------------------------
# Denoiser: x_hat_theta(z_t, t)
# ---------------------------------------------------------------------------
# Standard UNet that predicts clean x from noisy z_t.
# In NDM the loss compares F_phi(x_hat, s) vs F_phi(x, s), so this network
# still outputs x-space predictions — same interface as DDPM's x-predictor.


class Denoiser(nn.Module):
    """
    x_hat_theta(z_t, t): predict clean x from noisy observation z_t.

    Input:  z_t in R^(B, C, H, W),  t in R^(B,)
    Output: x_hat in R^(B, C, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 2),
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        channels = [base_channels * m for m in channel_mults]

        # Input projection
        self.enc_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_blocks.append(ResBlock(channels[i], time_emb_dim))
            self.downsamples.append(Downsample(channels[i]))
            if channels[i] != channels[i + 1]:
                self.downsamples[-1] = nn.Sequential(
                    Downsample(channels[i]),
                    nn.Conv2d(channels[i], channels[i + 1], 1),
                )

        # Bottleneck
        self.mid1 = ResBlock(channels[-1], time_emb_dim)
        self.mid2 = ResBlock(channels[-1], time_emb_dim)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            in_ch = channels[i + 1]
            out_ch = channels[i]
            self.upsamples.append(
                nn.Sequential(
                    Upsample(in_ch),
                    nn.Conv2d(in_ch, out_ch, 1),
                )
                if in_ch != out_ch
                else Upsample(in_ch)
            )
            self.dec_blocks.append(ResBlock(out_ch, time_emb_dim))

        # Output
        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_channels, 1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))

        h = self.enc_in(z_t)

        skips = []
        for resblock, down in zip(self.enc_blocks, self.downsamples, strict=False):
            h = resblock(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for resblock, up, skip in zip(self.dec_blocks, self.upsamples, reversed(skips), strict=False):
            h = up(h)
            h = h + skip
            h = resblock(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------


class NoiseSchedule(nn.Module):
    """
    DDPM variance-preserving schedule (same as paper's default).
    alpha_t^2 + sigma_t^2 = 1  (signal-to-noise tradeoff)

    alpha_t: signal coefficient  (1 at t=0, ~0 at t=T)
    sigma_t: noise std           (0 at t=0, ~1 at t=T)
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):  # noqa: N803
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)  # ᾱ_t = prod_{s=1}^{t} (1-β_s)

        # alpha_t in NDM notation = sqrt(ᾱ_t)
        self.register_buffer("alpha", alpha_bar.sqrt())
        self.register_buffer("sigma", (1.0 - alpha_bar).sqrt())
        self.T = T

    def get(self, t_idx: torch.Tensor):
        """Return alpha_t and sigma_t for integer timestep indices (1-indexed)."""
        idx = (t_idx - 1).clamp(0, self.T - 1)
        return self.alpha[idx], self.sigma[idx]


# ---------------------------------------------------------------------------
# Full NDM
# ---------------------------------------------------------------------------


class NDM(nn.Module):
    """
    Neural Diffusion Model wrapper.

    Holds:
      - F_phi  : learned forward transformer
      - denoiser: x_hat_theta
      - schedule: noise schedule

    Exposes:
      - q_sample(x, t_idx, eps): sample z_t ~ q_phi(z_t | x)
      - loss(x): full ELBO loss (L_diff + L_prior + L_rec)
    """

    def __init__(
        self,
        in_channels: int = 1,
        T: int = 1000,  # noqa: N803
        fphi_base_ch: int = 32,
        denoiser_base_ch: int = 64,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.T = T
        self.schedule = NoiseSchedule(T)
        self.fphi = FPhi(in_channels, fphi_base_ch, time_emb_dim // 2)
        self.denoiser = Denoiser(in_channels, denoiser_base_ch, time_emb_dim=time_emb_dim)

    # ------------------------------------------------------------------
    # Helper: sample z_t from forward process marginal
    # q_phi(z_t | x) = N(alpha_t * F_phi(x,t), sigma_t^2 * I)
    # ------------------------------------------------------------------
    def q_sample(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (z_t, eps) where z_t = alpha_t * F_phi(x,t) + sigma_t * eps
        t_idx: integer timesteps, shape (B,), values in [1, T]
        """
        if eps is None:
            eps = torch.randn_like(x)

        t_cont = t_idx.float() / self.T  # continuous t in (0,1] for network input
        F_x_t = self.fphi(x, t_cont)  # (B, C, H, W)  # noqa: N806

        alpha_t, sigma_t = self.schedule.get(t_idx)
        alpha_t = alpha_t[:, None, None, None]
        sigma_t = sigma_t[:, None, None, None]

        z_t = alpha_t * F_x_t + sigma_t * eps
        return z_t, F_x_t

    # ------------------------------------------------------------------
    # Compute posterior mean for q_phi(z_s | z_t, x)
    # mu_{s|t} = alpha_s * F_phi(x,s) + sqrt(sigma_s^2 - sigma_tilde_s^2) / sigma_t
    #            * (z_t - alpha_t * F_phi(x,t))
    # ------------------------------------------------------------------
    def _posterior_mean(
        self,
        x: torch.Tensor,
        z_t: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        sigma_tilde_sq: torch.Tensor,
    ) -> torch.Tensor:
        s_cont = s_idx.float() / self.T
        t_cont = t_idx.float() / self.T

        F_x_s = self.fphi(x, s_cont)  # noqa: N806
        F_x_t = self.fphi(x, t_cont)  # noqa: N806

        alpha_s, sigma_s = self.schedule.get(s_idx)
        alpha_t, sigma_t = self.schedule.get(t_idx)

        alpha_s = alpha_s[:, None, None, None]
        sigma_s = sigma_s[:, None, None, None]
        alpha_t = alpha_t[:, None, None, None]
        sigma_t = sigma_t[:, None, None, None]
        sigma_tilde_sq = sigma_tilde_sq[:, None, None, None]

        coeff = (sigma_s**2 - sigma_tilde_sq).sqrt() / sigma_t
        mu = alpha_s * F_x_s + coeff * (z_t - alpha_t * F_x_t)
        return mu, F_x_s, F_x_t

    # ------------------------------------------------------------------
    # Full ELBO loss  (Eq. 8-9 in the paper)
    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute the NDM ELBO loss for a batch of clean images x.

        Returns dict with keys: 'loss', 'ldiff', 'lprior', 'lrec'
        """
        B, C, H, W = x.shape  # noqa: N806
        device = x.device

        # Sample timestep t uniformly from [1, T]
        t_idx = torch.randint(1, self.T + 1, (B,), device=device)
        s_idx = (t_idx - 1).clamp(min=1)  # s = t-1, clamped to >=1

        # ---- Sample z_t from forward process ----
        eps = torch.randn_like(x)
        z_t, F_x_t = self.q_sample(x, t_idx, eps)  # noqa: N806

        # ---- Denoiser predicts x from z_t ----
        t_cont = t_idx.float() / self.T
        x_hat = self.denoiser(z_t, t_cont)  # x_hat_theta(z_t, t)

        # ---- Compute sigma_tilde^2 ----
        # DDPM posterior variance: sigma_tilde^2_{s|t} = sigma_s^2 * (1 - alpha_t^2 / alpha_s^2)
        # Derived from: Var[z_s | z_t, x] in the Gaussian forward process.
        # Both alpha_s, alpha_t are scalars per batch element here.
        alpha_s, sigma_s = self.schedule.get(s_idx)
        alpha_t, sigma_t = self.schedule.get(t_idx)

        alpha_s_sq = alpha_s**2
        alpha_t_sq = alpha_t**2
        sigma_tilde_sq = sigma_s**2 * (1.0 - alpha_t_sq / alpha_s_sq.clamp(min=1e-8))
        sigma_tilde_sq = sigma_tilde_sq.clamp(min=1e-8)  # numerical safety

        # ---- L_diff: KL between forward posterior and reverse (Eq. 9) ----
        # Since both posteriors are Gaussian with the same variance sigma_tilde^2,
        # KL reduces to: (1 / 2*sigma_tilde^2) * || mu_true - mu_pred ||^2
        #
        # mu_true = alpha_s*F_phi(x,s)    + c * (z_t - alpha_t*F_phi(x,t))
        # mu_pred = alpha_s*F_phi(x_hat,s) + c * (z_t - alpha_t*F_phi(x_hat,t))
        # where c = sqrt(sigma_s^2 - sigma_tilde^2) / sigma_t
        #
        # The z_t terms cancel, leaving Eq. 9:
        # || alpha_s*(F_phi(x,s) - F_phi(x_hat,s)) - c*alpha_t*(F_phi(x,t) - F_phi(x_hat,t)) ||^2
        s_cont = s_idx.float() / self.T
        t_cont2 = t_idx.float() / self.T

        F_x_s_true = self.fphi(x, s_cont)  # F_phi(x, s)  # noqa: N806
        F_x_t_true = self.fphi(x, t_cont2)  # F_phi(x, t)  — note: same t as z_t was sampled at  # noqa: N806
        F_x_s_pred = self.fphi(x_hat, s_cont)  # F_phi(x_hat, s)  # noqa: N806
        F_x_t_pred = self.fphi(x_hat, t_cont2)  # F_phi(x_hat, t)  # noqa: N806

        alpha_s_4d = alpha_s[:, None, None, None]
        alpha_t_4d = alpha_t[:, None, None, None]
        sigma_tilde_sq_4d = sigma_tilde_sq[:, None, None, None]  # noqa: F841

        # coefficient c = sqrt(sigma_s^2 - sigma_tilde^2) / sigma_t
        c = ((sigma_s**2 - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t.clamp(min=1e-8))[:, None, None, None]

        diff_s = alpha_s_4d * (F_x_s_true - F_x_s_pred)
        diff_t = c * alpha_t_4d * (F_x_t_true - F_x_t_pred)

        ldiff = (diff_s - diff_t).pow(2).mean()

        # ---- L_prior: KL( q_phi(z_T|x) || N(0,I) ) ----
        # q_phi(z_T|x) = N(alpha_T * F_phi(x,T), sigma_T^2 * I)
        T_idx = torch.full((B,), self.T, device=device, dtype=torch.long)  # noqa: N806
        T_cont = torch.ones(B, device=device)  # noqa: N806
        F_x_T = self.fphi(x, T_cont)  # noqa: N806
        alpha_T, sigma_T = self.schedule.get(T_idx)  # noqa: N806
        alpha_T = alpha_T[:, None, None, None]  # noqa: N806
        sigma_T = sigma_T[:, None, None, None]  # noqa: N806

        # KL( N(mu, sigma^2 I) || N(0, I) ) = 0.5 * (mu^2 + sigma^2 - 1 - log sigma^2)
        lprior = 0.5 * ((alpha_T * F_x_T) ** 2 + sigma_T**2 - 1 - (sigma_T**2).log()).mean()

        # ---- L_rec: -log p_theta(x | z_0) ----
        # At t=1 (s=0), z_0 ≈ x. We approximate with a Gaussian reconstruction.
        # Simplified: MSE between x and x_hat from z_1
        t1_idx = torch.ones(B, device=device, dtype=torch.long)
        z_1, _ = self.q_sample(x, t1_idx)
        t1_cont = t1_idx.float() / self.T
        x_hat_1 = self.denoiser(z_1, t1_cont)
        lrec = F.mse_loss(x_hat_1, x)

        total = ldiff + lprior + lrec
        return {
            "loss": total,
            "ldiff": ldiff.detach(),
            "lprior": lprior.detach(),
            "lrec": lrec.detach(),
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, device: torch.device, steps: int | None = None) -> torch.Tensor:
        """
        Ancestral sampling from z_T ~ N(0,I) back to x.
        steps: if None, uses full T steps; otherwise subsamples uniformly.
        """
        T = self.T  # noqa: N806
        step_seq = list(range(T, 0, -1)) if steps is None else list(range(T, 0, -(T // steps)))[:steps]

        # Determine image shape from a dummy forward pass
        dummy = torch.zeros(1, 1, 28, 28, device=device)
        with torch.no_grad():
            _ = self.fphi(dummy, torch.zeros(1, device=device))

        z = torch.randn(n, 1, 28, 28, device=device)

        for i, t in enumerate(tqdm(step_seq, desc="Sampling", unit="step")):
            t_idx = torch.full((n,), t, device=device, dtype=torch.long)
            s = step_seq[i + 1] if i + 1 < len(step_seq) else 1
            s_idx = torch.full((n,), s, device=device, dtype=torch.long)

            t_cont = t_idx.float() / T
            x_hat = self.denoiser(z, t_cont)

            alpha_s, sigma_s = self.schedule.get(s_idx)
            alpha_t, sigma_t = self.schedule.get(t_idx)

            # DDPM posterior variance: sigma_s^2 * (1 - alpha_t^2 / alpha_s^2)
            alpha_s_sq = alpha_s**2
            alpha_t_sq = alpha_t**2
            sigma_tilde_sq = sigma_s**2 * (1.0 - alpha_t_sq / alpha_s_sq.clamp(min=1e-8))
            sigma_tilde_sq = sigma_tilde_sq.clamp(min=1e-8)

            s_cont = s_idx.float() / T
            F_xs = self.fphi(x_hat, s_cont)  # noqa: N806
            F_xt = self.fphi(x_hat, t_cont)  # noqa: N806

            alpha_s_4d = alpha_s[:, None, None, None]
            alpha_t_4d = alpha_t[:, None, None, None]

            coeff = ((sigma_s**2 - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t.clamp(min=1e-8))[:, None, None, None]
            mu = alpha_s_4d * F_xs + coeff * (z - alpha_t_4d * F_xt)

            noise = torch.randn_like(z) if t > 1 else torch.zeros_like(z)
            z = mu + sigma_tilde_sq[:, None, None, None].clamp(min=0).sqrt() * noise

        return z.clamp(-1, 1)
