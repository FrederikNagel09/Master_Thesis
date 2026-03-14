import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

# =============================================================================
# Data Transformation Networks F_phi(x, t)
# =============================================================================


class MLPTransformation(nn.Module):
    """
    MLP-based transformation network F_phi(x, t) for MNIST.

    Takes a flattened image x (784-dim) and scalar time t, and outputs a
    transformed image of the same shape.

    Architecture: concatenate [x, t] -> MLP -> output (same dim as x)
    """

    def __init__(self, data_dim: int = 784, hidden_dims: list = None, t_embed_dim: int = 32):  # noqa: RUF013
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        # Small MLP to embed scalar time t into a richer representation
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
        )

        layers = []
        in_dim = data_dim + t_embed_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.SiLU()]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (batch, 784)  - flattened MNIST image
            t: (batch, 1)    - normalized time in [0, 1]
        Returns:
            (batch, 784) - transformed image, same shape as input
        """
        t_emb = self.t_embed(t)
        xt = torch.cat([x, t_emb], dim=-1)
        f_bar = self.net(xt)  # raw network output F_bar_phi
        return (1 - t) * x + t * f_bar


class UNetTransformation(nn.Module):
    """
    U-Net F_phi(x, t) for NDM on MNIST.

    Replaces the naive channel-concatenation time conditioning with:
      - Sinusoidal time embedding projected to t_dim
      - AdaGN (scale+shift) injection at every encoder and decoder block
      - Residual connections inside each block
      - Skip connections across the U-Net

    Input/output contract identical to the original:
        x: (batch, 784)  ->  (batch, 784)
        t: (batch, 1)    normalised to [0, 1]

    The t=0 identity constraint is enforced in forward():
        F_phi(x, 0) = x  exactly,  via  (1-t)*x + t*f_bar
    """

    def __init__(self, t_dim: int = 64):
        super().__init__()
        chs = [16, 32, 64, 128, 128]

        # ── Time embedding ────────────────────────────────────────────────────
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        # Each level: ResBlock (with time conditioning) + optional downsample
        self.enc0 = TimeConditionedResBlock(1, chs[0], t_dim)
        self.enc1 = TimeConditionedResBlock(chs[0], chs[1], t_dim)
        self.enc2 = TimeConditionedResBlock(chs[1], chs[2], t_dim)
        self.enc3 = TimeConditionedResBlock(chs[2], chs[3], t_dim)
        self.enc4 = TimeConditionedResBlock(chs[3], chs[4], t_dim)  # bottleneck

        self.down0 = nn.MaxPool2d(2)  # 28 -> 14
        self.down1 = nn.MaxPool2d(2)  # 14 ->  7
        self.down2 = nn.MaxPool2d(2, padding=1)  #  7 ->  4
        self.down3 = nn.MaxPool2d(2)  #  4 ->  2

        # ── Decoder ───────────────────────────────────────────────────────────
        # skip connections double the input channels at each level
        self.dec3 = TimeConditionedResBlock(chs[4] + chs[3], chs[3], t_dim)
        self.dec2 = TimeConditionedResBlock(chs[3] + chs[2], chs[2], t_dim)
        self.dec1 = TimeConditionedResBlock(chs[2] + chs[1], chs[1], t_dim)
        self.dec0 = TimeConditionedResBlock(chs[1] + chs[0], chs[0], t_dim)

        self.up3 = nn.ConvTranspose2d(chs[4], chs[4], kernel_size=2, stride=2)  #  2 ->  4
        self.up2 = nn.ConvTranspose2d(chs[3], chs[3], kernel_size=2, stride=2)  #  4 ->  8 (trimmed to 7)
        self.up1 = nn.ConvTranspose2d(chs[2], chs[2], kernel_size=2, stride=2)  #  7 -> 14
        self.up0 = nn.ConvTranspose2d(chs[1], chs[1], kernel_size=2, stride=2)  # 14 -> 28

        # ── Output projection ─────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(chs[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 784)
        t: (batch, 1)   normalised time in [0, 1]
        returns: (batch, 784)
        """
        batch = x.shape[0]
        h = x.view(batch, 1, 28, 28)  # (batch, 1, 28, 28)

        t_emb = self.t_embed(t)  # (batch, t_dim)

        # ── Encoder ───────────────────────────────────────────────────────────
        s0 = self.enc0(h, t_emb)  # (batch, 32,  28, 28)
        s1 = self.enc1(self.down0(s0), t_emb)  # (batch, 64,  14, 14)
        s2 = self.enc2(self.down1(s1), t_emb)  # (batch, 128,  7,  7)
        s3 = self.enc3(self.down2(s2), t_emb)  # (batch, 256,  4,  4)
        s4 = self.enc4(self.down3(s3), t_emb)  # (batch, 256,  2,  2)  <- bottleneck

        # ── Decoder with skip connections ─────────────────────────────────────
        # up3: 2->4, concat with s3 (4x4)
        d = self.up3(s4)  # (batch, 256, 4, 4)
        d = self.dec3(torch.cat([d, s3], dim=1), t_emb)  # (batch, 256, 4, 4)

        # up2: 4->8, but s2 is 7x7 — crop to match
        d = self.up2(d)  # (batch, 128, 8, 8)
        d = d[:, :, :7, :7]  # (batch, 128, 7, 7)
        d = self.dec2(torch.cat([d, s2], dim=1), t_emb)  # (batch, 128, 7, 7)

        # up1: 7->14, matches s1 exactly
        d = self.up1(d)  # (batch, 64, 14, 14)
        d = self.dec1(torch.cat([d, s1], dim=1), t_emb)  # (batch, 64, 14, 14)

        # up0: 14->28, matches s0 exactly
        d = self.up0(d)  # (batch, 32, 28, 28)
        d = self.dec0(torch.cat([d, s0], dim=1), t_emb)  # (batch, 32, 28, 28)

        # ── Output ────────────────────────────────────────────────────────────
        f_bar = self.out_conv(d)  # (batch, 1, 28, 28)
        f_bar = f_bar.view(batch, -1)  # (batch, 784)

        # Enforce F_phi(x, 0) = x exactly (Appendix C.2)
        return (1 - t) * x + t * f_bar


class UnetNDM(nn.Module):
    """
    Noise prediction network epsilon_theta(z_t, t) for NDM.

    Same input/output contract as the original Unet:
        x: (batch, 784)  ->  (batch, 784)
        t: (batch, 1)

    Architecture matches UNetTransformation exactly (sinusoidal time embedding,
    AdaGN ResBlocks, skip connections) so both networks share the same
    structure, as specified in Section 4.1 of the NDM paper.

    The only difference from UNetTransformation is the output:
        - UNetTransformation returns (1-t)*x + t*f_bar  (identity constraint)
        - UnetNDM returns the raw predicted noise epsilon_hat  (no constraint)
    """

    def __init__(self, t_dim: int = 64):
        super().__init__()
        chs = [16, 32, 64, 128, 128]

        # ── Time embedding ────────────────────────────────────────────────────
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc0 = TimeConditionedResBlock(1, chs[0], t_dim)
        self.enc1 = TimeConditionedResBlock(chs[0], chs[1], t_dim)
        self.enc2 = TimeConditionedResBlock(chs[1], chs[2], t_dim)
        self.enc3 = TimeConditionedResBlock(chs[2], chs[3], t_dim)
        self.enc4 = TimeConditionedResBlock(chs[3], chs[4], t_dim)  # bottleneck

        self.down0 = nn.MaxPool2d(2)  # 28 -> 14
        self.down1 = nn.MaxPool2d(2)  # 14 ->  7
        self.down2 = nn.MaxPool2d(2, padding=1)  #  7 ->  4
        self.down3 = nn.MaxPool2d(2)  #  4 ->  2

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec3 = TimeConditionedResBlock(chs[4] + chs[3], chs[3], t_dim)
        self.dec2 = TimeConditionedResBlock(chs[3] + chs[2], chs[2], t_dim)
        self.dec1 = TimeConditionedResBlock(chs[2] + chs[1], chs[1], t_dim)
        self.dec0 = TimeConditionedResBlock(chs[1] + chs[0], chs[0], t_dim)

        self.up3 = nn.ConvTranspose2d(chs[4], chs[4], kernel_size=2, stride=2)  #  2 ->  4
        self.up2 = nn.ConvTranspose2d(chs[3], chs[3], kernel_size=2, stride=2)  #  4 ->  8 (trimmed to 7)
        self.up1 = nn.ConvTranspose2d(chs[2], chs[2], kernel_size=2, stride=2)  #  7 -> 14
        self.up0 = nn.ConvTranspose2d(chs[1], chs[1], kernel_size=2, stride=2)  # 14 -> 28

        # ── Output projection ─────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(chs[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 784)   noisy latent z_t
        t: (batch, 1)     normalised time in [0, 1]
        returns: (batch, 784)  predicted noise epsilon_hat
        """
        batch = x.shape[0]
        h = x.view(batch, 1, 28, 28)

        t_emb = self.t_embed(t)  # (batch, t_dim)

        # ── Encoder ───────────────────────────────────────────────────────────
        s0 = self.enc0(h, t_emb)  # (batch, 32,  28, 28)
        s1 = self.enc1(self.down0(s0), t_emb)  # (batch, 64,  14, 14)
        s2 = self.enc2(self.down1(s1), t_emb)  # (batch, 128,  7,  7)
        s3 = self.enc3(self.down2(s2), t_emb)  # (batch, 256,  4,  4)
        s4 = self.enc4(self.down3(s3), t_emb)  # (batch, 256,  2,  2)

        # ── Decoder ───────────────────────────────────────────────────────────
        d = self.up3(s4)  # (batch, 256, 4, 4)
        d = self.dec3(torch.cat([d, s3], dim=1), t_emb)  # (batch, 256, 4, 4)

        d = self.up2(d)  # (batch, 128, 8, 8)
        d = d[:, :, :7, :7]  # (batch, 128, 7, 7)
        d = self.dec2(torch.cat([d, s2], dim=1), t_emb)  # (batch, 128, 7, 7)

        d = self.up1(d)  # (batch, 64, 14, 14)
        d = self.dec1(torch.cat([d, s1], dim=1), t_emb)  # (batch, 64, 14, 14)

        d = self.up0(d)  # (batch, 32, 28, 28)
        d = self.dec0(torch.cat([d, s0], dim=1), t_emb)  # (batch, 32, 28, 28)

        # ── Output ────────────────────────────────────────────────────────────
        out = self.out_conv(d)  # (batch, 1, 28, 28)
        return out.view(batch, -1)  # (batch, 784)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch, 1) normalised to [0, 1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t * freqs.unsqueeze(0) * 1000  # scale t to [0, 1000] range
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (batch, dim)


# =============================================================================
# Neural Diffusion Model
# =============================================================================
class TimeConditionedResBlock(nn.Module):
    """
    Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm,
    with time embedding injected as a scale+shift into the first GroupNorm
    (AdaGN style, as used in ADM/Dhariwal & Nichol 2021).
    Includes a residual connection with a 1x1 conv if channel dims differ.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)

        # Projects time embedding to scale and shift for AdaGN on gn1
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_ch * 2),  # split into scale + shift
        )

        # Residual projection if channel count changes
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (batch, t_dim)
        h = self.conv1(x)
        h = self.gn1(h)

        # AdaGN: modulate normalised features with time-derived scale/shift
        t_out = self.t_proj(t_emb)  # (batch, out_ch*2)
        scale, shift = t_out.chunk(2, dim=1)  # each (batch, out_ch)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)

        return h + self.res_conv(x)


class NeuralDiffusionModel(nn.Module):
    """
    Neural Diffusion Model (NDM).

    Generalises DDPM by introducing a learnable, time-dependent data
    transformation F_phi(x, t) in the forward process.
    """

    def __init__(
        self,
        network: nn.Module,
        F_phi: nn.Module,  # noqa: N803
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 100,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
    ):
        super().__init__()

        # epsilon_theta: noise predictor network
        self.network = network
        # learnable data transformation network
        self.F_phi = F_phi

        # beta 1 start value
        self.beta_1 = beta_1
        # beta T end value
        self.beta_T = beta_T

        # Amount of time steps
        self.T = T

        # Controlls something?
        self.sigma_tilde_factor = sigma_tilde_factor  # in [0,1]; 0 = deterministic DDIM

        # DDPM variance-preserving noise schedule
        beta = torch.linspace(beta_1, beta_T, T)  # goes from 0.0001 to 0.02 in T steps
        alpha = 1.0 - beta  # goes from 0.9999 to 0.98 in T steps

        # Accumulated product of alpha: alpha_bar_t = Π_{s=1}^t alpha_s
        alpha_cumprod = alpha.cumprod(dim=0)  # goes to 0 as t increases, since alpha < 1

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)  # alpha_bar
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq", 1.0 - alpha_cumprod)  # sigma_t^2
        self.register_buffer("sigma", (1.0 - alpha_cumprod).sqrt())

    def _sigma_tilde_sq(self, s_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        sigma_s_sq = self.sigma_sq[s_idx]
        sigma_t_sq = self.sigma_sq[t_idx]
        alpha_t_sq = self.alpha_cumprod[t_idx]
        alpha_s_sq = self.alpha_cumprod[s_idx]

        base = (sigma_t_sq - alpha_t_sq / alpha_s_sq * sigma_s_sq) * sigma_s_sq / sigma_t_sq
        return self.sigma_tilde_factor * base

    def _sample_zt(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns z_t and the noise epsilon used."""
        Fx = self.F_phi(x, t_norm)  # noqa: N806
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(x)
        z_t = alpha_t * Fx + sigma_t * epsilon
        return z_t, epsilon, Fx

    def _l_diff(self, x, z_t, t_idx, t_norm, Fx_t):  # noqa: N803
        eps_hat = self.network(z_t, t_norm.unsqueeze(1))
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        s_idx = (t_idx - 1).clamp(min=0)
        s_norm = s_idx.float() / (self.T - 1)

        Fx_hat_t = self.F_phi(x_hat, t_norm.unsqueeze(1))  # noqa: N806
        Fx_hat_s = self.F_phi(x_hat, s_norm.unsqueeze(1))  # noqa: N806
        Fx_s = self.F_phi(x, s_norm.unsqueeze(1))  # noqa: N806

        alpha_s = self.sqrt_alpha_cumprod[s_idx].unsqueeze(1)
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).unsqueeze(1)  # compute once
        coeff = (self.sigma_sq[s_idx].unsqueeze(1) - sigma_tilde_sq).clamp(min=0).sqrt()
        coeff = coeff / self.sigma[t_idx].unsqueeze(1).clamp(min=1e-6)

        diff = alpha_s * (Fx_s - Fx_hat_s) + coeff * alpha_t * (Fx_hat_t - Fx_t)
        l_diff = (diff**2).sum(dim=-1) / (2.0 * sigma_tilde_sq.squeeze(1).clamp(min=1e-8))
        return l_diff

    def _l_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Closed-form KL between N(alpha_T * F(x,T), sigma_T^2 I) and N(0, I).
        """
        T_idx = self.T - 1  # noqa: N806
        t_norm_T = torch.ones(x.shape[0], 1, device=x.device)  # noqa: N806
        Fx_T = self.F_phi(x, t_norm_T)  # (batch, 784)  # noqa: N806 ####################################################################

        sigma_T_sq = self.sigma_sq[T_idx]  # scalar  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # scalar  # noqa: N806
        d = x.shape[-1]  # 784

        # Eq. 20:  0.5 * [ d*(sigma_T^2 - log(sigma_T^2) - 1) + alpha_T^2 * ||F||^2 ]
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (Fx_T**2).sum(dim=-1))
        return kl  # (batch,)

    def _l_rec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss at t=0.
        """
        t0_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        t0_norm = torch.zeros(x.shape[0], 1, device=x.device)
        z0, _, _ = self._sample_zt(x, t0_idx, t0_norm)

        # Gaussian reconstruction: -log N(x; z0, I) ∝ 0.5 ||x - z0||^2
        l_rec = 0.5 * ((x - z0) ** 2).sum(dim=-1)  # (batch,)
        return l_rec

    def negative_elbo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates the negative ELBO:
            L = E[ L_prior + L_rec + L_diff ]

        using a single Monte-Carlo sample over t (same as DDPM training).

        Parameters:
            x: (batch, 784)
        Returns:
            (batch,) negative ELBO per sample
        """
        batch_size = x.shape[0]

        # --- Sample random time step ---
        t_idx = torch.randint(1, self.T + 1, (batch_size,), device=x.device) - 1  # 0-indexed
        t_norm = t_idx.float() / (self.T - 1)

        # --- Forward process: z_t ~ q_phi(z_t | x) ---
        z_t, _, Fx_t = self._sample_zt(x, t_idx, t_norm.unsqueeze(1))  # noqa: N806

        # --- Three terms of the objective ---
        l_diff = self._l_diff(x, z_t, t_idx, t_norm, Fx_t)  # (batch,)
        l_prior = self._l_prior(x)  # (batch,)
        # l_rec = self._l_rec(x)  # (batch,)

        elbo = l_diff + l_prior
        return elbo.mean(), l_diff.mean(), l_prior.mean()

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.negative_elbo(x)

    @torch.no_grad()
    def sample(self, shape: tuple) -> torch.Tensor:
        """
        Ancestral sampling from the NDM.

        Algorithm 2:
            z_T ~ N(0, I)
            for t = T, ..., 1:
                x_hat = x_hat_theta(z_t, t)          [noise -> x_hat]
                z_{t-1} ~ q_phi(z_{t-1} | z_t, x_hat)
            x ~ p(x | z_0)  [identity at t=0, so return z_0]

        Parameters:
            shape: (n_samples, 784)
        """
        device = self.sqrt_alpha_cumprod.device
        z_t = torch.randn(shape, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM Sampling", total=self.T):
            t_idx = torch.full((shape[0],), t, dtype=torch.long, device=device)
            t_norm = torch.full((shape[0], 1), t / max(self.T - 1, 1), device=device)

            # --- Predict x_hat from z_t (Eq. 34, Appendix C) ---
            eps_hat = self.network(z_t, t_norm)
            alpha_t = self.sqrt_alpha_cumprod[t].unsqueeze(0)
            sigma_t = self.sigma[t].unsqueeze(0)
            x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                # p(x|z_0) = identity (F_phi constrained to identity at t=0)
                z_t = x_hat
                break

            # --- Sample z_{t-1} ~ q_phi(z_{t-1} | z_t, x_hat) (Eq. 7/15) ---
            s = t - 1
            s_idx = torch.full((shape[0],), s, dtype=torch.long, device=device)
            s_norm = torch.full((shape[0], 1), s / max(self.T - 1, 1), device=device)

            Fx_hat_s = self.F_phi(x_hat, s_norm)  # F_phi(x_hat, s)  # noqa: N806
            Fx_hat_t = self.F_phi(x_hat, t_norm)  # F_phi(x_hat, t)  # noqa: N806

            # All scalars kept as (1, 1) so they broadcast cleanly with (batch, 784)
            alpha_s = self.sqrt_alpha_cumprod[s].view(1, 1)  # (1, 1)
            sigma_s_sq = self.sigma_sq[s].view(1, 1)  # (1, 1)
            sigma_t_val = self.sigma[t].view(1, 1)  # (1, 1)
            alpha_t_val = self.sqrt_alpha_cumprod[t].view(1, 1)  # (1, 1)
            sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx)[0].view(1, 1)  # scalar → (1,1)

            # Mean of q_phi(z_s | z_t, x_hat) — Eq. 7
            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t_val.clamp(min=1e-6)
            mu = alpha_s * Fx_hat_s + coeff * (z_t - alpha_t_val * Fx_hat_t)

            noise = torch.randn_like(z_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(z_t)
            z_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        return z_t
