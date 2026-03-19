import sys

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

sys.path.append(".")
from src.models.inr_vae_hypernet import INR

LAMBDA_REC = 10.0
LAMBDA_PRIOR = 1e-3


# ─────────────────────────────────────────────
#  F_phi: Named WeightEncoder
#  MLP that maps a flattened MNIST image (784 dims) to a weight vector
#  Theta of size M, representing the INR parameters for that image.
# ─────────────────────────────────────────────
class WeightEncoder(nn.Module):
    """
    MLP that maps a flattened MNIST image (784 dims) to a weight vector
    Theta of size M (the INR parameter space).
    Architecture scales all hidden layer widths from a single `base_width` parameter.
    The network progressively expands then contracts:
        784 → base_width*2 → base_width*3 → base_width*3 → base_width*2 → M
    Args:
        weight_dim : Output size M — must match inr.num_weights.
        layer_width: Single knob that scales all hidden layer sizes (default: 64).
        dropout    : Dropout probability applied between hidden layers (default: 0.2).
    """

    def __init__(self, weight_dim: int, layer_width: int = 64, dropout: float = 0.2):
        super().__init__()
        self.M = weight_dim
        self.base_width = layer_width
        in_dim = 28 * 28  # 784
        w = layer_width
        hidden_dims = [w * 2, w * 3, w * 3, w * 2]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        # Final projection to M — no activation, raw continuous output
        layers.append(nn.Linear(prev, weight_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (B, 1, 28, 28) or (B, 784)
        Returns:
            theta : Tensor of shape (B, M)
        """
        if x.dim() > 2:
            x = x.flatten(start_dim=1)  # (B, 784)
        x = x / 255.0 if x.max() > 1.0 else x  # normalise if raw uint8
        return self.net(x)


# ─────────────────────────────────────────────
#  Noise predictor MLP
# ─────────────────────────────────────────────
class NoisePredictorMLP(nn.Module):
    """
    Predicts the noise epsilon from (z_t, t_norm).
    Input:  z_t    (B, num_weights)
            t_norm (B, 1)           -- time in [0, 1]
    Output: eps    (B, num_weights)
    Architecture mirrors WeightEncoder:
        (num_weights + 1) → w*2 → w*3 → w*3 → w*2 → num_weights
    Time is concatenated to z_t before the first layer.
    """

    def __init__(self, num_weights: int, layer_width: int = 64, dropout: float = 0.2):
        super().__init__()
        w = layer_width
        hidden_dims = [w * 2, w * 3, w * 3, w * 2]
        layers = []
        prev = num_weights + 1  # +1 for t_norm
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_weights))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, z_t: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        z_t    : (B, num_weights)
        t_norm : (B, 1)
        returns: (B, num_weights)
        """
        x = torch.cat([z_t, t_norm], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
#  NDM-INR  (NeuralDiffusionModel + INR decoder)
# ─────────────────────────────────────────────
class NeuralDiffusionINR(nn.Module):
    """
    Neural Diffusion Model (NDM) with an INR decoder.

    Forward process:
        z_t = sqrt(alpha_bar_t) * W(x) + sqrt(1 - alpha_bar_t) * epsilon
    where W(x) = weight_encoder(x) maps an image to INR weight space.

    ELBO (per sample):
        L = L_diff  +  L_rec  +  L_prior * 1[t == T-1]

    L_diff  : noise prediction MSE, computed at every t.
    L_rec   : INR reconstruction MSE at t=0 — image decoded from W(x) via INR,
              compared to the original normalised pixel values.  Computed at every t.
    L_prior : closed-form KL(q(z_T|x) || N(0,I)), only added when t == T-1.
    """

    def __init__(
        self,
        noise_predictor: nn.Module,
        weight_encoder: nn.Module,
        inr: nn.Module,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 100,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
    ):
        super().__init__()
        self.noise_predictor = noise_predictor
        self.weight_encoder = weight_encoder
        self.inr = inr
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.sigma_tilde_factor = sigma_tilde_factor

        # DDPM variance-preserving noise schedule
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq", 1.0 - alpha_cumprod)
        self.register_buffer("sigma", (1.0 - alpha_cumprod).sqrt())

    # ── schedule helpers ──────────────────────────────────────────────────

    def _sigma_tilde_sq(self, s_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Variance of q(z_s | z_t, x_hat) for given s and t indices."""
        sigma_s_sq = self.sigma_sq[s_idx]
        sigma_t_sq = self.sigma_sq[t_idx]
        alpha_t_sq = self.alpha_cumprod[t_idx]
        alpha_s_sq = self.alpha_cumprod[s_idx]
        base = (sigma_t_sq - alpha_t_sq / alpha_s_sq * sigma_s_sq) * sigma_s_sq / sigma_t_sq
        return self.sigma_tilde_factor * base

    # ── forward process ───────────────────────────────────────────────────

    def _sample_theta_t(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode x → Wx, then sample z_t via the forward diffusion process.
        Returns (theta_t, epsilon, Wx).
        """
        Wx = self.weight_encoder(x)  # (B, M)  # noqa: N806
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)  # (B, 1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)  # (B, 1)
        epsilon = torch.randn_like(Wx)
        theta_t = alpha_t * Wx + sigma_t * epsilon
        return theta_t, epsilon, Wx

    # ── loss terms ────────────────────────────────────────────────────────

    def _l_prior(self, Wx: torch.Tensor) -> torch.Tensor:  # noqa: N803
        """
        Closed-form KL between N(sqrt(alpha_T)*W(x), sigma_T^2 I) and N(0, I).
        Returns per-sample KL of shape (B,).
        """
        T_idx = self.T - 1  # noqa: N806
        sigma_T_sq = self.sigma_sq[T_idx]  # scalar  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # scalar  # noqa: N806
        d = Wx.shape[-1]
        # 0.5 * [ d*(sigma_T^2 - log(sigma_T^2) - 1) + alpha_T^2 * ||W(x)||^2 ]
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (Wx**2).sum(dim=-1))
        return kl  # (B,)

    def _l_rec(
        self,
        coords: torch.Tensor,
        pixels: torch.Tensor,
        Wx: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """
        INR reconstruction loss.

        Decodes W(x) through the INR at the given pixel coordinates and
        computes MSE against the normalised pixel values.

        Args:
            x      : (B, 1, 28, 28) — original images (values in [0, 1])
            coords : (B, N, 2)      — pixel coordinates in [-1, 1]
            pixels : (B, N, 1)      — target pixel values in [0, 1]
            Wx     : (B, M)         — weight vector from weight_encoder(x)

        Returns:
            scalar MSE loss
        """
        # Decode: INR(coords, Wx) → (B, N, 1)
        pixels_hat = self.inr(coords, Wx)
        return F.mse_loss(pixels_hat, pixels)

    def _l_diff(
        self,
        theta_t: torch.Tensor,
        epsilon: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Noise prediction MSE.  Returns scalar."""
        eps_hat = self.noise_predictor(theta_t, t_norm)  # (B, M)
        return F.mse_loss(eps_hat, epsilon)

    # ── ELBO ──────────────────────────────────────────────────────────────

    def negative_elbo(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimates the negative ELBO:

            L = L_diff  +  L_rec  +  L_prior * 1[t == T-1]

        L_diff and L_rec are included for every sampled t.
        L_prior is only added when the sampled t equals T-1.

        Args:
            x      : (B, 1, 28, 28)  images normalised to [0, 1]
            coords : (B, N, 2)        pixel coordinates in [-1, 1]
            pixels : (B, N, 1)        pixel values in [0, 1]

        Returns:
            (total_loss, l_diff, l_prior, l_rec)  — all scalars
        """
        B = x.shape[0]  # noqa: N806

        # Sample a random time step per element, 0-indexed
        t_idx = torch.randint(0, self.T, (B,), device=x.device)  # (B,)
        t_norm = (t_idx.float() / max(self.T - 1, 1)).unsqueeze(1)  # (B, 1)

        # Forward process: z_t ~ q(z_t | x)
        theta_t, epsilon, Wx = self._sample_theta_t(x, t_idx)  # noqa: N806

        # Temporary diagnostic — remove after a few batches
        if torch.rand(1).item() < 0.01:  # log ~1% of steps
            print(f"  Wx mean={Wx.mean().item():.3f}  std={Wx.std().item():.3f}  norm={Wx.norm(dim=-1).mean().item():.3f}")

        # ── L_diff: noise prediction, always included ──────────────────
        l_diff = self._l_diff(theta_t, epsilon, t_norm)

        # ── L_rec: INR reconstruction from W(x), always included ──────
        l_rec = self._l_rec(coords, pixels, Wx)

        # ── L_rec: INR reconstruction from W(x), always included ──────
        l_rec = self._l_rec(coords, pixels, Wx)

        l_prior_per_sample = self._l_prior(Wx)  # (B,)
        at_T = (t_idx == self.T - 1).float()  # (B,)  # noqa: N806
        l_prior_real = (at_T * l_prior_per_sample).mean()
        l_prior = l_prior_per_sample.mean()  # Always include prior for stability

        total = l_diff + l_rec + l_prior_real

        return total, l_diff, l_prior, l_rec

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.negative_elbo(x, coords, pixels)

    # ── Sampling ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """
        Ancestral sampling from the NDM-INR.

        Algorithm:
            z_T ~ N(0, I)
            for t = T, ..., 1:
                x_hat = (z_t - sigma_t * eps_hat) / alpha_t   [denoise]
                z_{t-1} ~ q(z_{t-1} | z_t, x_hat)
            pixels = INR(coords, z_0)

        Args:
            coords   : (P, 2) coordinate grid, values in [-1, 1]
            n_samples: number of images to generate

        Returns:
            pixels : (n_samples, P, 1) — pixel values in [0, 1]
        """
        device = self.sqrt_alpha_cumprod.device
        M = self.weight_encoder.M  # noqa: N806
        z_t = torch.randn(n_samples, M, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM sampling", total=self.T):
            t_norm = torch.full((n_samples, 1), t / max(self.T - 1, 1), device=device)
            t_idx = torch.full((n_samples,), t, dtype=torch.long, device=device)

            # Predict noise, then denoise to get theta_hat
            eps_hat = self.noise_predictor(z_t, t_norm)
            alpha_t = self.sqrt_alpha_cumprod[t].view(1, 1)
            sigma_t = self.sigma[t].view(1, 1)
            theta_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                z_t = theta_hat
                break

            s = t - 1
            s_idx = torch.full((n_samples,), s, dtype=torch.long, device=device)

            alpha_s = self.sqrt_alpha_cumprod[s].view(1, 1)
            sigma_s_sq = self.sigma_sq[s].view(1, 1)
            sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx)[0].view(1, 1)

            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t.clamp(min=1e-6)
            mu = alpha_s * theta_hat + coeff * (z_t - alpha_t * theta_hat)

            noise = torch.randn_like(z_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(z_t)
            z_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        # Decode final weights through INR
        coords_batch = coords.unsqueeze(0).expand(n_samples, -1, -1)  # (N, P, 2)
        pixels = self.inr(coords_batch, z_t)  # (N, P, 1)
        return pixels


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    layer_width = 128
    print("=" * 70)
    print("NDMInr — full forward pass")
    print("=" * 70)

    inr = INR(coord_dim=2, hidden_dim=32, n_hidden=3, out_dim=1)
    weight_encoder = WeightEncoder(weight_dim=inr.num_weights, layer_width=layer_width)
    noise_net = NoisePredictorMLP(num_weights=inr.num_weights, layer_width=layer_width)
    ndm = NeuralDiffusionINR(
        weight_encoder=weight_encoder,
        noise_predictor=noise_net,
        inr=inr,
        T=100,
    )

    B = 8
    n_pixels = 784
    image = torch.rand(B, 1, 28, 28)  # [0, 1] floats
    coords = torch.rand(B, n_pixels, 2) * 2 - 1  # [-1, 1]
    pixels = torch.rand(B, n_pixels, 1)  # [0, 1] floats (not binary)

    loss, l_diff, l_prior, l_rec = ndm(image, coords, pixels)

    total_params = sum(p.numel() for p in ndm.parameters())
    enc_params = sum(p.numel() for p in weight_encoder.parameters())
    noise_params = sum(p.numel() for p in noise_net.parameters())
    inr_params = sum(p.numel() for p in inr.parameters())

    print(f"  inr.num_weights  = {inr.num_weights}")
    print(f"  layer_width      = {layer_width}")
    print()
    print(f"  params — WeightEncoder : {enc_params:>10,}")
    print(f"  params — NoiseMLP      : {noise_params:>10,}")
    print(f"  params — INR           : {inr_params:>10,}")
    print(f"  params — total         : {total_params:>10,}")
    print()
    print(f"  loss={loss.item():.4f}  l_diff={l_diff.item():.4f}  l_prior={l_prior.item():.4f}  l_rec={l_rec.item():.4f}")
    print("=" * 70)
