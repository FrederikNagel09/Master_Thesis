import sys

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

sys.path.append(".")

from src.models.inr_vae_hypernet import INR


# ─────────────────────────────────────────────
#  F_phi: Named WeightEncoder
#  MLP that maps INR weight vector θ to the "feature space" used by the diffusion process.
# ─────────────────────────────────────────────
class WeightEncoder(nn.Module):
    """
    MLP that maps a flattened MNIST image (784 dims) to a weight vector Theta of size M.

    Architecture scales all hidden layer widths from a single `base_width` parameter.
    The network progressively expands then contracts:
        784 → base_width*4 → base_width*8 → base_width*4 → base_width → M

    Args:
        M          : Output size of the weight vector Theta.
        base_width : Single knob that scales all hidden layer sizes (default: 64).
        dropout    : Dropout probability applied between hidden layers (default: 0.2).
    """

    def __init__(self, weight_dim: int, layer_width: int = 64, dropout: float = 0.2):
        super().__init__()
        self.M = weight_dim
        self.base_width = layer_width

        in_dim = 28 * 28  # 784

        # Layer widths derived from a single base_width
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
    Neural Diffusion Model (NDM).

    Generalises DDPM by introducing a learnable, time-dependent data
    transformation F_phi(x, t) in the forward process.
    """

    def __init__(
        self,
        noise_predictor: nn.Module,
        weight_encoder: nn.Module,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 100,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
    ):
        super().__init__()

        # noise predictor network
        self.noise_predictor = noise_predictor

        # Encodes images to a weight vector Theta_0
        self.weight_encoder = weight_encoder

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
        """Computes the variance of q_phi(z_s | z_t, x_hat) for given s and t indices."""
        sigma_s_sq = self.sigma_sq[s_idx]
        sigma_t_sq = self.sigma_sq[t_idx]
        alpha_t_sq = self.alpha_cumprod[t_idx]
        alpha_s_sq = self.alpha_cumprod[s_idx]

        base = (sigma_t_sq - alpha_t_sq / alpha_s_sq * sigma_s_sq) * sigma_s_sq / sigma_t_sq
        return self.sigma_tilde_factor * base

    def _sample_theta_t(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns theta_t and the noise epsilon used."""

        # Send image throught weight encoder
        Wx = self.weight_encoder(x, t_norm)  # noqa: N806

        # Initialize constants and noise sample
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(x)

        # Compute theta_t using the forward process equation
        theta_t = alpha_t * Wx + sigma_t * epsilon

        return theta_t, epsilon, Wx

    def _l_diff(self, theta_t, epsilon, t_norm):
        # Predict noise given theta_t and time step
        eps_hat = self.noise_predictor(theta_t, t_norm.unsqueeze(1))

        loss = F.mse_loss(eps_hat, epsilon, reduction="none")

        # Take average across batch
        l_diff = loss.mean(dim=[1])

        return l_diff

    def _l_prior(self, Wx: torch.Tensor) -> torch.Tensor:  # noqa: N803
        """
        Closed-form KL between N(alpha_T * W(x), sigma_T^2 I) and N(0, I).
        """
        T_idx = self.T - 1  # noqa: N806

        sigma_T_sq = self.sigma_sq[T_idx]  # scalar  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # scalar  # noqa: N806
        d = Wx.shape[-1]  # 784

        # Eq. 20:  0.5 * [ d*(sigma_T^2 - log(sigma_T^2) - 1) + alpha_T^2 * ||F||^2 ]
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (Wx**2).sum(dim=-1))
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
        theta_t, epsilon, Wx = self._sample_theta_t(x, t_idx, t_norm.unsqueeze(1))  # noqa: N806

        # --- Three terms of the objective ---
        # Error in noise prediction (L_diff)
        l_diff = self._l_diff(theta_t, epsilon, t_norm.unsqueeze(1))

        # KL divergence to the prior at t=T (L_prior)
        l_prior = self._l_prior(Wx)  # (batch,)

        prior_mask = 0.20 * (t_idx == self.T - 1).float()
        elbo = l_diff + prior_mask * l_prior

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
        noise_net=noise_net,
        inr=inr,
        T=100,
        rec_weight=1.0,
    )

    B = 8
    n_pixels = 784
    image = torch.randn(B, 1, 28, 28)
    coords = torch.rand(B, n_pixels, 2) * 2 - 1  # coords in [-1, 1]
    pixels = torch.randint(0, 2, (B, n_pixels, 1)).float()

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
    print("=" * 70)
