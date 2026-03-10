import torch
import torch.nn as nn
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
        t_emb = self.t_embed(t)  # (batch, t_embed_dim)
        xt = torch.cat([x, t_emb], dim=-1)  # (batch, 784 + t_embed_dim)
        return self.net(xt)  # (batch, 784)


class UNetTransformation(nn.Module):
    """
    U-Net-based transformation network F_phi(x, t) for MNIST.

    Same input/output contract as MLPTransformation:
        x: (batch, 784)  ->  output: (batch, 784)
        t: (batch, 1)

    Internally reshapes x to (batch, 1, 28, 28), concatenates a time channel,
    runs through a U-Net, then flattens back. This is the same architectural
    pattern used in the DDPM Unet above — here repurposed as the learnable
    transformation F_phi rather than the noise predictor.
    """

    def __init__(self):
        super().__init__()
        chs = [32, 64, 128, 256, 256]

        # Encoder (same as DDPM Unet)
        self._convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2, chs[0], kernel_size=3, padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                    nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
                    nn.SiLU(),
                ),
            ]
        )

        # Decoder
        self._tconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),
                ),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (batch, 784)
            t: (batch, 1)
        Returns:
            (batch, 784)
        """
        batch_size = x.shape[0]
        x2 = x.view(batch_size, 1, 28, 28)
        tt = t[:, :, None, None].expand(batch_size, 1, 28, 28)
        x2t = torch.cat([x2, tt], dim=1)  # (batch, 2, 28, 28)

        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat([signal, signals[-i]], dim=1)
                signal = tconv(signal)

        return signal.view(batch_size, -1)  # (batch, 784)


# =============================================================================
# Neural Diffusion Model
# =============================================================================


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
        self.network = network  # epsilon_theta: noise predictor
        self.F_phi = F_phi  # learnable data transformation

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.sigma_tilde_factor = sigma_tilde_factor  # in [0,1]; 0 = deterministic DDIM

        # DDPM variance-preserving noise schedule
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)

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
        Fx = self.F_phi(x, t_norm)  # (batch, 784)  # noqa: N806
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(x)
        z_t = alpha_t * Fx + sigma_t * epsilon
        return z_t, epsilon, Fx

    def _l_diff(
        self,
        x: torch.Tensor,
        z_t: torch.Tensor,
        t_idx: torch.Tensor,
        t_norm: torch.Tensor,
        Fx_t: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """
        KL divergence between forward posterior q_phi(z_{t-1}|z_t, x) and
        reverse p_theta(z_{t-1}|z_t), collapsed to an MSE on the transformed
        data
        """

        # Predict noise -> reconstruct x_hat -> get F_phi(x_hat, t)
        eps_hat = self.network(z_t, t_norm.unsqueeze(1))  # (batch, 784)
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        # s = t - 1  (0-indexed; clamp at 0)
        s_idx = (t_idx - 1).clamp(min=0)
        s_norm = s_idx.float() / (self.T - 1)

        Fx_hat_t = self.F_phi(x_hat, t_norm.unsqueeze(1))  # F_phi(x_hat, t)  # noqa: N806
        Fx_hat_s = self.F_phi(x_hat, s_norm.unsqueeze(1))  # F_phi(x_hat, s)  # noqa: N806
        Fx_s = self.F_phi(x, s_norm.unsqueeze(1))  # F_phi(x,    s)  # noqa: N806

        # All per-sample scalars unsqueezed to (batch, 1) for broadcasting with (batch, 784)
        alpha_s = self.sqrt_alpha_cumprod[s_idx].unsqueeze(1)  # (batch, 1)
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).unsqueeze(1)  # (batch, 1)
        coeff = (self.sigma_sq[s_idx].unsqueeze(1) - sigma_tilde_sq).clamp(min=0).sqrt()
        coeff = coeff / self.sigma[t_idx].unsqueeze(1).clamp(min=1e-6)  # (batch, 1)

        # MSE on the mean difference
        diff = alpha_s * (Fx_s - Fx_hat_s) + coeff * alpha_t * (Fx_hat_t - Fx_t)
        l_diff = 0.5 * (diff**2).sum(dim=-1)  # (batch,)
        return l_diff

    def _l_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Closed-form KL between N(alpha_T * F(x,T), sigma_T^2 I) and N(0, I).
        """
        T_idx = self.T - 1  # noqa: N806
        t_norm_T = torch.ones(x.shape[0], 1, device=x.device)  # noqa: N806
        Fx_T = self.F_phi(x, t_norm_T)  # (batch, 784)  # noqa: N806

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
        l_rec = self._l_rec(x)  # (batch,)

        return l_diff + l_prior + l_rec  # (batch,)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.negative_elbo(x).mean()

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
            sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).mean().view(1, 1)  # scalar → (1,1)

            # Mean of q_phi(z_s | z_t, x_hat) — Eq. 7
            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t_val.clamp(min=1e-6)
            mu = alpha_s * Fx_hat_s + coeff * (z_t - alpha_t_val * Fx_hat_t)

            noise = torch.randn_like(z_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(z_t)
            z_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        return z_t
