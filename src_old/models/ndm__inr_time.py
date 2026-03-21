import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

sys.path.append(".")


class INR(nn.Module):
    """
    A small MLP that predicts pixel values from (x,y) coordinates.
    Weights are supplied externally by the hypernetwork (VAE decoder).
    """

    def __init__(self, coord_dim=2, hidden_dim=20, n_hidden=2, out_dim=1):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim

        # Compute the total number of weights this INR needs
        dims = [coord_dim] + [hidden_dim] * n_hidden + [out_dim]
        self.weight_shapes = []
        self.bias_shapes = []
        total = 0
        for i in range(len(dims) - 1):
            self.weight_shapes.append((dims[i + 1], dims[i]))
            self.bias_shapes.append((dims[i + 1],))
            total += dims[i + 1] * dims[i] + dims[i + 1]
        self.num_weights = total

    def forward(self, coords, flat_weights):
        """
        coords:       (batch, n_pixels, 2)
        flat_weights: (batch, num_weights)
        returns:      (batch, n_pixels, 1)  — logits
        """
        batch = coords.shape[0]
        idx = 0
        x = coords  # (batch, n_pixels, coord_dim)

        for i, (ws, _bs) in enumerate(zip(self.weight_shapes, self.bias_shapes)):  # noqa: B905
            out_f, in_f = ws
            W = flat_weights[:, idx : idx + out_f * in_f].view(batch, out_f, in_f)  # noqa: N806
            idx += out_f * in_f
            b = flat_weights[:, idx : idx + out_f].view(batch, 1, out_f)
            idx += out_f

            # (batch, n_pixels, out_f)
            x = torch.bmm(x, W.transpose(1, 2)) + b

            # ReLU on hidden layers, sigmoid on last
            x = torch.relu(x) if i < len(self.weight_shapes) - 1 else torch.tanh(x)

        return x  # (batch, n_pixels, 1)


# =============================================================================
# Noise Predictor Network  epsilon_theta(z_t, t)
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t in [0,1] to a rich sinusoidal embedding, then projects it.
    Identical in spirit to the positional encoding used in DDPM / transformer
    models, but operating on continuous normalised time rather than integer steps.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        # Learnable projection on top of the fixed sinusoidal features
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (batch, 1)  normalised time in [0, 1]
        returns: (batch, embed_dim)
        """
        half = self.embed_dim // 2
        # frequencies: 10000^(2i/D) for i=0..half-1
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)  # (half,)
        args = t * freqs.unsqueeze(0)  # (batch, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch, embed_dim)
        return self.proj(emb)  # (batch, embed_dim)


class NoisePredictor(nn.Module):
    """
    MLP noise predictor  epsilon_theta(z_t, t).

    Operates entirely in *weight space* (the same space that F_phi maps into).
    For MNIST with the default INR this is `inr.num_weights` dimensional.

    Architecture
    ------------
    - Sinusoidal time embedding projected to `t_embed_dim`
    - Four residual blocks:  Linear -> LayerNorm -> SiLU -> Linear  (+ skip)
    - Time conditioning: add projected time embedding at the start of each block
    - Final linear readout with no activation (predicts raw noise)

    Parameters
    ----------
    weight_dim  : dimensionality of the weight vector (= inr.num_weights)
    hidden_dim  : width of every hidden layer            (default 512)
    n_blocks    : number of residual blocks              (default 4)
    t_embed_dim : dimensionality of the time embedding   (default 128)
    """

    def __init__(
        self,
        weight_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        t_embed_dim: int = 128,
    ):
        super().__init__()

        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        # --- Time embedding ---
        self.time_embed = SinusoidalTimeEmbedding(t_embed_dim)
        # Project time embedding to hidden_dim so we can add it inside each block
        self.time_proj = nn.Linear(t_embed_dim, hidden_dim)

        # --- Input projection: weight_dim -> hidden_dim ---
        self.input_proj = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- Residual blocks ---
        # Each block: LayerNorm -> Linear -> SiLU -> Linear  (+skip from block input)
        # Time conditioning is added before the first activation.
        self.blocks = nn.ModuleList()
        self.t_projs = nn.ModuleList()  # one time projection per block
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                    ]
                )
            )
            # separate time projections per block keep conditioning expressive
            self.t_projs.append(nn.Linear(t_embed_dim, hidden_dim))

        # --- Output projection: hidden_dim -> weight_dim ---
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, weight_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, weight_dim)   noisy weight vector at time t
        t : (batch, 1)            normalised time in [0, 1]

        Returns
        -------
        eps_hat : (batch, weight_dim)  predicted noise
        """
        t_emb = self.time_embed(t)  # (batch, t_embed_dim)
        t_global = self.time_proj(t_emb)  # (batch, hidden_dim) — for residual conditioning

        h = self.input_proj(z)  # (batch, hidden_dim)
        h = h + t_global  # inject time before first block

        for (norm, lin1, lin2), t_proj in zip(self.blocks, self.t_projs):  # noqa: B905
            residual = h
            h = norm(h)
            h = lin1(h)
            h = h + t_proj(t_emb)  # time conditioning inside each block
            h = F.silu(h)
            h = lin2(h)
            h = h + residual  # skip connection

        return self.output_proj(h)  # (batch, weight_dim)


# =============================================================================
# Data Transformation Network  F_phi(x, t)
# =============================================================================


class WeightEncoder(nn.Module):
    """
    MLP-based transformation network F_phi(x, t) for MNIST.

    Maps a flattened image x (784-dim) and normalised scalar time t to a
    *weight vector* in the INR parameter space (inr.num_weights - dim).

    Architecture
    ------------
    [x (784)] + [t_emb (t_embed_dim)]  ->  MLP  ->  weight vector (weight_dim)
    """

    def __init__(
        self,
        data_dim: int = 784,
        weight_dim: int = 501,  # must match inr.num_weights
        hidden_dims: list = None,  # noqa: RUF013
        t_embed_dim: int = 64,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.time_embed = SinusoidalTimeEmbedding(t_embed_dim)

        layers = []
        in_dim = data_dim + t_embed_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.SiLU()]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, weight_dim))  # output = INR weight vector
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 784)   flattened MNIST image
        t : (batch, 1)     normalised time in [0, 1]

        Returns
        -------
        (batch, weight_dim)  weight vector in INR parameter space
        """
        t_emb = self.time_embed(t)  # (batch, t_embed_dim)
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)


# =============================================================================
# Neural Diffusion Model
# =============================================================================


class NeuralDiffusionModel(nn.Module):
    """
    Neural Diffusion Model (NDM) with INR-based reconstruction.

    Forward process
    ---------------
      z_t = alpha_t * F_phi(x, t) + sigma_t * eps,   eps ~ N(0, I)

    F_phi maps images into *INR weight space*.  The diffusion process therefore
    operates entirely in weight space.

    Reconstruction (t=0)
    --------------------
      weights = F_phi(x, t=0)          # encode image -> weight vector
      x_recon = INR(coords, weights)   # decode weight vector -> pixel values
      l_rec   = 0.5 * ||x - x_recon||^2

    Sampling
    --------
      Sample z_0 via ancestral sampling, then decode: x = INR(coords, z_0).
    """

    def __init__(
        self,
        network: nn.Module,  # NoisePredictor  epsilon_theta(z_t, t)
        F_phi: nn.Module,  # noqa: N803 — WeightEncoder
        inr: INR,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 100,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
        data_dim: int = 784,  # pixel dimension (for coords / MSE)
        img_size: int = 28,  # spatial size (28 for MNIST)
    ):
        super().__init__()

        self.data_dim = data_dim
        self.img_size = img_size
        self.network = network  # epsilon_theta
        self.F_phi = F_phi  # image -> weight vector
        self.inr = inr  # weight vector + coords -> pixels

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.sigma_tilde_factor = sigma_tilde_factor

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

        # --- Pre-build pixel coordinate grid for MNIST (28x28) ---
        # Normalised to [-1, 1]; shape (1, 784, 2) — broadcast over batch
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (28,28) each
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (784, 2)
        self.register_buffer("coords", coords.unsqueeze(0))  # (1, 784, 2)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns z_t, epsilon, and F_phi(x, t) — all in weight space."""
        Fx = self.F_phi(x, t_norm)  # (batch, weight_dim)  # noqa: N806
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)  # (batch, 1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)  # (batch, 1)
        epsilon = torch.randn_like(Fx)  # noise in weight space
        z_t = alpha_t * Fx + sigma_t * epsilon
        return z_t, epsilon, Fx

    def _inr_decode(self, flat_weights: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of flat weight vectors to pixel images.

        Parameters
        ----------
        flat_weights : (batch, num_weights)

        Returns
        -------
        pixels : (batch, 784)   values in [0, 1]  (sigmoid output from INR)
        """
        batch = flat_weights.shape[0]
        coords = self.coords.expand(batch, -1, -1)  # (batch, 784, 2)
        pixels = self.inr(coords, flat_weights)  # (batch, 784, 1)
        return pixels.squeeze(-1)  # (batch, 784)

    # -------------------------------------------------------------------------
    # Loss terms
    # -------------------------------------------------------------------------

    def _l_diff(self, x, z_t, t_idx, t_norm, Fx_t):  # noqa: N803
        eps_hat = self.network(z_t, t_norm.unsqueeze(1))  # (batch, weight_dim)
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)

        # Predicted clean weight vector
        x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        s_idx = (t_idx - 1).clamp(min=0)
        s_norm = s_idx.float() / (self.T - 1)

        # With:
        x_hat_pixels = self._inr_decode(x_hat)  # weight -> pixel space
        Fx_hat_t = self.F_phi(x_hat_pixels, t_norm.unsqueeze(1))  # noqa: N806
        Fx_hat_s = self.F_phi(x_hat_pixels, s_norm.unsqueeze(1))  # noqa: N806
        Fx_s = self.F_phi(x, s_norm.unsqueeze(1))  # noqa: N806

        alpha_s = self.sqrt_alpha_cumprod[s_idx].unsqueeze(1)
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).unsqueeze(1)
        coeff = (self.sigma_sq[s_idx].unsqueeze(1) - sigma_tilde_sq).clamp(min=0).sqrt() / self.sigma[t_idx].unsqueeze(1).clamp(min=1e-6)

        diff = alpha_s * (Fx_s - Fx_hat_s) + coeff * alpha_t * (Fx_hat_t - Fx_t)
        l_diff = (diff**2).sum(dim=-1) / (2.0 * sigma_tilde_sq.squeeze(1).clamp(min=1e-8))
        return l_diff

    def _l_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Closed-form KL  N(alpha_T * F(x,T), sigma_T^2 I) || N(0,I)."""
        T_idx = self.T - 1  # noqa: N806
        t_norm_T = torch.ones(x.shape[0], 1, device=x.device)  # noqa: N806
        Fx_T = self.F_phi(x, t_norm_T)  # (batch, weight_dim)  # noqa: N806
        sigma_T_sq = self.sigma_sq[T_idx]  # scalar  # noqa: N806
        alpha_T_sq = self.alpha_cumprod[T_idx]  # scalar  # noqa: N806
        d = Fx_T.shape[-1]  # weight_dim
        kl = 0.5 * (d * (sigma_T_sq - torch.log(sigma_T_sq) - 1.0) + alpha_T_sq * (Fx_T**2).sum(dim=-1))
        return kl  # (batch,)

    def _l_rec(self, x: torch.Tensor) -> torch.Tensor:
        """
        INR-based reconstruction loss at t = 0.

        Pipeline
        --------
          1. Encode image to weight vector:  w = F_phi(x, t=0)
          2. Decode weight vector to pixels: x_recon = INR(coords, w)
          3. MSE in pixel space:             l_rec = 0.5 * ||x - x_recon||^2

        Parameters
        ----------
        x : (batch, 784)   flattened input image, values in [0, 1]

        Returns
        -------
        (batch,)  per-sample reconstruction loss
        """
        t0_norm = torch.zeros(x.shape[0], 1, device=x.device)
        weights = self.F_phi(x, t0_norm)  # (batch, weight_dim)
        x_recon = self._inr_decode(weights)  # (batch, 784)  in [0,1]
        l_rec = 0.5 * ((x - x_recon) ** 2).sum(dim=-1)  # (batch,)
        return l_rec

    # -------------------------------------------------------------------------
    # ELBO
    # -------------------------------------------------------------------------

    def negative_elbo(self, x: torch.Tensor, rec_scale: float = 1.0):
        """
        Estimates the negative ELBO:
            L = E[ l_diff ] + prior_mask * l_prior + l_rec

        Parameters
        ----------
        x : (batch, 784)

        Returns
        -------
        (scalar mean loss, l_diff mean, l_prior mean, l_rec mean)
        """
        batch_size = x.shape[0]

        # Sample random time step  t ~ Uniform{1, ..., T}
        t_idx = torch.randint(1, self.T + 1, (batch_size,), device=x.device) - 1
        t_norm = t_idx.float() / (self.T - 1)

        # Forward: z_t ~ q_phi(z_t | x)
        z_t, _, Fx_t = self._sample_zt(x, t_idx, t_norm.unsqueeze(1))  # noqa: N806

        # Three loss terms
        l_diff = self._l_diff(x, z_t, t_idx, t_norm, Fx_t)  # (batch,)
        l_prior = self._l_prior(x)  # (batch,)
        l_rec = self._l_rec(x)  # (batch,)

        prior_mask = (t_idx == self.T - 1).float()
        elbo = l_diff + prior_mask * l_prior + rec_scale * l_rec

        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    def loss(self, x: torch.Tensor, rec_scale: float = 1.0):
        return self.negative_elbo(x, rec_scale=rec_scale)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Ancestral sampling from the NDM, with INR decoding at t=0.

        Returns
        -------
        images : (n_samples, 784)  pixel values in [0, 1]
        """
        weight_dim = self.F_phi.net[-1].out_features  # infer from WeightEncoder output
        device = self.sqrt_alpha_cumprod.device

        z_t = torch.randn(n_samples, weight_dim, device=device)  # sample in weight space

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM Sampling", total=self.T):
            t_idx = torch.full((n_samples,), t, dtype=torch.long, device=device)
            t_norm = torch.full((n_samples, 1), t / max(self.T - 1, 1), device=device)

            # Predict noise, recover clean weight vector
            eps_hat = self.network(z_t, t_norm)  # (n, weight_dim)
            alpha_t = self.sqrt_alpha_cumprod[t].unsqueeze(0)
            sigma_t = self.sigma[t].unsqueeze(0)
            x_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                # x_hat is now a clean weight vector — decode through INR
                images = self._inr_decode(x_hat)  # (n_samples, 784)
                return images

            # Sample z_{t-1} ~ q_phi(z_{t-1} | z_t, x_hat)
            s = t - 1
            s_idx = torch.full((n_samples,), s, dtype=torch.long, device=device)
            s_norm = torch.full((n_samples, 1), s / max(self.T - 1, 1), device=device)

            # With:
            x_hat_pixels = self._inr_decode(x_hat)  # weight -> pixel space
            Fx_hat_t = self.F_phi(x_hat_pixels, t_norm)  # noqa: N806
            Fx_hat_s = self.F_phi(x_hat_pixels, s_norm)  # noqa: N806

            alpha_s = self.sqrt_alpha_cumprod[s].view(1, 1)
            sigma_s_sq = self.sigma_sq[s].view(1, 1)
            sigma_t_val = self.sigma[t].view(1, 1)
            alpha_t_val = self.sqrt_alpha_cumprod[t].view(1, 1)
            sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx)[0].view(1, 1)

            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t_val.clamp(min=1e-6)
            mu = alpha_s * Fx_hat_s + coeff * (z_t - alpha_t_val * Fx_hat_t)
            noise = torch.randn_like(z_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(z_t)
            z_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        # Should not reach here, but safety fallback
        return self._inr_decode(z_t)


# =============================================================================
# Convenience factory
# =============================================================================


def build_ndm(
    inr: INR,
    img_size: int = 28,
    f_phi_hidden: list = None,  # noqa: RUF013
    noise_hidden: int = 512,
    noise_blocks: int = 4,
    t_embed_dim: int = 128,
    beta_1: float = 1e-4,
    beta_T: float = 2e-2,  # noqa: N803
    T: int = 100,  # noqa: N803
    sigma_tilde_factor: float = 1.0,
) -> NeuralDiffusionModel:
    """
    Build a NeuralDiffusionModel from an already-constructed INR.

    Example
    -------
    >>> inr = INR(coord_dim=2, hidden_dim=20, n_hidden=2, out_dim=1)
    >>> model = build_ndm(inr)
    """
    if f_phi_hidden is None:
        f_phi_hidden = [512, 512, 512]

    data_dim = img_size * img_size  # 784
    weight_dim = inr.num_weights  # e.g. 501 for default INR

    f_phi = WeightEncoder(
        data_dim=data_dim,
        weight_dim=weight_dim,
        hidden_dims=f_phi_hidden,
        t_embed_dim=t_embed_dim // 2,  # lighter embedding for encoder
    )

    network = NoisePredictor(
        weight_dim=weight_dim,
        hidden_dim=noise_hidden,
        n_blocks=noise_blocks,
        t_embed_dim=t_embed_dim,
    )

    return NeuralDiffusionModel(
        network=network,
        F_phi=f_phi,
        inr=inr,
        beta_1=beta_1,
        beta_T=beta_T,
        T=T,
        sigma_tilde_factor=sigma_tilde_factor,
        data_dim=data_dim,
        img_size=img_size,
    )
