"""
DiffusionHyperINR: DDPM diffusion model as hypernetwork for INR on MNIST.

Pipeline:
  Training:
    x_0 (MNIST image) -> add noise -> x_t -> UNet denoiser -> x_0_hat (predicted clean image)
    x_0_hat -> HyperNetwork MLP -> flat INR weights
    coords -> INR (SIREN) with those weights -> pixel predictions
    Loss = MSE(predictions, pixels) + lambda_denoise * MSE(x_0_hat, x_0)

  Inference:
    pure noise -> full DDPM reverse chain -> generated MNIST image
    generated image -> HyperNetwork MLP -> flat INR weights
    coords (any resolution) -> INR -> pixel predictions
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# ---------------------------------------------------------------------------
# INR parameter helpers
# ---------------------------------------------------------------------------


def get_inr_param_shapes(h1: int, h2: int, h3: int) -> list[tuple[tuple, tuple]]:
    """Weight/bias shapes for each layer of the INR: Linear(2->h1->h2->h3->1)."""
    dims = [2, h1, h2, h3, 1]
    return [((dims[i + 1], dims[i]), (dims[i + 1],)) for i in range(len(dims) - 1)]


def count_inr_params(h1: int, h2: int, h3: int) -> int:
    """Total scalar parameters in the INR."""
    return sum(math.prod(ws) + math.prod(bs) for ws, bs in get_inr_param_shapes(h1, h2, h3))


# ---------------------------------------------------------------------------
# SIREN INR — stateless, weights supplied at forward time
# ---------------------------------------------------------------------------


class INRMLP(nn.Module):
    """
    3-layer SIREN MLP. NO trainable parameters — weights are supplied by the
    hypernetwork at runtime.

    Architecture:
        SineLayer(2  -> h1, is_first=True)
        SineLayer(h1 -> h2)
        SineLayer(h2 -> h3)
        Linear(h3 -> 1) + clamp(0,1)
    """

    def __init__(self, h1: int = 20, h2: int = 20, h3: int = 20, omega_0: float = 20.0):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.omega_0 = omega_0
        self.param_shapes = get_inr_param_shapes(h1, h2, h3)

    def _unpack_weights_batched(self, flat: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """flat: (B, P) -> list of (W:(B, out, in), b:(B, out)) per layer."""
        b = flat.shape[0]
        params = []
        offset = 0
        for w_shape, b_shape in self.param_shapes:
            w_size = math.prod(w_shape)
            b_size = math.prod(b_shape)
            w = flat[:, offset : offset + w_size].view(b, *w_shape)
            offset += w_size
            b_vec = flat[:, offset : offset + b_size].view(b, *b_shape)
            offset += b_size
            params.append((w, b_vec))
        return params

    def forward(self, coords: torch.Tensor, flat_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords:       (B, N, 2)
            flat_weights: (B, P)
        Returns:
            (B, N, 1)
        """
        params = self._unpack_weights_batched(flat_weights)
        x = coords

        for w, b in params[:-1]:
            x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
            x = torch.sin(self.omega_0 * x)

        w, b = params[-1]
        x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
        return x.clamp(0, 1)


# ---------------------------------------------------------------------------
# HyperNetwork MLP — image -> flat INR weights
# ---------------------------------------------------------------------------


class HyperNetwork(nn.Module):
    """
    3-layer ReLU MLP: flattened image (784,) -> flat INR weight vector (P,).
    This is the ONLY module with trainable parameters alongside the UNet.
    """

    def __init__(self, inr_param_count: int, image_size: int = 784, hyper_h: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size, hyper_h),
            nn.ReLU(),
            nn.Linear(hyper_h, hyper_h),
            nn.ReLU(),
            nn.Linear(hyper_h, hyper_h),
            nn.ReLU(),
            nn.Linear(hyper_h, inr_param_count),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 784) pixel values in [0, 1]
        Returns:
            (B, P) flat INR weight vectors
        """
        return self.net(image)


# ---------------------------------------------------------------------------
# UNet building blocks (DDPM)
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """
    UNet for DDPM. Configured for MNIST: img_size=32, c_in=1, c_out=1.
    Predicts the clean image x_0 directly (x_0-parameterization) rather
    than predicting noise — this makes the end-to-end gradient path cleaner.
    """

    def __init__(self, img_size=32, c_in=1, c_out=1, time_dim=256, channels=32):
        super().__init__()
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels * 2, emb_dim=time_dim)
        self.sa1 = SelfAttention(channels * 2, img_size // 2)
        self.down2 = Down(channels * 2, channels * 4, emb_dim=time_dim)
        self.sa2 = SelfAttention(channels * 4, img_size // 4)
        self.down3 = Down(channels * 4, channels * 4, emb_dim=time_dim)
        self.sa3 = SelfAttention(channels * 4, img_size // 8)

        self.bot1 = DoubleConv(channels * 4, channels * 8)
        self.bot2 = DoubleConv(channels * 8, channels * 8)
        self.bot3 = DoubleConv(channels * 8, channels * 4)

        self.up1 = Up(channels * 8, channels * 2, emb_dim=time_dim)
        self.sa4 = SelfAttention(channels * 2, img_size // 4)
        self.up2 = Up(channels * 4, channels, emb_dim=time_dim)
        self.sa5 = SelfAttention(channels, img_size // 2)
        self.up3 = Up(channels * 2, channels, emb_dim=time_dim)
        self.sa6 = SelfAttention(channels, img_size)

        self.outc = nn.Conv2d(channels, c_out, kernel_size=1)

    def pos_encoding(self, t, channels, device):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        # Sigmoid to keep output in [0, 1] — compatible with pixel-space images
        return torch.sigmoid(self.outc(x))


# ---------------------------------------------------------------------------
# Diffusion process helpers
# ---------------------------------------------------------------------------


class Diffusion:
    """
    DDPM diffusion process using x_0-parameterization.

    The UNet predicts the clean image x_0_hat directly from (x_t, t).
    This is equivalent to noise prediction but gives a more direct signal
    for the end-to-end INR reconstruction loss.

    img_size=32 (MNIST 28x28 padded to 32x32 for clean power-of-2 downsampling).
    """

    def __init__(
        self,
        T: int = 500,  # noqa: N803
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        img_size: int = 32,
        img_channels: int = 1,
        device: str = "cpu",
    ):
        self.T = T
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Forward process: add noise to x0 at timestep t.
        x0 should be in [0, 1].  We treat [0,1] as the clean space.

        Returns:
            x_t:   noisy image at timestep t
            noise: the noise that was added
        """
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alphas_bar[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return x_t, noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """Convert noise prediction to x_0 prediction."""
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alphas_bar[t]).view(-1, 1, 1, 1)
        return (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha_bar

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)

    @torch.no_grad()
    def p_sample_loop(self, unet: nn.Module, batch_size: int) -> torch.Tensor:
        """
        Full reverse diffusion loop — generates images from pure noise.
        Returns images in [0, 1], shape (B, 1, img_size, img_size).
        """
        logging.info(f"Sampling {batch_size} images via full reverse chain...")
        unet.eval()

        x = torch.randn(batch_size, self.img_channels, self.img_size, self.img_size, device=self.device)

        for i in tqdm(reversed(range(1, self.T)), total=self.T - 1, desc="Sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            alpha = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar = self.alphas_bar[t].view(-1, 1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1, 1)

            # UNet predicts x_0_hat directly (x_0 parameterization)
            x0_hat = unet(x, t).clamp(0, 1)

            # Reconstruct mean of p(x_{t-1} | x_t) using predicted x_0
            # DDPM posterior mean formula derived from x_0 prediction
            sqrt_alpha_bar_prev = torch.sqrt(self.alphas_bar[t - 1]).view(-1, 1, 1, 1) if i > 1 else torch.ones_like(alpha_bar)  # noqa: F841
            alpha_bar_prev = self.alphas_bar[t - 1].view(-1, 1, 1, 1) if i > 1 else torch.ones_like(alpha_bar)

            # Posterior mean
            coef_x0 = (torch.sqrt(alpha_bar_prev) * beta) / (1.0 - alpha_bar)
            coef_xt = (torch.sqrt(alpha) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)
            mean = coef_x0 * x0_hat + coef_xt * x

            if i > 1:
                posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                std = torch.sqrt(posterior_var)
                x = mean + std * torch.randn_like(x)
            else:
                x = mean

        unet.train()
        return x.clamp(0, 1)


# ---------------------------------------------------------------------------
# Full DiffusionHyperINR model
# ---------------------------------------------------------------------------


class DiffusionHyperINR(nn.Module):
    """
    Combined model: UNet (DDPM) + HyperNetwork MLP + SIREN INR.

    Trainable components:
        - unet:      predicts clean MNIST images from noisy inputs
        - hypernet:  maps a clean MNIST image (784,) -> flat INR weights (P,)

    Stateless components:
        - inr:       SIREN MLP, weights supplied at runtime by hypernet
        - diffusion: noise schedule helper (no parameters)

    Training forward pass (end-to-end):
        1. x0 (B,1,32,32) + t -> UNet -> x0_hat (B,1,32,32)   [denoising]
        2. x0_hat -> flatten (B,784) -> HyperNet -> weights (B,P)
        3. coords (B,N,2) -> INR(weights) -> pixel_preds (B,N,1)
        Returns x0_hat and pixel_preds for loss computation.

    Inference:
        Call `sample_and_reconstruct(coords, batch_size)`.
    """

    IMG_SIZE = 32  # UNet operates at 32x32; MNIST 28x28 is padded

    def __init__(
        self,
        h1: int = 32,
        h2: int = 32,
        h3: int = 32,
        omega_0: float = 1.0,
        hyper_h: int = 256,
        unet_channels: int = 32,
        T: int = 500,  # noqa: N803
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()

        # INR (stateless)
        self.inr = INRMLP(h1=h1, h2=h2, h3=h3, omega_0=omega_0)
        inr_param_count = count_inr_params(h1, h2, h3)

        # HyperNetwork: image (784,) -> INR weights (P,)
        # Input is the 32x32=1024 flattened image from UNet output
        self.hypernet = HyperNetwork(
            inr_param_count=inr_param_count,
            image_size=self.IMG_SIZE * self.IMG_SIZE,  # 1024
            hyper_h=hyper_h,
        )

        # UNet denoiser
        self.unet = UNet(
            img_size=self.IMG_SIZE,
            c_in=1,
            c_out=1,
            time_dim=256,
            channels=unet_channels,
        )

        # Diffusion schedule (not an nn.Module, no parameters)
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Diffusion object is device-aware; rebuilt in train/inference via _get_diffusion()
        self._diffusion_cache = {}

        print(f"INR param count       : {inr_param_count}")
        print(f"HyperNetwork params   : {sum(p.numel() for p in self.hypernet.parameters()):,}")
        print(f"UNet params           : {sum(p.numel() for p in self.unet.parameters()):,}")
        print(f"Total trainable params: {sum(p.numel() for p in self.parameters()):,}")

    def _get_diffusion(self, device: str) -> Diffusion:
        """Return (or create) a Diffusion helper for the given device."""
        if device not in self._diffusion_cache:
            self._diffusion_cache[device] = Diffusion(
                T=self.T,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                img_size=self.IMG_SIZE,
                img_channels=1,
                device=device,
            )
        return self._diffusion_cache[device]

    def forward(
        self,
        x0: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end training forward pass.

        Args:
            x0:     (B, 1, 32, 32) clean MNIST images, pixel values in [0, 1]
            coords: (B, N, 2)      pixel coordinates in [-1, 1]
            t:      (B,) integer timesteps; sampled randomly if None

        Returns:
            x0_hat:      (B, 1, 32, 32) — UNet's predicted clean image
            pixel_preds: (B, N, 1)      — INR pixel predictions
            t:           (B,)           — timesteps used (for logging)
        """
        device = x0.device
        diffusion = self._get_diffusion(str(device))

        # 1. Sample timesteps if not provided
        if t is None:
            t = diffusion.sample_timesteps(x0.shape[0])

        # 2. Forward diffusion: add noise
        x_t, _noise = diffusion.q_sample(x0, t)

        # 3. UNet predicts clean image x_0_hat from (x_t, t)
        x0_hat = self.unet(x_t, t)  # (B, 1, 32, 32), sigmoid output in [0,1]

        # 4. Flatten x0_hat -> hypernetwork -> INR weights
        img_flat = x0_hat.view(x0_hat.shape[0], -1)  # (B, 1024)
        flat_weights = self.hypernet(img_flat)  # (B, P)

        # 5. INR: coords -> pixel predictions
        pixel_preds = self.inr(coords, flat_weights)  # (B, N, 1)

        return x0_hat, pixel_preds, t

    @torch.no_grad()
    def sample_and_reconstruct(
        self,
        coords: torch.Tensor,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference: sample new images from noise and reconstruct at given coordinates.

        Args:
            coords:     (N, 2) or (B, N, 2) coordinate grid in [-1, 1]
            batch_size: number of images to generate
            device:     device string

        Returns:
            generated_images: (B, 1, 32, 32) generated MNIST images in [0, 1]
            pixel_preds:      (B, N, 1) INR pixel predictions
        """
        self.eval()
        diffusion = self._get_diffusion(device)

        # 1. Full reverse diffusion chain -> generated images
        generated = diffusion.p_sample_loop(self.unet, batch_size)  # (B, 1, 32, 32)

        # 2. Flatten -> hypernetwork -> INR weights
        img_flat = generated.view(batch_size, -1)  # (B, 1024)
        flat_weights = self.hypernet(img_flat)  # (B, P)

        # 3. INR reconstruction at given coordinates
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 2)
        coords = coords.to(device)

        pixel_preds = self.inr(coords, flat_weights)  # (B, N, 1)

        return generated, pixel_preds
