"""
DDPM Model: UNet + Diffusion classes adapted for MNIST (1-channel, 28x28 → resized to 32x32)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# ---------------------------------------------------------------------------
# UNet building blocks
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
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

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
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

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
    UNet for DDPM.
    For MNIST we use: img_size=32, c_in=1, c_out=1
    """

    def __init__(self, img_size=32, c_in=1, c_out=1, time_dim=256, device="cpu", channels=32):
        super().__init__()
        self.device = device
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

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

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

        return self.outc(x)


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------


class Diffusion:
    """
    DDPM diffusion process.
    img_size should match the UNet (default 32 for MNIST after resize).
    img_channels=1 for MNIST.
    """

    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, img_size=32, img_channels=1, device="cpu"):  # noqa: N803
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x, t):
        """Forward process: add noise to x at timestep t."""
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise

    def p_sample(self, model, x_t, t):
        """One reverse step: sample x_{t-1} from x_t."""
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar = self.alphas_bar[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)

        predicted_noise = model(x_t, t)
        mean = (1 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise)
        std = torch.sqrt(beta)
        noise = torch.randn_like(x_t) if t[0] > 1 else torch.zeros_like(x_t)
        return mean + std * noise

    @torch.no_grad()
    def p_sample_loop(self, model, batch_size, timesteps_to_save=None):
        """Full reverse loop: generate images from pure noise (Algorithm 2)."""
        logging.info(f"Sampling {batch_size} new images...")
        model.eval()
        intermediates = []

        x = torch.randn((batch_size, self.img_channels, self.img_size, self.img_size), device=self.device)
        for i in tqdm(reversed(range(1, self.T)), total=self.T - 1, position=0):
            t = (torch.ones(batch_size) * i).long().to(self.device)
            x = self.p_sample(model, x, t)
            if timesteps_to_save is not None and i in timesteps_to_save:
                intermediates.append(self._to_uint8(x))

        model.train()
        x = self._to_uint8(x)
        if timesteps_to_save is not None:
            intermediates.append(x)
            return x, intermediates
        return x

    def sample_timesteps(self, batch_size):
        return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)

    @staticmethod
    def _to_uint8(x):
        x = (x.clamp(-1, 1) + 1) / 2
        return (x * 255).type(torch.uint8)
