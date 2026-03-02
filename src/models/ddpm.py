"""
DDPM Model: UNet + Diffusion classes adapted for MNIST (1-channel, 28x28 → resized to 32x32)
"""

import logging

import torch
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


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
