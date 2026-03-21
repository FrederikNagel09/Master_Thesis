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
# Diffusion
# ---------------------------------------------------------------------------


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_t=2e-2, t=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()  # noqa: UP008
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_t
        self.T = t

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_t, t), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###

        # Get batch size
        batch_size = x.shape[0]

        # Sample time step for each image in batch and normalize to [0,1]
        t = torch.randint(1, self.T + 1, (batch_size,), device=x.device)
        t_norm = (t - 1) / (self.T - 1)

        # Sample noise epsilon
        epsilon = torch.randn_like(x)

        # Compute noisy x_t
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t - 1]).view(batch_size, 1)

        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alpha_cumprod[t - 1]).view(batch_size, 1)

        x_t = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * epsilon

        # Predict noise using the network
        epsilon_theta = self.network(x_t, t_norm.view(batch_size, 1))

        # Compute loss
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")

        # Take average across batch
        neg_elbo = loss.mean(dim=[1])

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.
        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_T ~ N(0, I)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Denoise from t=T down to t=1
        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling", total=self.T):
            # No noise at the final step (t=0 in 0-indexed = t=1 in paper)
            z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            # Correct 0-indexed lookups (no more [t-1] wrap-around bug)
            sqrt_alpha_t = torch.sqrt(self.alpha[t])
            one_minus_alpha_t = 1 - self.alpha[t]
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_cumprod[t])
            beta_t = self.beta[t]

            # Normalize t to [0, 1] consistently with negative_elbo (t+1 maps 0-idx back to 1-idx)
            t_norm = torch.full((shape[0],), t / (self.T - 1), device=x_t.device)
            pred_noise = self.network(x_t, t_norm.view(shape[0], 1))

            # DDPM reverse step (Algorithm 2, line 4)
            x_t = (1 / sqrt_alpha_t) * (x_t - (one_minus_alpha_t / sqrt_one_minus_alpha_bar) * pred_noise) + torch.sqrt(beta_t) * z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


class Unet(torch.nn.Module):
    """
    A simple U-Net architecture for MNIST that takes an input image and time
    """

    def __init__(self):
        super().__init__()
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                    torch.nn.SiLU(),  # (batch, 8, 28, 28)
                ),
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                    torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                    torch.nn.SiLU(),  # (batch, 16, 14, 14)
                ),
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                    torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                    torch.nn.SiLU(),  # (batch, 32, 7, 7)
                ),
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                    torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                    torch.nn.SiLU(),  # (batch, 64, 4, 4)
                ),
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                    torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                    torch.nn.SiLU(),  # (batch, 64, 2, 2)
                ),
            ]
        )
        self._tconvs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    # input is the output of convs[4]
                    torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                    torch.nn.SiLU(),
                ),
                torch.nn.Sequential(
                    # input is the output from the above sequential concated with the output from convs[3]
                    torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                    torch.nn.SiLU(),
                ),
                torch.nn.Sequential(
                    # input is the output from the above sequential concated with the output from convs[2]
                    torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    torch.nn.SiLU(),
                ),
                torch.nn.Sequential(
                    # input is the output from the above sequential concated with the output from convs[1]
                    torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                    torch.nn.SiLU(),
                ),
                torch.nn.Sequential(
                    # input is the output from the above sequential concated with the output from convs[0]
                    torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
                ),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
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
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal
