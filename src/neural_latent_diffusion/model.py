"""
NDM Latent Diffusion Model
==========================
Architecture overview
---------------------
1. **VAE** - encodes 1x32x32 MNIST images into a latent of shape
   (latent_channels x latent_size x latent_size).  Default: 4x4x4.
   The encoder/decoder are trained jointly with the diffusion loss via
   a weighted KL + reconstruction term, or you can pre-train the VAE
   separately and freeze it.

2. **NDM Transform F_φ** - a small time-conditioned MLP that maps a
   flattened latent + sinusoidal time embedding → transformed latent.
   This is the F_φ(x, t) from the NDM paper (Bartosh et al., 2024).
   The forward noising process operates on F_φ(z, t) instead of z,
   so the UNet learns to predict F_φ(ẑ, t) rather than ẑ directly.

3. **UNet** - same architecture as before but now operates on the
   (latent_channels x latent_size x latent_size) space.

4. **LatentNDMDiffusion** - handles
   - q_sample  : z → noised latent using F_φ
   - p_sample  : one reverse step
   - p_sample_loop : full generation loop (in latent space → decode → pixel)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (shared utility)
# ---------------------------------------------------------------------------


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t : (B,) long or float tensor of timesteps
    returns: (B, dim) sinusoidal embedding
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# VAE  (lightweight convolutional)
# ---------------------------------------------------------------------------


class VAEEncoder(nn.Module):
    """1x32x32  →  (latent_channels, latent_size, latent_size)"""

    def __init__(self, in_channels: int = 1, latent_channels: int = 4, latent_size: int = 4):  # noqa: ARG002
        super().__init__()
        # Encoder: progressively downsample
        # 32 → 16 → 8 → latent_size
        layers = []
        ch = in_channels
        out_channels_list = [32, 64, 128]
        for out_ch in out_channels_list:
            layers += [
                nn.Conv2d(ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            ]
            ch = out_ch
        # Final spatial reduction to latent_size
        # After 3 stride-2 convs: 32 → 4  (matches latent_size=4)
        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Conv2d(ch, latent_channels, 1)
        self.logvar_head = nn.Conv2d(ch, latent_channels, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)


class VAEDecoder(nn.Module):
    """(latent_channels, latent_size, latent_size)  →  1x32x32"""

    def __init__(self, out_channels: int = 1, latent_channels: int = 4, latent_size: int = 4):  # noqa: ARG002
        super().__init__()
        layers = []
        channel_list = [128, 64, 32]
        ch = latent_channels
        for out_ch in channel_list:
            layers += [
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            ]
            ch = out_ch
        self.decoder = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        return torch.tanh(self.out_conv(self.decoder(z)))


class VAE(nn.Module):
    def __init__(self, img_channels: int = 1, latent_channels: int = 4, latent_size: int = 4):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.encoder = VAEEncoder(img_channels, latent_channels, latent_size)
        self.decoder = VAEDecoder(img_channels, latent_channels, latent_size)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------------------------------------------------------
# NDM Transform  F_φ(z, t)
# ---------------------------------------------------------------------------


class NDMTransform(nn.Module):
    """
    Time-conditioned MLP that transforms a latent vector before noising.

    F_φ : R^(CxHxW) x [0,T]  →  R^(CxHxW)

    We flatten the spatial dims, apply a residual MLP with time conditioning,
    then reshape back.  The identity initialisation ensures F_φ ≈ identity
    at the start of training (important for stable warm-up).
    """

    def __init__(self, latent_dim: int, time_dim: int = 128, hidden_dim: int = 256, num_layers: int = 3):
        """
        latent_dim : C x H x W  (total flattened latent size)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Residual blocks
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection - initialise near zero so F_φ ≈ identity early
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z : (B, C, H, W) latent
        t : (B,) integer timesteps
        returns: (B, C, H, W) transformed latent
        """
        beta = z.shape[0]
        z_flat = z.view(beta, -1)  # (B, latent_dim)

        t_emb = sinusoidal_embedding(t, self.time_dim)  # (B, time_dim)
        t_emb = self.time_embed(t_emb)  # (B, hidden_dim)

        h = self.input_proj(z_flat) + t_emb  # (B, hidden_dim)
        for layer in self.layers:
            h = h + layer(h)  # residual

        delta = self.output_proj(h)  # (B, latent_dim)
        # Residual connection: F_φ(z, t) = z + delta
        out = z_flat + delta
        return out.view(beta, *z.shape[1:])


# ---------------------------------------------------------------------------
# UNet building blocks  (same as before, now operating on latent space)
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
# UNet  (now operates on latent_channels x latent_size x latent_size)
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """
    UNet for latent NDM.
    Default: img_size=4 (latent), c_in=c_out=latent_channels=4
    """

    def __init__(self, img_size: int = 4, c_in: int = 4, c_out: int = 4, time_dim: int = 256, device: str = "cpu", channels: int = 64):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels * 2, emb_dim=time_dim)
        self.sa1 = SelfAttention(channels * 2, img_size // 2)
        self.down2 = Down(channels * 2, channels * 4, emb_dim=time_dim)
        self.sa2 = SelfAttention(channels * 4, img_size // 4)

        self.bot1 = DoubleConv(channels * 4, channels * 8)
        self.bot2 = DoubleConv(channels * 8, channels * 4)

        self.up1 = Up(channels * 8, channels * 2, emb_dim=time_dim)
        self.sa3 = SelfAttention(channels * 2, img_size // 2)
        self.up2 = Up(channels * 4, channels, emb_dim=time_dim)
        self.sa4 = SelfAttention(channels, img_size)

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

        x3 = self.bot1(x3)
        x3 = self.bot2(x3)

        x = self.up1(x3, x2, t)
        x = self.sa3(x)
        x = self.up2(x, x1, t)
        x = self.sa4(x)

        return self.outc(x)


# ---------------------------------------------------------------------------
# Latent NDM Diffusion
# ---------------------------------------------------------------------------


class LatentNDMDiffusion:
    """
    Neural Diffusion Model operating in VAE latent space.

    Key difference from standard DDPM
    -----------------------------------
    The forward process noises F_φ(z, t) rather than z itself:

        q_φ(z_t | z) = N(z_t ; alpha_t · F_φ(z, t),  sigma_t² · I)

    The UNet therefore predicts  F_φ(ẑ, t)  (the transformed latent),
    and the training target is also F_φ(z, t).

    Per the NDM paper (eq. 9), the MSE loss is:

        L = || F_φ(z, t) - UNet(z_t, t) ||²

    which collapses to the standard DDPM noise-prediction loss when
    F_φ is the identity.

    Parameters
    ----------
    T            : total diffusion steps
    beta_start   : starting β value
    beta_end     : ending β value
    latent_shape : (C, H, W) of the latent space
    device       : torch device string
    """

    def __init__(
        self,
        T: int = 500,  # noqa: N803
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        latent_shape: tuple = (4, 4, 4),
        device: str = "cpu",
    ):
        self.T = T
        self.latent_shape = latent_shape
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    # ------------------------------------------------------------------
    # Forward process  (NDM eq. 6)
    # ------------------------------------------------------------------

    def q_sample(self, z: torch.Tensor, t: torch.Tensor, transform: NDMTransform):
        """
        Sample z_t ~ q_φ(z_t | z) = N(alpha_t·F_φ(z,t), sigma_t²·I)

        Returns
        -------
        z_t       : noised latent
        Fz        : F_φ(z, t)   — the training target for the UNet
        noise     : ε sampled
        """
        Fz = transform(z, t)  # (B, C, H, W) — transformed latent  # noqa: N806

        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1 - self.alphas_bar[t]).view(-1, 1, 1, 1)

        noise = torch.randn_like(Fz)
        z_t = sqrt_alpha_bar * Fz + sqrt_one_minus * noise
        return z_t, Fz, noise

    # ------------------------------------------------------------------
    # Reverse process  (predict F_φ(ẑ,t), recover ẑ, then step)
    # ------------------------------------------------------------------

    def p_sample(
        self,
        unet: UNet,
        transform: NDMTransform,  # noqa: ARG002
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        One reverse step: estimate z_{t-1} from z_t.

        The UNet predicts F_φ(ẑ, t).  We use this as a proxy for the
        de-noised transformed latent, then reconstruct the DDPM-style mean.
        """
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar = self.alphas_bar[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)

        # UNet predicts F_φ(ẑ, t)  [same role as predicted noise in DDPM]
        pred_Fz = unet(z_t, t)  # noqa: N806

        # DDPM mean formula using the transformed target
        mean = (1.0 / torch.sqrt(alpha)) * (z_t - (beta / torch.sqrt(1 - alpha_bar)) * pred_Fz)
        std = torch.sqrt(beta)
        noise = torch.randn_like(z_t) if t[0] > 1 else torch.zeros_like(z_t)
        return mean + std * noise

    # ------------------------------------------------------------------
    # Full generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_loop(
        self,
        unet: UNet,
        transform: NDMTransform,
        vae: VAE,
        batch_size: int,
        timesteps_to_save=None,
    ):
        """
        Full reverse chain: pure noise → latent → decode → pixel images.

        Returns uint8 images of shape (B, 1, H, W).
        """
        logging.info(f"Sampling {batch_size} new images (latent NDM)…")
        unet.eval()
        transform.eval()
        vae.eval()

        intermediates = []
        c, h, w = self.latent_shape
        z = torch.randn((batch_size, c, h, w), device=self.device)

        for i in tqdm(reversed(range(1, self.T)), total=self.T - 1, position=0):
            t = (torch.ones(batch_size, device=self.device) * i).long()
            z = self.p_sample(unet, transform, z, t)
            if timesteps_to_save is not None and i in timesteps_to_save:
                imgs = self._decode_to_uint8(vae, z)
                intermediates.append(imgs)

        # Final decode
        imgs = self._decode_to_uint8(vae, z)

        unet.train()
        transform.train()
        vae.train()

        if timesteps_to_save is not None:
            intermediates.append(imgs)
            return imgs, intermediates
        return imgs

    def _decode_to_uint8(self, vae: VAE, z: torch.Tensor) -> torch.Tensor:
        x = vae.decode(z)  # (B, 1, H, W) in [-1, 1]
        x = (x.clamp(-1, 1) + 1) / 2  # [0, 1]
        return (x * 255).to(torch.uint8)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)
