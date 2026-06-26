from torch import nn
import torch
from src.models.latent_diffusion.modules.trans_inr import make_coord_grid


class VAEWrapper(nn.Module):
    """
    Wraps encoder + TransInr decoder into a VAE.
    Supports both 2D image and 3D voxel data via is_3d flag.
    Encoder and decoder can be extracted individually after training.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        img_size: int,
        device: torch.device,
        is_3d: bool = False,
    ) -> None:
        super().__init__()
        self.latent_encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.device = device
        self.is_3d = is_3d
        data_shape = (img_size, img_size, img_size) if is_3d else (img_size, img_size)
        coord_grid = make_coord_grid(data_shape, (-1, 1))
        self.register_buffer("coord_grid", coord_grid)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]  # noqa: N806
        if self.is_3d:
            coords = self.coord_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1).to(self.device)
        else:
            coords = self.coord_grid.unsqueeze(0).repeat(B, 1, 1, 1).to(self.device)
        return self.decoder(z, coords)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass: encode → reparameterise → decode.

        Args:
            x (torch.Tensor): input (B, C, H, W) or (B, C, D, H, W)
        Returns:
            tuple: (x_recon, mu, logvar) each (B, ...)
        """
        mu, logvar = self.latent_encoder(x)
        z = self.latent_encoder.reparameterize(mu, logvar)
        x_recon = self._decode_latent(z)
        return x_recon, mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# KL ANNEALING
# ──────────────────────────────────────────────────────────────────────────────


def _get_beta(
    global_step: int,
    beta_final: float,
    warmup_steps: int,
    burnin_steps: int = 0,
) -> float:
    """
    Linear KL warmup with optional burn-in period.

    Args:
        global_step  (int):   current training step
        beta_final   (float): target beta
        warmup_steps (int):   steps to ramp from 0 → beta_final after burnin
        burnin_steps (int):   steps to hold at 0 before ramping
    Returns:
        float: current beta
    """
    if global_step < burnin_steps:
        return 0.0
    return beta_final * min(1.0, (global_step - burnin_steps) / warmup_steps)
