from torch import nn
import torch
from src.models.latent_diffusion.modules.trans_inr import make_coord_grid


class TwoStageLDM(nn.Module):
    """
    Two-stage LDM: frozen VAE encoder/decoder + trainable noise predictor.
    Supports both 2D image and 3D voxel data via is_3d flag.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        noise_predictor: nn.Module,
        img_size: int,
        latent_dim: int,
        latent_size: tuple[int, int],
        T: int,  # noqa: N803
        beta_1: float,
        beta_T: float,  # noqa: N803
        device: torch.device,
        is_3d: bool = False,
    ) -> None:
        super().__init__()
        self.latent_encoder = encoder
        self.decoder = decoder
        self.noise_predictor = noise_predictor
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.T = T
        self.device = device
        self.is_3d = is_3d

        # Freeze encoder and decoder
        for p in self.latent_encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        # Noise schedule buffers
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq", 1.0 - alpha_cumprod)
        self.register_buffer("sigma", (1.0 - alpha_cumprod).sqrt())

        data_shape = (img_size, img_size, img_size) if is_3d else (img_size, img_size)
        coord_grid = make_coord_grid(data_shape, (-1, 1))
        self.register_buffer("coord_grid", coord_grid, persistent=False)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]  # noqa: N806
        if self.is_3d:
            coords = self.coord_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1).to(self.device)
        else:
            coords = self.coord_grid.unsqueeze(0).repeat(B, 1, 1, 1).to(self.device)
        return self.decoder(z, coords)

    @torch.no_grad()
    def compute_rec_loss(self, val_loader) -> float:
        self.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            x = batch[0].to(self.device)
            B = x.shape[0]

            # Force shape correction directly on 'x' to prevent flat vectors
            if len(x.shape) == 2:
                if self.is_3d:
                    x = x.view(B, -1, self.img_size, self.img_size, self.img_size)
                else:
                    x = x.view(B, -1, self.img_size, self.img_size)

            # 1. Encode
            encoded = self.latent_encoder(x)
            
            if hasattr(encoded, "sample"):
                z = encoded.sample()
            elif isinstance(encoded, tuple):
                z = encoded[0]
            else:
                z = encoded

            # 2. Decode
            x_reconstructed = self._decode_latent(z)

            # 3. MSE Loss
            loss = torch.nn.functional.mse_loss(x_reconstructed.view_as(x), x)
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return 0.0

        return total_loss / num_batches

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean latent at timestep t.

        Args:
            z0    (torch.Tensor): clean latent (B, latent_dim, H, W)
            t     (torch.Tensor): timestep indices (B,) in [0, T-1]
            noise (torch.Tensor): noise tensor same shape as z0
        Returns:
            torch.Tensor: noisy latent z_t, same shape as z0
        """
        sqrt_ac = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sigma = self.sigma[t].view(-1, 1, 1, 1)
        return sqrt_ac * z0 + sigma * noise

    @torch.no_grad()
    def p_sample_loop(self, n_samples: int) -> torch.Tensor:
        """
        Full reverse diffusion chain to generate outputs.

        Args:
            n_samples (int): number of samples to generate
        Returns:
            torch.Tensor: generated outputs (n_samples, C, ...) in model output range
        """
        H, W = self.latent_size  # noqa: N806
        z = torch.randn(n_samples, self.latent_dim, H, W, device=self.device)
        for t_idx in reversed(range(self.T)):
            t_tensor = torch.full(
                (n_samples, 1),
                t_idx / (self.T - 1),
                device=self.device,
                dtype=torch.float32,
            )
            t_int = torch.full(
                (n_samples,), t_idx, device=self.device, dtype=torch.long
            )
            eps_pred = self.noise_predictor(z, t_tensor)
            alpha_t = self.alpha[t_int].view(-1, 1, 1, 1)
            beta_t = self.beta[t_int].view(-1, 1, 1, 1)
            sigma_t = self.sigma[t_int].view(-1, 1, 1, 1)
            z = (1.0 / alpha_t.sqrt()) * (z - (beta_t / sigma_t) * eps_pred)
            if t_idx > 0:
                z = z + beta_t.sqrt() * torch.randn_like(z)
        return self._decode_latent(z)
