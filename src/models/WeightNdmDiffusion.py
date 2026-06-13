from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm
import random

if TYPE_CHECKING:
    import numpy as np

from src.models.WeightDiffusion import WeightDiffusion
from src.configs.general_config import GLOBAL_DEBUG_BOOL, probability_threshold


class WeightNDMDiffusion(WeightDiffusion):
    """
    WeightDiffusion with an NDM-style weight-space transformation F_phi(theta, t).

    Extends WeightDiffusion by applying a learnable transformation to theta_prime
    before the forward diffusion process:
        z_t = alpha_t * F_phi(theta_prime, t) + sigma_t * epsilon

    F_phi is constrained to be identity at t=0:
        F_phi(theta, 0) = theta  exactly

    Gradient flow:
        encoder  <--  _l_rec  only        (theta_prime detached before F_phi)
        F_phi    <--  _l_diff only
        denoiser <--  _l_diff only
    """

    def __init__(
        self,
        NoisePredictor: nn.Module,  # noqa: N803
        WeightEncoder: nn.Module,  # noqa: N803
        F_phi: nn.Module,  # noqa: N803
        coord_grid: torch.Tensor,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,  # noqa: N803
        T: int = 1000,  # noqa: N803
        sigma_tilde_factor: float = 1.0,
        data_dim: int = 784,
        img_size: int = 28,
        normalize: bool = True,
        lambda_kl: float = 5e-3,
        probablistic: bool = False,
        stop_gradient_flow: bool = True,
    ):
        super().__init__(
            NoisePredictor=NoisePredictor,
            WeightEncoder=WeightEncoder,
            coord_grid=coord_grid,
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            sigma_tilde_factor=sigma_tilde_factor,
            data_dim=data_dim,
            img_size=img_size,
            normalize=normalize,
            lambda_kl=lambda_kl,
            probablistic=probablistic,
            stop_gradient_flow=stop_gradient_flow,
        )
        self.F_phi = F_phi

    # -------------------------------------------------------------------------
    # Override: forward process uses F_phi
    # -------------------------------------------------------------------------
    def _construct_theta_t(
        self,
        theta_prime: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs z_t using F_phi(theta_prime, t) as the clean signal.

        Args:
            theta_prime: (B, modulation_dim) clean (normalized, detached) weights
            t_idx:       (B,) integer timestep indices
        Returns:
            z_t:     (B, modulation_dim) noisy weights
            epsilon: (B, modulation_dim) noise sample
        """
        t_norm = (t_idx.float() / (self.T - 1)).unsqueeze(1)  # (B, 1)
        Fx = self.F_phi(theta_prime, t_norm)  # noqa: N806

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("==================== DEBUG: Forward Process with F_phi ====================")
            print(f"F_phi stats: mean={Fx.mean():.4f}, std={Fx.std():.4f}, min={Fx.min():.4f}, max={Fx.max():.4f}")
            print("===========================================================================")
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(theta_prime)

        z_t = alpha_t * Fx + sigma_t * epsilon
        return z_t, epsilon

    # -------------------------------------------------------------------------
    # Override: v-prediction target uses F_phi(theta_prime, t) as clean signal
    # -------------------------------------------------------------------------
    def _l_diff(
        self,
        theta_t: torch.Tensor,
        t_norm: torch.Tensor,
        epsilon: torch.Tensor,
        x0: torch.Tensor,
        t_idx: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        V-prediction loss where the clean signal is F_phi(x0, t) not x0.

        Args:
            theta_t: (B, modulation_dim) noisy weights at timestep t
            t_norm:  (B,) timestep normalised to [0, 1]
            epsilon: (B, modulation_dim) noise used to corrupt F_phi(x0, t)
            x0:      (B, modulation_dim) clean weights (theta_prime)
            t_idx:   (B,) integer timestep indices
        Returns:
            (B,) per-sample MSE loss
        """
        v_hat = self.denoiser(theta_t, t_norm.unsqueeze(1))

        sqrt_ab = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sqrt_1mab = self.sigma[t_idx].unsqueeze(1)

        # Clean signal for v-target is F_phi(x0, t), not x0
        t_norm_col = t_norm.unsqueeze(1)  # (B, 1)
        Fx0 = self.F_phi(x0, t_norm_col)  # noqa: N806

        v_target = sqrt_ab * epsilon - sqrt_1mab * Fx0

        return F.mse_loss(v_hat, v_target, reduction="none").mean(dim=-1)  # (B,)

    # -------------------------------------------------------------------------
    # Override: reverse diffusion recovers theta via F_phi identity at t=0
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def sample_weight(
        self,
        n_samples: int = 1,
        collect_snapshots: bool = False,
        debug: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, np.ndarray]]:
        """
        Reverse diffusion with v-prediction, using F_phi in the forward model.

        Args:
            n_samples:         Number of weight samples to generate.
            collect_snapshots: If True, return weight distributions at T_VALUES too.
        Returns:
            curr_theta: (n_samples, modulation_dim) sampled weights
            snapshots:  {t_value: np.ndarray} only if collect_snapshots=True
        """
        weight_dim = self.weight_encoder.modulation_dim
        device = self.sqrt_alpha_cumprod.device
        T_values = {self.T - 1, 3 * self.T // 4, self.T // 2, self.T // 4, 0}  # noqa: N806
        snapshots: dict[int, np.ndarray] = {}

        curr_theta = torch.randn(n_samples, weight_dim, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling", total=self.T):
            t_norm = torch.full((n_samples, 1), t / (self.T - 1), device=device)

            v_hat = self.denoiser(curr_theta, t_norm)

            sqrt_ab = self.sqrt_alpha_cumprod[t]
            sqrt_1mab = self.sigma[t]

            # 1. Recover the transformed signal F_phi(x0, t), NOT x0
            Fx0_hat = sqrt_ab * curr_theta - sqrt_1mab * v_hat
            
            # 2. Recover the noise directly
            eps_hat = sqrt_1mab * curr_theta + sqrt_ab * v_hat  

            if t == 0:
                # Assuming your identity constraint F_phi(theta, 0) == theta holds perfectly,
                # Fx0_hat at t=0 is effectively x0.
                curr_theta = Fx0_hat
                if collect_snapshots and t in T_values:
                    snapshots[t] = curr_theta.detach().cpu().numpy().flatten()
                break

            # DO NOT re-apply self.F_phi here. Fx0_hat is already transformed.

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]

            # DDPM posterior mean using F_phi(x0_hat, t) as the clean signal
            coeff1 = 1.0 / torch.sqrt(alpha_t)
            coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

            # Use the eps_hat we got for free from v-prediction inversion
            mean = coeff1 * (curr_theta - coeff2 * eps_hat)

            curr_theta = mean + torch.sqrt(beta_t) * torch.randn_like(curr_theta) if t > 0 else mean

            if collect_snapshots and t in T_values:
                snapshots[t] = curr_theta.detach().cpu().numpy().flatten()

        if self.normalize:
            curr_theta = self.scaler(curr_theta, reverse=True, training=False)

        if collect_snapshots:
            return curr_theta, snapshots
        return curr_theta
