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
    WeightDiffusion with NDM-style F_phi in both forward process and loss.

    Forward process: z_t = alpha_t * F_phi(theta_prime, t) + sigma_t * epsilon
    Loss:   NDM ELBO — epsilon prediction + posterior mean matching in F_phi-space.
    Sample: epsilon → x0_hat → F_phi → NDM posterior mean.

    NOTE: denoiser is trained as epsilon predictor (not v-predictor as in WeightDiffusion).

    Gradient routing (stop_gradient_flow=True):
        encoder  <-- _l_rec only   (theta_prime detached before _construct_theta_t)
        F_phi    <-- _l_diff only
        denoiser <-- _l_diff only
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
    # Override: forward process uses F_phi(theta_prime, t) as clean signal
    # -------------------------------------------------------------------------
    def _construct_theta_t(
        self,
        theta_prime: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_t = alpha_t * F_phi(theta_prime, t) + sigma_t * epsilon.

        Args:
            theta_prime: (B, modulation_dim) clean weights
            t_idx:       (B,) integer timestep indices
        Returns:
            z_t:     (B, modulation_dim) noisy weights
            epsilon: (B, modulation_dim) noise sample
        """
        t_norm = (t_idx.float() / (self.T - 1)).unsqueeze(1)  # (B, 1)
        Fx = self.F_phi(theta_prime, t_norm)  # noqa: N806
        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)
        epsilon = torch.randn_like(theta_prime)
        return alpha_t * Fx + sigma_t * epsilon, epsilon


    def negative_elbo(self, x: torch.Tensor, lambda_kl: float = 1.0) -> torch.Tensor:
        batch_size = x.shape[0]
        t_idx  = torch.randint(1, self.T, (batch_size,), device=x.device)
        t_norm = t_idx.float() / (self.T - 1)

        if self.probablistic:
            mean, logvar = self.weight_encoder(x)
            theta_prime_raw = self.weight_encoder._reparameterize(mean, logvar)
        else:
            theta_prime_raw = self.weight_encoder(x)

        theta_prime = self.scaler(theta_prime_raw, reverse=False) if self.normalize else theta_prime_raw
        theta_prime = theta_prime.detach() if self.stop_gradient_flow else theta_prime

        theta_t, epsilon = self._construct_theta_t(theta_prime, t_idx)

        l_diff = self._l_diff(theta_t, t_norm, epsilon, theta_prime, t_idx)

        theta  = self.weight_encoder.decode_modulations(theta_prime_raw)
        l_rec  = self._l_rec(x, theta)

        if self.probablistic:
            l_prior = self._l_entropy(logvar)
            elbo = l_diff + l_rec - lambda_kl * l_prior
        else:
            l_prior = torch.zeros_like(l_diff)
            elbo = l_diff + l_rec

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("############### ELBO DEBUG ###############")
            print("theta_prime_raw stats:", theta_prime_raw.mean().item(), theta_prime_raw.std().item())
            print("theta_prime stats:    ", theta_prime.mean().item(), theta_prime.std().item())
            print("theta_t stats:        ", theta_t.mean().item(), theta_t.std().item())
            print("theta stats:          ", theta.mean().item(), theta.std().item())
            print("###########################################")


        return elbo.mean(), l_diff.mean(), l_prior.mean(), l_rec.mean()

    # -------------------------------------------------------------------------
    # Override: NDM ELBO loss — epsilon prediction + posterior mean matching
    # -------------------------------------------------------------------------
    def _l_diff(
        self,
        theta_t: torch.Tensor,
        t_norm: torch.Tensor,
        epsilon: torch.Tensor,  # unused here; kept for parent interface compatibility
        x0: torch.Tensor,
        t_idx: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        NDM ELBO diffusion loss. Denoiser predicts epsilon; loss measures how
        well the predicted posterior mean matches the true one in F_phi-space.

        Args:
            theta_t: (B, D) noisy weights z_t
            t_norm:  (B,) normalized timestep in [0, 1]
            epsilon: (B, D) unused — kept for parent-class interface compatibility
            x0:      (B, D) clean weights (theta_prime, detached when stop_gradient_flow=True)
            t_idx:   (B,) integer timestep indices
        Returns:
            (B,) per-sample loss
        """
        # Denoiser acts as epsilon predictor (not v-predictor as in parent class)
        eps_hat = self.denoiser(theta_t, t_norm.unsqueeze(1))

        alpha_t = self.sqrt_alpha_cumprod[t_idx].unsqueeze(1)  # (B, 1)
        sigma_t = self.sigma[t_idx].unsqueeze(1)               # (B, 1)

        # Recover clean signal estimate from epsilon prediction
        x0_hat = (theta_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        # Previous timestep indices/norms
        s_idx = (t_idx - 1).clamp(min=0)
        s_norm = s_idx.float() / (self.T - 1)  # (B,)

        # Apply F_phi to all four needed (signal, time) pairs
        # Gradients: Fx0_hat_* flows through F_phi + denoiser; Fx0_* flows through F_phi only
        Fx0_hat_t = self.F_phi(x0_hat, t_norm.unsqueeze(1))   # F_phi(x0_hat, t)
        Fx0_hat_s = self.F_phi(x0_hat, s_norm.unsqueeze(1))   # F_phi(x0_hat, s)
        Fx0_s     = self.F_phi(x0, s_norm.unsqueeze(1))       # F_phi(x0, s) — ground truth  # noqa: E241
        Fx0_t     = self.F_phi(x0, t_norm.unsqueeze(1))       # F_phi(x0, t) — ground truth  # noqa: E241

        alpha_s        = self.sqrt_alpha_cumprod[s_idx].unsqueeze(1)       # (B, 1)  # noqa: E241
        sigma_tilde_sq = self._sigma_tilde_sq(s_idx, t_idx).unsqueeze(1)  # (B, 1)

        # sqrt(sigma_s^2 - sigma_tilde^2) / sigma_t
        coeff = (self.sigma_sq[s_idx].unsqueeze(1) - sigma_tilde_sq).clamp(min=0).sqrt()
        coeff = coeff / sigma_t.clamp(min=1e-6)

        # Gap between true and predicted posterior mean in F_phi-transformed space
        diff = alpha_s * (Fx0_s - Fx0_hat_s) + coeff * alpha_t * (Fx0_hat_t - Fx0_t)

        if GLOBAL_DEBUG_BOOL and random.random() < probability_threshold:
            print("############### L_DIFF DEBUG ###############")
            print("t_idx stats:          ", t_idx.float().mean().item(), t_idx.float().std().item())
            print("x0_hat stats:         ", x0_hat.mean().item(), x0_hat.std().item())
            print("Fx0_hat_t stats:      ", Fx0_hat_t.mean().item(), Fx0_hat_t.std().item())
            print("Fx0_hat_s stats:      ", Fx0_hat_s.mean().item(), Fx0_hat_s.std().item())
            print("Fx0_s stats:          ", Fx0_s.mean().item(), Fx0_s.std().item())
            print("Fx0_t stats:          ", Fx0_t.mean().item(), Fx0_t.std().item())
            print("diff stats:           ", diff.mean().item(), diff.std().item())
            print("###########################################")

        return (diff ** 2).sum(dim=-1) / (2.0 * sigma_tilde_sq.squeeze(1).clamp(min=1e-8))

    # -------------------------------------------------------------------------
    # Override: NDM reverse diffusion — eps → x0_hat → F_phi → posterior mean
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def sample_weight(
        self,
        n_samples: int = 1,
        collect_snapshots: bool = False,
        debug: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, np.ndarray]]:
        """
        NDM reverse diffusion: epsilon prediction → x0_hat → NDM posterior mean.

        Args:
            n_samples:         Number of weight samples to generate.
            collect_snapshots: If True, also return weight distributions at T_VALUES.
        Returns:
            z_t:       (n_samples, modulation_dim) sampled weights
            snapshots: {t_value: np.ndarray} only if collect_snapshots=True
        """
        weight_dim = self.weight_encoder.modulation_dim
        device = self.sqrt_alpha_cumprod.device
        T_values = {self.T - 1, 3 * self.T // 4, self.T // 2, self.T // 4, 0}  # noqa: N806
        snapshots: dict[int, np.ndarray] = {}
        T_minus_1 = max(self.T - 1, 1)  # noqa: N806

        z_t = torch.randn(n_samples, weight_dim, device=device)

        for t in tqdm(range(self.T - 1, -1, -1), desc="NDM Sampling", total=self.T):
            t_norm = torch.full((n_samples, 1), t / T_minus_1, device=device)

            # Predict epsilon, recover clean estimate
            eps_hat = self.denoiser(z_t, t_norm)
            alpha_t = self.sqrt_alpha_cumprod[t]
            sigma_t = self.sigma[t]
            x0_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                # F_phi identity at t=0 → x0_hat is the final sample
                z_t = x0_hat
                if collect_snapshots and t in T_values:
                    snapshots[t] = z_t.detach().cpu().numpy().flatten()
                break

            s = t - 1
            s_norm = torch.full((n_samples, 1), s / T_minus_1, device=device)

            # Batch F_phi(x0_hat, s) and F_phi(x0_hat, t) into one forward pass
            Fx_2x = self.F_phi(
                torch.cat([x0_hat, x0_hat], dim=0),
                torch.cat([s_norm, t_norm], dim=0),
            )  # (2N, D)
            Fx0_hat_s, Fx0_hat_t = Fx_2x.chunk(2, dim=0)

            sigma_tilde_sq = self._sigma_tilde_sq(
                torch.tensor([s], device=device),
                torch.tensor([t], device=device),
            )[0]

            alpha_s = self.sqrt_alpha_cumprod[s]
            coeff = (self.sigma_sq[s] - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t.clamp(min=1e-6)

            # NDM posterior mean: alpha_s * F_phi(x0_hat, s) + coeff * (z_t - alpha_t * F_phi(x0_hat, t))
            mu = alpha_s * Fx0_hat_s + coeff * (z_t - alpha_t * Fx0_hat_t)
            noise = sigma_tilde_sq.clamp(min=0).sqrt() * torch.randn_like(z_t)
            z_t = mu + noise if sigma_tilde_sq.item() > 0 else mu

            if collect_snapshots and t in T_values:
                snapshots[t] = z_t.detach().cpu().numpy().flatten()
            
            if (t % 100 == 0 and GLOBAL_DEBUG_BOOL and debug) or (t == 0 and GLOBAL_DEBUG_BOOL and debug):
                print("################## Sampling: ##############################")
                print(f"Sampling step {t}/{self.T}:")
                print("x0_hat stats:         ", x0_hat.mean().item(), x0_hat.std().item())
                print("Fx0_hat_s stats:      ", Fx0_hat_s.mean().item(), Fx0_hat_s.std().item())
                print("Fx0_hat_t stats:      ", Fx0_hat_t.mean().item(), Fx0_hat_t.std().item())
                print("z_t stats:            ", z_t.mean().item(), z_t.std().item()) 
                print("###########################################################\n")

        if self.normalize:
            z_t = self.scaler(z_t, reverse=True, training=False)

        if collect_snapshots:
            return z_t, snapshots
        return z_t
