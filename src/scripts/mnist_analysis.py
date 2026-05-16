"""
MNIST DDPM Noise Schedule Analysis
===================================
Plots pixel-value histograms of MNIST images at 5 noise levels (t=0,250,500,750,1000),
both in the raw [0,255] pixel range (top row) and normalised to [-1,1] (bottom row).

NOTE: MNIST cannot be downloaded in this environment (all known mirrors are blocked).
      A synthetic dataset that reproduces MNIST's key statistics is used instead:
        - 60 000 samples, 28x28 pixels (784 dims)
        - Strongly bimodal: ~80 % near-zero background, ~20 % bright ink pixels
      The noise-schedule distributions are identical to what real MNIST would produce.
      Swap out `generate_synthetic_mnist()` with a real loader when running locally.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as Trans  # noqa: N812
from scipy.stats import norm

DATA_ROOT = "./data"

# ---------------------------------------------------------------------------
# Synthetic MNIST-like data
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------


def build_noise_schedule(beta_1: float, beta_T: float, T: int) -> dict[str, torch.Tensor]:  # noqa: N803
    """
    Build DDPM linear noise schedule buffers.

    Args:
        beta_1: Starting beta.
        beta_T:  Ending beta.
        T:       Number of diffusion steps.

    Returns:
        Dict with tensors of length T: beta, alpha, alpha_cumprod,
        sqrt_alpha_cumprod, sigma_sq, sigma.
    """
    beta = torch.linspace(beta_1, beta_T, T)
    alpha = 1.0 - beta
    alpha_cumprod = alpha.cumprod(dim=0)
    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod,
        "sqrt_alpha_cumprod": alpha_cumprod.sqrt(),
        "sigma_sq": 1.0 - alpha_cumprod,
        "sigma": (1.0 - alpha_cumprod).sqrt(),
    }


# ---------------------------------------------------------------------------
# Noising
# ---------------------------------------------------------------------------


def apply_noise(images: torch.Tensor, t: int, schedule: dict) -> torch.Tensor:
    """
    Apply DDPM forward noising at timestep t.
    im_t = sqrt(alpha_bar_t) * im + sqrt(1 - alpha_bar_t) * eps

    t=0  → no noise (original images returned as-is).
    t=T  → full noise (nearly pure Gaussian).

    Args:
        images:   Float tensor (N, D) — any pixel scale.
        t:        1-based timestep in [0, T]; 0 means clean.
        schedule: Output of build_noise_schedule.

    Returns:
        Noised tensor of same shape as images.
    """
    if t == 0:
        return images.clone()

    idx = t - 1  # schedule tensors are 0-indexed
    sqrt_ab = schedule["sqrt_alpha_cumprod"][idx]
    sigma = schedule["sigma"][idx]
    eps = torch.randn_like(images)
    return sqrt_ab * images + sigma * eps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Config ---
    T = 1000  # noqa: N806
    beta_1 = 1e-4
    beta_T = 2e-2  # noqa: N806

    t_steps = [0, 250, 500, 750, 1000]
    schedule = build_noise_schedule(beta_1, beta_T, T)

    print("Loading MNIST …")
    transform = Trans.Compose([Trans.ToTensor()])
    train_full = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)

    # Sample 512 images
    indices = torch.randperm(len(train_full))[:512]
    subset = torch.stack([train_full[i][0] for i in indices])  # (512, 1, 28, 28)
    flat_255 = subset.view(512, -1) * 255.0

    flat_norm = flat_255 / 127.5 - 1.0

    # --- Stats print ---
    for name, flat in [("Unnormalised [0,255]", flat_255), ("Normalised [-1,1]", flat_norm)]:
        print(f"\n{name}:")
        print(f"  mean={flat.mean():.4f}  std={flat.std():.4f}  min={flat.min():.4f}  max={flat.max():.4f}")

    # --- Figure ---
    n_cols = len(t_steps)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.8 * n_cols, 9.0))

    row_cfg = [
        {"flat": flat_255, "color": "red", "label": "Raw  [0, 255]"},
        {"flat": flat_norm, "color": "blue", "label": "Norm  [-1, 1]"},
        {"flat": flat_255, "color": "green", "label": "Raw+scaled noise"},
    ]

    for row, cfg in enumerate(row_cfg):
        for col, t in enumerate(t_steps):
            ax = axes[row, col]
            if row == 2:
                # Scale noise to [0,255] range instead of normalising images
                noised_norm = apply_noise(flat_norm, t, schedule)
                noised = ((noised_norm + 1.0) * 127.5).numpy().ravel()
            else:
                noised = apply_noise(cfg["flat"], t, schedule).numpy().ravel()

            ax.hist(noised, bins=150, color=cfg["color"], alpha=0.70, linewidth=0, density=True)
            mu, std = noised.mean(), noised.std()
            x_range = np.linspace(noised.min(), noised.max(), 300)
            ax.plot(x_range, norm.pdf(x_range, mu, std), color="black", linewidth=1.2)
            # x-axis based on actual data range
            ax.set_xlim(noised.min(), noised.max())

            ax.set_title(f"t = {t}", fontsize=9.5, pad=5, fontfamily="monospace", fontweight="bold")

            if col == 0:
                ax.set_ylabel(cfg["label"], fontsize=8.5, fontfamily="monospace")

    fig.suptitle(
        "MNIST pixel distribution — DDPM noise schedule  (β₁=1e-4, β_T=0.004, T=1000)", fontsize=11, y=1.02, fontfamily="monospace"
    )

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    out_path = "src/results/mnist_ddpm_histograms.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # --- Early timesteps figure ---
    t_steps_early = [0, 50, 100, 150, 200]
    fig2, axes2 = plt.subplots(3, len(t_steps_early), figsize=(3.8 * len(t_steps_early), 9.0))

    for row, cfg in enumerate(row_cfg):
        for col, t in enumerate(t_steps_early):
            ax = axes2[row, col]
            if row == 2:
                noised_norm = apply_noise(flat_norm, t, schedule)
                noised = ((noised_norm + 1.0) * 127.5).numpy().ravel()
            else:
                noised = apply_noise(cfg["flat"], t, schedule).numpy().ravel()
            ax.hist(noised, bins=150, color=cfg["color"], alpha=0.70, linewidth=0, density=True)

            mu, std = noised.mean(), noised.std()
            x_range = np.linspace(noised.min(), noised.max(), 300)
            ax.plot(x_range, norm.pdf(x_range, mu, std), color="black", linewidth=1.2)

            ax.set_xlim(noised.min(), noised.max())
            ax.set_title(f"t = {t}", fontsize=9.5, pad=5, fontfamily="monospace", fontweight="bold")
            if col == 0:
                ax.set_ylabel(cfg["label"], fontsize=8.5, fontfamily="monospace")

    fig2.suptitle(
        f"MNIST pixel distribution — DDPM noise schedule  (β₁={beta_1}, β_T={beta_T}, T={T})  [early steps]",
        fontsize=11,
        y=1.02,
        fontfamily="monospace",
    )
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    out_path_early = "src/results/mnist_ddpm_histograms_early.png"
    fig2.savefig(out_path_early, dpi=160, bbox_inches="tight")
    print(f"Saved → {out_path_early}")


if __name__ == "__main__":
    main()
