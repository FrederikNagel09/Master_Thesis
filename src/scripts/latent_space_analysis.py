"""
latent_space_analysis.py
Latent-space analysis for the VAE-INR model, up to three one-stage Latent
Diffusion variants, and up to two two-stage LDM variants:
  1. PCA scatter (real-image encodings, colored by class label) with a KDE
     density-background, one subplot per model, in a single 2x2 figure.
  2. Per-model interpolation: pick two real images of different class, encode
     to z1/z2, linearly interpolate in latent space, decode all 10 points,
     plot as a single row.

Usage
-----

CUDA_VISIBLE_DEVICES=1 python src/scripts/latent_space_analysis.py \
    --vae_config_path src/results/VAE_Baseline/VAE_Baseline_config.json \
    --vae_checkpoint_path src/results/VAE_Baseline/VAE_Baseline_checkpoint.pt \
    --latent_config_paths src/train_results/latent-diffusion/metadata/config.json \
    --two_stage_config_paths src/train_results/latent_two_stage_fixed/latent_two_stage_fixed_ldm_config.json src/train_results/two_stage_convergence/two_stage_convergence_ldm_config.json\
    --n_pca_samples 8 \
    --n_background_samples 8
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from src.scripts.get_all_plot_results import build_vae_model, make_coord_grid

PCA_HEADLINES_DEFAULT = ["(a) VAE-INR", "(b) Latent One-Stage", "(c) Latent Two-Stage Fixed", "(d) Latent Two-Stage Converged"]
INTERP_TITLES_DEFAULT = ["VAE-INR", "Latent One-Stage", "Latent Two-Stage Fixed", "Latent Two-Stage Converged"]


# ── Path helpers ───────────────────────────────────────────────────────────────
def _extract_run_name(config_path: str) -> str:
    """
    Extract run name from .../<run_name>/metadata/config.json.

    Args:
        config_path: Path to config JSON.
    Returns:
        run_name string.
    """
    parts = os.path.normpath(config_path).split(os.sep)
    try:
        idx = parts.index("metadata")
        return parts[idx - 1]
    except (ValueError, IndexError):
        raise ValueError(f"Could not extract run name from: {config_path}\nExpected: .../<run_name>/metadata/config.json")  # noqa: B904


def _extract_two_stage_checkpoint(config_path: str, run_name: str) -> str:
    """
    Derive the two-stage LDM checkpoint path from the config file's directory.
    Convention: <config_dir>/<run_name>_ldm_checkpoint.pt

    Args:
        config_path: Path to the flat two-stage config JSON.
        run_name:    Run name read from the config (config["run_name"]).
    Returns:
        Absolute path to the LDM checkpoint.
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.join(config_dir, f"{run_name}_ldm_checkpoint.pt")


# ── Slerp ──────────────────────────────────────────────────────────────────────
def slerp(z1: torch.Tensor, z2: torch.Tensor, alphas: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical linear interpolation between two single latent tensors (any
    shape, e.g. (1, latent_dim, H', W')), evaluated at each alpha in [0, 1].
    Falls back to linear interpolation when z1/z2 are (near) parallel or
    antiparallel, where the slerp formula is numerically unstable.

    Args:
        z1:     (1, ...) start latent.
        z2:     (1, ...) end latent, same shape as z1.
        alphas: (n_steps,) interpolation positions in [0, 1].
        eps:    Threshold on sin(omega) below which we fall back to linear interpolation.
    Returns:
        z_interp: (n_steps, ...) interpolated latents, z1 at alpha=0, z2 at alpha=1.
    """
    orig_shape = z1.shape[1:]
    z1_flat = z1.reshape(-1)
    z2_flat = z2.reshape(-1)

    cos_omega = torch.dot(z1_flat, z2_flat) / (z1_flat.norm() * z2_flat.norm() + eps)
    cos_omega = cos_omega.clamp(-1.0, 1.0)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    alphas_flat = alphas.view(-1)

    if sin_omega.abs() < eps:
        alphas_b = alphas.view(-1, *([1] * len(orig_shape)))
        return (1 - alphas_b) * z1 + alphas_b * z2

    coeff1 = (torch.sin((1 - alphas_flat) * omega) / sin_omega).view(-1, *([1] * len(orig_shape)))
    coeff2 = (torch.sin(alphas_flat * omega) / sin_omega).view(-1, *([1] * len(orig_shape)))
    return coeff1 * z1 + coeff2 * z2


# ── Collect (z, label) pairs by encoding real images ───────────────────────────
@torch.no_grad()
def collect_latents(
    model,
    model_type: str,
    loader: torch.utils.data.DataLoader,
    n_samples: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode real images into flattened latent vectors, paired with class labels.

    Args:
        model:      Trained VAEWrapper, LatentDiffusion, or TwoStageLDM model.
        model_type: "vae", "ldm", or "two_stage".
        loader:     DataLoader yielding (image, label) batches.
        n_samples:  Number of samples to collect (stops once reached).
        device:     Device string.
    Returns:
        z_flat: (n_samples, D) flattened latent vectors.
        labels: (n_samples,) integer class labels.
    """
    z_list, label_list = [], []
    n_collected = 0

    for x, y in loader:
        x = x.to(device)
        if x.dim() == 2:
            img_size = round(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], 1, img_size, img_size)

        # All three model types now expose encode() -> (z, mu, logvar)
        z, _, _ = model.encode(x)

        z_list.append(z.reshape(z.shape[0], -1).cpu().numpy())
        label_list.append(y.cpu().numpy())
        n_collected += x.shape[0]
        if n_collected >= n_samples:
            break

    z_flat = np.concatenate(z_list, axis=0)[:n_samples]
    labels = np.concatenate(label_list, axis=0)[:n_samples]
    return z_flat, labels


# ── Draw background samples from the model's own generative process ───────────
@torch.no_grad()
def sample_background_latents(
    model,
    model_type: str,
    n_samples: int,
    device: str,
    vae_config: dict | None = None,
    max_batch_size: int = 512,
) -> np.ndarray:
    """
    Draw latent samples from the model's generative process in memory-safe batches.

    Args:
        model:         Trained VAEWrapper, LatentDiffusion, or TwoStageLDM.
        model_type:    "vae", "ldm", or "two_stage".
        n_samples:     Total number of latent samples to draw.
        device:        Device string.
        vae_config:    VAE config dict (required when model_type == "vae").
        max_batch_size: Max samples per batch to avoid OOM.
    Returns:
        z_flat: (n_samples, D) flattened latent samples.
    """
    z_list = []
    samples_left = n_samples

    while samples_left > 0:
        current_batch_size = min(samples_left, max_batch_size)

        if model_type == "vae":
            latent_dim = vae_config["latent_dim"]
            latent_size = vae_config["latent_size"]
            z_batch = torch.randn(current_batch_size, latent_dim, latent_size, latent_size, device=device)
        else:
            # Both "ldm" and "two_stage" expose _sample_latent() -> (B, latent_dim, H', W')
            z_batch = model._sample_latent(current_batch_size, collect_snapshots=False, debug=False) if model_type == "ldm" \
                else model._sample_latent(current_batch_size)

        z_list.append(z_batch.reshape(z_batch.shape[0], -1).cpu().numpy())
        samples_left -= current_batch_size

    return np.concatenate(z_list, axis=0)


# ── PCA + KDE background for a single model (one subplot) ──────────────────────
def plot_pca_subplot(
    ax: plt.Axes,
    z_flat: np.ndarray,
    labels: np.ndarray,
    z_background: np.ndarray,
    title: str,
    grid_res: int = 150,
    interp_path_flat: np.ndarray | None = None,
    slerp_path_flat: np.ndarray | None = None,
) -> None:
    """
    Fit PCA(2) on real encoded data, project independent model-drawn samples
    into that same basis for a KDE density background, then scatter the real
    points (colored by class label) on top.

    Args:
        ax:                Matplotlib axes to draw on.
        z_flat:            (N, D) flattened latents from real images.
        labels:            (N,) integer class labels for z_flat.
        z_background:      (M, D) flattened latents from the model's generative process.
        title:             Subplot headline.
        grid_res:          Resolution of the KDE background grid (per axis).
        interp_path_flat:  Optional (n_steps, D) LINEAR interpolation path.
        slerp_path_flat:   Optional (n_steps, D) SLERP interpolation path.
    Returns:
        scatter handle (for colorbar).
    """
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_flat)
    bg_2d = pca.transform(z_background)

    x_min, x_max = bg_2d[:, 0].min(), bg_2d[:, 0].max()
    y_min, y_max = bg_2d[:, 1].min(), bg_2d[:, 1].max()
    x_pad = 0.1 * (x_max - x_min + 1e-8)
    y_pad = 0.1 * (y_max - y_min + 1e-8)

    xx, yy = np.mgrid[x_min - x_pad : x_max + x_pad : complex(0, grid_res), y_min - y_pad : y_max + y_pad : complex(0, grid_res)]
    kde = gaussian_kde(bg_2d.T)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    ax.contourf(xx, yy, density, levels=15, cmap="viridis", alpha=0.5)

    n_classes = int(labels.max()) + 1
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", vmin=0, vmax=n_classes - 1, s=6, alpha=0.85, linewidths=0)

    endpoints_2d = None
    if interp_path_flat is not None:
        interp_2d = pca.transform(interp_path_flat)
        endpoints_2d = interp_2d[[0, -1]]
        ax.plot(interp_2d[:, 0], interp_2d[:, 1], linestyle="-", color="darkorange", linewidth=1.3, zorder=4, label="Linear")
        ax.scatter(interp_2d[1:-1, 0], interp_2d[1:-1, 1], color="darkorange", s=12, marker="o", zorder=5)

    if slerp_path_flat is not None:
        slerp_2d = pca.transform(slerp_path_flat)
        if endpoints_2d is None:
            endpoints_2d = slerp_2d[[0, -1]]
        ax.plot(slerp_2d[:, 0], slerp_2d[:, 1], linestyle="-", color="mediumvioletred", linewidth=1.3, zorder=4, label="Slerp")
        ax.scatter(slerp_2d[1:-1, 0], slerp_2d[1:-1, 1], color="mediumvioletred", s=12, marker="o", zorder=5)

    if endpoints_2d is not None:
        ax.scatter(endpoints_2d[:, 0], endpoints_2d[:, 1], color="gray", s=22, marker="o", edgecolors="black", linewidths=0.7, zorder=6)
        ax.annotate(r"$z_1$", endpoints_2d[0], ha="center", va="center", fontsize=6, color="black", fontweight="bold", zorder=7, xytext=(0, -9), textcoords="offset points")
        ax.annotate(r"$z_2$", endpoints_2d[1], ha="center", va="center", fontsize=6, color="black", fontweight="bold", zorder=7, xytext=(0, -9), textcoords="offset points")
        ax.legend(loc="best", fontsize=6, framealpha=0.8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=8)
    ax.tick_params(labelsize=7)

    return scatter


# ── 2x2 PCA grid across all models ─────────────────────────────────────────────
def plot_pca_grid(
    data_tuples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    headlines: list[str],
    save_path: str,
) -> None:
    """
    Build a 2x2 grid of PCA+KDE subplots, one per model.

    Args:
        data_tuples:  List of (z_flat, labels, z_background, interp_path_flat,
                      slerp_path_flat) tuples, one per model (max 4).
        headlines:    Subplot titles, same length/order as data_tuples.
        save_path:    Output PNG path.
    Returns:
        None
    """
    n_models = len(data_tuples)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    last_scatter = None
    for idx in range(4):
        if idx < n_models:
            z_flat, labels, z_background, interp_path_flat, slerp_path_flat = data_tuples[idx]
            last_scatter = plot_pca_subplot(axes[idx], z_flat, labels, z_background, headlines[idx], interp_path_flat=interp_path_flat, slerp_path_flat=slerp_path_flat)
        else:
            axes[idx].axis("off")

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes, location="right", shrink=0.7, pad=0.02)
        cbar.set_label("Digit class", fontsize=9)
        cbar.set_ticks(range(10))

    fig.suptitle("Latent Space PCA Projections (background = model-generated sample density)", fontsize=13, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA grid saved -> {save_path}")


# ── Per-model 1x2 PCA comparison figure ───────────────────────────────────────
def plot_model_pca_comparison(
    z_flat: np.ndarray,
    labels: np.ndarray,
    z_background: np.ndarray,
    title_prefix: str,
    save_path: str,
    grid_res: int = 150,
    interp_path_flat: np.ndarray | None = None,
    slerp_path_flat: np.ndarray | None = None,
) -> None:
    """
    Standalone 1x2 figure: left = real encodings over real-data KDE, right =
    generative samples over the same KDE. Both subplots share PCA basis and axis limits.

    Args:
        z_flat:           (N, D) real image latents (defines PCA basis + KDE).
        labels:           (N,) class labels for z_flat.
        z_background:     (M, D) latents from the model's generative process.
        title_prefix:     Model name for the figure title.
        save_path:        Output PNG path.
        grid_res:         KDE grid resolution per axis.
        interp_path_flat: Optional (n_steps, D) linear interpolation path.
        slerp_path_flat:  Optional (n_steps, D) slerp interpolation path.
    Returns:
        None
    """
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_flat)
    bg_2d = pca.transform(z_background)

    x_min = min(z_2d[:, 0].min(), bg_2d[:, 0].min())
    x_max = max(z_2d[:, 0].max(), bg_2d[:, 0].max())
    y_min = min(z_2d[:, 1].min(), bg_2d[:, 1].min())
    y_max = max(z_2d[:, 1].max(), bg_2d[:, 1].max())
    x_pad = 0.08 * (x_max - x_min + 1e-8)
    y_pad = 0.08 * (y_max - y_min + 1e-8)
    xlims = (x_min - x_pad, x_max + x_pad)
    ylims = (y_min - y_pad, y_max + y_pad)

    xx, yy = np.mgrid[xlims[0]:xlims[1]:complex(0, grid_res), ylims[0]:ylims[1]:complex(0, grid_res)]
    kde = gaussian_kde(z_2d.T)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    n_classes = int(labels.max()) + 1

    ax1.set_title("Real Image Encodings", fontsize=11, fontweight="bold")
    ax1.contourf(xx, yy, density, levels=8, cmap="summer", alpha=1.0)
    scatter = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", vmin=0, vmax=n_classes - 1, s=7, alpha=0.80, linewidths=0, zorder=2)

    endpoints_2d = None
    if interp_path_flat is not None:
        interp_2d = pca.transform(interp_path_flat)
        endpoints_2d = interp_2d[[0, -1]]
        ax1.plot(interp_2d[:, 0], interp_2d[:, 1], linestyle="-", color="darkorange", linewidth=1.5, zorder=4, label="Linear")
        ax1.scatter(interp_2d[1:-1, 0], interp_2d[1:-1, 1], color="darkorange", s=14, marker="o", zorder=5)

    if slerp_path_flat is not None:
        slerp_2d = pca.transform(slerp_path_flat)
        if endpoints_2d is None:
            endpoints_2d = slerp_2d[[0, -1]]
        ax1.plot(slerp_2d[:, 0], slerp_2d[:, 1], linestyle="-", color="mediumvioletred", linewidth=1.5, zorder=4, label="Slerp")
        ax1.scatter(slerp_2d[1:-1, 0], slerp_2d[1:-1, 1], color="mediumvioletred", s=14, marker="o", zorder=5)

    if endpoints_2d is not None:
        ax1.scatter(endpoints_2d[:, 0], endpoints_2d[:, 1], color="white", s=30, marker="o", edgecolors="black", linewidths=1.0, zorder=6)
        ax1.legend(loc="best", fontsize=7, framealpha=0.8)

    ax2.set_title("Model Generated Samples", fontsize=11, fontweight="bold")
    ax2.contourf(xx, yy, density, levels=8, cmap="summer", alpha=1.0)
    ax2.scatter(bg_2d[:, 0], bg_2d[:, 1], color="black", s=4, alpha=0.4, label="Sampled Points", zorder=2)
    ax2.legend(loc="best", fontsize=7, framealpha=0.8)

    pc1_label = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    pc2_label = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    for ax in (ax1, ax2):
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel(pc1_label, fontsize=8)
        ax.set_ylabel(pc2_label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(f"Latent PCA Alignment (Background = Real Data Density): {title_prefix}", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(bottom=0.22, wspace=0.2)

    cbar_ax = fig.add_axes([0.15, 0.08, 0.70, 0.03])
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Digit Class Label", fontsize=9, fontweight="bold")
    cbar.set_ticks(range(n_classes))
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison plot -> {save_path}")


# ── Interpolation helpers ───────────────────────────────────────────────────────
@torch.no_grad()
def get_interpolation_pair(loader: torch.utils.data.DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull two completely random images of different classes from the loader.

    Args:
        loader: DataLoader yielding (image, label) batches.
        device: Device string.
    Returns:
        x1, x2: (1, C, H, W) tensors of two differently-labeled images.
    """
    x1, x2 = None, None
    label1 = None

    for x, y in loader:
        if x.dim() == 2:
            img_size = round(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], 1, img_size, img_size)
        idx1 = random.randint(0, x.shape[0] - 1)
        x1 = x[idx1 : idx1 + 1].to(device)
        label1 = int(y[idx1].item())
        break

    for x, y in loader:
        if x.dim() == 2:
            img_size = round(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], 1, img_size, img_size)
        indices = list(range(x.shape[0]))
        random.shuffle(indices)
        for idx2 in indices:
            if int(y[idx2].item()) != label1:
                x2 = x[idx2 : idx2 + 1].to(device)
                return x1, x2

    raise RuntimeError("Could not find a second distinct class in the loader.")


@torch.no_grad()
def interpolate_and_decode(
    model,
    model_type: str,
    x1: torch.Tensor,
    x2: torch.Tensor,
    n_steps: int,
    coord_grid: torch.Tensor,
    channels: int,
    device: str,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode x1/x2, interpolate z in latent space (linear or slerp), decode each step.

    Args:
        model:      Trained VAEWrapper, LatentDiffusion, or TwoStageLDM.
        model_type: "vae", "ldm", or "two_stage".
        x1, x2:    (1, C, H, W) endpoint images.
        n_steps:   Total points including endpoints.
        coord_grid: (H, W, 2) coordinate grid for the decoder.
        channels:  Number of image channels.
        device:    Device string.
        method:    "linear" or "slerp".
    Returns:
        images:        (n_steps, H, W) or (n_steps, H, W, C) decoded images in [0,1].
        z_interp_flat: (n_steps, D) flattened interpolation path.
    """
    # Uniform encode interface across all three model types
    z1, _, _ = model.encode(x1)
    z2, _, _ = model.encode(x2)

    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, *([1] * (z1.dim() - 1)))

    if method == "slerp":
        z_interp = slerp(z1, z2, alphas)
    else:
        z_interp = (1 - alphas) * z1 + alphas * z2

    z_interp_flat = z_interp.reshape(z_interp.shape[0], -1).cpu().numpy()

    x_hat = model.decoder(z_interp, coord_grid)
    x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1).cpu().float()

    if channels == 1:
        return x_hat.squeeze(1).numpy(), z_interp_flat
    return x_hat.permute(0, 2, 3, 1).numpy(), z_interp_flat


def plot_interpolation_row(images: np.ndarray, channels: int, title: str, save_path: str) -> None:
    """
    Plot a single row of interpolated reconstructions with a headline.

    Args:
        images:    (n_steps, H, W) or (n_steps, H, W, C) images in [0,1].
        channels:  Number of image channels.
        title:     Headline above the row.
        save_path: Output PNG path.
    Returns:
        None
    """
    n_steps = images.shape[0]
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.5, 1.8), gridspec_kw={"wspace": 0.0})

    for i, ax in enumerate(axes):
        img = images[i]
        if channels == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        else:
            ax.imshow(img, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.axis("off")
        if i == 0:
            ax.set_title("z1 (real)", fontsize=7)
        elif i == n_steps - 1:
            ax.set_title("z2 (real)", fontsize=7)

    fig.suptitle(f"Latent Interpolation: {title}", fontsize=11, fontweight="bold", y=1.05)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Interpolation row saved -> {save_path}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Latent space analysis (PCA + interpolation) for VAE and Latent Diffusion models.")

    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--latent_config_paths", type=str, nargs="+", default=[], help="One-stage LDM config paths (max 3).")
    parser.add_argument("--two_stage_config_paths", type=str, nargs="+", default=[], help="Two-stage LDM config paths (max 2).")
    parser.add_argument("--n_pca_samples", type=int, default=2000)
    parser.add_argument("--n_background_samples", type=int, default=2000)
    parser.add_argument("--n_interp_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if len(args.latent_config_paths) > 3:
        parser.error("Max 3 --latent_config_paths.")
    if len(args.two_stage_config_paths) > 2:
        parser.error("Max 2 --two_stage_config_paths.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
    from src.utility.model_builders.model_builder import build_model as build_ldm_model
    from src.utility.model_builders.util.twostage_builder import build_ldm as build_two_stage_ldm

    device = _get_device()
    output_dir = os.path.join("src", "results", "latent_space_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Latent Space Analysis  |  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # ── Load VAE config + dataset ─────────────────────────────────────────────
    with open(args.vae_config_path) as f:
        vae_config = json.load(f)

    print("  Building dataset ...")
    _, val_dataset, data_config = build_dataset(
        dataset_name=vae_config["dataset"],
        data_root="data/",
        subset_frac=1.0,
        single_class=False,
    )
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    coord_grid = make_coord_grid((img_size, img_size), (-1, 1), device=device)

    pca_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)
    interp_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    # ── Build VAE ─────────────────────────────────────────────────────────────
    print("--- Loading VAE-INR Model ---")
    vae_model = build_vae_model(vae_config, channels, img_size, device)
    vae_ckpt = torch.load(args.vae_checkpoint_path, map_location=device)
    vae_model.load_state_dict(vae_ckpt["model_state_dict"])
    vae_model.eval()
    models = [("vae", vae_model, "VAE-INR")]

    # ── Build one-stage LDM variants ──────────────────────────────────────────
    if args.latent_config_paths:
        print(f"--- Loading One-Stage LDM Suite ({len(args.latent_config_paths)} variants) ---")
        for p in args.latent_config_paths:
            with open(p) as f:
                l_cfg = json.load(f)
            l_hparams = SimpleNamespace(**l_cfg["hparams"])
            l_data_cfg = l_cfg["data"]
            l_data_config = {
                "dataset": l_cfg["dataset"],
                "channels": l_data_cfg["channels"],
                "img_size": l_data_cfg["img_size"],
                "data_dim": l_data_cfg["data_dim"],
            }
            run_name = _extract_run_name(p)
            print(f"  Building & loading: {run_name} ...")
            l_model = build_ldm_model(l_hparams, l_data_config).to(device)
            l_ckpt = torch.load(l_cfg["paths"]["weights"], map_location=device)
            l_model.load_state_dict(l_ckpt["model_state_dict"])
            l_model.eval()
            models.append(("ldm", l_model, run_name))

    # ── Build two-stage LDM variants ──────────────────────────────────────────
    if args.two_stage_config_paths:
        print(f"--- Loading Two-Stage LDM Suite ({len(args.two_stage_config_paths)} variants) ---")
        for p in args.two_stage_config_paths:
            with open(p) as f:
                ts_cfg = json.load(f)

            run_name = ts_cfg["run_name"]
            ckpt_path = _extract_two_stage_checkpoint(p, run_name)

            # build_ldm expects args-style access for T/beta; wrap flat config as namespace
            ts_args = SimpleNamespace(
                T=ts_cfg["T"],
                beta_1=ts_cfg["beta_1"],
                beta_T=ts_cfg["beta_T"],
            )

            print(f"  Building & loading: {run_name} ...")
            ts_model = build_two_stage_ldm(
                hparams=ts_cfg,
                args=ts_args,
                channels=channels,
                img_size=img_size,
                device=device,
            )
            ts_ckpt = torch.load(ckpt_path, map_location=device)
            ts_model.load_state_dict(ts_ckpt["model_state_dict"])
            ts_model.eval()
            models.append(("two_stage", ts_model, run_name))

    n_models = len(models)
    pca_headlines = PCA_HEADLINES_DEFAULT[:n_models]
    interp_titles = INTERP_TITLES_DEFAULT[:n_models]

    # ── Fetch interpolation endpoints ─────────────────────────────────────────
    print("\n--- Fetching interpolation endpoints ---")
    x1, x2 = get_interpolation_pair(interp_loader, device)

    # ── Per-model analysis loop ───────────────────────────────────────────────
    print("\n--- Processing Latent Space Analysis & Interpolations ---")
    for (model_type, model, run_name), title in zip(models, interp_titles, strict=False):
        print(f"\n[Model: {run_name}]")
        safe_name = run_name.lower().replace(" ", "_").replace("-", "_")

        print(f"  Encoding {args.n_pca_samples} real images...")
        z_flat, labels = collect_latents(model, model_type, pca_loader, args.n_pca_samples, device)

        print(f"  Drawing {args.n_background_samples} generative samples...")
        z_background = sample_background_latents(
            model, model_type, args.n_background_samples, device,
            vae_config=vae_config if model_type == "vae" else None,
        )

        print("  Running Linear interpolation...")
        img_lin, path_lin = interpolate_and_decode(model, model_type, x1, x2, args.n_interp_steps, coord_grid, channels, device, method="linear")
        plot_interpolation_row(img_lin, channels, title=f"{title} (Linear)", save_path=os.path.join(output_dir, f"interpolation_{safe_name}_linear.png"))

        print("  Running Slerp interpolation...")
        img_slerp, path_slerp = interpolate_and_decode(model, model_type, x1, x2, args.n_interp_steps, coord_grid, channels, device, method="slerp")
        plot_interpolation_row(img_slerp, channels, title=f"{title} (Slerp)", save_path=os.path.join(output_dir, f"interpolation_{safe_name}_slerp.png"))

        print("  Generating paired PCA plots...")
        plot_model_pca_comparison(
            z_flat=z_flat, labels=labels, z_background=z_background,
            title_prefix=title, save_path=os.path.join(output_dir, f"pca_comparison_{safe_name}.png"),
            interp_path_flat=path_lin, slerp_path_flat=path_slerp,
        )

    print("\nLatent Space Analysis Complete.")


if __name__ == "__main__":
    main()