"""
weight_space_analysis.py
Weight-space analysis across all three model families: VAE-INR, Latent
Diffusion (up to 3 variants), and Weight Diffusion (up to 3 variants),
plus two-stage variants of both Latent and Weight Diffusion (up to 2 each).

CUDA_VISIBLE_DEVICES=1 python src/scripts/weight_space_analysis.py \
    --vae_config_path src/results/VAE_Baseline/VAE_Baseline_config.json \
    --vae_checkpoint_path src/results/VAE_Baseline/VAE_Baseline_checkpoint.pt \
    --latent_config_paths src/train_results/latent-diffusion/metadata/config.json \
    --two_stage_config_paths src/train_results/latent_two_stage_fixed/latent_two_stage_fixed_ldm_config.json src/train_results/two_stage_convergence/two_stage_convergence_ldm_config.json \
    --weight_config_paths src/train_results/weight-diffusion/metadata/config.json \
    --two_stage_weight_config_paths src/train_results/wd_two_stage_fixed/wd_two_stage_fixed_wd_config.json src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_wd_config.json\
    --n_pca_samples 2024 \
    --n_interp_steps 10

"""
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
from sklearn.decomposition import PCA

from src.scripts.get_all_plot_results import build_vae_model, make_coord_grid


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


def _safe_name(run_name: str) -> str:
    """
    Sanitize a run name for use in a filename.

    Args:
        run_name: Arbitrary run name string.
    Returns:
        Lowercased, filesystem-safe string.
    """
    return run_name.lower().replace(" ", "_").replace("-", "_")


def _extract_two_stage_wd_checkpoint(config_path: str, run_name: str) -> str:
    """
    Derive the two-stage WeightDiffusion weights path from the config file's directory.
    Convention: <config_dir>/<run_name>_wd_weights.pt

    Args:
        config_path: Path to the flat two-stage WD config JSON.
        run_name:    Run name read from config["run_name"].
    Returns:
        Absolute path to the WD weights file.
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.join(config_dir, f"{run_name}_wd_weights.pt")


def _reshape_if_flat(x: torch.Tensor, channels: int = 1) -> torch.Tensor:
    """
    Reshape a flat (B, data_dim) image tensor to (B, C, H, W) if needed.

    Args:
        x:        (B, data_dim) or (B, C, H, W) tensor.
        channels: Number of image channels.
    Returns:
        (B, C, H, W) tensor.
    """
    if x.dim() == 2:
        img_size = round((x.shape[1] // channels) ** 0.5)
        x = x.view(x.shape[0], channels, img_size, img_size)
    return x


# ── Slerp ──────────────────────────────────────────────────────────────────────
def slerp(w1: torch.Tensor, w2: torch.Tensor, alphas: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical linear interpolation between two single vectors, evaluated at
    each alpha in [0, 1]. Falls back to linear when w1/w2 are near parallel
    or antiparallel.

    Args:
        w1:     (1, D) start vector.
        w2:     (1, D) end vector.
        alphas: (n_steps, 1) interpolation positions in [0, 1].
        eps:    Threshold on sin(omega) below which we fall back to linear.
    Returns:
        w_interp: (n_steps, D) interpolated vectors.
    """
    w1_flat = w1.reshape(-1)
    w2_flat = w2.reshape(-1)
    cos_omega = torch.dot(w1_flat, w2_flat) / (w1_flat.norm() * w2_flat.norm() + eps)
    cos_omega = cos_omega.clamp(-1.0, 1.0)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < eps:
        return (1 - alphas) * w1 + alphas * w2
    coeff1 = torch.sin((1 - alphas) * omega) / sin_omega
    coeff2 = torch.sin(alphas * omega) / sin_omega
    return coeff1 * w1 + coeff2 * w2


# ── Weight vector extraction ───────────────────────────────────────────────────
@torch.no_grad()
def collect_weight_vectors(
    model,
    model_type: str,
    x_pca: torch.Tensor,
    y_pca: torch.Tensor,
    coord_grid: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract flattened modulated SIREN weight vectors from a fixed set of real
    images, paired with class labels.

    Args:
        model:      Trained VAEWrapper, LatentDiffusion, TwoStageLDM, WeightDiffusion,
                    or two-stage WeightDiffusion model.
        model_type: "vae", "ldm", "two_stage", "weight_diffusion", or "two_stage_weight".
        x_pca:      (N, C, H, W) fixed image batch, identical across all models.
        y_pca:      (N,) integer class labels matching x_pca.
        coord_grid: (H, W, 2) coordinate grid (used by vae/ldm/two_stage decoders).
        device:     Device string.
        batch_size: Sub-batch size for memory control.
    Returns:
        weight_flat: (N, D) flattened weight vectors.
        labels:      (N,) integer class labels as numpy.
    """
    w_list = []
    n_total = x_pca.shape[0]

    for start in range(0, n_total, batch_size):
        x = x_pca[start : start + batch_size].to(device)

        if model_type in ("weight_diffusion", "two_stage_weight"):
            # Both one-stage and two-stage WD share the same encode/decode_modulations interface
            theta_prime_raw, _, _ = model.encode(x)
            w_flat = model.weight_encoder.decode_modulations(theta_prime_raw)
        else:
            # vae / ldm / two_stage: encode to z, expose weights via decoder
            z, _, _ = model.encode(x)
            coord_batched = coord_grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            _, w_flat = model.decoder.forward_with_weights(z, coord_batched)

        w_list.append(w_flat.cpu().numpy())

    weight_flat = np.concatenate(w_list, axis=0)
    labels = y_pca.cpu().numpy()
    return weight_flat, labels


# ── Weight vector sampling from model's own generative process ─────────────────
@torch.no_grad()
def sample_weight_vectors(
    model,
    model_type: str,
    n_samples: int,
    coord_grid: torch.Tensor,
    device: str,
    vae_config: dict | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Draw flattened modulated SIREN weight vectors from the model's own
    generative process, batched to bound memory.

    Args:
        model:      Trained VAEWrapper, LatentDiffusion, TwoStageLDM, WeightDiffusion,
                    or two-stage WeightDiffusion model.
        model_type: "vae", "ldm", "two_stage", "weight_diffusion", or "two_stage_weight".
        n_samples:  Total number of weight vectors to draw.
        coord_grid: (H, W, 2) coordinate grid (used by vae/ldm/two_stage decoders).
        device:     Device string.
        vae_config: Required when model_type == "vae"; needs latent_dim/latent_size.
        batch_size: Samples drawn per chunk.
    Returns:
        weight_flat: (n_samples, D) flattened weight vectors.
    """
    w_list = []
    n_remaining = n_samples

    while n_remaining > 0:
        b = min(batch_size, n_remaining)

        if model_type == "vae":
            latent_dim = vae_config["latent_dim"]
            latent_size = vae_config["latent_size"]
            z = torch.randn(b, latent_dim, latent_size, latent_size, device=device)
            coord_batched = coord_grid.unsqueeze(0).expand(b, -1, -1, -1)
            _, w_flat = model.decoder.forward_with_weights(z, coord_batched)
        elif model_type == "ldm":
            z = model._sample_latent(b, collect_snapshots=False, debug=False)  # noqa: SLF001
            coord_batched = coord_grid.unsqueeze(0).expand(b, -1, -1, -1)
            _, w_flat = model.decoder.forward_with_weights(z, coord_batched)
        elif model_type == "two_stage":
            # Two-stage LDM: reverse diffusion -> z -> decoder weights
            z = model._sample_latent(b)
            coord_batched = coord_grid.unsqueeze(0).expand(b, -1, -1, -1)
            _, w_flat = model.decoder.forward_with_weights(z, coord_batched)
        elif model_type in ("weight_diffusion", "two_stage_weight"):
            # Both share the same sample_weight -> decode_modulations interface
            theta_prime = model.sample_weight(b)
            w_flat = model.weight_encoder.decode_modulations(theta_prime)
        
        w_list.append(w_flat.cpu())
        n_remaining -= b

    return torch.cat(w_list, dim=0).numpy()


# ── Single PCA subplot helper ──────────────────────────────────────────────────
def _draw_pca_subplot(
    ax: plt.Axes,
    w_2d: np.ndarray,
    labels: np.ndarray | None,
    interp_path_2d: np.ndarray | None,
    panel_title: str,
    slerp_path_2d: np.ndarray | None = None,
) -> object:
    """
    Draw one PCA scatter panel (points already projected to 2D) plus optional
    linear and Slerp interpolation paths.

    Args:
        ax:              Matplotlib axes to draw on.
        w_2d:            (N, 2) already-PCA-projected points.
        labels:          (N,) integer class labels, or None for unlabeled.
        interp_path_2d:  (n_steps, 2) projected LINEAR path, or None.
        panel_title:     Subplot title.
        slerp_path_2d:   (n_steps, 2) projected SLERP path, or None.
    Returns:
        scatter artist (for colorbar), or None if labels is None.
    """
    scatter = None
    if labels is not None:
        n_classes = int(labels.max()) + 1
        scatter = ax.scatter(w_2d[:, 0], w_2d[:, 1], c=labels, cmap="tab10", vmin=0, vmax=n_classes - 1, s=8, alpha=0.85, linewidths=0)
    else:
        ax.scatter(w_2d[:, 0], w_2d[:, 1], color="black", s=8, alpha=0.6, linewidths=0)

    if interp_path_2d is not None:
        ax.plot(interp_path_2d[:, 0], interp_path_2d[:, 1], linestyle="-", color="darkorange", linewidth=1.4, zorder=4, label="Linear")
        ax.scatter(interp_path_2d[1:-1, 0], interp_path_2d[1:-1, 1], color="darkorange", s=15, marker="o", zorder=5)

    if slerp_path_2d is not None:
        ax.plot(slerp_path_2d[:, 0], slerp_path_2d[:, 1], linestyle="-", color="mediumvioletred", linewidth=1.4, zorder=4, label="Slerp")
        ax.scatter(slerp_path_2d[1:-1, 0], slerp_path_2d[1:-1, 1], color="mediumvioletred", s=15, marker="o", zorder=5)

    if interp_path_2d is not None:
        endpoints_2d = interp_path_2d[[0, -1]]
        ax.scatter(endpoints_2d[:, 0], endpoints_2d[:, 1], color="gray", s=28, marker="o", edgecolors="black", linewidths=0.8, zorder=6)
        ax.annotate(r"$\theta_1$", endpoints_2d[0], ha="center", va="center", fontsize=6, color="black", fontweight="bold", zorder=7, xytext=(0, -10), textcoords="offset points")
        ax.annotate(r"$\theta_2$", endpoints_2d[1], ha="center", va="center", fontsize=6, color="black", fontweight="bold", zorder=7, xytext=(0, -10), textcoords="offset points")

    if interp_path_2d is not None or slerp_path_2d is not None:
        ax.legend(loc="best", fontsize=7, framealpha=0.8)

    ax.set_title(panel_title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    return scatter


# ── Combined PCA figure: reconstruction (left) + sample (right) ───────────────
def plot_weight_pca_combined(
    weight_flat_real: np.ndarray,
    labels_real: np.ndarray,
    interp_path_real_flat: np.ndarray,
    weight_flat_sample: np.ndarray,
    interp_path_sample_flat: np.ndarray,
    title: str,
    save_path: str,
    slerp_path_real_flat: np.ndarray | None = None,
    slerp_path_sample_flat: np.ndarray | None = None,
    grid_res: int = 150,
) -> None:
    """
    Fit PCA(2) once on real-image weight vectors, project sampled vectors into
    that same basis, render side-by-side with a shared real-data KDE background.

    Args:
        weight_flat_real:      (N, D) real-image weight vectors (defines PCA basis + KDE).
        labels_real:           (N,) class labels for real vectors.
        interp_path_real_flat: (n_steps, D) linear interp path from real endpoints.
        weight_flat_sample:    (M, D) model-generated weight vectors.
        interp_path_sample_flat: (n_steps, D) linear interp path from sampled endpoints.
        title:                 Model name for figure title.
        save_path:             Output PNG path.
        slerp_path_real_flat:  Optional (n_steps, D) slerp path from real endpoints.
        slerp_path_sample_flat: Optional (n_steps, D) slerp path from sampled endpoints.
        grid_res:              KDE grid resolution per axis.
    Returns:
        None
    """
    from scipy.stats import gaussian_kde

    pca = PCA(n_components=2)
    w_2d_real = pca.fit_transform(weight_flat_real)
    w_2d_sample = pca.transform(weight_flat_sample)
    interp_2d_real = pca.transform(interp_path_real_flat)
    interp_2d_sample = pca.transform(interp_path_sample_flat)
    slerp_2d_real = pca.transform(slerp_path_real_flat) if slerp_path_real_flat is not None else None
    slerp_2d_sample = pca.transform(slerp_path_sample_flat) if slerp_path_sample_flat is not None else None

    # Shared axis limits across both panels
    x_parts = [w_2d_real[:, 0], w_2d_sample[:, 0], interp_2d_real[:, 0], interp_2d_sample[:, 0]]
    y_parts = [w_2d_real[:, 1], w_2d_sample[:, 1], interp_2d_real[:, 1], interp_2d_sample[:, 1]]
    if slerp_2d_real is not None:
        x_parts.append(slerp_2d_real[:, 0])
        y_parts.append(slerp_2d_real[:, 1])
    if slerp_2d_sample is not None:
        x_parts.append(slerp_2d_sample[:, 0])
        y_parts.append(slerp_2d_sample[:, 1])

    all_x = np.concatenate(x_parts)
    all_y = np.concatenate(y_parts)
    x_pad = 0.05 * (all_x.max() - all_x.min() + 1e-8)
    y_pad = 0.05 * (all_y.max() - all_y.min() + 1e-8)
    xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
    ylim = (all_y.min() - y_pad, all_y.max() + y_pad)

    xx, yy = np.mgrid[xlim[0]:xlim[1]:complex(0, grid_res), ylim[0]:ylim[1]:complex(0, grid_res)]
    kde = gaussian_kde(w_2d_real.T)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))

    axes[0].contourf(xx, yy, density, levels=8, cmap="summer", alpha=1.0, zorder=1)
    scatter = _draw_pca_subplot(axes[0], w_2d_real, labels_real, interp_2d_real, "Reconstruction (real images)", slerp_path_2d=slerp_2d_real)

    axes[1].contourf(xx, yy, density, levels=8, cmap="summer", alpha=1.0, zorder=1)
    _draw_pca_subplot(axes[1], w_2d_sample, None, interp_2d_sample, "Sample (model-generated)", slerp_path_2d=slerp_2d_sample)

    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)", fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)
        ax.grid(True, linestyle=":", alpha=0.3, zorder=3)

    fig.suptitle(f"Weight Space PCA (Background = Real Density): {title}", fontsize=13, fontweight="bold")
    if scatter is not None:
        n_classes = int(labels_real.max()) + 1
        cbar = fig.colorbar(scatter, ax=axes, location="bottom", orientation="horizontal", shrink=0.5, pad=0.1)
        cbar.set_label("Digit class", fontsize=9)
        cbar.set_ticks(range(n_classes))

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Weight PCA saved -> {save_path}")


# ── Fixed PCA image batch (shared across all models) ──────────────────────────
def draw_fixed_pca_batch(
    loader: torch.utils.data.DataLoader,
    n_samples: int,
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a single fixed batch of real images + labels, reused across all models.

    Args:
        loader:    DataLoader yielding (image, label) batches.
        n_samples: Number of images to collect.
        channels:  Number of image channels.
    Returns:
        x_pca: (n_samples, C, H, W) fixed image batch.
        y_pca: (n_samples,) integer class labels.
    """
    x_list, y_list = [], []
    n_collected = 0
    for x, y in loader:
        x = _reshape_if_flat(x, channels)
        x_list.append(x)
        y_list.append(y)
        n_collected += x.shape[0]
        if n_collected >= n_samples:
            break
    x_pca = torch.cat(x_list, dim=0)[:n_samples]
    y_pca = torch.cat(y_list, dim=0)[:n_samples]
    return x_pca, y_pca


# ── Interpolation pair selection ───────────────────────────────────────────────
def get_interpolation_pair(
    dataset: torch.utils.data.Dataset,
    channels: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pick two random images of different classes from the dataset.

    Args:
        dataset:  Dataset yielding (image, label) pairs.
        channels: Number of image channels.
        device:   Device string.
    Returns:
        x1, x2: (1, C, H, W) tensors of two differently-labeled images.
    """
    n = len(dataset)
    idx1 = random.randrange(n)
    x1, y1 = dataset[idx1]
    label1 = int(y1)
    while True:
        idx2 = random.randrange(n)
        x2, y2 = dataset[idx2]
        if int(y2) != label1:
            break
    x1 = _reshape_if_flat(x1.unsqueeze(0), channels).to(device)
    x2 = _reshape_if_flat(x2.unsqueeze(0), channels).to(device)
    return x1, x2


# ── Per-image weight vector extraction ────────────────────────────────────────
@torch.no_grad()
def get_weight_vector(
    model,
    model_type: str,
    x: torch.Tensor,
    coord_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Extract a single flattened weight vector for one image.

    Args:
        model:      Trained model.
        model_type: "vae", "ldm", "two_stage", "weight_diffusion", or "two_stage_weight".
        x:          (1, C, H, W) input image.
        coord_grid: (H, W, 2) coordinate grid.
    Returns:
        weight_flat: (1, D) flattened weight vector.
    """
    if model_type in ("weight_diffusion", "two_stage_weight"):
        theta_prime_raw, _, _ = model.encode(x)
        return model.weight_encoder.decode_modulations(theta_prime_raw)

    z, _, _ = model.encode(x)
    coord_batched = coord_grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    _, w_flat = model.decoder.forward_with_weights(z, coord_batched)
    return w_flat


# ── Weight vector decoding to pixels ──────────────────────────────────────────
@torch.no_grad()
def decode_weight_vector(
    model,
    model_type: str,
    weight_flat: torch.Tensor,
    coord_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Decode a (possibly interpolated) flat weight vector back to pixel space.

    Args:
        model:       Trained model.
        model_type:  "vae", "ldm", "two_stage", "weight_diffusion", or "two_stage_weight".
        weight_flat: (B, D) flattened weight vectors.
        coord_grid:  (H, W, 2) coordinate grid.
    Returns:
        pixels: (B, C, H, W) decoded images, NOT yet un-normalized to [0,1].
    """
    B = weight_flat.shape[0]  # noqa: N806
    coord_batched = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

    if model_type in ("weight_diffusion", "two_stage_weight"):
        pixels_flat = model._inr_decode(weight_flat, coords=coord_batched)  # noqa: SLF001
        img_size = coord_grid.shape[0]
        channels = pixels_flat.shape[1] // (img_size * img_size)
        return pixels_flat.reshape(B, channels, img_size, img_size)

    # vae / ldm / two_stage: unflatten into per-layer param dict, query INR directly
    decoder = model.decoder
    params = {}
    offset = 0
    for name, shape in decoder.inr.param_shapes.items():
        numel = shape[0] * shape[1]
        params[name] = weight_flat[:, offset : offset + numel].reshape(B, shape[0], shape[1])
        offset += numel
    decoder.inr.set_params(params)
    pred = decoder.inr(coord_batched)  # (B, H, W, C_out)
    return pred.permute(0, 3, 1, 2).contiguous()


def plot_weight_interpolation_row(
    images: np.ndarray,
    channels: int,
    title: str,
    save_path: str,
) -> None:
    """
    Plot a single row of weight-space-interpolated reconstructions.

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
            ax.set_title("theta1 (real)", fontsize=7)
        elif i == n_steps - 1:
            ax.set_title("theta2 (real)", fontsize=7)
    fig.suptitle(f"Weight-Space Interpolation: {title}", fontsize=11, fontweight="bold", y=1.05)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Weight interpolation row saved -> {save_path}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Weight space analysis across VAE, Latent, and Weight Diffusion models.")

    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--latent_config_paths", type=str, nargs="+", default=[], help="One-stage LDM config paths (max 3).")
    parser.add_argument("--two_stage_config_paths", type=str, nargs="+", default=[], help="Two-stage LDM config paths (max 2).")
    parser.add_argument("--weight_config_paths", type=str, nargs="+", default=[], help="One-stage WeightDiffusion config paths (max 3).")
    parser.add_argument("--two_stage_weight_config_paths", type=str, nargs="+", default=[], help="Two-stage WeightDiffusion config paths (max 2).")
    parser.add_argument("--n_pca_samples", type=int, default=2000)
    parser.add_argument("--n_interp_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if len(args.latent_config_paths) > 3:
        parser.error("Max 3 --latent_config_paths.")
    if len(args.two_stage_config_paths) > 2:
        parser.error("Max 2 --two_stage_config_paths.")
    if len(args.weight_config_paths) > 3:
        parser.error("Max 3 --weight_config_paths.")
    if len(args.two_stage_weight_config_paths) > 2:
        parser.error("Max 2 --two_stage_weight_config_paths.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
    from src.utility.model_builders.model_builder import build_model as build_ldm_model
    from src.utility.model_builders.util.twostage_builder import build_ldm as build_two_stage_ldm
    from src.scripts.two_stage_weight_training import build_full_wd_model

    device = _get_device()
    output_dir = os.path.join("src", "results", "weight_space_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Weight Space Analysis  |  Output: {output_dir}")
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
    data_dim = data_config["data_dim"]
    coord_grid = make_coord_grid((img_size, img_size), (-1, 1), device=device)

    pca_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)

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
            l_data_config = {"dataset": l_cfg["dataset"], "channels": l_cfg["data"]["channels"], "img_size": l_cfg["data"]["img_size"], "data_dim": l_cfg["data"]["data_dim"]}
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
            ckpt_path = os.path.join(os.path.dirname(os.path.abspath(p)), f"{run_name}_ldm_checkpoint.pt")
            ts_args = SimpleNamespace(T=ts_cfg["T"], beta_1=ts_cfg["beta_1"], beta_T=ts_cfg["beta_T"])
            print(f"  Building & loading: {run_name} ...")
            ts_model = build_two_stage_ldm(hparams=ts_cfg, args=ts_args, channels=channels, img_size=img_size, device=device)
            ts_ckpt = torch.load(ckpt_path, map_location=device)
            ts_model.load_state_dict(ts_ckpt["model_state_dict"])
            ts_model.eval()
            models.append(("two_stage", ts_model, run_name))

    # ── Build one-stage WeightDiffusion variants ───────────────────────────────
    if args.weight_config_paths:
        print(f"--- Loading One-Stage WeightDiffusion Suite ({len(args.weight_config_paths)} variants) ---")
        for p in args.weight_config_paths:
            with open(p) as f:
                w_cfg = json.load(f)
            w_hparams = SimpleNamespace(**w_cfg["hparams"])
            w_data_config = {"dataset": w_cfg["dataset"], "channels": w_cfg["data"]["channels"], "img_size": w_cfg["data"]["img_size"], "data_dim": w_cfg["data"]["data_dim"]}
            run_name = _extract_run_name(p)
            print(f"  Building & loading: {run_name} ...")
            w_model = build_ldm_model(w_hparams, w_data_config).to(device)
            w_ckpt = torch.load(w_cfg["paths"]["weights"], map_location=device)
            state_dict = {k: v for k, v in w_ckpt["model_state_dict"].items() if k != "coords"}
            w_model.load_state_dict(state_dict, strict=False)
            w_model.eval()
            models.append(("weight_diffusion", w_model, run_name))

    # ── Build two-stage WeightDiffusion variants ───────────────────────────────
    if args.two_stage_weight_config_paths:
        print(f"--- Loading Two-Stage WeightDiffusion Suite ({len(args.two_stage_weight_config_paths)} variants) ---")
        for p in args.two_stage_weight_config_paths:
            with open(p) as f:
                tsw_cfg = json.load(f)
            run_name = tsw_cfg["run_name"]
            ckpt_path = _extract_two_stage_wd_checkpoint(p, run_name)
            tsw_args = SimpleNamespace(T=tsw_cfg["T"], beta_1=tsw_cfg["beta_1"], beta_T=tsw_cfg["beta_T"])
            print(f"  Building & loading: {run_name} ...")
            tsw_model = build_full_wd_model(
                hparams=tsw_cfg,
                args=tsw_args,
                channels=channels,
                img_size=img_size,
                data_dim=data_dim,
                device=device,
            )
            tsw_ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = {k: v for k, v in tsw_ckpt["full_model_state_dict"].items() if k != "coords"}
            tsw_model.load_state_dict(state_dict, strict=False)
            tsw_model.eval()
            models.append(("two_stage_weight", tsw_model, run_name))

    # ── Shared setup ───────────────────────────────────────────────────────────
    print("\n--- Selecting interpolation pair (real images) ---")
    x1, x2 = get_interpolation_pair(val_dataset, channels, device)

    print(f"\n--- Drawing fixed PCA batch ({args.n_pca_samples} real images, shared across all models) ---")
    x_pca, y_pca = draw_fixed_pca_batch(pca_loader, args.n_pca_samples, channels)

    # ── Per-model analysis loop ────────────────────────────────────────────────
    print("\n--- Computing weight-space PCA + interpolation ---")
    alphas = torch.linspace(0, 1, args.n_interp_steps, device=device).view(-1, 1)

    for model_type, model, run_name in models:
        print(f"\n[Model: {run_name}]")

        print(f"  Extracting weight vectors ({args.n_pca_samples} real images)...")
        weight_flat_real, labels_real = collect_weight_vectors(model, model_type, x_pca, y_pca, coord_grid, device)

        w1_real = get_weight_vector(model, model_type, x1, coord_grid)
        w2_real = get_weight_vector(model, model_type, x2, coord_grid)

        print(f"  Sampling {args.n_pca_samples} weight vectors from generative process...")
        weight_flat_sample = sample_weight_vectors(
            model, model_type, args.n_pca_samples, coord_grid, device,
            vae_config=vae_config if model_type == "vae" else None,
        )

        idx1, idx2 = random.sample(range(weight_flat_sample.shape[0]), 2)
        w1_sample = torch.from_numpy(weight_flat_sample[idx1 : idx1 + 1]).to(device)
        w2_sample = torch.from_numpy(weight_flat_sample[idx2 : idx2 + 1]).to(device)

        print(f"  Interpolating in weight space (linear + slerp)...")
        w_interp_real = (1 - alphas) * w1_real + alphas * w2_real
        w_interp_sample = (1 - alphas) * w1_sample + alphas * w2_sample
        w_slerp_real = slerp(w1_real, w2_real, alphas)
        w_slerp_sample = slerp(w1_sample, w2_sample, alphas)

        print(f"  Generating PCA figure...")
        plot_weight_pca_combined(
            weight_flat_real, labels_real,
            w_interp_real.cpu().numpy(), weight_flat_sample,
            w_interp_sample.cpu().numpy(),
            title=run_name,
            save_path=os.path.join(output_dir, f"weight_pca_{_safe_name(run_name)}.png"),
            slerp_path_real_flat=w_slerp_real.cpu().numpy(),
            slerp_path_sample_flat=w_slerp_sample.cpu().numpy(),
        )

        rows = [
            (w_interp_real, "reconstruction_linear"),
            (w_interp_sample, "sample_linear"),
            (w_slerp_real, "reconstruction_slerp"),
            (w_slerp_sample, "sample_slerp"),
        ]
        for w_interp, tag in rows:
            x_hat = decode_weight_vector(model, model_type, w_interp, coord_grid)
            x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1).cpu().float()
            images = x_hat.squeeze(1).numpy() if channels == 1 else x_hat.permute(0, 2, 3, 1).numpy()
            plot_weight_interpolation_row(
                images, channels,
                title=f"{run_name} ({tag})",
                save_path=os.path.join(output_dir, f"weight_interp_{_safe_name(run_name)}_{tag}.png"),
            )

    print("\nWeight Space Analysis Complete.")


if __name__ == "__main__":
    main()