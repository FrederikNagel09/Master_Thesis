"""
weight_space_analysis.py
Weight-space analysis across all three model families: VAE-INR, Latent
Diffusion (up to 3 variants), and Weight Diffusion (up to 3 variants).

For every model, the modulated SIREN weight vector is extracted per real
image (NOT the upstream latent z) and analyzed directly in weight space:
  1. PCA scatter (real-image weight vectors, colored by class label).
     One PNG per model (no shared prior/sample background here — see
     latent_space_analysis.py for that diagnostic, which is VAE/LDM-specific).
  2. Interpolation: pick two real images of different class, get their two
     weight vectors, linearly interpolate in WEIGHT space across N steps,
     decode each step through the shared SIREN INR, plot as a row.
     One PNG per model.

Usage
-----
python src/scripts/weight_space_analysis.py \
    --vae_config_path src/results/vae_testing_beta01/vae_testing_beta01_config.json \
    --vae_checkpoint_path src/results/vae_testing_beta01/vae_testing_beta01_checkpoint.pt \
    --latent_config_paths src/train_results/Latent-Diffusion-Probabilistic-1616/metadata/config.json src/train_results/Latent-Probabilistic-two-stage/metadata/config.json\
    --weight_config_paths src/train_results/Weight-Diffusion-Probabilistic/metadata/config.json \
    --n_pca_samples 2048
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
from sklearn.decomposition import PCA

# Reuse model-building / coord-grid helpers already written for the eval suite
from src.scripts.get_all_plot_results import build_vae_model, make_coord_grid


# ── Path helper (mirrors eval_visual.py / latent_space_analysis.py) ────────────
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


def _reshape_if_flat(x: torch.Tensor, channels: int = 1) -> torch.Tensor:
    """
    Reshape a flat (B, data_dim) image tensor to (B, C, H, W) if needed.

    Args:
        x:        (B, data_dim) or (B, C, H, W) tensor.
        channels: Number of image channels (default 1 for MNIST).
    Returns:
        (B, C, H, W) tensor.
    """
    if x.dim() == 2:
        img_size = round((x.shape[1] // channels) ** 0.5)
        x = x.view(x.shape[0], channels, img_size, img_size)
    return x


def slerp(w1: torch.Tensor, w2: torch.Tensor, alphas: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical linear interpolation between two single vectors, evaluated at
    each alpha in [0, 1]. Falls back to linear interpolation when w1/w2 are
    (near) parallel or antiparallel, where the slerp formula is numerically
    unstable (sin(omega) -> 0).

    Args:
        w1:     (1, D) start vector.
        w2:     (1, D) end vector.
        alphas: (n_steps, 1) interpolation positions in [0, 1].
        eps:    Threshold on sin(omega) below which we fall back to linear interpolation.
    Returns:
        w_interp: (n_steps, D) interpolated vectors, w1 at alpha=0, w2 at alpha=1.
    """
    w1_flat = w1.reshape(-1)
    w2_flat = w2.reshape(-1)

    cos_omega = torch.dot(w1_flat, w2_flat) / (w1_flat.norm() * w2_flat.norm() + eps)
    cos_omega = cos_omega.clamp(-1.0, 1.0)  # guard against floating point drift outside [-1, 1]
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    if sin_omega.abs() < eps:
        # w1, w2 are (near) parallel/antiparallel: slerp is undefined/unstable, use linear instead
        return (1 - alphas) * w1 + alphas * w2

    coeff1 = torch.sin((1 - alphas) * omega) / sin_omega  # (n_steps, 1)
    coeff2 = torch.sin(alphas * omega) / sin_omega  # (n_steps, 1)
    return coeff1 * w1 + coeff2 * w2


# ── Weight vector extraction (branches by model family) ────────────────────────
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
    Extract flattened modulated SIREN weight vectors from a FIXED set of real
    images (shared across all models), paired with class labels.

    Args:
        model:      Trained VAEWrapper, LatentDiffusion, or WeightDiffusion model.
        model_type: "vae", "ldm", or "weight_diffusion".
        x_pca:      (N, C, H, W) fixed image batch, identical across all models.
        y_pca:      (N,) integer class labels matching x_pca.
        coord_grid: (H, W, 2) coordinate grid (only used by vae/ldm decoders).
        device:     Device string.
        batch_size: Sub-batch size for the forward pass (memory control only).
    Returns:
        weight_flat: (N, D) flattened weight vectors.
        labels:      (N,) integer class labels (same as y_pca, returned as numpy).
    """
    w_list = []
    n_total = x_pca.shape[0]

    for start in range(0, n_total, batch_size):
        x = x_pca[start : start + batch_size].to(device)

        if model_type == "weight_diffusion":
            theta_prime_raw, _, _ = model.encode(x)  # compressed code, NOT the actual weight vector
            w_flat = model.weight_encoder.decode_modulations(theta_prime_raw)  # true modulated SIREN weights
        else:
            # vae / ldm: encode to z, then run the decoder's weight-exposing forward pass
            z, _, _ = model.encode(x)
            coord_batched = coord_grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            _, w_flat = model.decoder.forward_with_weights(z, coord_batched)

        w_list.append(w_flat.cpu().numpy())

    weight_flat = np.concatenate(w_list, axis=0)
    labels = y_pca.cpu().numpy()
    return weight_flat, labels


# ── Weight vector sampling: draw from the model's OWN generative process ───────
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
    generative process (NOT from real images): N(0,I) prior + decoder for the
    VAE, reverse diffusion + decoder for LDM, reverse diffusion + decode_modulations
    for weight diffusion. Batched to bound memory.

    Args:
        model:      Trained VAEWrapper, LatentDiffusion, or WeightDiffusion model.
        model_type: "vae", "ldm", or "weight_diffusion".
        n_samples:  Total number of weight vectors to draw.
        coord_grid: (H, W, 2) coordinate grid (only used by vae/ldm decoders).
        device:     Device string.
        vae_config: Required when model_type == "vae"; needs latent_dim/latent_size.
        batch_size: Number of samples drawn per chunk (memory control).
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
            if model._normalize:  # noqa: SLF001
                z = model._denormalize_z(z)  # noqa: SLF001
            coord_batched = coord_grid.unsqueeze(0).expand(b, -1, -1, -1)
            _, w_flat = model.decoder.forward_with_weights(z, coord_batched)

        else:  # weight_diffusion
            theta_prime = model.sample_weight(b)
            w_flat = model.weight_encoder.decode_modulations(theta_prime)

        w_list.append(w_flat.cpu())
        n_remaining -= b

    return torch.cat(w_list, dim=0).numpy()


# ── Single PCA subplot helper (draws onto a given Axes, points already 2D) ─────
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
    linear and Slerp interpolation paths, onto a given matplotlib Axes.

    Args:
        ax:              Matplotlib axes to draw on.
        w_2d:            (N, 2) already-PCA-projected background points.
        labels:          (N,) integer class labels, or None for unlabeled.
        interp_path_2d:  (n_steps, 2) already-PCA-projected LINEAR interpolation path, or None.
        panel_title:     Subplot title.
        slerp_path_2d:   (n_steps, 2) already-PCA-projected SLERP interpolation path, or None.
                         Shares the same theta1/theta2 endpoints as interp_path_2d.
    Returns:
        scatter: The background scatter artist (for an optional colorbar), or None if labels is None.
    """
    scatter = None
    if labels is not None:
        n_classes = int(labels.max()) + 1
        scatter = ax.scatter(w_2d[:, 0], w_2d[:, 1], c=labels, cmap="tab10", vmin=0, vmax=n_classes - 1, s=8, alpha=0.85, linewidths=0)
    else:
        ax.scatter(w_2d[:, 0], w_2d[:, 1], color="black", s=8, alpha=0.6, linewidths=0)

    if interp_path_2d is not None:
        linear_color = "darkorange"
        ax.plot(interp_path_2d[:, 0], interp_path_2d[:, 1], linestyle="-", color=linear_color, linewidth=1.4, zorder=4, label="Linear")
        ax.scatter(interp_path_2d[1:-1, 0], interp_path_2d[1:-1, 1], color=linear_color, s=15, marker="o", zorder=5)

    if slerp_path_2d is not None:
        slerp_color = "mediumvioletred"  # distinct from linear's orange, class tab10 colors, and the neutral background
        ax.plot(slerp_path_2d[:, 0], slerp_path_2d[:, 1], linestyle="-", color=slerp_color, linewidth=1.4, zorder=4, label="Slerp")
        ax.scatter(slerp_path_2d[1:-1, 0], slerp_path_2d[1:-1, 1], color=slerp_color, s=15, marker="o", zorder=5)

    # Endpoints (shared by both paths): drawn once, in neutral gray so they read as
    # "the two real points" rather than belonging to either path's color
    if interp_path_2d is not None:
        endpoints_2d = interp_path_2d[[0, -1]]
        ax.scatter(endpoints_2d[:, 0], endpoints_2d[:, 1], color="gray", s=28, marker="o", edgecolors="black", linewidths=0.8, zorder=6)
        ax.annotate(
            r"$\theta_1$",
            endpoints_2d[0],
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            fontweight="bold",
            zorder=7,
            xytext=(0, -10),
            textcoords="offset points",
        )
        ax.annotate(
            r"$\theta_2$",
            endpoints_2d[1],
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            fontweight="bold",
            zorder=7,
            xytext=(0, -10),
            textcoords="offset points",
        )

    if interp_path_2d is not None or slerp_path_2d is not None:
        ax.legend(loc="best", fontsize=7, framealpha=0.8)

    ax.set_title(panel_title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    return scatter


# ── Combined PCA figure: reconstruction (left) + sample (right), shared basis ──
# ── Combined PCA figure: reconstruction (left) + sample (right), shared basis ──
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
    Fit PCA(2) ONCE on the real-image (reconstruction) weight vectors, then
    project the sampled weight vectors into that SAME basis — so the two
    panels are positionally comparable. Renders side-by-side: left panel =
    real-image weights (colored by class) + real-data interpolation path(s);
    right panel = model-generated weights (neutral color) + sample-data
    interpolation path(s). 
    
    Both panels display a synchronized KDE background density map estimated 
    from the real image weight vector distribution.
    """
    from scipy.stats import gaussian_kde

    pca = PCA(n_components=2)
    w_2d_real = pca.fit_transform(weight_flat_real)  # basis defined HERE, on real data only
    w_2d_sample = pca.transform(weight_flat_sample)  # projected into the SAME basis

    interp_2d_real = pca.transform(interp_path_real_flat)
    interp_2d_sample = pca.transform(interp_path_sample_flat)

    slerp_2d_real = pca.transform(slerp_path_real_flat) if slerp_path_real_flat is not None else None
    slerp_2d_sample = pca.transform(slerp_path_sample_flat) if slerp_path_sample_flat is not None else None

    # ── Shared axis limits across BOTH panels ─────────────────────────────────
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

    # ── Compute KDE Background from REAL Data ─────────────────────────────────
    xx, yy = np.mgrid[xlim[0]:xlim[1]:complex(0, grid_res), ylim[0]:ylim[1]:complex(0, grid_res)]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    kde = gaussian_kde(w_2d_real.T)
    density = kde(grid_coords).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))

    # ── Render Subplots with Unified Density Background ──────────────────────
    # Left Plot: Paint KDE density under real scatter points
    axes[0].contourf(xx, yy, density, levels=8, cmap="summer", alpha=1.0, zorder=1)
    scatter = _draw_pca_subplot(
        axes[0], w_2d_real, labels_real, interp_2d_real, "Reconstruction (real images)", slerp_path_2d=slerp_2d_real
    )
    
    # Right Plot: Paint the exact same KDE density under generative sample points
    axes[1].contourf(xx, yy, density, levels=8, cmap="summer", alpha=1.0, zorder=1)
    _draw_pca_subplot(axes[1], w_2d_sample, None, interp_2d_sample, "Sample (model-generated)", slerp_path_2d=slerp_2d_sample)

    # ── Format and Clean Axes ─────────────────────────────────────────────────
    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)", fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)  # Keep panels perfectly square
        # Ensure scatter/plots remain readable against strong alpha=1.0 contours
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


# ── Fixed PCA image batch (drawn ONCE, shared identically across all models) ───
def draw_fixed_pca_batch(loader: torch.utils.data.DataLoader, n_samples: int, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a single fixed batch of real images + labels, to be reused identically
    across every model's weight extraction (so PCA scatters are comparable).

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


# ── Interpolation pair selection: two genuinely random images, different class ─
def get_interpolation_pair(
    dataset: torch.utils.data.Dataset, channels: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pick one image at complete random and record its class label. Then keep
    picking a second image at complete random until its label differs from
    the first. No sorting, no encounter-order bias.

    Args:
        dataset:  Dataset yielding (image, label) pairs, indexable by int.
        channels: Number of image channels.
        device:   Device string.
    Returns:
        x1, x2: (1, C, H, W) tensors of two randomly-chosen, differently-labeled images.
    """
    n = len(dataset)

    idx1 = random.randrange(n)
    x1, y1 = dataset[idx1]
    label1 = int(y1)

    while True:
        idx2 = random.randrange(n)
        x2, y2 = dataset[idx2]
        label2 = int(y2)
        if label2 != label1:
            break

    x1 = _reshape_if_flat(x1.unsqueeze(0), channels).to(device)
    x2 = _reshape_if_flat(x2.unsqueeze(0), channels).to(device)
    return x1, x2


@torch.no_grad()
def get_weight_vector(
    model,
    model_type: str,
    x: torch.Tensor,
    coord_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Extract a single flattened weight vector for one image, branching by model family.

    Args:
        model:      Trained VAEWrapper, LatentDiffusion, or WeightDiffusion model.
        model_type: "vae", "ldm", or "weight_diffusion".
        x:          (1, C, H, W) input image.
        coord_grid: (H, W, 2) coordinate grid (only used by vae/ldm decoders).
    Returns:
        weight_flat: (1, D) flattened weight vector.
    """
    if model_type == "weight_diffusion":
        theta_prime_raw, _, _ = model.encode(x)
        theta = model.weight_encoder.decode_modulations(theta_prime_raw)
        return theta
    z, _, _ = model.encode(x)
    coord_batched = coord_grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    _, w_flat = model.decoder.forward_with_weights(z, coord_batched)
    return w_flat


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
        model:       Trained VAEWrapper, LatentDiffusion, or WeightDiffusion model.
        model_type:  "vae", "ldm", or "weight_diffusion".
        weight_flat: (B, D) flattened weight vectors.
        coord_grid:  (H, W, 2) coordinate grid.
    Returns:
        pixels: (B, C, H, W) decoded images, NOT yet un-normalized to [0,1].
    """
    B = weight_flat.shape[0]  # noqa: N806
    coord_batched = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

    if model_type == "weight_diffusion":
        coords_flat = coord_batched.reshape(B, -1, 2)  # _inr_decode expects (B, H, W, 2) or flat; matches trans_coord convention
        pixels_flat = model._inr_decode(weight_flat, coords=coord_batched)  # noqa: SLF001 (B, H*W*C)
        img_size = coord_grid.shape[0]
        channels = pixels_flat.shape[1] // (img_size * img_size)
        return pixels_flat.reshape(B, channels, img_size, img_size)

    # vae / ldm: unflatten weight_flat into the per-layer param dict, then query the INR directly
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


def plot_weight_interpolation_row(images: np.ndarray, channels: int, title: str, save_path: str) -> None:
    """
    Plot a single row of weight-space-interpolated reconstructions with a headline.

    Args:
        images:    (n_steps, H, W) or (n_steps, H, W, C) images in [0,1].
        channels:  Number of image channels.
        title:     Headline above the row (model name).
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
        label = "theta1 (real)" if i == 0 else "theta2 (real)" if i == n_steps - 1 else None
        if label:
            ax.set_title(label, fontsize=7)

    fig.suptitle(f"Weight-Space Interpolation: {title}", fontsize=11, fontweight="bold", y=1.05)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Weight interpolation row saved -> {save_path}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Weight space analysis (PCA + interpolation) across VAE, Latent, and Weight models.")

    parser.add_argument("--vae_config_path", type=str, required=True, help="Path to VAE _config.json.")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to VAE checkpoint .pt.")
    parser.add_argument(
        "--latent_config_paths", type=str, nargs="+", default=[], help="Paths to Latent Diffusion config.json files (Max 3)."
    )
    parser.add_argument(
        "--weight_config_paths", type=str, nargs="+", default=[], help="Paths to Weight Diffusion config.json files (Max 3)."
    )
    parser.add_argument("--n_pca_samples", type=int, default=2000, help="Number of weight vectors to gather for the PCA scatter.")
    parser.add_argument("--n_interp_steps", type=int, default=10, help="Number of points along the interpolation path (incl. endpoints).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible image/loader sampling.")

    args = parser.parse_args()

    if len(args.latent_config_paths) > 3:
        parser.error("You can provide a maximum of 3 latent_config_paths.")
    if len(args.weight_config_paths) > 3:
        parser.error("You can provide a maximum of 3 weight_config_paths.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
    from src.utility.model_builders import build_model as build_ldm_model

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
    coord_grid = make_coord_grid((img_size, img_size), (-1, 1), device=device)

    pca_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)

    # ── Build + load VAE ───────────────────────────────────────────────────────
    print("--- Loading VAE-INR Model ---")
    vae_model = build_vae_model(vae_config, channels, img_size, device)
    vae_ckpt = torch.load(args.vae_checkpoint_path, map_location=device)
    vae_model.load_state_dict(vae_ckpt["model_state_dict"])
    vae_model.eval()

    models = [("vae", vae_model, "VAE-INR")]

    # ── Build + load Latent Diffusion variants ─────────────────────────────────
    if args.latent_config_paths:
        print(f"--- Loading Latent Diffusion Suite ({len(args.latent_config_paths)} variants) ---")
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

    # ── Build + load Weight Diffusion variants ──────────────────────────────────
    if args.weight_config_paths:
        print(f"--- Loading Weight Diffusion Suite ({len(args.weight_config_paths)} variants) ---")
        for p in args.weight_config_paths:
            with open(p) as f:
                w_cfg = json.load(f)
            w_hparams = SimpleNamespace(**w_cfg["hparams"])
            w_data_cfg = w_cfg["data"]
            w_data_config = {
                "dataset": w_cfg["dataset"],
                "channels": w_data_cfg["channels"],
                "img_size": w_data_cfg["img_size"],
                "data_dim": w_data_cfg["data_dim"],
            }
            run_name = _extract_run_name(p)
            print(f"  Building & loading: {run_name} ...")
            w_model = build_ldm_model(w_hparams, w_data_config).to(device)
            w_ckpt = torch.load(w_cfg["paths"]["weights"], map_location=device)
            w_model.load_state_dict(w_ckpt["model_state_dict"])
            w_model.eval()
            models.append(("weight_diffusion", w_model, run_name))

    # ── Shared setup: both modes are always computed ────────────────────────────
    print("\n--- Selecting interpolation pair (real images) ---")
    x1, x2 = get_interpolation_pair(val_dataset, channels, device)

    print(f"\n--- Drawing fixed PCA batch ({args.n_pca_samples} real images, shared across all models) ---")
    x_pca, y_pca = draw_fixed_pca_batch(pca_loader, args.n_pca_samples, channels)

    # ── PCA (reconstruction + sample, shared basis) + interpolation rows ───────
    print("\n--- Computing weight-space PCA + interpolation ---")
    for model_type, model, run_name in models:
        # --- Reconstruction population: real images -> weight vectors ---
        print(f"  Extracting weight vectors for {run_name} ({args.n_pca_samples} real images) ...")
        weight_flat_real, labels_real = collect_weight_vectors(model, model_type, x_pca, y_pca, coord_grid, device)

        # theta1/theta2 (real): same two real images (x1, x2) for every model,
        # each model encodes them into its OWN weight vectors
        w1_real = get_weight_vector(model, model_type, x1, coord_grid)
        w2_real = get_weight_vector(model, model_type, x2, coord_grid)

        # --- Sample population: model's own generative process -> weight vectors ---
        print(f"  Sampling {args.n_pca_samples} weight vectors from {run_name}'s generative process ...")
        weight_flat_sample = sample_weight_vectors(
            model, model_type, args.n_pca_samples, coord_grid, device, vae_config=vae_config if model_type == "vae" else None
        )
        # theta1/theta2 (sample): two vectors drawn at random from the sampled pool above
        idx1, idx2 = random.sample(range(weight_flat_sample.shape[0]), 2)
        w1_sample = torch.from_numpy(weight_flat_sample[idx1 : idx1 + 1]).to(device)
        w2_sample = torch.from_numpy(weight_flat_sample[idx2 : idx2 + 1]).to(device)

        print(f"  Interpolating in weight space with {run_name} (linear + slerp) ...")
        alphas = torch.linspace(0, 1, args.n_interp_steps, device=device).view(-1, 1)
        w_interp_real = (1 - alphas) * w1_real + alphas * w2_real  # (n_steps, D)
        w_interp_sample = (1 - alphas) * w1_sample + alphas * w2_sample  # (n_steps, D)

        w_slerp_real = slerp(w1_real, w2_real, alphas)  # (n_steps, D), same endpoints as w_interp_real
        w_slerp_sample = slerp(w1_sample, w2_sample, alphas)  # (n_steps, D), same endpoints as w_interp_sample

        plot_weight_pca_combined(
            weight_flat_real,
            labels_real,
            w_interp_real.cpu().numpy(),
            weight_flat_sample,
            w_interp_sample.cpu().numpy(),
            title=run_name,
            save_path=os.path.join(output_dir, f"weight_pca_{_safe_name(run_name)}.png"),
            slerp_path_real_flat=w_slerp_real.cpu().numpy(),
            slerp_path_sample_flat=w_slerp_sample.cpu().numpy(),
        )

        # --- Decode + plot all four interpolation rows (linear/slerp x real/sample) ---
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
                images,
                channels,
                title=f"{run_name} ({tag})",
                save_path=os.path.join(output_dir, f"weight_interp_{_safe_name(run_name)}_{tag}.png"),
            )

    print("\nWeight Space Analysis Complete.")


if __name__ == "__main__":
    main()