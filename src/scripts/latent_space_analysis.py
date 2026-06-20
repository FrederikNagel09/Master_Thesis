"""
latent_space_analysis.py
Latent-space analysis for the VAE-INR model and up to three Latent Diffusion
variants:
  1. PCA scatter (real-image encodings, colored by class label) with a KDE
     density-background, one subplot per model, in a single 2x2 figure.
  2. Per-model interpolation: pick two real images of different class, encode
     to z1/z2, linearly interpolate in latent space, decode all 10 points,
     plot as a single row.

Usage
-----
python src/scripts/latent_space_analysis.py \
    --vae_config_path src/train_results2/vae_testing/vae_testing_config.json \
    --vae_checkpoint_path src/train_results2/vae_testing/vae_testing_checkpoint.pt \
    --latent_config_paths src/train_results/Latent-Diffusion-Deterministic/metadata/config.json src/train_results/Latent-Diffusion-Probabilistic-1616/metadata/config.json src/train_results/Latent-Diffusion-Probabilistic-3212/metadata/config.json\
    --n_pca_samples 800 \
    --n_background_samples 2048 \
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

# Reuse model-building / coord-grid helpers already written for the eval suite
from src.scripts.get_all_plot_results import build_vae_model, make_coord_grid

PCA_HEADLINES_DEFAULT = ["(a) VAE-INR", "(b) Latent Deterministic", "(c) Latent Probabilistic", "(d) Latent Two-Stage"]
INTERP_TITLES_DEFAULT = ["VAE-INR", "Latent Deterministic", "Latent Probabilistic", "Latent Two-Stage"]


# ── Path helper (mirrors eval_visual.py) ───────────────────────────────────────
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
        model:      Trained VAEWrapper or LatentDiffusion model.
        model_type: "vae" or "ldm".
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
        if model_type == "vae":
            z, _, _ = model.encode(x)  # VAEWrapper.encode -> (mu, None, None)
        else:
            z, _, _ = model.encode(x)  # LatentDiffusion.encode -> (z_raw, mu, logvar)

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
) -> np.ndarray:
    """
    Draw latent samples from the model's generative process — N(0,I) prior for
    the VAE, reverse diffusion for the LDM — to use as an independent density
    background (NOT derived from the scattered real-data points).

    Args:
        model:      Trained VAEWrapper or LatentDiffusion model.
        model_type: "vae" or "ldm".
        n_samples:  Number of latent samples to draw.
        device:     Device string.
        vae_config: Required when model_type == "vae"; needs latent_dim/latent_size.
    Returns:
        z_flat: (n_samples, D) flattened latent vectors.
    """
    if model_type == "vae":
        latent_dim = vae_config["latent_dim"]
        latent_size = vae_config["latent_size"]
        z = torch.randn(n_samples, latent_dim, latent_size, latent_size, device=device)
    else:
        # LatentDiffusion: full reverse-diffusion sampling, pre-decode latent only
        z = model._sample_latent(n_samples, collect_snapshots=False, debug=False)
        if model._normalize:
            z = model._denormalize_z(z)

    return z.reshape(z.shape[0], -1).cpu().numpy()


# ── PCA + KDE background for a single model (one subplot) ──────────────────────
def plot_pca_subplot(
    ax: plt.Axes,
    z_flat: np.ndarray,
    labels: np.ndarray,
    z_background: np.ndarray,
    title: str,
    grid_res: int = 150,
) -> None:
    """
    Fit PCA(2) on real encoded data, project independent model-drawn samples
    into that same basis for a KDE density background, then scatter the real
    points (colored by class label) on top.

    Args:
        ax:           Matplotlib axes to draw on.
        z_flat:       (N, D) flattened latents from real images (defines PCA basis, scattered).
        labels:       (N,) integer class labels for z_flat.
        z_background: (M, D) flattened latents drawn from the model's own
                      generative process (prior for VAE, reverse diffusion for LDM).
                      Projected into the same PCA basis, used only for the KDE background.
        title:        Subplot headline.
        grid_res:     Resolution of the KDE background grid (per axis).
    Returns:
        None
    """
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_flat)  # (N, 2) — basis defined by real encoded data
    bg_2d = pca.transform(z_background)  # (M, 2) — independent generative samples, same basis

    # ── KDE background: density of the model's OWN generative samples ─────────
    # (independent of the scattered points below — not circular)
    x_min, x_max = bg_2d[:, 0].min(), bg_2d[:, 0].max()
    y_min, y_max = bg_2d[:, 1].min(), bg_2d[:, 1].max()
    x_pad = 0.1 * (x_max - x_min + 1e-8)
    y_pad = 0.1 * (y_max - y_min + 1e-8)

    xx, yy = np.mgrid[x_min - x_pad : x_max + x_pad : complex(0, grid_res), y_min - y_pad : y_max + y_pad : complex(0, grid_res)]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    kde = gaussian_kde(bg_2d.T)
    density = kde(grid_coords).reshape(xx.shape)

    ax.contourf(xx, yy, density, levels=15, cmap="viridis", alpha=0.5)

    # ── Scatter, colored by class label ────────────────────────────────────────
    n_classes = int(labels.max()) + 1
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", vmin=0, vmax=n_classes - 1, s=6, alpha=0.85, linewidths=0)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=8)
    ax.tick_params(labelsize=7)

    return scatter


# ── 2x2 PCA grid across all models ─────────────────────────────────────────────
def plot_pca_grid(
    data_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    headlines: list[str],
    save_path: str,
) -> None:
    """
    Build a 2x2 grid of PCA+KDE subplots, one per model.

    Args:
        data_triples: List of (z_flat, labels, z_background) tuples, one per
                       model (max 4). z_background is drawn from the model's own
                       generative process (prior for VAE, reverse diffusion for LDM).
        headlines:     Subplot titles, same length/order as data_triples.
        save_path:     Output PNG path.
    Returns:
        None
    """
    n_models = len(data_triples)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    last_scatter = None
    for idx in range(4):
        if idx < n_models:
            z_flat, labels, z_background = data_triples[idx]
            last_scatter = plot_pca_subplot(axes[idx], z_flat, labels, z_background, headlines[idx])
        else:
            axes[idx].axis("off")  # empty cell if fewer than 4 models

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes, location="right", shrink=0.7, pad=0.02)
        cbar.set_label("Digit class", fontsize=9)
        cbar.set_ticks(range(10))

    fig.suptitle("Latent Space PCA Projections (background = model-generated sample density)", fontsize=13, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA grid saved -> {save_path}")


# ── Interpolation: encode two images, lerp in latent space, decode all ─────────
@torch.no_grad()
def get_interpolation_pair(loader: torch.utils.data.DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull two images of different class from the loader.

    Args:
        loader: DataLoader yielding (image, label) batches.
        device: Device string.
    Returns:
        x1, x2: (1, C, H, W) tensors of two differently-labeled images.
    """
    found = {}
    for x, y in loader:
        if x.dim() == 2:
            img_size = round(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], 1, img_size, img_size)
        for i in range(x.shape[0]):
            label = int(y[i].item())
            if label not in found:
                found[label] = x[i : i + 1]
            if len(found) >= 2:
                labels = sorted(found.keys())[:2]
                return found[labels[0]].to(device), found[labels[1]].to(device)
    raise RuntimeError("Could not find two distinct classes in the loader.")


@torch.no_grad()
def interpolate_and_decode(
    model,
    model_type: str,  # noqa: ARG001
    x1: torch.Tensor,
    x2: torch.Tensor,
    n_steps: int,
    coord_grid: torch.Tensor,
    channels: int,
    device: str,
) -> np.ndarray:
    """
    Encode x1/x2, linearly interpolate z in latent space, decode each step.

    Args:
        model:      Trained VAEWrapper or LatentDiffusion model.
        model_type: "vae" or "ldm".
        x1, x2:     (1, C, H, W) endpoint images.
        n_steps:    Total number of points including both endpoints.
        coord_grid: (H, W, 2) coordinate grid for the decoder.
        channels:   Number of image channels.
        device:     Device string.
    Returns:
        images: (n_steps, H, W) or (n_steps, H, W, C) decoded images in [0,1].
    """
    z1, _, _ = model.encode(x1)
    z2, _, _ = model.encode(x2)

    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, *([1] * (z1.dim() - 1)))
    z_interp = (1 - alphas) * z1 + alphas * z2  # (n_steps, *z.shape[1:])

    x_hat = model.decoder(z_interp, coord_grid)
    x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)

    x_hat = x_hat.cpu().float()
    if channels == 1:
        return x_hat.squeeze(1).numpy()
    return x_hat.permute(0, 2, 3, 1).numpy()


def plot_interpolation_row(images: np.ndarray, channels: int, title: str, save_path: str) -> None:
    """
    Plot a single row of interpolated reconstructions with a headline.

    Args:
        images:    (n_steps, H, W) or (n_steps, H, W, C) images in [0,1].
        channels:  Number of image channels.
        title:     Headline above the row (e.g. model name).
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
        label = "z1 (real)" if i == 0 else "z2 (real)" if i == n_steps - 1 else None
        if label:
            ax.set_title(label, fontsize=7)

    fig.suptitle(f"Latent Interpolation: {title}", fontsize=11, fontweight="bold", y=1.05)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Interpolation row saved -> {save_path}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Latent space analysis (PCA + interpolation) for VAE and Latent Diffusion models.")

    parser.add_argument("--vae_config_path", type=str, required=True, help="Path to VAE _config.json.")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to VAE checkpoint .pt.")
    parser.add_argument(
        "--latent_config_paths", type=str, nargs="+", default=[], help="Paths to Latent Diffusion config.json files (Max 3)."
    )
    parser.add_argument("--n_pca_samples", type=int, default=2000, help="Number of real images to encode for the PCA scatter.")
    parser.add_argument(
        "--n_background_samples",
        type=int,
        default=2000,
        help="Number of samples drawn from the model's own generative process (prior/reverse diffusion) for the KDE background.",
    )
    parser.add_argument("--n_interp_steps", type=int, default=10, help="Number of points along the interpolation path (incl. endpoints).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible image/loader sampling.")

    args = parser.parse_args()

    if len(args.latent_config_paths) > 3:
        parser.error("You can provide a maximum of 3 latent_config_paths.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
    from src.utility.model_builders import build_model as build_ldm_model

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

    # Two loaders: a large-batch one for PCA collection, a small one for interpolation pair fetching
    pca_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)
    interp_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

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

    n_models = len(models)
    pca_headlines = PCA_HEADLINES_DEFAULT[:n_models]
    interp_titles = INTERP_TITLES_DEFAULT[:n_models]

    # ── PCA + KDE for each model ────────────────────────────────────────────────
    print("\n--- Computing PCA projections ---")
    data_triples = []
    for model_type, model, run_name in models:
        print(f"  Encoding {args.n_pca_samples} real images with {run_name} ...")
        z_flat, labels = collect_latents(model, model_type, pca_loader, args.n_pca_samples, device)

        print(f"  Drawing {args.n_background_samples} generative samples with {run_name} ...")
        z_background = sample_background_latents(
            model, model_type, args.n_background_samples, device, vae_config=vae_config if model_type == "vae" else None
        )

        data_triples.append((z_flat, labels, z_background))

    plot_pca_grid(data_triples, pca_headlines, save_path=os.path.join(output_dir, "latent_pca_grid.png"))

    # ── Interpolation for each model ────────────────────────────────────────────
    print("\n--- Computing interpolations ---")
    x1, x2 = get_interpolation_pair(interp_loader, device)

    for (model_type, model, run_name), title in zip(models, interp_titles, strict=False):
        print(f"  Interpolating with {run_name} ...")
        images = interpolate_and_decode(model, model_type, x1, x2, args.n_interp_steps, coord_grid, channels, device)
        safe_name = run_name.lower().replace(" ", "_").replace("-", "_")
        plot_interpolation_row(images, channels, title=title, save_path=os.path.join(output_dir, f"interpolation_{safe_name}.png"))

    print("\nLatent Space Analysis Complete.")


if __name__ == "__main__":
    main()
