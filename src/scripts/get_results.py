"""
eval_visual.py
Generates two evaluation plots for a trained INR-based model:
  1. A single-row of N upscaled samples at a custom resolution.
  2. Side-by-side comparison of upscaled reconstructions (top row)
     vs. bilinear/bicubic interpolated originals (bottom row).

Usage
-----
python src/scripts/get_results.py \
    --config_path src/train_results/Latent-Diffusion-Deterministic-new/metadata/config.json \
    --sample_scale 128 \
    --n_samples 10 \
    --n_recon 8 \
    --interp_mode bilinear
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(".")

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    import numpy as np

warnings.filterwarnings("ignore", message="The operator 'aten::im2col' is not currently supported on the MPS backend")
# ── Coord grid (mirrors make_coord_grid in the decoder) ───────────────────────


def make_coord_grid(shape: tuple[int, ...], range: list | tuple, device=None) -> torch.Tensor:
    """
    Build a coordinate grid matching the model's internal convention.

    Args:
        shape:  Spatial resolution, e.g. (H, W).
        range:  Coordinate range [minv, maxv] or per-dim [[min0,max0], ...].
        device: Target device.
    Returns:
        grid: (*shape, len(shape)) float tensor.
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s  # noqa: E741
        if isinstance(range[0], (list, tuple)):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l  # noqa: E741
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid


# ── Sampling at custom resolution ─────────────────────────────────────────────


@torch.no_grad()
def sample_at_scale(
    model,
    n_samples: int,
    scale: int,
    device: str,
    channels: int,
) -> np.ndarray:
    """
    Sample from the model and decode at a custom spatial resolution.

    Args:
        model:     Trained model with _sample_latent, _denormalize_z, decoder.
        n_samples: Number of images to generate.
        scale:     Target H=W resolution.
        device:    Device string.
        channels:  Number of image channels.
    Returns:
        images: (n_samples, scale, scale) for grayscale or (n_samples, scale, scale, 3) for RGB, float32 in [0,1].
    """
    # Build custom coord grid at target scale
    coord = make_coord_grid((scale, scale), (-1, 1), device=device)  # (scale, scale, 2)

    # Sample latents exactly as the model normally would
    z = model._sample_latent(n_samples, collect_snapshots=False, debug=False)
    if model._normalize:
        z = model._denormalize_z(z)

    # Decode with custom coord instead of model's fixed self.coord_grid
    x_hat = model.decoder(z, coord)  # (B, C, scale, scale)
    x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)  # un-normalize to [0,1]

    x_hat = x_hat.cpu().float()
    if channels == 1:
        return x_hat.squeeze(1).numpy()  # (N, H, W)
    else:
        return x_hat.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)


# ── Reconstruction at custom resolution ───────────────────────────────────────


@torch.no_grad()
def reconstruct_at_scale(
    model,
    x: torch.Tensor,
    scale: int,
    device: str,
    channels: int,
) -> np.ndarray:
    """
    Encode a batch of images and decode at a custom spatial resolution.

    Args:
        model:    Trained model with encode() and decoder.
        x:        (B, C, H, W) input images on device.
        scale:    Target H=W resolution for the reconstruction.
        device:   Device string.
        channels: Number of image channels.
    Returns:
        recons: (B, scale, scale) or (B, scale, scale, C) float32 in [0,1].
    """
    coord = make_coord_grid((scale, scale), (-1, 1), device=device)  # (scale, scale, 2)

    # Val loader returns flat (B, data_dim) — reshape to (B, C, H, W) for the encoder
    if x.dim() == 2:
        img_size = round((x.shape[1] // channels) ** 0.5)
        x = x.view(x.shape[0], channels, img_size, img_size)

    z_raw, _, _ = model.encode(x)
    x_hat = model.decoder(z_raw, coord)  # (B, C, scale, scale)
    x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)

    x_hat = x_hat.cpu().float()
    if channels == 1:
        return x_hat.squeeze(1).numpy()
    else:
        return x_hat.permute(0, 2, 3, 1).numpy()


# ── Path helper ───────────────────────────────────────────────────────────────


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
        raise ValueError(  # noqa: B904
            f"Could not extract run name from: {config_path}\n" "Expected: .../<run_name>/metadata/config.json"
        )


# ── Plot 1: upscaled sample row ───────────────────────────────────────────────


def plot_sample_row(
    model,
    n_samples: int,
    scale: int,
    device: str,
    channels: int,
    epoch: int,
    run_dir: str,
) -> None:
    """
    Generate and save a single-row plot of upscaled model samples.

    Args:
        model:     Trained model.
        n_samples: Number of samples in the row.
        scale:     Target resolution (scale x scale).
        device:    Device string.
        channels:  Number of image channels.
        epoch:     Epoch number (used in filename).
        run_dir:   Output directory.
    Returns:
        None
    """
    print(f"  Generating {n_samples} samples at {scale}x{scale} …")
    images = sample_at_scale(model, n_samples, scale, device, channels)

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 1.5, 1.5))
    for ax, img in zip(axes, images):  # noqa: B905
        if channels == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")

    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    save_path = os.path.join(run_dir, f"samples_upscaled_{scale}x{scale}_ep{epoch}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Upscaled samples saved → {save_path}")


# ── Plot 2: reconstruction vs interpolated original ───────────────────────────


def plot_recon_vs_interp(
    model,
    val_loader: torch.utils.data.DataLoader,
    n_images: int,
    scale: int,
    device: str,
    channels: int,
    interp_mode: str,
    epoch: int,
    run_dir: str,
) -> None:
    """
    Two-row comparison: top = model reconstructions at `scale`, bottom = interpolated originals at `scale`.

    Args:
        model:       Trained model.
        val_loader:  Validation DataLoader for sourcing original images.
        n_images:    Number of images in each row.
        scale:       Target resolution for both reconstruction and interpolation.
        device:      Device string.
        channels:    Number of image channels.
        interp_mode: Interpolation mode for upscaling originals, 'bilinear' or 'bicubic'.
        epoch:       Epoch number (used in filename).
        run_dir:     Output directory.
    Returns:
        None
    """
    print(f"  Fetching {n_images} validation images …")

    # Grab exactly n_images from the val loader
    x_batch = []
    for batch in val_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_batch.append(imgs)
        if sum(b.shape[0] for b in x_batch) >= n_images:
            break
    x = torch.cat(x_batch, dim=0)[:n_images]  # (N, C, H, W)

    # Reshape if flat — derive img_size from tensor shape, no config needed
    if x.dim() == 2:
        img_size = round((x.shape[1] // channels) ** 0.5)
        x = x.view(x.shape[0], channels, img_size, img_size)

    x = x.to(device)

    # ── Reconstructions at target scale ───────────────────────────────────────
    print(f"  Reconstructing at {scale}x{scale} …")
    recons = reconstruct_at_scale(model, x, scale, device, channels)

    # ── Interpolate originals to target scale ─────────────────────────────────
    print(f"  Interpolating originals to {scale}x{scale} ({interp_mode}) …")
    align = False if interp_mode == "nearest" else True  # noqa: SIM211
    originals_up = F.interpolate(
        x.cpu().float(),
        size=(scale, scale),
        mode=interp_mode,
        align_corners=align,
    )
    # Un-normalize: training data is in [-1,1], bring to [0,1]
    originals_up = (originals_up * 0.5 + 0.5).clamp(0, 1)

    if channels == 1:  # noqa: SIM108
        originals_up = originals_up.squeeze(1).numpy()  # (N, H, W)
    else:
        originals_up = originals_up.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)

    # ── Unscaled originals (native resolution) ───────────────────────────────
    originals_native = (x.cpu().float() * 0.5 + 0.5).clamp(0, 1)
    if channels == 1:  # noqa: SIM108
        originals_native = originals_native.squeeze(1).numpy()  # (N, H, W)
    else:
        originals_native = originals_native.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)

    # ── Build 3-row figure with a title text row above each image row ─────────
    # Layout: 6 rows total — alternating title row (small) + image row
    n_rows = 3
    title_h = 0.18  # relative height for each title row
    image_h = 1.0  # relative height for each image row
    row_heights = []
    for _ in range(n_rows):
        row_heights += [title_h, image_h]

    fig, axes = plt.subplots(
        n_rows * 2,
        n_images,
        figsize=(n_images * 1.5, n_rows * 1.5 + 0.2),
        gridspec_kw={"height_ratios": row_heights},
    )

    rows_data = [
        (f"Reconstruction ({scale}x{scale})", recons),
        (f"Original upscaled {scale}x{scale} ({interp_mode})", originals_up),
        ("Original (native resolution)", originals_native),
    ]

    for group_idx, (label, imgs) in enumerate(rows_data):
        title_row = group_idx * 2  # 0, 2, 4
        image_row = group_idx * 2 + 1  # 1, 3, 5

        # Title spans all columns — use leftmost ax, hide the rest
        for col in range(n_images):
            axes[title_row, col].axis("off")
        axes[title_row, 0].text(
            0.0,
            0.5,
            label,
            transform=axes[title_row, 0].transAxes,
            fontsize=7,
            va="center",
            ha="left",
            fontweight="bold",
        )

        # Image row
        for col in range(n_images):
            ax = axes[image_row, col]
            img = imgs[col]
            if channels == 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

    plt.subplots_adjust(hspace=0.03, wspace=0.02)
    save_path = os.path.join(run_dir, f"recon_vs_interp_{scale}x{scale}_ep{epoch}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Recon vs interp saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Upscaled sample + reconstruction vs interpolation plots.")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--sample_scale", type=int, default=32, help="Target resolution for upscaled samples and reconstructions.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples in the sample row.")
    parser.add_argument("--n_recon", type=int, default=8, help="Number of images in the reconstruction comparison.")
    parser.add_argument(
        "--interp_mode", type=str, default="bicubic", choices=["bilinear", "bicubic"], help="Interpolation method for upscaling originals."
    )
    args = parser.parse_args()

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
    from src.utility.model_builders import build_model

    with open(args.config_path) as f:
        config = json.load(f)

    hparams = SimpleNamespace(**config["hparams"])
    data_cfg = config["data"]
    data_config = {
        "dataset": config["dataset"],
        "channels": data_cfg["channels"],
        "img_size": data_cfg["img_size"],
        "data_dim": data_cfg["data_dim"],
    }
    epoch = config["epochs"]["end"]
    channels = data_config["channels"]

    device = _get_device()
    run_name = _extract_run_name(args.config_path)
    run_dir = os.path.join("src", "results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Eval Visual  |  run={run_name}  |  device={device}")
    print(f"  Scale={args.sample_scale}  |  n_samples={args.n_samples}  |  n_recon={args.n_recon}")
    print(f"  Output dir: {run_dir}")
    print(f"{'=' * 60}\n")

    # ── Build & load model ────────────────────────────────────────────────────
    print("  Building model …")
    model = build_model(hparams, data_config).to(device)

    weights_path = config["paths"]["weights"]
    print(f"  Loading weights from {weights_path} …")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Val loader (for reconstructions) ─────────────────────────────────────
    print("  Building validation dataset …")
    _, val_dataset, _ = build_dataset(
        dataset_name=data_config["dataset"],
        data_root=hparams.data_root,
        subset_frac=hparams.subset_frac,
        single_class=hparams.single_class,
        single_class_label=hparams.single_class_label,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.n_recon,
        shuffle=True,
        drop_last=False,
        num_workers=hparams.num_workers,
    )

    # ── Run plots ─────────────────────────────────────────────────────────────
    plot_sample_row(
        model=model,
        n_samples=args.n_samples,
        scale=args.sample_scale,
        device=device,
        channels=channels,
        epoch=epoch,
        run_dir=run_dir,
    )

    plot_recon_vs_interp(
        model=model,
        val_loader=val_loader,
        n_images=args.n_recon,
        scale=args.sample_scale,
        device=device,
        channels=channels,
        interp_mode=args.interp_mode,
        epoch=epoch,
        run_dir=run_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
