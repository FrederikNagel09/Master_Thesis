"""
eval_visual.py
Generates two evaluation plots for a trained INR-based model:
  1. A single-row of N upscaled samples at a custom resolution.
  2. Side-by-side comparison of upscaled reconstructions (top row)
     vs. bilinear/bicubic interpolated originals (bottom row).

Usage
-----
python src/scripts/get_results.py \
    --config_path src/train_results/Latent-Diffusion-Deterministic/metadata/config.json \
    --sample_scale 128 \
    --n_samples 10 \
    --n_recon 8 \
    --interp_mode bilinear

Usage
-----
python src/scripts/get_all_plot_results.py \
    --vae_config_path src/results/VAE_Baseline/VAE_Baseline_config.json \
    --vae_checkpoint_path src/results/VAE_Baseline/VAE_Baseline_checkpoint.pt \
    --latent_config_paths src/train_results/latent-diffusion/metadata/config.json src/train_results/latent_two_stage_fixed/latent_two_stage_fixed_ldm_config.json src/train_results/two_stage_convergence/two_stage_convergence_ldm_config.json\
    --weight_config_paths src/train_results/weight-diffusion/metadata/config.json src/train_results/wd_two_stage_fixed/wd_two_stage_fixed_wd_config.json src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_wd_config.json \
    --sample_scale 128

python src/scripts/get_all_plot_results.py \
    --vae_config_path src/train_results2/vae_testing/vae_testing_config.json \
    --vae_checkpoint_path src/train_results2/vae_testing/vae_testing_checkpoint.pt \
    --latent_config_paths src/train_results/Test-Latent-1/metadata/config.json src/train_results/Test-Latent-2/metadata/config.json \
    --weight_config_paths src/train_results/Test-Weight-1/metadata/config.json src/train_results/Test-Weight-2/metadata/config.json \
    --sample_scale 128
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import matplotlib.gridspec as gridspec

sys.path.append(".")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from src.models.two_stage_models.latent_two_stage import TwoStageLDM

warnings.filterwarnings(
    "ignore",
    message="The operator 'aten::im2col' is not currently supported on the MPS backend",
)


HEADLINES_VAE = ["(a) Originals", "(b) VAE-INR Reconstructions"]

HEADLINES_LATENT = [
    "(a) Originals",
    "(b) Latent One-Stage",
    "(c) Latent Two-Stage Fixed",
    "(d) Latent Two-Stage Convergence",
]

HEADLINES_WEIGHT = [
    "(a) Originals",
    "(b) Weight One-Stage",
    "(c) Weight Two-Stage Fixed",
    "(d) Weight Two-Stage Convergence",
]

HEADLINE_VAE_SAMPLES = "VAE-INR Samples"

HEADLINES_LATENT_SAMPLES = [
    "(a) Latent One-Stage",
    "(b) Latent Two-Stage Fixed",
    "(c) Latent Two-Stage Convergence",
]

HEADLINES_WEIGHT_SAMPLES = [
    "(a) Weight One-Stage",
    "(b) Weight Two-Stage Fixed",
    "(c) Weight Two-Stage Convergence",
]

NAMES = ["One-Stage", "Two-Stage Fixed", "Two-Stage Convergence"]


# ── VAE 3-Panel Generation Comparison Grid ────────────────────────────────────
def plot_vae_sample_grid(
    model,
    vae_config: dict,
    input_scale: int,
    device: str,
    channels: int,
    save_path: str,
) -> None:
    """
    Samples 25 images and displays them as three 5x5 grids side-by-side.
    """
    GRID_SIDE = 6  # noqa: N806

    N_GRID = GRID_SIDE * GRID_SIDE  # 25 samples  # noqa: N806

    latent_dim = vae_config["latent_dim"]
    latent_size = vae_config["latent_size"]
    base_res = vae_config.get("img_size", 28)

    # Sample 25 latent codes
    z = torch.randn(N_GRID, latent_dim, latent_size, latent_size, device=device)
    scales = [base_res, input_scale, input_scale * 2]
    panel_imgs = []

    with torch.no_grad():
        for s in scales:
            coord = make_coord_grid((s, s), (-1, 1), device=device)
            x_hat = model.decoder(z, coord)
            x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
            panel_imgs.append(_to_numpy_images(x_hat, channels))

    fig = plt.figure(figsize=(20, 5.8))

    # Outer layout for the 3 panels
    outer_gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.18)

    for p_idx in range(3):
        # Inner 5x5 grid space layout
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            GRID_SIDE, GRID_SIDE, subplot_spec=outer_gs[p_idx], wspace=0.0, hspace=0.0
        )

        for idx in range(N_GRID):
            r = idx // GRID_SIDE
            c = idx % GRID_SIDE
            ax = fig.add_subplot(inner_gs[r, c])
            img = panel_imgs[p_idx][idx]

            if channels == 1:
                ax.imshow(
                    img,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    aspect="auto",
                )
            else:
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            ax.axis("off")

    # Centered headline labels adjusted snugly beneath the 5x5 blocks
    fig.text(
        0.24,
        0.08,
        f"{base_res}x{base_res}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.515,
        0.08,
        f"{input_scale}x{input_scale}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.79,
        0.08,
        f"{input_scale*2}x{input_scale*2}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"  VAE 5x5 scale grid saved → {save_path}")


# ── Multi-Model Variant Generation Grid ────────────────────────────────────────
def plot_multi_sample_grid(
    model,
    cfg: dict,
    model_type: str,
    input_scale: int,
    device: str,
    channels: int,
    save_path: str,
) -> None:
    """
    Generates a 6x6 scale comparison grid for a specific variant.
    Samples once, decodes at three scales so images are identical across panels.
    """
    GRID_SIDE = 6  # noqa: N806
    N_GRID = GRID_SIDE * GRID_SIDE  # noqa: N806

    base_res = cfg.get("data", {}).get("img_size", 28)
    hparams = cfg.get("hparams", cfg)
    scales = [base_res, input_scale, input_scale * 2]

    # ── Sample ONCE, decode at each scale ─────────────────────────────────────
    with torch.no_grad():
        if model_type == "ldm":
            z = model._sample_latent(N_GRID) if isinstance(model, TwoStageLDM) \
                else model._sample_latent(N_GRID, collect_snapshots=False, debug=False)
        elif model_type == "weight_diffusion":
            theta_prime = model.sample_weight(N_GRID)
            theta = model.weight_encoder.decode_modulations(theta_prime)
        else:
            # VAE: sample z ~ N(0, I)
            latent_dim = hparams["latent_dim"] if isinstance(hparams, dict) else hparams.latent_dim
            latent_size = hparams["latent_size"] if isinstance(hparams, dict) else hparams.latent_size
            z = torch.randn(N_GRID, latent_dim, latent_size, latent_size, device=device)

        panel_imgs = []
        for s in scales:
            coord = make_coord_grid((s, s), (-1, 1), device=device)
            if model_type == "ldm":
                x_hat = model.decoder(z, coord)
                x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
            elif model_type == "weight_diffusion":
                coord_batched = coord.unsqueeze(0).expand(N_GRID, -1, -1, -1)
                pixels = model._inr_decode(theta, coords=coord_batched)
                x_hat = pixels.reshape(N_GRID, channels, s, s)
                x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
            else:
                x_hat = model.decoder(z, coord)
                x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
            panel_imgs.append(_to_numpy_images(x_hat, channels))

    # ── Render figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 5.8))
    outer_gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.18)

    for p_idx in range(3):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            GRID_SIDE, GRID_SIDE, subplot_spec=outer_gs[p_idx], wspace=0.0, hspace=0.0
        )
        for idx in range(N_GRID):
            r = idx // GRID_SIDE
            c = idx % GRID_SIDE
            ax = fig.add_subplot(inner_gs[r, c])
            img = panel_imgs[p_idx][idx]
            if channels == 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            else:
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            ax.axis("off")

    fig.text(0.24, 0.08, f"{base_res}x{base_res}", ha="center", va="center", fontsize=11, fontweight="bold")
    fig.text(0.515, 0.08, f"{input_scale}x{input_scale}", ha="center", va="center", fontsize=11, fontweight="bold")
    fig.text(0.79, 0.08, f"{input_scale*2}x{input_scale*2}", ha="center", va="center", fontsize=11, fontweight="bold")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"  Variant 6x6 grid saved → {save_path}")


# ── Multi-Variant Training Curves (Dynamic Min/Max Y-limits) ──────────────────
def plot_multi_training_curves(
    config_paths: list[str],
    configs: list[dict],
    model_type: str,  # noqa: ARG001
    save_path: str,
    plot_every_n: int = 100,
    total_ylim: float = 1.0,
    diff_ylim: float = 1.0,
    rec_ylim: float = 1.0,
) -> None:
    """
    Plots 'total', 'diff', and 'rec' losses from all variants.
    The y-axis minimum dynamically adjusts to the lowest point plotted across models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ("total", "Total Loss", total_ylim),
        ("diff", "Diffusion Loss (diff)", diff_ylim),
        ("rec", "Reconstruction Loss (rec)", rec_ylim),
    ]

    # First pass: Load all histories so we can compute global minimums per panel
    loaded_histories = []
    for c_path in config_paths:
        base_dir = os.path.dirname(c_path)
        history_path = os.path.join(base_dir, "training_graph_data.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                loaded_histories.append(json.load(f))
        else:
            loaded_histories.append(None)

    # Second pass: Draw panels and apply dynamic ymin
    for p_idx, (key, title, y_max) in enumerate(panels):
        ax = axes[p_idx]
        ax.set_title(title)

        all_plotted_values = []

        for idx, (cfg, history) in enumerate(
            zip(configs, loaded_histories, strict=False)
        ):
            if history is None:
                continue

            run_label = cfg.get("run_name", f"Variant {idx+1}")
            steps = history.get("steps", [])
            data = history.get(key, [])

            if steps and data and any(v != 0.0 for v in data):
                plot_steps = steps[::plot_every_n]
                downsampled_data = data[::plot_every_n]

                ax.plot(
                    plot_steps,
                    downsampled_data,
                    label=run_label,
                    linewidth=1.0,
                    alpha=0.85,
                )
                all_plotted_values.extend(downsampled_data)

        # Calculate dynamic y-axis minimum if data exists
        if all_plotted_values:
            min_val = min(all_plotted_values)
            # Add a small 5% margin below the min value so it doesn't hug the bottom edge
            y_min = min_val - 0.05 * abs(min_val)
            # Clip at 0 if your loss metric stays strictly positive
            if min_val >= 0:
                y_min = max(0.0, y_min)
        else:
            y_min = 0.0

        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss")
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Multi-variant training curves saved → {save_path}")


# ── Multi-Model Sample Rows Plot (Custom Headlines, Manual Y-Coordinates) ──────
def plot_multi_sample_rows(
    models: list,
    configs: list[dict],
    model_type: str,
    scale: int,
    device: str,
    channels: int,
    headlines: list[str],  # Input a full list of up to 3 headline strings
    save_path: str,
) -> None:
    """
    Generates a stacked grid where each row represents one variant's unconditional samples.
    Features zero column spacing and absolute manual control over headline positioning.
    """
    N_SAMPLES = 10  # noqa: N806
    n_variants = len(models)

    fig, axes = plt.subplots(
        n_variants,
        N_SAMPLES,
        figsize=(N_SAMPLES * 1.5, n_variants * 1.5),
        gridspec_kw={"wspace": 0.0},
    )
    if n_variants == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, (model, cfg) in enumerate(zip(models, configs, strict=False)):
        images = sample_at_scale(
            model,
            model_type,
            N_SAMPLES,
            scale,
            device,
            channels,
            cfg.get("hparams", cfg),
        )

        for c_idx in range(N_SAMPLES):
            ax = axes[r_idx, c_idx]
            img = images[c_idx]
            if channels == 1:
                ax.imshow(
                    img,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    aspect="auto",
                )
            else:
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            ax.axis("off")

    # ── MANUAL SAMPLE HEADLINE Y-COORDINATES ──────────────────────────────────
    # Adjust these 3 values to control the text height positions below each row
    # index 0 = Variant 1 headline position
    # index 1 = Variant 2 headline position
    # index 2 = Variant 3 headline position
    MANUAL_Y_POSITIONS = [0.64, 0.36, 0.07]  # noqa: N806
    # ───────────────────────────────────────────────────────────────────────────

    # Place text blocks using manual configurations
    for i in range(n_variants):
        y_pos = MANUAL_Y_POSITIONS[i]
        fig.text(
            0.5,
            y_pos,
            headlines[i],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.subplots_adjust(hspace=0.4)  # Leave a slight vertical gap for the text lines
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"  Multi-variant upscaled samples saved → {save_path}")


# ── Multi-Model Comparison Plot (Custom Headlines, No Column Spacing) ───────────
def plot_multi_recon_vs_interp(
    models: list,
    configs: list[dict],
    model_type: str,
    val_loader: torch.utils.data.DataLoader,
    scale: int,
    device: str,
    channels: int,
    headlines: list[
        str
    ],  # Input required: [Headline 1, Headline 2, Headline 3, Headline 4]
    save_path: str,
) -> None:
    """
    Creates a comparison plot with zero spacing between columns.
    A full list of customizable headline strings must be provided.
    """
    N_IMAGES = 10  # noqa: N806
    n_variants = len(models)
    total_rows = 1 + n_variants

    # Ensure provided headlines match generated rows
    if len(headlines) != total_rows:
        raise ValueError(
            f"You provided {len(headlines)} headlines, but the plot generated {total_rows} rows. "
            "Please ensure headlines list length matches (1 + n_variants)."
        )

    # Validation image fetching
    x_batch = []
    for batch in val_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_batch.append(imgs)
        if sum(b.shape[0] for b in x_batch) >= N_IMAGES:
            break
    x = torch.cat(x_batch, dim=0)[:N_IMAGES]
    if x.dim() == 2:
        img_size = round((x.shape[1] // channels) ** 0.5)
        x = x.view(x.shape[0], channels, img_size, img_size)
    x = x.to(device)

    # Shared Bilinear Base
    originals_up = F.interpolate(
        x.cpu().float(), size=(scale, scale), mode="bilinear", align_corners=False
    )
    originals_up = (originals_up * 0.5 + 0.5).clamp(0, 1)
    originals_up = _to_numpy_images(originals_up, channels)

    # ── Figure Layout & Tightness ──
    fig, axes = plt.subplots(
        total_rows,
        N_IMAGES,
        figsize=(N_IMAGES * 1.5, total_rows * 1.6 + 0.4),
        gridspec_kw={"wspace": 0.0},
    )
    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)  # Safety reshape

    # Render Row 1: Bilinear
    for col in range(N_IMAGES):
        ax = axes[0, col]
        if channels == 1:
            ax.imshow(
                originals_up[col],
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="auto",
            )
        else:
            ax.imshow(
                originals_up[col],
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="auto",
            )
        ax.axis("off")

    # Render Subsequent Rows: Model Variants
    for r_idx, (model, cfg) in enumerate(zip(models, configs, strict=False)):  # noqa: B007
        recons = reconstruct_at_scale(model, model_type, x, scale, device, channels)
        for col in range(N_IMAGES):
            ax = axes[r_idx + 1, col]
            if channels == 1:
                ax.imshow(
                    recons[col],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    aspect="auto",
                )
            else:
                ax.imshow(
                    recons[col], vmin=0, vmax=1, interpolation="nearest", aspect="auto"
                )
            ax.axis("off")

    # ── MANUAL HEADLINE Y-COORDINATES ─────────────────────────────────────────
    # Adjust these 4 values manually to position each headline vertically (0.0 to 1.0)
    # index 0 = Bilinear row headline position
    # index 1 = Variant 1 row headline position
    # index 2 = Variant 2 row headline position
    # index 3 = Variant 3 row headline position
    MANUAL_Y_POSITIONS = [0.71, 0.50, 0.29, 0.08]  # noqa: N806
    # ───────────────────────────────────────────────────────────────────────────

    # Render Row 1 Headline (Bilinear)
    fig.text(
        0.5,
        MANUAL_Y_POSITIONS[0],
        headlines[0],
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Render Model Rows Headlines
    for i in range(n_variants):
        headline_idx = i + 1
        y_pos = MANUAL_Y_POSITIONS[headline_idx]
        fig.text(
            0.5,
            y_pos,
            headlines[headline_idx],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.subplots_adjust(hspace=0.5)  # Vertical gap remains for readability
    fig.savefig(
        save_path, dpi=150, bbox_inches="tight", pad_inches=0.0
    )  # pad_inches removes fig-level borders
    plt.close(fig)
    print(f"  Tight multi-variant comparison plot saved → {save_path}")


# ── Custom 3-Panel VAE Training Curves (With Dynamic Ymin) ────────────────────
def plot_vae_training_curves(
    history: dict[str, list[float]],
    epoch_reached: int,
    save_path: str,
    plot_every_n: int = 100,
    elbo_ylim: float = 100.0,
    recon_ylim: float = 100.0,
    kl_ylim: float = 20.0,
) -> None:
    """
    Saves a 3-panel VAE training graph with distinct y-axis caps.
    The y-axis minimum dynamically adjusts to the lowest value in history per panel.
    """
    if epoch_reached <= 0 or len(history.get("elbo", [])) == 0:
        print(
            "  [Warning] History empty or epoch_reached is 0. Skipping training curves."
        )
        return

    steps_per_epoch = len(history["elbo"]) // epoch_reached
    max_ticks = 10
    tick_step = max(1, epoch_reached // max_ticks)
    tick_positions = [
        i * steps_per_epoch // plot_every_n
        for i in range(0, epoch_reached + 1, tick_step)
    ]
    tick_labels = [str(i) for i in range(0, epoch_reached + 1, tick_step)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("elbo", "Total ELBO", "tab:blue", elbo_ylim),
        ("recon", "Reconstruction Loss", "tab:orange", recon_ylim),
        ("kl", "KL Loss", "tab:green", kl_ylim),
    ]

    for ax, (key, title, color, y_max) in zip(axes, panels, strict=False):
        data = history.get(key, [])
        if not data:
            continue

        downsampled = data[::plot_every_n]
        ax.plot(
            range(len(downsampled)), downsampled, color=color, linewidth=0.8, alpha=0.85
        )

        # Calculate dynamic y-axis minimum
        min_val = min(downsampled)
        y_min = min_val - 0.05 * abs(min_val)
        if min_val >= 0:
            y_min = max(0.0, y_min)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("VAE Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved → {save_path}")


# ── VAE Sample Row Plot (Custom Headline, Zero Spacing) ───────────────────────
def plot_vae_sample_row(
    model,
    vae_config: dict,
    scale: int,
    device: str,
    channels: int,
    headline: str,  # Pass a single string manually here
    save_path: str,
) -> None:
    """
    Generates 10 unconditional random samples from the VAE latent space
    and decodes them side-by-side with zero spacing and a custom headline.
    """
    N_SAMPLES = 10  # noqa: N806
    latent_dim = vae_config["latent_dim"]
    latent_size = vae_config["latent_size"]

    z = torch.randn(N_SAMPLES, latent_dim, latent_size, latent_size, device=device)
    coord = make_coord_grid((scale, scale), (-1, 1), device=device)

    with torch.no_grad():
        x_hat = model.decoder(z, coord)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)

    images = _to_numpy_images(x_hat, channels)

    fig, axes = plt.subplots(
        1, N_SAMPLES, figsize=(N_SAMPLES * 1.5, 1.5), gridspec_kw={"wspace": 0.0}
    )
    for ax, img in zip(axes, images, strict=False):
        if channels == 1:
            ax.imshow(
                img, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto"
            )
        else:
            ax.imshow(img, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.axis("off")

    # Manual placement of the single VAE headline below the row
    fig.text(
        0.5, 0.01, headline, ha="center", va="center", fontsize=10, fontweight="bold"
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"  Upscaled samples row saved → {save_path}")


# ── VAE 2-Row Comparison Plot (Custom Headlines, No Column Spacing) ─────────────
def plot_vae_recon_vs_interp(
    model,
    val_loader: torch.utils.data.DataLoader,
    scale: int,
    device: str,
    channels: int,
    headlines: list[str],  # Input required: [Headline Row 1, Headline Row 2]
    save_path: str,
) -> None:
    """
    Creates a tight comparison plot with zero spacing between columns.
    Headline strings must be provided via the `headlines` list argument.
    """
    N_IMAGES = 10  # noqa: N806

    # Validation image fetching logic
    x_batch = []
    for batch in val_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_batch.append(imgs)
        if sum(b.shape[0] for b in x_batch) >= N_IMAGES:
            break
    x = torch.cat(x_batch, dim=0)[:N_IMAGES]
    if x.dim() == 2:
        img_size = round((x.shape[1] // channels) ** 0.5)
        x = x.view(x.shape[0], channels, img_size, img_size)
    x = x.to(device)

    # Computations
    with torch.no_grad():
        z, _, _ = model.encode(x)
        coord = make_coord_grid((scale, scale), (-1, 1), device=device)
        x_hat = model.decoder(z, coord)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
    recons = _to_numpy_images(x_hat, channels)

    originals_up = F.interpolate(
        x.cpu().float(), size=(scale, scale), mode="bilinear", align_corners=False
    )
    originals_up = (originals_up * 0.5 + 0.5).clamp(0, 1)
    originals_up = _to_numpy_images(originals_up, channels)

    # ── Figure Layout & Tightness ──
    fig, axes = plt.subplots(
        2, N_IMAGES, figsize=(N_IMAGES * 1.5, 3.4), gridspec_kw={"wspace": 0.0}
    )

    for col in range(N_IMAGES):
        # Top Row
        ax_top = axes[0, col]
        img_top = originals_up[col]
        if channels == 1:
            ax_top.imshow(
                img_top,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="auto",
            )
        else:
            ax_top.imshow(
                img_top, vmin=0, vmax=1, interpolation="nearest", aspect="auto"
            )
        ax_top.axis("off")

        # Bottom Row
        ax_bot = axes[1, col]
        img_bot = recons[col]
        if channels == 1:
            ax_bot.imshow(
                img_bot,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="auto",
            )
        else:
            ax_bot.imshow(
                img_bot, vmin=0, vmax=1, interpolation="nearest", aspect="auto"
            )
        ax_bot.axis("off")

    # Add headlines positioned below the relevant rows
    fig.text(
        0.5,
        0.52,
        headlines[0],
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.04,
        headlines[1],
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    plt.subplots_adjust(hspace=0.4)  # Vertical gap remains for label readability
    fig.savefig(
        save_path, dpi=150, bbox_inches="tight", pad_inches=0.0
    )  # pad_inches removes fig-level borders
    plt.close(fig)
    print(f"  Tight VAE comparison plot saved → {save_path}")


# ── Coord grid (mirrors make_coord_grid in the decoder) ───────────────────────


def make_coord_grid(
    shape: tuple[int, ...], range: list | tuple, device=None
) -> torch.Tensor:
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


# ── VAE model builder ─────────────────────────────────────────────────────────


def build_vae_model(vae_config: dict, channels: int, img_size: int, device: str):
    """
    Build and return a VAEWrapper from a saved VAE _config.json dict.

    Args:
        vae_config: Flat config dict loaded from <run_name>_config.json.
        channels:   Number of image channels (from build_dataset).
        img_size:   Spatial image size (from build_dataset).
        device:     Device string.
    Returns:
        model: VAEWrapper on device, weights NOT yet loaded.
    """
    import torch.nn as nn

    from src.models.latent_diffusion.modules.LatentEncoder import ResNetLatentEncoder
    from src.models.latent_diffusion.modules.trans_inr import TransInr

    class VAEWrapper(nn.Module):
        """Thin wrapper combining ResNetLatentEncoder + TransInr decoder."""

        def __init__(self, encoder, decoder, img_size, device):
            super().__init__()
            self.latent_encoder = encoder
            self.decoder = decoder
            self.img_size = img_size
            self.device = device
            self.register_buffer(
                "coord_grid", make_coord_grid((img_size, img_size), (-1, 1))
            )

        def encode(self, x):
            """Returns (mu, logvar) — mirrors LDM encode() signature for reconstruct_at_scale."""
            mu, logvar = self.latent_encoder(x)  # noqa: RUF059
            # Return mu as the deterministic latent (no sampling during eval)
            return mu, None, None

    latent_dim = vae_config["latent_dim"]
    latent_size = vae_config["latent_size"]

    encoder = ResNetLatentEncoder(
        in_channels=channels,
        latent_dim=latent_dim,
        latent_size=(latent_size, latent_size),
        hidden_dim=vae_config["latent_enc_hidden_dim"],
    )
    decoder = TransInr(
        tokenizer={
            "target": "src.models.tokenizers.latent_tokenizer.LatentTokenizer",
            "params": {
                "latent_dim": latent_dim,
                "latent_size": latent_size,
                "patch_size": vae_config["latent_patch_size"],
                "dim": vae_config["dec_trans_dim"],
                "n_head": vae_config["dec_trans_n_head"],
                "head_dim": vae_config["dec_trans_head_dim"],
            },
        },
        inr={
            "target": "src.models.inr.siren.SIREN",
            "params": {
                "depth": vae_config["inr_layers"],
                "in_dim": 2,
                "out_dim": channels,
                "hidden_dim": vae_config["inr_hidden_dim"],
                "out_bias": 0.5,
            },
        },
        data_shape=(img_size, img_size),
        n_groups=vae_config["dec_trans_n_groups"],
        transformer={
            "target": "src.models.utils.transformer.Transformer",
            "params": {
                "dim": vae_config["dec_trans_dim"],
                "encoder_depth": vae_config["dec_trans_enc_depth"],
                "decoder_depth": vae_config["dec_trans_dec_depth"],
                "n_head": vae_config["dec_trans_n_head"],
                "head_dim": vae_config["dec_trans_head_dim"],
                "ff_dim": vae_config["dec_trans_ff_dim"],
            },
        },
        update_strategy=vae_config["dec_trans_update_strategy"],
    )
    return VAEWrapper(encoder, decoder, img_size, device).to(device)


# ── Sampling at custom resolution ─────────────────────────────────────────────


def _to_numpy_images(x_hat: torch.Tensor, channels: int) -> np.ndarray:
    """
    Convert a (B, C, H, W) float tensor in [0,1] to a numpy image array.

    Args:
        x_hat:    (B, C, H, W) float tensor already clamped to [0,1].
        channels: Number of image channels.
    Returns:
        images: (B, H, W) for grayscale or (B, H, W, C) for RGB.
    """
    x_hat = x_hat.cpu().float()
    if channels == 1:
        return x_hat.squeeze(1).numpy()
    return x_hat.permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def sample_at_scale(
    model,
    model_type: str,
    n_samples: int,
    scale: int,
    device: str,
    channels: int,
    hparams,
) -> np.ndarray:
    """
    Sample from the model and decode at a custom spatial resolution.

    Args:
        model:      Trained model (LDM, VAEWrapper, or WeightDiffusion).
        model_type: "ldm", "vae", or "weight_diffusion".
        n_samples:  Number of images to generate.
        scale:      Target H=W resolution.
        device:     Device string.
        channels:   Number of image channels.
        hparams:    Namespace/dict with latent shape info (used for VAE).
    Returns:
        images: (n_samples, scale, scale) or (n_samples, scale, scale, C) in [0,1].
    """
    coord = make_coord_grid((scale, scale), (-1, 1), device=device)  # (scale, scale, 2)

    if model_type == "ldm":
        z = model._sample_latent(n_samples) if isinstance(model, TwoStageLDM) else model._sample_latent(n_samples, collect_snapshots=False, debug=False)
        
        x_hat = model.decoder(z, coord)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)

    elif model_type == "weight_diffusion":
        # Sample weight vectors via reverse diffusion, then decode via SIREN at custom scale
        # coord must be (B, H, W, 2) — _inr_decode handles the batch expand internally
        theta = model.sample_weight(n_samples)  # (B, weight_dim)
        theta = model.weight_encoder.decode_modulations(theta)  # structured params
        # Pass custom coord grid; _inr_decode expects (B, H, W, 2) or (1, H, W, 2)
        coord_batched = coord.unsqueeze(0).expand(
            n_samples, -1, -1, -1
        )  # (B, scale, scale, 2)
        pixels = model._inr_decode(theta, coords=coord_batched)  # (B, scale*scale)
        x_hat = pixels.reshape(n_samples, channels, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)

    else:
        # VAE: sample z ~ N(0, I) directly then decode via TransInr decoder
        latent_dim = (
            hparams["latent_dim"] if isinstance(hparams, dict) else hparams.latent_dim
        )
        latent_size = (
            hparams["latent_size"] if isinstance(hparams, dict) else hparams.latent_size
        )
        z = torch.randn(n_samples, latent_dim, latent_size, latent_size, device=device)
        x_hat = model.decoder(z, coord)  # (B, C, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)

    return _to_numpy_images(x_hat, channels)


# ── Reconstruction at custom resolution ───────────────────────────────────────


@torch.no_grad()
def reconstruct_at_scale(
    model,
    model_type: str,
    x: torch.Tensor,
    scale: int,
    device: str,
    channels: int,
) -> np.ndarray:
    """
    Encode a batch of images and decode at a custom spatial resolution.

    Args:
        model:      Trained model (LDM, VAEWrapper, or WeightDiffusion).
        model_type: "ldm", "vae", or "weight_diffusion".
        x:          (B, C, H, W) input images on device.
        scale:      Target H=W resolution for the reconstruction.
        device:     Device string.
        channels:   Number of image channels.
    Returns:
        recons: (B, scale, scale) or (B, scale, scale, C) float32 in [0,1].
    """
    coord = make_coord_grid((scale, scale), (-1, 1), device=device)  # (scale, scale, 2)

    # Reshape flat inputs to (B, C, H, W) for all paths
    if x.dim() == 2:
        img_size = round((x.shape[1] // channels) ** 0.5)
        x = x.view(x.shape[0], channels, img_size, img_size)

    if model_type == "weight_diffusion":
        # WeightDiffusion encodes to flat weight vectors, decodes via SIREN
        B = x.shape[0]  # noqa: N806
        theta_prime_raw, _, _ = model.encode(x)  # (B, weight_dim)
        theta = model.weight_encoder.decode_modulations(
            theta_prime_raw
        )  # structured params
        coord_batched = coord.unsqueeze(0).expand(B, -1, -1, -1)  # (B, scale, scale, 2)
        pixels = model._inr_decode(theta, coords=coord_batched)  # (B, scale*scale)
        x_hat = pixels.reshape(B, channels, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
    else:
        # LDM and VAEWrapper both expose encode() → (z, ?, ?) and a TransInr decoder
        z, _, _ = model.encode(x)
        x_hat = model.decoder(z, coord)  # (B, C, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)

    return _to_numpy_images(x_hat, channels)


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
        raise ValueError(
            f"Could not extract run name from: {config_path}\n"
            "Expected: .../<run_name>/metadata/config.json"
        )  # noqa: B904


# ── Plot 1: upscaled sample row ───────────────────────────────────────────────


def plot_sample_row(
    model,
    model_type: str,
    hparams,
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
        model:      Trained model.
        model_type: "ldm" or "vae".
        hparams:    Namespace/dict with latent shape info (used for VAE sampling).
        n_samples:  Number of samples in the row.
        scale:      Target resolution (scale x scale).
        device:     Device string.
        channels:   Number of image channels.
        epoch:      Epoch number (used in filename).
        run_dir:    Output directory.
    Returns:
        None
    """
    print(f"  Generating {n_samples} samples at {scale}x{scale} …")
    images = sample_at_scale(
        model, model_type, n_samples, scale, device, channels, hparams
    )

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 1.5, 1.5))
    for ax, img in zip(axes, images, strict=False):
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
    model_type: str,
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
    Three-row comparison: reconstructions, interpolated originals, native originals.

    Args:
        model:       Trained model.
        model_type:  "ldm", "vae", or "weight_diffusion".
        val_loader:  Validation DataLoader for sourcing original images.
        n_images:    Number of images in each row.
        scale:       Target resolution for reconstruction and interpolation.
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
    recons = reconstruct_at_scale(model, model_type, x, scale, device, channels)

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
    parser = argparse.ArgumentParser(
        description="Reworked Evaluation Visualizations for INR models."
    )

    # VAE inputs (Required)
    parser.add_argument(
        "--vae_config_path", type=str, required=True, help="Path to VAE _config.json."
    )
    parser.add_argument(
        "--vae_checkpoint_path",
        type=str,
        required=True,
        help="Path to VAE checkpoint .pt.",
    )

    # Latent Diffusion inputs (Optional, accepts up to 3 paths)
    parser.add_argument(
        "--latent_config_paths",
        type=str,
        nargs="+",
        default=[],
        help="Paths to Latent Diffusion config.json files (Max 3).",
    )

    # Weight Diffusion inputs (Optional, accepts up to 3 paths)
    parser.add_argument(
        "--weight_config_paths",
        type=str,
        nargs="+",
        default=[],
        help="Paths to Weight Diffusion config.json files (Max 3).",
    )

    # Scale input
    parser.add_argument(
        "--sample_scale",
        type=int,
        default=32,
        help="Target resolution for upscaled samples and reconstructions.",
    )

    args = parser.parse_args()

    # Enforce strict input limits
    if len(args.latent_config_paths) > 3:
        parser.error("You can provide a maximum of 3 latent_config_paths.")
    if len(args.weight_config_paths) > 3:
        parser.error("You can provide a maximum of 3 weight_config_paths.")

    # Hardcoded configuration constants
    N_SAMPLES = 10  # noqa: F841, N806
    N_RECON = 10  # noqa: N806
    INTERP_MODE = "bilinear"  # noqa: F841, N806

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device

    device = _get_device()

    # ── 1. UNIFIED OUTPUT DIRECTORY SETUP ─────────────────────────────────────
    # We load the VAE config first to establish the primary run name and output folder
    with open(args.vae_config_path) as f:
        vae_config = json.load(f)

    output_dir = os.path.join("src", "results", "final_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Unified Eval Visual Suite  |  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # ── 2. DATASET & DATALOADER SETUP ─────────────────────────────────────────
    print("  Building validation dataset ...")
    _, val_dataset, data_config = build_dataset(
        dataset_name=vae_config["dataset"],
        data_root="data/",
        subset_frac=1.0,
        single_class=False,
    )
    channels = data_config["channels"]
    img_size = data_config["img_size"]

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=N_RECON,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # ── 3. PROCESS VAE MODEL ──────────────────────────────────────────────────

    print("--- Processing VAE-INR Model ---")
    print("  Building VAE model ...")
    vae_model = build_vae_model(vae_config, channels, img_size, device)

    print(f"  Loading VAE checkpoint from {args.vae_checkpoint_path} ...")
    vae_ckpt = torch.load(args.vae_checkpoint_path, map_location=device)
    vae_model.load_state_dict(vae_ckpt["model_state_dict"])
    vae_model.eval()

    epoch_reached = vae_ckpt.get("epoch_reached", 0)
    history = vae_ckpt.get("history", {})

    # Plot 1: Training Curves (with explicit individual y-limits)
    plot_vae_training_curves(
        history=history,
        epoch_reached=epoch_reached,
        save_path=os.path.join(output_dir, "vae_training_curves.png"),
        plot_every_n=1,
        elbo_ylim=30.0,
        recon_ylim=30.0,
        kl_ylim=120.0,
    )

    # Plot 2: Upscaled Sample Row
    plot_vae_sample_row(
        model=vae_model,
        vae_config=vae_config,
        scale=args.sample_scale,
        device=device,
        channels=channels,
        headline=HEADLINE_VAE_SAMPLES,
        save_path=os.path.join(output_dir, "vae_samples_upscaled.png"),
    )

    # Reworked call using controllable list
    plot_vae_recon_vs_interp(
        model=vae_model,
        val_loader=val_loader,
        scale=args.sample_scale,
        device=device,
        channels=channels,
        headlines=HEADLINES_VAE,  # Mandatory input list
        save_path=os.path.join(output_dir, "vae_reconstructions.png"),
    )

    plot_vae_sample_grid(
        model=vae_model,
        vae_config=vae_config,
        input_scale=args.sample_scale,
        device=device,
        channels=channels,
        save_path=os.path.join(output_dir, "vae_sample_grid.png"),
    )

    # ── 4. LATENT & WEIGHT VARIATIONS ─────────────────────────────────────────
    # ── 4. PROCESS LATENT DIFFUSION MODELS ────────────────────────────────────
    if args.latent_config_paths:
        print(
            f"\n--- Processing Latent Diffusion Suite ({len(args.latent_config_paths)} variants) ---"
        )
        from src.utility.model_builders.model_builder import build_model as build_ldm_model
        from src.utility.model_builders.util.twostage_builder import build_ldm as build_two_stage_ldm

        latent_models = []
        latent_configs = []

        for idx, p in enumerate(args.latent_config_paths):
            with open(p) as f:
                l_cfg = json.load(f)

            if idx == 0:
                # One-stage: nested config with hparams/data/paths
                l_hparams = SimpleNamespace(**l_cfg["hparams"])
                l_data_cfg = l_cfg["data"]
                l_data_config = {
                    "dataset": l_cfg["dataset"],
                    "channels": l_data_cfg["channels"],
                    "img_size": l_data_cfg["img_size"],
                    "data_dim": l_data_cfg["data_dim"],
                }
                l_cfg["run_name"] = _extract_run_name(p)
                print(f"  Building & loading (one-stage): {l_cfg['run_name']} ...")
                l_model = build_ldm_model(l_hparams, l_data_config).to(device)
                l_ckpt = torch.load(l_cfg["paths"]["weights"], map_location=device)
                l_model.load_state_dict(l_ckpt["model_state_dict"])
            else:
                # Two-stage: flat config, checkpoint in same dir as config
                run_name = l_cfg["run_name"]
                ckpt_path = os.path.join(
                    os.path.dirname(os.path.abspath(p)),
                    f"{run_name}_ldm_checkpoint.pt"
                )
                ts_args = SimpleNamespace(
                    T=l_cfg["T"], beta_1=l_cfg["beta_1"], beta_T=l_cfg["beta_T"]
                )
                print(f"  Building & loading (two-stage): {run_name} ...")
                l_model = build_two_stage_ldm(
                    hparams=l_cfg, args=ts_args, channels=channels, img_size=img_size, device=device
                )
                l_ckpt = torch.load(ckpt_path, map_location=device)
                l_model.load_state_dict(l_ckpt["model_state_dict"])
                # Normalise cfg shape so downstream plot functions can read run_name/hparams
                l_cfg = {"run_name": run_name, "hparams": l_cfg, "data": {"img_size": img_size}}

            l_model.eval()
            latent_models.append(l_model)
            latent_configs.append(l_cfg)
        
        # Generate Composite Visuals
        plot_multi_training_curves(
            config_paths=args.latent_config_paths,
            configs=latent_configs,
            model_type="ldm",
            save_path=os.path.join(output_dir, "latent_training_curves.png"),
            plot_every_n=1,
            total_ylim=50,  # Modify these caps directly here
            diff_ylim=0.1,
            rec_ylim=50,
        )
        active_latent_sample_headlines = HEADLINES_LATENT_SAMPLES[: len(latent_configs)]

        plot_multi_sample_rows(
            models=latent_models,
            configs=latent_configs,
            model_type="ldm",
            scale=args.sample_scale,
            device=device,
            channels=channels,
            headlines=active_latent_sample_headlines,
            save_path=os.path.join(output_dir, "latent_samples_upscaled.png"),
        )
        required_rows = 1 + len(latent_configs)
        active_latent_headlines = HEADLINES_LATENT[:required_rows]

        plot_multi_recon_vs_interp(
            models=latent_models,
            configs=latent_configs,
            model_type="ldm",
            val_loader=val_loader,
            scale=args.sample_scale,
            device=device,
            channels=channels,
            headlines=active_latent_headlines,  # Dynamically sliced list
            save_path=os.path.join(output_dir, "latent_reconstructions.png"),
        )

        # Loop through each active variant separately
        for idx, (model, cfg) in enumerate(
            zip(latent_models, latent_configs, strict=False)
        ):
            run_name = NAMES[idx] if idx < len(NAMES) else f"variant_{idx+1}"
            plot_multi_sample_grid(
                model=model,
                cfg=cfg,
                model_type="ldm",
                input_scale=args.sample_scale,
                device=device,
                channels=channels,
                save_path=os.path.join(output_dir, f"latent_grid_{run_name}.png"),
            )

    # ── 5. PROCESS WEIGHT DIFFUSION MODELS ────────────────────────────────────
    if args.weight_config_paths:
        print(
            f"\n--- Processing Weight Diffusion Suite ({len(args.weight_config_paths)} variants) ---"
        )
        from src.utility.model_builders.model_builder import build_model as build_ldm_model
        from src.scripts.two_stage_weight_training import build_full_wd_model

        weight_models = []
        weight_configs = []

        for idx, p in enumerate(args.weight_config_paths):
            with open(p) as f:
                w_cfg = json.load(f)

            if idx == 0:
                # One-stage: nested config with hparams/data/paths
                w_hparams = SimpleNamespace(**w_cfg["hparams"])
                w_data_cfg = w_cfg["data"]
                w_data_config = {
                    "dataset": w_cfg["dataset"],
                    "channels": w_data_cfg["channels"],
                    "img_size": w_data_cfg["img_size"],
                    "data_dim": w_data_cfg["data_dim"],
                }
                w_cfg["run_name"] = _extract_run_name(p)
                print(f"  Building & loading (one-stage): {w_cfg['run_name']} ...")
                w_model = build_ldm_model(w_hparams, w_data_config).to(device)
                w_ckpt = torch.load(w_cfg["paths"]["weights"], map_location=device)
                state_dict = {k: v for k, v in w_ckpt["model_state_dict"].items() if k != "coords"}
                w_model.load_state_dict(state_dict, strict=False)
            else:
                # Two-stage: flat config, checkpoint in same dir as config
                run_name = w_cfg["run_name"]
                ckpt_path = os.path.join(
                    os.path.dirname(os.path.abspath(p)),
                    f"{run_name}_wd_weights.pt"
                )
                tsw_args = SimpleNamespace(
                    T=w_cfg["T"], beta_1=w_cfg["beta_1"], beta_T=w_cfg["beta_T"]
                )
                print(f"  Building & loading (two-stage): {run_name} ...")
                w_model = build_full_wd_model(
                    hparams=w_cfg,
                    args=tsw_args,
                    channels=channels,
                    img_size=img_size,
                    data_dim=data_config["data_dim"],
                    device=device,
                )
                w_ckpt = torch.load(ckpt_path, map_location=device)
                state_dict = {k: v for k, v in w_ckpt["full_model_state_dict"].items() if k != "coords"}
                w_model.load_state_dict(state_dict, strict=False)
                # Normalise cfg shape so downstream plot functions can read run_name/hparams
                w_cfg = {"run_name": run_name, "hparams": w_cfg, "data": {"img_size": img_size}}

            w_model.eval()
            weight_models.append(w_model)
            weight_configs.append(w_cfg)

        # Generate Composite Visuals
        plot_multi_training_curves(
            config_paths=args.weight_config_paths,
            configs=weight_configs,
            model_type="weight_diffusion",
            save_path=os.path.join(output_dir, "weight_training_curves.png"),
            plot_every_n=1,
            total_ylim=1500,  # Modify these caps directly here
            diff_ylim=1.5,
            rec_ylim=50,
        )
        active_weight_sample_headlines = HEADLINES_WEIGHT_SAMPLES[: len(weight_configs)]

        plot_multi_sample_rows(
            models=weight_models,
            configs=weight_configs,
            model_type="weight_diffusion",
            scale=args.sample_scale,
            device=device,
            channels=channels,
            headlines=active_weight_sample_headlines,
            save_path=os.path.join(output_dir, "weight_samples_upscaled.png"),
        )
        required_rows = 1 + len(weight_configs)
        active_weight_headlines = HEADLINES_WEIGHT[:required_rows]

        plot_multi_recon_vs_interp(
            models=weight_models,
            configs=weight_configs,
            model_type="weight_diffusion",
            val_loader=val_loader,
            scale=args.sample_scale,
            device=device,
            channels=channels,
            headlines=active_weight_headlines,
            save_path=os.path.join(output_dir, "weight_reconstructions.png"),
        )
        for idx, (model, cfg) in enumerate(
            zip(weight_models, weight_configs, strict=False)
        ):
            run_name = NAMES[idx] if idx < len(NAMES) else f"variant_{idx+1}"
            plot_multi_sample_grid(
                model=model,
                cfg=cfg,
                model_type="weight_diffusion",
                input_scale=args.sample_scale,
                device=device,
                channels=channels,
                save_path=os.path.join(output_dir, f"weight_grid_{run_name}.png"),
            )

    print("\nVAE Processing Complete.")


if __name__ == "__main__":
    main()
