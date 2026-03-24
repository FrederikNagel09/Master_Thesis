"""
results_sample_grid.py
Generates a side-by-side 6x6 sample grid for up to three trained models.

Usage
-----
python src/scripts/sample_comparison.py \
    --ndm     src/train_results/ndm_unet_mnist/metadata/config.json \
    --inr_vae src/train_results/vae_inr_mnist/metadata/config.json \
    --ndm_inr src/train_results/ndm_inr_mlp_mnist/metadata/config.json \
    --out      src/results/sample_grid.png

All three config paths are optional — omit any model you don't want included.
The figure adjusts to however many models are provided.
"""

import argparse
import os
import sys

sys.path.append(".")

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from src.configs.results_config import MODEL_COLORS, MODEL_LABELS, SAMPLE_COMPARISON_GRID_SIZE
from src.utility.general import _draw_grid, _get_device
from src.utility.inference import sample

if TYPE_CHECKING:
    import torch

# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side sample grids.")
    parser.add_argument("--ndm", type=str, default=None, help="Path to NDM config.json")
    parser.add_argument("--inr_vae", type=str, default=None, help="Path to INR-VAE config.json")
    parser.add_argument("--ndm_inr", type=str, default=None, help="Path to NDM-INR config.json")
    parser.add_argument("--out", type=str, default="src/results/sample_grid.png", help="Output path for the saved figure")
    args = parser.parse_args()

    # ── Collect requested models ──────────────────────────────────────────────
    requested = {}
    for key in ("ndm", "inr_vae", "ndm_inr"):
        path = getattr(args, key)
        if path is not None:
            requested[key] = path

    if not requested:
        print("No config paths provided. Pass at least one of --ndm, --inr_vae, --ndm_inr.")
        sys.exit(1)

    device = _get_device()
    n_models = len(requested)
    n_samples = SAMPLE_COMPARISON_GRID_SIZE * SAMPLE_COMPARISON_GRID_SIZE

    print(f"\nGenerating {SAMPLE_COMPARISON_GRID_SIZE}x{SAMPLE_COMPARISON_GRID_SIZE} sample grids for: {list(requested.keys())}")
    print(f"Device: {device}\n")

    # ── Sample from each model ────────────────────────────────────────────────
    results: dict[str, tuple[torch.Tensor, int]] = {}  # model_key -> (images, channels)
    for model_key, config_path in requested.items():
        print(f"── Sampling {MODEL_LABELS[model_key]} ──")
        images = sample(
            model_name=model_key,
            config_path=config_path,
            n_samples=n_samples,
            device=device,
        )  # (N, C, H, W) in [0, 1]
        channels = images.shape[1]
        results[model_key] = (images, channels)
        print()

    # ── Build figure ──────────────────────────────────────────────────────────
    img_inches = 1.0  # inches per image cell
    gap = 0.6  # inches between model grids

    fig_w = n_models * SAMPLE_COMPARISON_GRID_SIZE * img_inches + (n_models - 1) * gap
    fig_h = SAMPLE_COMPARISON_GRID_SIZE * img_inches + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    title_pad = 0.4  # inches for model title
    total_h = fig_h

    for m_idx, (model_key, (images, channels)) in enumerate(results.items()):
        # Compute left offset for this model's grid block
        block_w = SAMPLE_COMPARISON_GRID_SIZE * img_inches
        left_start = (m_idx * (block_w + gap)) / fig_w

        # Draw SAMPLE_COMPARISON_GRID_SIZE x SAMPLE_COMPARISON_GRID_SIZE images
        axes = []
        for r in range(SAMPLE_COMPARISON_GRID_SIZE):
            for c in range(SAMPLE_COMPARISON_GRID_SIZE):
                left = left_start + (c * img_inches) / fig_w
                bottom = (title_pad + (SAMPLE_COMPARISON_GRID_SIZE - 1 - r) * img_inches) / total_h
                width = img_inches / fig_w
                height = img_inches / total_h

                ax = fig.add_axes([left, bottom, width, height])
                axes.append(ax)

        _draw_grid(axes, images, channels)
        # Centre of this model's block in figure coordinates
        block_centre_x = (m_idx * (block_w + gap) + block_w / 2) / fig_w
        title_y = (SAMPLE_COMPARISON_GRID_SIZE * img_inches + title_pad * 0.4) / fig_h + 0.05
        fig.text(
            block_centre_x,
            title_y,
            MODEL_LABELS[model_key],
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=MODEL_COLORS[model_key],
        )

    fig.suptitle("Generated Samples", fontsize=15, fontweight="bold", y=1.04)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sample grid saved → {args.out}")


if __name__ == "__main__":
    main()
