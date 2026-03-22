"""
results_upscaling.py
Generates a multi-resolution upscaling figure for INR-based models.

For each model (inr_vae, ndm_inr), samples N latent codes and renders
each at multiple resolutions. The same latent code is used across all
resolutions in a row, demonstrating the INR's continuous rendering ability.

Layout
------
              28px    64px    128px   256px
INR-VAE  | [img]   [img]   [img]   [img]
         | [img]   [img]   [img]   [img]
         | ...
─────────────────────────────────────────
NDM-INR  | [img]   [img]   [img]   [img]
         | [img]   [img]   [img]   [img]
         | ...

Usage
-----
python src/scripts/upscaling_comparison.py \
    --inr_vae src/train_results/vae_inr_mnist/metadata/config.json \
    --ndm_inr src/train_results/ndm_inr_mlp_mnist/metadata/config.json \
    --out      src/results/upscaling.png

Both config paths are optional — omit any model you don't want included.
"""

import argparse
import os
import sys

sys.path.append(".")

import matplotlib.pyplot as plt

from src.utility.general import _get_device, _load_model
from src.utility.inference import (
    _sample_at_resolutions_inr_vae,
    _sample_at_resolutions_ndm_inr,
)

# =============================================================================
# Config
# =============================================================================

N_ROWS = 3
RESOLUTIONS = [28, 64, 128, 256, 512, 1024]
MODEL_LABELS = {
    "inr_vae": "VAE-INR",
    "ndm_inr": "NDM-INR",
}
MODEL_COLORS = {
    "ndm": "#2a6fdb",
    "inr_vae": "#e07b39",
    "ndm_inr": "#2ca05a",
}

# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-resolution upscaling figure.")
    parser.add_argument("--inr_vae", type=str, default=None, help="Path to INR-VAE config.json")
    parser.add_argument("--ndm_inr", type=str, default=None, help="Path to NDM-INR config.json")
    parser.add_argument("--out", type=str, default="src/results/upscaling.png", help="Output path for the saved figure")
    parser.add_argument("--n_rows", type=int, default=N_ROWS, help="Number of sample rows per model")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to render at")
    args = parser.parse_args()

    requested = {}
    for key in ("inr_vae", "ndm_inr"):
        path = getattr(args, key)
        if path is not None:
            requested[key] = path

    if not requested:
        print("No config paths provided. Pass at least one of --inr_vae, --ndm_inr.")
        sys.exit(1)

    device = _get_device()
    n_rows = args.n_rows
    resolutions = args.resolutions
    n_res = len(resolutions)
    n_models = len(requested)

    print(f"\nUpscaling figure — models: {list(requested.keys())}")
    print(f"Resolutions : {resolutions}")
    print(f"Rows/model  : {n_rows}")
    print(f"Device      : {device}\n")

    # ── Sample all models ─────────────────────────────────────────────────────
    all_rows: dict[str, list] = {}
    all_channels: dict[str, int] = {}

    for model_key, config_path in requested.items():
        print(f"── Loading {MODEL_LABELS[model_key]} ──")
        model, data_config = _load_model(config_path, device)
        channels = data_config["channels"]
        all_channels[model_key] = channels

        print(f"  Rendering {n_rows} rows at {resolutions} …")
        if model_key == "inr_vae":
            rows = _sample_at_resolutions_inr_vae(model, n_rows, resolutions, channels, device)
        else:
            rows = _sample_at_resolutions_ndm_inr(model, n_rows, resolutions, channels, device)

        all_rows[model_key] = rows
        print()

    # ── Build figure ──────────────────────────────────────────────────────────
    label_width = 0.55  # inches for model name label on left
    res_label_h = 0.30  # inches for resolution labels on top
    img_inches = 1.0  # inches per image cell
    model_gap = 0.35  # inches between model sections
    title_pad = 0.35  # inches for figure title

    fig_w = label_width + n_res * img_inches
    fig_h = title_pad + res_label_h + n_models * n_rows * img_inches + (n_models - 1) * model_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Resolution labels along the top
    for j, res in enumerate(resolutions):
        x = (label_width + (j + 0.5) * img_inches) / fig_w
        y = 1.0 - (title_pad / fig_h) - (res_label_h * 0.5 / fig_h)
        fig.text(x, y, f"{res}x{res}", ha="center", va="center", fontsize=9, fontweight="bold", color="#333333")

    # Draw each model section
    for m_idx, (model_key, rows) in enumerate(all_rows.items()):
        channels = all_channels[model_key]

        # Vertical offset: how far down from top this model section starts
        v_offset = title_pad / fig_h + res_label_h / fig_h + m_idx * (n_rows * img_inches + model_gap) / fig_h

        # Model name label centred vertically on left
        label_y = 1.0 - v_offset - (n_rows * img_inches * 0.5) / fig_h
        fig.text(
            (label_width * 0.5) / fig_w,
            label_y,
            MODEL_LABELS[model_key],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=MODEL_COLORS[model_key],
            rotation=90,
        )

        # Separator line between model sections (except before first)
        if m_idx > 0:
            line_y = 1.0 - v_offset + (model_gap * 0.5 / fig_h)
            fig.add_artist(
                plt.Line2D(
                    [label_width / fig_w, 1.0],
                    [line_y, line_y],
                    color="#cccccc",
                    linewidth=0.8,
                    transform=fig.transFigure,
                )
            )

        # Draw images — no gap between rows within a model
        for r, row_imgs in enumerate(rows):
            for j, img in enumerate(row_imgs):
                left = (label_width + j * img_inches) / fig_w
                bottom = 1.0 - v_offset - (r + 1) * img_inches / fig_h
                width = img_inches / fig_w
                height = img_inches / fig_h

                ax = fig.add_axes([left, bottom, width, height])
                if channels == 1:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
                ax.axis("off")

    fig.suptitle("Multi-Resolution Upscaling", fontsize=13, fontweight="bold", y=0.99)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Upscaling figure saved → {args.out}")


if __name__ == "__main__":
    main()
