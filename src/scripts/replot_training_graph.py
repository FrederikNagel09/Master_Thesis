"""
Replot training curves from a saved training_graph_data.json metadata file.

Usage:
    python src/scripts/replot_training_graph.py \
        --metadata src/train_results/latent-diffusion/metadata/training_graph_data.json \
        --y_lim_total 40 \
        --y_lim_diff 0.01 \
        --y_lim_prior -3 \
        --y_lim_rec 40 \
        --plot_every_n 10
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


# ── Panel config ──────────────────────────────────────────────────────────────
_PANELS: list[tuple[str, str, str]] = [
    ("total", "Total Loss",      "tab:blue"),
    ("diff",  "Diffusion Loss",  "tab:orange"),
    ("prior", "Prior Loss",      "tab:green"),
    ("rec",   "Reconstruction",  "tab:red"),
]


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    p = argparse.ArgumentParser(description="Replot training curves from a JSON metadata file")
    p.add_argument("--metadata", type=str, required=True, help="Path to training_graph_data.json")
    p.add_argument("--y_lim_total", type=float, default=None, help="Y-axis cap for total loss panel")
    p.add_argument("--y_lim_diff",  type=float, default=None, help="Y-axis cap for diffusion loss panel")
    p.add_argument("--y_lim_prior", type=float, default=None, help="Y-axis cap for prior loss panel")
    p.add_argument("--y_lim_rec",   type=float, default=None, help="Y-axis cap for reconstruction loss panel")
    p.add_argument("--plot_every_n", type=int,   default=1,    help="Downsample factor (default: 1 = no downsampling)")
    return p.parse_args()


def load_metadata(path: str) -> dict[str, list[float]]:
    """
    Load history dict from a JSON metadata file.

    Args:
        path: path to training_graph_data.json
    Returns:
        history dict with keys: steps, total, diff, prior, rec, lr
    """
    with open(path) as f:
        return json.load(f)


def _plot_loss_panel(
    ax: plt.Axes,
    steps: list[float],
    values: list[float],
    title: str,
    color: str,
    y_lim: float | None,
    plot_every_n: int,
) -> None:
    """
    Draw a single loss panel onto an Axes object.

    Args:
        ax:           target matplotlib Axes
        steps:        fractional epoch x-values
        values:       loss y-values (same length as steps)
        title:        panel title string
        color:        line color string
        y_lim:        upper y-axis cap; None = auto-scale
        plot_every_n: downsample factor
    Returns:
        None
    """
    xs = steps[::plot_every_n]
    ys = values[::plot_every_n]

    ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0, top=y_lim)  # top=None → matplotlib auto-scales
    ax.grid(True, linestyle="--", alpha=0.4)


def _add_lr_twin(ax: plt.Axes, steps: list[float], lr: list[float], plot_every_n: int) -> None:
    """
    Overlay a learning-rate curve on a twin y-axis.

    Args:
        ax:           host Axes to twin from
        steps:        fractional epoch x-values
        lr:           learning rate values (same length as steps)
        plot_every_n: downsample factor
    Returns:
        None
    """
    ax2 = ax.twinx()
    ax2.plot(steps[::plot_every_n], lr[::plot_every_n], color="gray", linewidth=0.6, alpha=0.5, linestyle="--")
    ax2.set_ylabel("LR", color="gray", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)


def plot_training_curves(
    history: dict[str, list[float]],
    y_lims: dict[str, float | None],
    plot_every_n: int,
    save_path: str,
) -> None:
    """
    Build and save the multi-panel training curve figure.

    Args:
        history:      history dict with keys steps, total, diff, prior, rec, lr
        y_lims:       per-panel y-axis caps keyed by loss name; None = auto-scale
        plot_every_n: downsample factor
        save_path:    full path to write the output .png
    Returns:
        None
    """
    steps = history["steps"]

    # Drop panels where all values are zero (component not used in this run)
    active = [
        (key, title, color)
        for key, title, color in _PANELS
        if any(v != 0.0 for v in history.get(key, [0.0]))
    ]

    n = len(active)
    if n == 0:
        print("No active loss components found — nothing to plot.")
        return

    # 2x2 grid for 4 panels, single row otherwise
    if n == 4:
        fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes_grid.flatten().tolist()
    else:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

    fig.suptitle("Training Curves (replot)", fontsize=14, fontweight="bold", y=1.02)

    has_lr = bool(history.get("lr"))

    for ax, (key, title, color) in zip(axes, active, strict=False):
        _plot_loss_panel(ax, steps, history[key], title, color, y_lims.get(key), plot_every_n)

        if key == "total" and has_lr:
            _add_lr_twin(ax, steps, history["lr"], plot_every_n)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


def main() -> None:
    args = parse_args()

    history = load_metadata(args.metadata)
    print(f"Loaded metadata: {len(history['steps'])} logged steps")

    y_lims: dict[str, float | None] = {
        "total": args.y_lim_total,
        "diff":  args.y_lim_diff,
        "prior": args.y_lim_prior,
        "rec":   args.y_lim_rec,
    }

    out_dir = os.path.dirname(os.path.abspath(args.metadata))
    save_path = os.path.join(out_dir, "training_graph_replot.png")

    plot_training_curves(
        history=history,
        y_lims=y_lims,
        plot_every_n=args.plot_every_n,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()