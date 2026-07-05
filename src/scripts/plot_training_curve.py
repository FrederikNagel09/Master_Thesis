"""
Replot training curves from a VAE checkpoint with a y-axis cap at 100.

Usage:
    python src/scripts/plot_training_curve.py \
        --checkpoint src/train_results/Latent-two_stage_fixed/Latent-two_stage_fixed_vae_checkpoint.pt \
        --y_lim 50 

    python src/scripts/plot_training_curves.py --checkpoint Master_Thesis/src/results/vae_baseline/vae_baseline_checkpoint.pt --steps_per_epoch 391
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    p = argparse.ArgumentParser(description="Replot VAE training curves from a checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pt file")
    p.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Steps per epoch for x-axis tick labels. Inferred from history length / epoch_reached if omitted.",
    )
    p.add_argument("--plot_every_n", type=int, default=100, help="Downsample factor for plotting (default: 100)")
    p.add_argument("--y_lim", type=float, default=20, help="Upper y-axis limit (default: 100)")
    return p.parse_args()


def load_checkpoint(path: str) -> tuple[dict[str, list[float]], int]:
    """
    Loads history and epoch count from a checkpoint file.

    Args:
        path: path to the .pt checkpoint file
    Returns:
        (history, epoch_reached) — loss history dict and last completed epoch
    """
    ckpt = torch.load(path, map_location="cpu")
    history = ckpt["history"]
    epoch_reached = ckpt["epoch_reached"]
    return history, epoch_reached


def plot_training_curves(
    history: dict[str, list[float]],
    steps_per_epoch: int,
    total_epochs: int,
    save_path: str,
    plot_every_n: int = 100,
    y_lim: float = 100.0,
) -> None:
    """
    Saves a 3-panel training graph (total ELBO, recon loss, KL loss) with a y-axis cap.

    Args:
        history:          dict with keys "elbo", "recon", "kl" containing per-step values
        steps_per_epoch:  optimizer steps per epoch (used to compute x-axis ticks)
        total_epochs:     total global epochs completed
        save_path:        full file path to save the .png
        plot_every_n:     downsample factor to reduce noise in the plot
        y_lim:            upper limit for the y-axis
    Returns:
        None
    """
    max_ticks = 10
    tick_step = max(1, total_epochs // max_ticks)
    tick_positions = [i * steps_per_epoch // plot_every_n for i in range(0, total_epochs + 1, tick_step)]
    tick_labels = [str(i) for i in range(0, total_epochs + 1, tick_step)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("elbo", "Total ELBO", "tab:blue"),
        ("recon", "Reconstruction Loss", "tab:orange"),
        ("kl", "KL Loss", "tab:green"),
    ]

    for ax, (key, title, color) in zip(axes, panels):  # noqa: B905
        downsampled = history[key][::plot_every_n]
        ax.plot(range(len(downsampled)), downsampled, color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_ylim(bottom=0, top=y_lim)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


def main() -> None:
    args = parse_args()

    history, epoch_reached = load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: epoch_reached={epoch_reached}, steps in history={len(history['elbo'])}")

    # Infer steps_per_epoch from history length if not provided
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = len(history["elbo"]) // epoch_reached
        print(f"Inferred steps_per_epoch: {steps_per_epoch} (pass --steps_per_epoch to override)")

    # Save into the same folder as the checkpoint
    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0].removesuffix("_checkpoint")
    save_path = os.path.join(out_dir, f"{run_name}_training_curves_replot.png")

    plot_training_curves(
        history=history,
        steps_per_epoch=steps_per_epoch,
        total_epochs=epoch_reached,
        save_path=save_path,
        plot_every_n=args.plot_every_n,
        y_lim=args.y_lim,
    )


if __name__ == "__main__":
    main()