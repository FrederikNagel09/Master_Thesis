import math
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")


def make_coord_grid(height: int, width: int) -> torch.Tensor:
    """Build a (H*W, 2) coordinate grid normalized to [-1, 1]."""
    rows = torch.linspace(-1, 1, height)
    cols = torch.linspace(-1, 1, width)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)
    return coords


def mse_to_psnr(mse: float) -> float:
    """Convert MSE (on [0,1] pixel values) to PSNR in dB."""
    if mse == 0:
        return float("inf")
    return -10 * math.log10(mse)


def _save_plot(history: dict, name: str, graph_dir: str, current_epoch: int, total_epochs: int):
    """Save a training loss plot, updated after every epoch."""
    epochs = range(1, current_epoch + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_mse"], label="Train MSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"HyperINR Training — {name}")
    ax.legend()
    ax.set_xlim(1, total_epochs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{name}.png"), dpi=150)
    plt.close(fig)


def _save_reconstruction_hyper(model, dataset, name: str, graph_dir: str, device: str = "cpu", num_examples: int = 4):
    """
    Save a grid of original vs reconstructed images using the HyperINR model.

    Samples `num_examples` images from the dataset, runs them through the
    hypernetwork and INR, and plots originals alongside reconstructions.

    Args:
        model:        HyperINR instance
        dataset:      MNISTHyperDataset
        name:         Run name used for the output filename
        graph_dir:    Directory to save the figure
        device:       Device to run inference on
        num_examples: How many images to show side-by-side
    """
    underlying = dataset.dataset if hasattr(dataset, "dataset") else dataset
    h, w = underlying.image_shape

    indices = list(range(min(num_examples, len(dataset))))
    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))

    with torch.no_grad():
        for col, idx in enumerate(indices):
            image_flat, coords, _ = dataset[idx]

            image_flat = image_flat.unsqueeze(0).to(device)  # (1, 784)
            coords_in = coords.unsqueeze(0).to(device)  # (1, 784, 2)

            preds = model(image_flat, coords_in)  # (1, 784, 1)
            recon = preds.squeeze(0).squeeze(-1).cpu().numpy().reshape(h, w)
            original = image_flat.squeeze(0).cpu().numpy().reshape(h, w)

            axes[0, col].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"Original {idx}")
            axes[0, col].axis("off")

            axes[1, col].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title(f"Recon {idx}")
            axes[1, col].axis("off")

    plt.suptitle(f"HyperINR Reconstructions — {name}")
    plt.tight_layout()

    out_path = os.path.join(graph_dir, f"{name}_reconstruction.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Reconstruction grid saved to {out_path}")
