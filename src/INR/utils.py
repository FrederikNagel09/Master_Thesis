import os
import re
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")
import math
import sys

sys.path.append(".")


def parse_weights_name(weights_path: str) -> tuple[int, int, int, int]:
    """
    Extract image index and layer sizes from a filename like:
        src/INR/weights/trial_0_20_20_20.pth
    Returns: (img_index, h1, h2, h3)
    """
    stem = os.path.basename(weights_path).replace(".pth", "")
    match = re.search(r"_(\d+)_(\d+)_(\d+)_(\d+)$", stem)
    if not match:
        raise ValueError(f"Could not parse index and layer sizes from filename: {stem}\nExpected format: <name>_<index>_<h1>_<h2>_<h3>.pth")
    idx, h1, h2, h3 = (int(x) for x in match.groups())
    return idx, h1, h2, h3


def make_coord_grid(height: int, width: int) -> torch.Tensor:
    """Build a (H*W, 2) coordinate grid normalized to [-1, 1]."""
    rows = torch.linspace(-1, 1, height)
    cols = torch.linspace(-1, 1, width)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)
    return coords


def compute_layer_sizes(num_pixels: int) -> tuple[int, int, int]:
    """
    Compute three hidden layer sizes such that the total parameter count
    of the MLP (2 -> h1 -> h2 -> h3 -> 1) approximately matches num_pixels.

    Total params = (2*h1 + h1) + (h1*h2 + h2) + (h2*h3 + h3) + (h3*1 + 1)
                 ≈ 3*h + h^2 + h^2 + h   (for h1=h2=h3=h)
                 = 2*h^2 + 4*h

    Solving 2h^2 + 4h = N  →  h = (-4 + sqrt(16 + 8N)) / 4
    """
    h = int((-4 + math.sqrt(16 + 8 * num_pixels)) / 4)
    h = max(h, 8)  # minimum sanity floor
    return h + 2, h + 2, h + 2


# Add this function:
def _save_reconstruction(model, dataset, name, graph_dir, device):
    name = name.split("_")[0]

    height, width = dataset.image_shape
    model.eval().to(device)
    with torch.no_grad():
        coords = dataset.coords.to(device)  # all (H*W, 2) coords
        preds = model(coords).cpu().numpy().reshape(height, width)
    original = dataset.image  # already (H, W) float32 in [0,1]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(preds, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{name}_reconstruction.png"), dpi=150)
    plt.close(fig)
    print(f"Reconstruction saved to {graph_dir}/{name}_reconstruction.png")


# Add this function near the top of train.py:
def mse_to_psnr(mse: float) -> float:
    """Convert MSE (on [0,1] pixel values) to PSNR in dB."""
    if mse == 0:
        return float("inf")
    return -10 * math.log10(mse)  # add: import math at top


def _save_plot(history: dict, name: str, graph_dir: str, current_epoch: int, total_epochs: int):
    epochs = range(1, current_epoch + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_mse"], label="Train MSE", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"INR Training — {name}")
    ax.legend()
    ax.set_xlim(1, total_epochs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{name}.png"), dpi=150)
    plt.close(fig)
