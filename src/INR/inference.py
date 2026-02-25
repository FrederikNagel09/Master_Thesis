import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")

from src.INR.dataloader import MNISTCoordDataset
from src.INR.model import INRMLP


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


def main():
    parser = argparse.ArgumentParser(description="INR inference — reconstruct an image at any resolution.")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the .pth weights file, e.g. src/INR/weights/trial_0_20_20_20.pth"
    )
    parser.add_argument("--height", type=int, required=True, help="Output image height in pixels.")
    parser.add_argument("--width", type=int, required=True, help="Output image width in pixels.")
    parser.add_argument(
        "--mnist_dir",
        type=str,
        default="data/MNIST/raw",
        help="Path to MNIST raw directory for loading the original image (default: data/MNIST/raw).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="src/INR/graphs", help="Directory to save the output plot (default: src/INR/graphs)."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Parse weights filename to get index + layer sizes
    # ------------------------------------------------------------------
    img_idx, h1, h2, h3 = parse_weights_name(args.weights)
    print(f"Image index : {img_idx}")
    print(f"Layer sizes : h1={h1}, h2={h2}, h3={h3}")
    print(f"Output size : {args.height} x {args.width}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    model = INRMLP(h1=h1, h2=h2, h3=h3)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # ------------------------------------------------------------------
    # 3. Build coordinate grid at requested resolution and run inference
    # ------------------------------------------------------------------
    coords = make_coord_grid(args.height, args.width)  # (H*W, 2)

    with torch.no_grad():
        preds = model(coords).numpy().reshape(args.height, args.width)  # (H, W)

    # ------------------------------------------------------------------
    # 4. Load original image (always at native 28x28)
    # ------------------------------------------------------------------
    dataset = MNISTCoordDataset(mnist_raw_dir=args.mnist_dir, image_index=img_idx)
    original = dataset.image  # (28, 28) float32 in [0, 1]

    # ------------------------------------------------------------------
    # 5. Plot original vs reconstruction
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original (28x28)")
    axes[0].axis("off")

    axes[1].imshow(preds, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Reconstruction ({args.height}x{args.width})")
    axes[1].axis("off")

    plt.suptitle(f"INR Reconstruction — image index {img_idx}", y=1.02)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"img_{img_idx}_{args.height}x{args.width}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
