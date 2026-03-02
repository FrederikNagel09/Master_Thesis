import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")

from src_old.implicit_neural_representation.dataloader import MNISTCoordDataset
from src_old.implicit_neural_representation.model import INRMLP
from src_old.implicit_neural_representation.utils import make_coord_grid, parse_weights_name


def run_inference():
    """
    Runs inference using a trained INRMLP model to reconstruct an image at any resolution, 
    and plots the original vs reconstructed images side by side.

    Usage example:
    python src/implicit_neural_representation/inference.py \
        --weights src/implicit_neural_representation/weights/image_2_20_20_20.pth \
        --height 512 \
        --width 512 \
        --mnist_dir data/MNIST/raw \
        --out_dir src/implicit_neural_representation/results

    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="INR inference — reconstruct an image at any resolution.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the .pth weights file, e.g. src/implicit_neural_representation/weights/trial_0_20_20_20.pth",
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
        "--out_dir",
        type=str,
        default="src/implicit_neural_representation/results",
        help="Directory to save the output plot (default: src/implicit_neural_representation/results).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Parse weights filename to get index + layer sizes
    # ------------------------------------------------------------------
    img_idx, h1, h2, h3 = parse_weights_name(args.weights)

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
    # 4. Load original image
    # ------------------------------------------------------------------
    dataset = MNISTCoordDataset(mnist_raw_dir=args.mnist_dir, image_index=img_idx)
    original = dataset.image

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
    run_inference()
