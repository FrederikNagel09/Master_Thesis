import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")

from src_old.inr_mlp_hypernetwork.dataloader import MNISTCoordDataset
from src_old.inr_mlp_hypernetwork.model import HyperINR
from src_old.inr_mlp_hypernetwork.utils import make_coord_grid


def run_inference():
    """
    Run inference with a trained HyperINR model.
    Picks 5 random MNIST images, shows originals on top and upscaled reconstructions below.

    Usage:
    python src/inr_hypernetwork/inference.py \
        --weights src/inr_hypernetwork/weights/hyper_run_inr32_hyper256.pth \
        --height 512 \
        --width 512 \
        --inr_h 32 \
        --hyper_h 256
    """
    parser = argparse.ArgumentParser(description="HyperINR inference.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--inr_h", type=int, default=32)
    parser.add_argument("--hyper_h", type=int, default=256)
    parser.add_argument("--omega_0", type=float, default=1.0)
    parser.add_argument("--mnist_dir", type=str, default="data/MNIST/raw")
    parser.add_argument("--out_dir", type=str, default="src/inr_hypernetwork/results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model = HyperINR(h1=args.inr_h, h2=args.inr_h, h3=args.inr_h, omega_0=args.omega_0, hyper_h=args.hyper_h)
    model.hypernet.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # ------------------------------------------------------------------
    # 2. Pick 5 random image indices and load them
    # ------------------------------------------------------------------
    rng = random.Random()
    indices = rng.sample(range(10000), 5)  # sample from test set size

    originals = []
    images_flat = []
    for idx in indices:
        ds = MNISTCoordDataset(mnist_raw_dir=args.mnist_dir, image_index=idx)
        originals.append(ds.image)
        images_flat.append(ds.image_flat)

    images_flat = torch.stack(images_flat, dim=0)  # (5, 784)

    # ------------------------------------------------------------------
    # 3. Build coordinate grid at requested resolution
    # ------------------------------------------------------------------
    coords = make_coord_grid(args.height, args.width)  # (H*W, 2)
    coords_batch = coords.unsqueeze(0).expand(5, -1, -1)  # (5, H*W, 2)

    # ------------------------------------------------------------------
    # 4. Run inference on all 5 images at once
    # ------------------------------------------------------------------
    with torch.no_grad():
        preds = model(images_flat, coords_batch)  # (5, H*W, 1)

    recons = preds.squeeze(-1).numpy().reshape(5, args.height, args.width)

    # ------------------------------------------------------------------
    # 5. Plot: originals on top row, reconstructions on bottom row
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for col in range(5):
        axes[0, col].imshow(originals[col], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Original #{indices[col]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(recons[col], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{args.height}x{args.width}")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)

    plt.suptitle("HyperINR — Original vs Upscaled Reconstruction", fontsize=14)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"hyper_5samples_{args.height}x{args.width}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    run_inference()
