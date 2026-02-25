import argparse
import math
import os
import sys

import torch

sys.path.append(".")

from src.INR.dataloader import MNISTCoordDataset
from src.INR.model import INRMLP
from src.INR.train import train

WEIGHTS_DIR = "src/INR/weights"


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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Train an INR MLP on a single MNIST image.")
    parser.add_argument("--index", type=int, default=0, help="Index of the MNIST image to fit (default: 0).")
    parser.add_argument(
        "--name", type=str, default="trial_", help="Base name for the run. The image index is appended automatically (default: 'trial_')."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of pixels used for validation (default: 0.1).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset = MNISTCoordDataset(mnist_raw_dir="data/MNIST/raw", image_index=args.index)

    height, width = dataset.image_shape
    num_pixels = height * width
    print(f"Image index : {args.index}")
    print(f"Image shape : {height} x {width}  ({num_pixels} pixels)")

    # ------------------------------------------------------------------
    # 2. Compute layer sizes so #params ≈ #pixels
    # ------------------------------------------------------------------
    h1, h2, h3 = compute_layer_sizes(num_pixels)
    model = INRMLP(h1=h1, h2=h2, h3=h3)
    num_params = count_parameters(model)

    print(f"Layer sizes : h1={h1}, h2={h2}, h3={h3}")
    print(f"Parameters  : {num_params:,}  (target: {num_pixels:,})")

    # ------------------------------------------------------------------
    # 3. Build run name:  <base><index>_<h1>_<h2>_<h3>
    # ------------------------------------------------------------------
    run_name = f"{args.name}{args.index}_{h1}_{h2}_{h3}"
    print(f"Run name    : {run_name}\n")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    model = train(
        model=model,
        dataset=dataset,
        name=run_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # ------------------------------------------------------------------
    # 5. Save weights
    # ------------------------------------------------------------------
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weights_path = os.path.join(WEIGHTS_DIR, f"{run_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")


if __name__ == "__main__":
    main()
