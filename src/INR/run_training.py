import argparse
import os
import sys

import torch

sys.path.append(".")

from src.INR.dataloader import MNISTCoordDataset
from src.INR.model import INRMLP
from src.INR.train import train
from src.INR.utils import compute_layer_sizes

WEIGHTS_DIR = "src/INR/weights"


def run_training():
    """
    Run training of an INR MLP on a single MNIST image, with layer sizes chosen to match the number of pixels.

    Usage:
    python src/INR/run_training.py --index 5 --name image_ --epochs 1500 --batch_size 32 --lr 1e-3
    """
    # Args parsing initialization
    parser = argparse.ArgumentParser(description="Train an INR MLP on a single MNIST image.")
    parser.add_argument("--index", type=int, default=0, help="Index of the MNIST image to fit (default: 0).")
    parser.add_argument(
        "--name", type=str, default="img_", help="Base name for the run. The image index is appended automatically (default: 'trial_')."
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
    # ------------------------------------------------------------------
    # 2. Compute layer sizes so #params ≈ #pixels
    # ------------------------------------------------------------------
    h1, h2, h3 = compute_layer_sizes(num_pixels)
    model = INRMLP(h1=h1, h2=h2, h3=h3)
    # ------------------------------------------------------------------
    # 3. Build run name:  <base><index>_<h1>_<h2>_<h3>
    # ------------------------------------------------------------------
    run_name = f"{args.name}{args.index}_{h1}_{h2}_{h3}"
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
    run_training()
