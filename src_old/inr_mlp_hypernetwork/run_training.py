import argparse
import os
import sys

import torch

sys.path.append(".")

from src_old.inr_mlp_hypernetwork.dataloader import MNISTHyperDataset
from src_old.inr_mlp_hypernetwork.model import HyperINR
from src_old.inr_mlp_hypernetwork.train import train

WEIGHTS_DIR = "src/inr_hypernetwork/weights"


def run_training():
    """
    Train a HyperINR model on MNIST.

    The hypernetwork learns to map a full MNIST image to a set of INR weights.
    The INR then maps pixel coordinates to pixel values using those weights.

    Usage:
    python src/inr_hypernetwork/run_training.py \
        --name hyper_run \
        --epochs 5 \
        --batch_size 32 \
        --lr 1e-3 \
        --inr_h 32 \
        --hyper_h 256
    """
    parser = argparse.ArgumentParser(description="Train HyperINR on MNIST.")
    parser.add_argument("--name", type=str, default="hyper", help="Base name for the run.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    parser.add_argument("--inr_h", type=int, default=20, help="Hidden size of each INR layer (default: 20).")
    parser.add_argument("--hyper_h", type=int, default=256, help="Hidden size of the hypernetwork (default: 256).")
    parser.add_argument("--omega_0", type=float, default=1, help="SIREN omega_0 (default: 20.0).")
    parser.add_argument("--split", type=str, default="train", help="MNIST split to train on: 'train' or 'test' (default: train).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    dataset = MNISTHyperDataset(mnist_raw_dir="data/MNIST/raw", split=args.split)
    print(f"Dataset size: {len(dataset)} images")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = HyperINR(
        h1=args.inr_h,
        h2=args.inr_h,
        h3=args.inr_h,
        omega_0=args.omega_0,
        hyper_h=args.hyper_h,
    )

    run_name = f"{args.name}_inr{args.inr_h}_hyper{args.hyper_h}"

    # ------------------------------------------------------------------
    # 3. Train
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
    # 4. Save weights (hypernetwork only — the INR has no weights to save)
    # ------------------------------------------------------------------
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weights_path = os.path.join(WEIGHTS_DIR, f"{run_name}.pth")
    torch.save(model.hypernet.state_dict(), weights_path)
    print(f"Hypernetwork weights saved to: {weights_path}")


if __name__ == "__main__":
    run_training()
