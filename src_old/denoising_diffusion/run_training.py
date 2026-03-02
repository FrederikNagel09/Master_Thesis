"""
run_training.py  -  entry point for DDPM MNIST training.

Example usage:
    python src/denoising_diffusion/run_training.py --num_epochs 1 --batch_size 128 --lr 1e-3
"""

import argparse
import sys

import torch

sys.path.append(".")

from src_old.denoising_diffusion.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM on MNIST")

    # Diffusion
    parser.add_argument("--T", type=int, default=500, help="Total diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta schedule start value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta schedule end value")

    # Architecture
    parser.add_argument("--img_size", type=int, default=32, help="Spatial size images are resized to (must be power-of-2 divisible by 8)")
    parser.add_argument("--channels", type=int, default=32, help="Base channel count for UNet")
    parser.add_argument("--time_dim", type=int, default=256, help="Timestep embedding dimension")

    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")

    # Paths
    parser.add_argument("--experiment_name", type=str, default="ddpm_mnist", help="Run name (used in filenames)")
    parser.add_argument("--weights_dir", type=str, default="src/DDPM/weights", help="Where to save model weights")
    parser.add_argument("--graphs_dir", type=str, default="src/DDPM/graphs", help="Where to save loss plot")
    parser.add_argument("--results_dir", type=str, default="src/DDPM/results", help="Where to save sampled images")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for MNIST download")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu', 'cuda', 'mps'. Auto-detected if not specified.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Training on device: {device}")

    train(
        device=device,
        T=args.T,
        img_size=args.img_size,
        channels=args.channels,
        time_dim=args.time_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        experiment_name=args.experiment_name,
        weights_dir=args.weights_dir,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
