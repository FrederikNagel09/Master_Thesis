import argparse
import os
import sys

import torch
from torch.utils.data import Subset

sys.path.append(".")

from src_old.inr_ddpm_hypernetwork.dataloader import MNISTHyperDataset
from src_old.inr_ddpm_hypernetwork.model import DiffusionHyperINR
from src_old.inr_ddpm_hypernetwork.train import train

WEIGHTS_DIR = "src/inr_ddpm_hypernetwork/weights"


def run_training():
    """
    Train a DiffusionHyperINR model end-to-end on MNIST.

    The UNet denoiser and hypernetwork MLP are trained jointly:
        - Denoising loss: UNet learns to predict clean images from noisy ones
        - Reconstruction loss: gradients flow back through the full pipeline
          (hypernetwork -> INR weights -> pixel reconstruction)

    Inference pipeline after training:
        noise -> UNet (full reverse chain) -> MNIST image -> HyperNet -> INR weights
        -> query INR at any (x,y) resolution

    Usage:
    python src/inr_ddpm_hypernetwork/run_training.py \
        --name diffusion_hyper_run \
        --epochs 10 \
        --batch_size 32 \
        --lr 1e-4 \
        --inr_h 32 \
        --hyper_h 256 \
        --unet_channels 32 \
        --T 1000 \
        --lambda_denoise 0.5
    """
    parser = argparse.ArgumentParser(description="Train DiffusionHyperINR on MNIST.")
    parser.add_argument("--name", type=str, default="diffusion_hyper", help="Base name for the run.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--inr_h", type=int, default=32, help="Hidden dim of each INR layer.")
    parser.add_argument("--hyper_h", type=int, default=256, help="Hidden dim of the hypernetwork.")
    parser.add_argument("--unet_channels", type=int, default=32, help="Base channel count for the UNet.")
    parser.add_argument("--omega_0", type=float, default=15.0, help="SIREN omega_0.")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion timesteps.")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Diffusion beta start.")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Diffusion beta end.")
    parser.add_argument(
        "--lambda_denoise",
        type=float,
        default=1.0,
        help="Weight for the denoising loss. "
        "Higher values push the UNet toward better image generation; "
        "lower values prioritize INR reconstruction. (default: 1.0)",
    )
    parser.add_argument("--split", type=str, default="train", help="MNIST split: 'train' or 'test'.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    dataset = MNISTHyperDataset(mnist_raw_dir="data/MNIST/raw", split=args.split)

    # Subset: use first N images (or random sample)
    subset_frac = 0.01
    n = int(len(dataset) * subset_frac)
    dataset = Subset(dataset, range(n))

    print(f"Dataset size: {len(dataset)} images")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = DiffusionHyperINR(
        h1=args.inr_h,
        h2=args.inr_h,
        h3=args.inr_h,
        omega_0=args.omega_0,
        hyper_h=args.hyper_h,
        unet_channels=args.unet_channels,
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    run_name = f"{args.name}_inr{args.inr_h}_hyper{args.hyper_h}_unet{args.unet_channels}"

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
        lambda_denoise=args.lambda_denoise,
    )

    # ------------------------------------------------------------------
    # 4. Save weights
    # ------------------------------------------------------------------
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    unet_path = os.path.join(WEIGHTS_DIR, f"{run_name}_unet.pth")
    hypernet_path = os.path.join(WEIGHTS_DIR, f"{run_name}_hypernet.pth")

    torch.save(model.unet.state_dict(), unet_path)
    torch.save(model.hypernet.state_dict(), hypernet_path)

    print(f"UNet weights saved to       : {unet_path}")
    print(f"HyperNetwork weights saved to: {hypernet_path}")


if __name__ == "__main__":
    run_training()
