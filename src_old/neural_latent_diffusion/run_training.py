"""
run_training.py  -  entry point for Latent NDM MNIST training.

Example usage:
    python src/neural_latent_diffusion/run_training.py \
        --num_epochs_vae 1 \
        --num_epochs_ndm 1 \
        --batch_size 16 \
        --T 10 \
        --latent_channels 8 \
        --latent_size 8 \
        --ndm_hidden_dim 16 \
        --ndm_num_layers 2 \
        --ndm_time_dim 4 \
        --unet_channels 8 \
        --time_dim 8 \
        --beta_kl 0.001 \
        --lambda_vae 0.1 \
        --subset_pct 0.01


Two-phase training:
  Phase 1 - VAE warm-up    (num_epochs_vae epochs, only VAE trained)
  Phase 2 - Joint NDM      (num_epochs_ndm epochs, UNet + F_φ + VAE trained)
"""

import argparse
import json
import os
import sys

import torch

sys.path.append(".")

from src_old.neural_latent_diffusion.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent NDM on MNIST")
    # Diffusion
    parser.add_argument("--T", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    # VAE architecture
    parser.add_argument("--latent_channels", type=int, default=8, help="Number of channels in the VAE latent space")
    parser.add_argument("--latent_size", type=int, default=8, help="Spatial size of the VAE latent (4 → 4x4 latent grid)")
    # NDM Transform
    parser.add_argument("--ndm_hidden_dim", type=int, default=256, help="Hidden dim of the NDM MLP transform F_φ")
    parser.add_argument("--ndm_num_layers", type=int, default=3, help="Number of residual layers in F_φ")
    parser.add_argument("--ndm_time_dim", type=int, default=128, help="Time embedding dim inside F_φ")
    # UNet
    parser.add_argument("--img_size", type=int, default=32, help="Pixel-space image size (resize target)")
    parser.add_argument("--unet_channels", type=int, default=64, help="Base channel count for the UNet")
    parser.add_argument("--time_dim", type=int, default=256, help="Time embedding dimension for the UNet")
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs_vae", type=int, default=5, help="VAE warm-up epochs (Phase 1)")
    parser.add_argument("--num_epochs_ndm", type=int, default=10, help="Joint NDM training epochs (Phase 2)")
    parser.add_argument("--beta_kl", type=float, default=0.01, help="KL weight in VAE loss")
    parser.add_argument("--lambda_vae", type=float, default=0.1, help="Weight of VAE loss during joint NDM training")
    # Paths
    parser.add_argument("--experiment_name", type=str, default="latent_ndm_mnist")
    parser.add_argument("--weights_dir", type=str, default="src/neural_latent_diffusion/weights")
    parser.add_argument("--graphs_dir", type=str, default="src/neural_latent_diffusion/graphs")
    parser.add_argument("--results_dir", type=str, default="src/neural_latent_diffusion/results")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--subset_pct", type=float, default=1.0, help="Fraction of dataset to use for faster testing")
    # Device
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps  (auto-detected if omitted)")
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
    print(
        f"Architecture: VAE latent {args.latent_channels}x{args.latent_size}x{args.latent_size}  |  "
        f"UNet channels={args.unet_channels}  |  F_φ hidden={args.ndm_hidden_dim}x{args.ndm_num_layers}"
    )
    print(
        f"Schedule: {args.num_epochs_vae} VAE warm-up + {args.num_epochs_ndm} joint NDM epochs  |  "
        f"T={args.T}  β_kl={args.beta_kl}  λ_vae={args.lambda_vae}"
    )

    _ = train(
        device=device,
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        ndm_hidden_dim=args.ndm_hidden_dim,
        ndm_num_layers=args.ndm_num_layers,
        ndm_time_dim=args.ndm_time_dim,
        img_size=args.img_size,
        unet_channels=args.unet_channels,
        time_dim=args.time_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs_vae=args.num_epochs_vae,
        num_epochs_ndm=args.num_epochs_ndm,
        beta_kl=args.beta_kl,
        lambda_vae=args.lambda_vae,
        experiment_name=args.experiment_name,
        weights_dir=args.weights_dir,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        data_root=args.data_root,
        subset_pct=args.subset_pct,
    )

    # ------------------------------------------------------------------
    # Save experiment config — all hyperparameters + weight paths in one
    # JSON file so inference only needs a single --config argument.
    # ------------------------------------------------------------------
    epoch = args.num_epochs_ndm  # final epoch index used when saving weights
    config = {
        # Hyperparameters
        "T": args.T,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "latent_channels": args.latent_channels,
        "latent_size": args.latent_size,
        "ndm_hidden_dim": args.ndm_hidden_dim,
        "ndm_num_layers": args.ndm_num_layers,
        "ndm_time_dim": args.ndm_time_dim,
        "img_size": args.img_size,
        "unet_channels": args.unet_channels,
        "time_dim": args.time_dim,
        # Weight paths (mirror the names used inside train())
        "vae_weights": os.path.join(args.weights_dir, f"{args.experiment_name}_vae_final.pt"),
        "unet_weights": os.path.join(args.weights_dir, f"{args.experiment_name}_unet_epoch{epoch:03d}.pt"),
        "transform_weights": os.path.join(args.weights_dir, f"{args.experiment_name}_transform_epoch{epoch:03d}.pt"),
        # Misc
        "experiment_name": args.experiment_name,
        "results_dir": args.results_dir,
    }

    os.makedirs(args.weights_dir, exist_ok=True)
    config_path = os.path.join(args.weights_dir, f"{args.experiment_name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Experiment config saved to: {config_path}")


if __name__ == "__main__":
    main()
