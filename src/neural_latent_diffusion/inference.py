"""
inference.py  -  load a trained Latent NDM checkpoint and generate a 4x4 image grid.

All three components (VAE, NDM transform F_φ, UNet) must be provided.

Example usage:
    python src/neural_latent_diffusion/inference.py \\
        --vae_weights   src/neural_latent_diffusion/weights/latent_ndm_mnist_vae_final.pt \\
        --unet_weights  src/neural_latent_diffusion/weights/latent_ndm_mnist_unet_epoch010.pt \\
        --transform_weights src/neural_latent_diffusion/weights/latent_ndm_mnist_transform_epoch010.pt

    # Override output path:
    python src/neural_latent_diffusion/inference.py \\
        --vae_weights   src/neural_latent_diffusion/weights/latent_ndm_mnist_vae_final.pt \\
        --unet_weights  src/neural_latent_diffusion/weights/latent_ndm_mnist_unet_epoch010.pt \\
        --transform_weights src/neural_latent_diffusion/weights/latent_ndm_mnist_transform_epoch010.pt \\
        --out_path src/neural_latent_diffusion/results/sample_grid.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision

sys.path.append(".")

from src.neural_latent_diffusion.model import VAE, LatentNDMDiffusion, NDMTransform, UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Latent NDM MNIST inference - generate a 4x4 grid")

    # Required weight paths
    parser.add_argument("--vae_weights", type=str, required=True, help="Path to saved VAE weights (.pt)")
    parser.add_argument("--unet_weights", type=str, required=True, help="Path to saved UNet weights (.pt)")
    parser.add_argument("--transform_weights", type=str, required=True, help="Path to saved NDM transform F_φ weights (.pt)")

    parser.add_argument("--out_path", type=str, default="src/neural_latent_diffusion/results/inference_grid.png")

    # Architecture - must match training
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--unet_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--ndm_hidden_dim", type=int, default=256)
    parser.add_argument("--ndm_num_layers", type=int, default=3)
    parser.add_argument("--ndm_time_dim", type=int, default=128)
    parser.add_argument("--T", type=int, default=500)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=16, help="Number of images to generate (default 16 → 4x4 grid)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Running inference on: {device}")

    # -----------------------------------------------------------------------
    # Build and load models
    # -----------------------------------------------------------------------
    vae = VAE(img_channels=1, latent_channels=args.latent_channels, latent_size=args.latent_size).to(device)
    vae.load_state_dict(torch.load(args.vae_weights, map_location=device))
    vae.eval()
    print(f"VAE loaded from: {args.vae_weights}")

    latent_dim = args.latent_channels * args.latent_size * args.latent_size
    transform = NDMTransform(
        latent_dim=latent_dim,
        time_dim=args.ndm_time_dim,
        hidden_dim=args.ndm_hidden_dim,
        num_layers=args.ndm_num_layers,
    ).to(device)
    transform.load_state_dict(torch.load(args.transform_weights, map_location=device))
    transform.eval()
    print(f"NDM Transform F_φ loaded from: {args.transform_weights}")

    unet = UNet(
        img_size=args.latent_size,
        c_in=args.latent_channels,
        c_out=args.latent_channels,
        time_dim=args.time_dim,
        device=device,
        channels=args.unet_channels,
    ).to(device)
    unet.load_state_dict(torch.load(args.unet_weights, map_location=device))
    unet.eval()
    print(f"UNet loaded from: {args.unet_weights}")

    # -----------------------------------------------------------------------
    # Diffusion sampler
    # -----------------------------------------------------------------------
    latent_shape = (args.latent_channels, args.latent_size, args.latent_size)
    diffusion = LatentNDMDiffusion(
        T=args.T,
        beta_start=1e-4,
        beta_end=0.02,
        latent_shape=latent_shape,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------
    nrow = max(1, int(args.n_images**0.5))
    sampled = diffusion.p_sample_loop(unet, transform, vae, batch_size=args.n_images)

    # Build and save grid
    grid = torchvision.utils.make_grid(sampled, nrow=nrow, padding=2)
    ndarr = grid.permute(1, 2, 0).cpu().numpy().squeeze()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(ndarr, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("Latent NDM - Generated MNIST digits")
    plt.tight_layout()
    plt.savefig(args.out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"{nrow}x{nrow} grid saved to: {args.out_path}")


if __name__ == "__main__":
    main()
