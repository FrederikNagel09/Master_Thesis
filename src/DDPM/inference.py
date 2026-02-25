"""
inference.py  -  load a trained DDPM checkpoint and generate a 4x4 image grid.

Example usage:
    python src/DDPM/inference.py --Master_Thesis/src/DDPM/weights/ddpm_mnist_epoch030.pt
    python src/DDPM/inference.py --weights src/DDPM/weights/ddpm_mnist_epoch010.pt --out_path src/DDPM/results/sample_grid.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision

sys.path.append(".")

from src.DDPM.model import Diffusion, UNet


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM MNIST inference - generate a 4x4 grid")

    parser.add_argument("--weights", type=str, required=True, help="Path to the saved model weights (.pt file)")
    parser.add_argument(
        "--out_path", type=str, default="src/DDPM/results/inference_grid.png", help="Output path for the generated image grid"
    )

    # Must match the values used during training
    parser.add_argument("--img_size", type=int, default=32, help="Spatial size used during training")
    parser.add_argument("--channels", type=int, default=32, help="Base channel count used during training")
    parser.add_argument("--time_dim", type=int, default=256, help="Time embedding dim used during training")
    parser.add_argument("--T", type=int, default=500, help="Diffusion timesteps used during training")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu / cuda / mps)")

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

    # Load model
    model = UNet(img_size=args.img_size, c_in=1, c_out=1, time_dim=args.time_dim, channels=args.channels, device=device).to(device)

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from: {args.weights}")

    # Diffusion sampler
    diffusion = Diffusion(T=args.T, beta_start=1e-4, beta_end=0.02, img_size=args.img_size, img_channels=1, device=device)

    # Generate 16 images (4x4 grid)
    sampled = diffusion.p_sample_loop(model, batch_size=16)  # uint8, shape (16, 1, H, W)

    # Build grid
    grid = torchvision.utils.make_grid(sampled, nrow=4, padding=2)
    ndarr = grid.permute(1, 2, 0).cpu().numpy().squeeze()  # H x W (grayscale)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(ndarr, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("DDPM - Generated MNIST digits")
    plt.tight_layout()
    plt.savefig(args.out_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"4x4 grid saved to: {args.out_path}")


if __name__ == "__main__":
    main()
