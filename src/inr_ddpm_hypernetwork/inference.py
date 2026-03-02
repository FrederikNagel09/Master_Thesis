import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")

from src.inr_ddpm_hypernetwork.model import DiffusionHyperINR
from src.inr_ddpm_hypernetwork.utils import make_coord_grid


def run_inference():
    """
    Run inference with a trained DiffusionHyperINR model.

    Two modes:
        generate:     Sample 5 new images from pure noise, reconstruct at target resolution.
        reconstruct:  Load real MNIST images, pass through UNet (single denoise step), reconstruct.

    Example usage — generate new digits at 512x512:
    python src/inr_ddpm_hypernetwork/inference.py \
        --unet_weights  src/inr_ddpm_hypernetwork/weights/run_unet.pth \
        --hyper_weights src/inr_ddpm_hypernetwork/weights/run_hypernet.pth \
        --height 512 --width 512 \
        --inr_h 32 --hyper_h 256 --unet_channels 32 \
        --mode generate
    """
    parser = argparse.ArgumentParser(description="DiffusionHyperINR inference.")
    parser.add_argument("--unet_weights", type=str, required=True)
    parser.add_argument("--hyper_weights", type=str, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--inr_h", type=int, default=32)
    parser.add_argument("--hyper_h", type=int, default=256)
    parser.add_argument("--unet_channels", type=int, default=32)
    parser.add_argument("--omega_0", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "reconstruct"],
        help="'generate': sample from noise. 'reconstruct': denoise real images.",
    )
    parser.add_argument("--mnist_dir", type=str, default="data/MNIST/raw")
    parser.add_argument("--out_dir", type=str, default="src/inr_ddpm_hypernetwork/results")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of images to generate/show.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running inference on: {device}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model = DiffusionHyperINR(
        h1=args.inr_h,
        h2=args.inr_h,
        h3=args.inr_h,
        omega_0=args.omega_0,
        hyper_h=args.hyper_h,
        unet_channels=args.unet_channels,
        T=args.T,
    )
    model.unet.load_state_dict(torch.load(args.unet_weights, map_location="cpu"))
    model.hypernet.load_state_dict(torch.load(args.hyper_weights, map_location="cpu"))
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Build target coordinate grid at requested resolution
    # ------------------------------------------------------------------
    coords = make_coord_grid(args.height, args.width).to(device)  # (H*W, 2)

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    if args.mode == "generate":
        # Full reverse diffusion chain -> generate new MNIST-like digits
        generated_images, pixel_preds = model.sample_and_reconstruct(
            coords=coords,
            batch_size=args.n_samples,
            device=device,
        )
        # generated_images: (N, 1, 32, 32);  pixel_preds: (N, H*W, 1)

        source_images = generated_images[:, 0, 2:30, 2:30].cpu().numpy()  # crop 32->28 for display
        recons = pixel_preds.squeeze(-1).cpu().numpy().reshape(args.n_samples, args.height, args.width)
        source_title = "Generated (UNet)"
        source_label = "Generated"

    else:  # reconstruct mode
        import random

        from src.inr_ddpm_hypernetwork.dataloader import MNISTCoordDataset

        rng = random.Random()
        indices = rng.sample(range(10000), args.n_samples)

        images_32 = []
        for idx in indices:
            ds = MNISTCoordDataset(mnist_raw_dir=args.mnist_dir, image_index=idx)
            images_32.append(ds.image_32)

        images_32 = torch.stack(images_32, dim=0).to(device)  # (N, 1, 32, 32)
        coords_batch = coords.unsqueeze(0).expand(args.n_samples, -1, -1)  # (N, H*W, 2)

        with torch.no_grad():
            diffusion = model._get_diffusion(device)
            # Use t=1 (nearly clean) for reconstruction to demonstrate the pipeline
            t = torch.ones(args.n_samples, dtype=torch.long, device=device)
            x_t, _ = diffusion.q_sample(images_32, t)
            x0_hat = model.unet(x_t, t)
            img_flat = x0_hat.view(args.n_samples, -1)
            flat_weights = model.hypernet(img_flat)
            pixel_preds = model.inr(coords_batch, flat_weights)

        source_images = images_32[:, 0, 2:30, 2:30].cpu().numpy()
        recons = pixel_preds.squeeze(-1).cpu().numpy().reshape(args.n_samples, args.height, args.width)
        source_title = "Original MNIST"
        source_label = "Original"

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, args.n_samples, figsize=(3 * args.n_samples, 6))

    for col in range(args.n_samples):
        axes[0, col].imshow(source_images[col], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{source_title} #{col}")
        axes[0, col].axis("off")

        axes[1, col].imshow(recons[col], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"INR {args.height}x{args.width}")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel(source_label, fontsize=12)
    axes[1, 0].set_ylabel("INR Reconstruction", fontsize=12)

    plt.suptitle(f"DiffusionHyperINR — {source_title} vs INR at {args.height}x{args.width}", fontsize=13)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"diffusion_hyper_{args.mode}_{args.height}x{args.width}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    run_inference()
