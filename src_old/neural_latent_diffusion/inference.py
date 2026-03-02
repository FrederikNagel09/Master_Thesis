"""
inference.py  -  load a trained Latent NDM checkpoint and generate an image grid.
All model architecture and weight paths are read from the JSON config saved by train.py.

Example usage:
    python src/neural_latent_diffusion/inference.py \
        --config src/neural_latent_diffusion/weights/latent_ndm_mnist_config.json

    # Override output path and number of images:
    python src/neural_latent_diffusion/inference.py \
        --config src/neural_latent_diffusion/weights/latent_ndm_mnist_config.json \
        --out_path src/neural_latent_diffusion/results/sample_grid.png \
        --n_images 25
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision

sys.path.append(".")

from src_old.neural_latent_diffusion.model import VAE, LatentNDMDiffusion, NDMTransform, UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Latent NDM MNIST inference - generate a grid of images")
    # Single config file produced by training replaces all individual weight/arch flags
    parser.add_argument("--config", type=str, required=True, help="Path to experiment JSON config saved by train.py")
    # Inference-only options
    parser.add_argument("--out_path", type=str, default=None, help="Output image path (default: <results_dir>/inference_grid.png)")
    parser.add_argument("--n_images", type=int, default=16, help="Number of images to generate (default 16 → 4x4 grid)")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps  (auto-detected if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load experiment config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = json.load(f)
    print(f"Loaded config from: {args.config}")

    # Device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Running inference on: {device}")

    # Output path falls back to results_dir from config
    if args.out_path is not None:  # noqa: SIM108
        out_path = args.out_path
    else:
        out_path = os.path.join(cfg["results_dir"], "inference_grid.png")

    # ------------------------------------------------------------------
    # Build and load models using architecture params from config
    # ------------------------------------------------------------------
    vae = VAE(
        img_channels=1,
        latent_channels=cfg["latent_channels"],
        latent_size=cfg["latent_size"],
    ).to(device)
    vae.load_state_dict(torch.load(cfg["vae_weights"], map_location=device))
    vae.eval()
    print(f"VAE loaded from: {cfg['vae_weights']}")

    latent_dim = cfg["latent_channels"] * cfg["latent_size"] * cfg["latent_size"]
    transform = NDMTransform(
        latent_dim=latent_dim,
        time_dim=cfg["ndm_time_dim"],
        hidden_dim=cfg["ndm_hidden_dim"],
        num_layers=cfg["ndm_num_layers"],
    ).to(device)
    transform.load_state_dict(torch.load(cfg["transform_weights"], map_location=device))
    transform.eval()
    print(f"NDM Transform F_φ loaded from: {cfg['transform_weights']}")

    unet = UNet(
        img_size=cfg["latent_size"],
        c_in=cfg["latent_channels"],
        c_out=cfg["latent_channels"],
        time_dim=cfg["time_dim"],
        device=device,
        channels=cfg["unet_channels"],
    ).to(device)
    unet.load_state_dict(torch.load(cfg["unet_weights"], map_location=device))
    unet.eval()
    print(f"UNet loaded from: {cfg['unet_weights']}")

    # ------------------------------------------------------------------
    # Diffusion sampler
    # ------------------------------------------------------------------
    latent_shape = (cfg["latent_channels"], cfg["latent_size"], cfg["latent_size"])
    diffusion = LatentNDMDiffusion(
        T=cfg["T"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        latent_shape=latent_shape,
        device=device,
    )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    nrow = max(1, int(args.n_images**0.5))
    sampled = diffusion.p_sample_loop(unet, transform, vae, batch_size=args.n_images)

    grid = torchvision.utils.make_grid(sampled, nrow=nrow, padding=2)
    ndarr = grid.permute(1, 2, 0).cpu().numpy().squeeze()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(ndarr, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title(f"Latent NDM — {cfg['experiment_name']}")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"{nrow}x{nrow} grid saved to: {out_path}")


if __name__ == "__main__":
    main()
