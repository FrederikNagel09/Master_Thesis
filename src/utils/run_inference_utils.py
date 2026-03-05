import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")

from src.dataloaders.MNISTCoord import MNISTCoordDataset
from src.models.inr_siren import SirenINR
from src.utils.general_utils import make_coord_grid


def run_inference_siren_inr(args, config):
    """
    Runs inference using a trained INR model to reconstruct an image at any resolution,
    and plots the original vs reconstructed images side by side.
    """

    # ------------------------------------------------------------------
    # 1. Parse weights filename to get index + layer sizes
    # ------------------------------------------------------------------
    img_idx = config["index"]
    h1 = config["h1"]
    h2 = config["h2"]
    h3 = config["h3"]
    omega_0 = config["omega_0"]  # default to 30.0 if not present

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    model = SirenINR(h1=h1, h2=h2, h3=h3, omega_0=omega_0)
    model.load_state_dict(torch.load(config["weights_path"], map_location="cpu"))
    model.eval()

    # ------------------------------------------------------------------
    # 3. Build coordinate grid at requested resolution and run inference
    # ------------------------------------------------------------------
    coords = make_coord_grid(args.height, args.width)  # (H*W, 2)

    with torch.no_grad():
        preds = model(coords).numpy().reshape(args.height, args.width)  # (H, W)

    # ------------------------------------------------------------------
    # 4. Load original image
    # ------------------------------------------------------------------
    dataset = MNISTCoordDataset(mnist_raw_dir="data/MNIST/raw", image_index=img_idx)
    original = dataset.image

    # ------------------------------------------------------------------
    # 5. Plot original vs reconstruction
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original (28x28)")
    axes[0].axis("off")

    axes[1].imshow(preds, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Reconstruction ({args.height}x{args.width})")
    axes[1].axis("off")

    plt.suptitle(f"INR Reconstruction — image index {img_idx}", y=1.02)
    plt.tight_layout()

    out_dir = "src/results/basic_inr/samples"
    out_path = os.path.join(out_dir, f"img_{img_idx}_{args.height}x{args.width}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


def run_inference_inr_mlp_hypernet(args, config):
    """
    Run inference with a trained HyperINR model.
    Picks 5 random MNIST images, shows originals on top and upscaled reconstructions below.
    """
    # Imports:
    import random

    from src.models.inr_mlp_hypernet import HyperINR

    h1 = config["h1"]
    h2 = config["h2"]
    h3 = config["h3"]
    omega_0 = config["omega_0"]
    hyper_h = config["hyper_h"]
    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model = HyperINR(h1=h1, h2=h2, h3=h3, omega_0=omega_0, hyper_h=hyper_h)
    model.load_state_dict(torch.load(config["weights_path"], map_location="cpu"))
    model.eval()

    # ------------------------------------------------------------------
    # 2. Pick 5 random image indices and load them
    # ------------------------------------------------------------------
    rng = random.Random()
    indices = rng.sample(range(10000), 5)  # sample from test set size

    originals = []
    images_flat = []
    for idx in indices:
        ds = MNISTCoordDataset(mnist_raw_dir="data/MNIST/raw", image_index=idx)
        originals.append(ds.image)
        images_flat.append(ds.image_flat)

    images_flat = torch.stack(images_flat, dim=0)  # (5, 784)

    # ------------------------------------------------------------------
    # 3. Build coordinate grid at requested resolution
    # ------------------------------------------------------------------
    coords = make_coord_grid(args.height, args.width)  # (H*W, 2)
    coords_batch = coords.unsqueeze(0).expand(5, -1, -1)  # (5, H*W, 2)

    # ------------------------------------------------------------------
    # 4. Run inference on all 5 images at once
    # ------------------------------------------------------------------
    with torch.no_grad():
        preds = model(images_flat, coords_batch)  # (5, H*W, 1)

    recons = preds.squeeze(-1).numpy().reshape(5, args.height, args.width)

    # ------------------------------------------------------------------
    # 5. Plot: originals on top row, reconstructions on bottom row
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for col in range(5):
        axes[0, col].imshow(originals[col], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Original #{indices[col]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(recons[col], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{args.height}x{args.width}")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)

    plt.suptitle("HyperINR — Original vs Upscaled Reconstruction", fontsize=14)
    plt.tight_layout()

    out_dir = "src/results/hypernet_inr/samples"
    out_path = os.path.join(out_dir, f"hypernetINR_5_samples_{args.height}x{args.width}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


def run_inference_ddpm(args):
    # Imports:
    import torchvision

    from src.models.ddpm import Diffusion
    from src.models.unet import UNet

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


def run_inference_ndm(args, config):
    import numpy as np

    from src.models.ndm import NDM

    # ---- Load model ----
    model = NDM(
        in_channels=1,
        T=config["T"],
        fphi_base_ch=config["fphi_ch"],
        denoiser_base_ch=config["denoiser_ch"],
        time_emb_dim=config["time_emb_dim"],
    )
    model.load_state_dict(torch.load(config["weights_path"], map_location="cpu"))
    model.eval()

    # ---- Sample ----
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    n = args.grid_size**2
    with torch.no_grad():
        samples = model.sample(n, device=torch.device(device), steps=args.sample_steps)

    # ---- Build grid and save ----
    import matplotlib.pyplot as plt

    n = args.grid_size**2
    imgs = ((samples * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
    rows = [np.concatenate([imgs[r * args.grid_size + c, 0] for c in range(args.grid_size)], axis=1) for r in range(args.grid_size)]
    grid = np.concatenate(rows, axis=0)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ax.set_title(f"NDM Samples — {args.grid_size**2} images", fontsize=12)

    out_dir = "src/results/ndm/samples"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.grid_size}x{args.grid_size}_{config['name']}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved to: {out_path}")


def run_inference_vae(args, config):
    """
    Runs inference using a trained VAE model to sample a grid of images from the prior,
    and saves the result.
    """
    # imports:
    from src.models.prior import GaussianPrior, MoGPrior, VAMPPrior
    from src.models.vae import VAE
    from src.models.vae_coders import BernoulliDecoder, GaussianEncoder
    # Imports for encoder, decoder, and prior

    # ------------------------------------------------------------------
    # 1. Parse config
    # ------------------------------------------------------------------
    latent_dim = config["latent_dim"]
    hidden_dims = config["hidden_dims"]
    prior_type = config["prior"]
    device = config.get("device", "cpu")

    # ------------------------------------------------------------------
    # 2. Rebuild model architecture
    # ------------------------------------------------------------------
    encoder = GaussianEncoder(input_dim=784, latent_dim=latent_dim, hidden_dims=hidden_dims)
    decoder = BernoulliDecoder(latent_dim=latent_dim, output_shape=(28, 28), hidden_dims=hidden_dims)

    if prior_type == "gaussian":
        prior = GaussianPrior(latent_dim=latent_dim)
    elif prior_type == "mog":
        prior = MoGPrior(latent_dim=latent_dim)
    elif prior_type == "vampp":
        prior = VAMPPrior()
    else:
        raise ValueError(f"Unsupported prior: {prior_type}")

    model = VAE(encoder=encoder, decoder=decoder, prior=prior)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()
    model.to(device)

    # ------------------------------------------------------------------
    # 3. Sample n x n grid from prior
    # ------------------------------------------------------------------
    n = args.grid_size  # grid will be n x n
    n_total = n * n

    with torch.no_grad():
        samples = model.sample(n_samples=n_total).cpu().numpy()  # (n_total, 28, 28)

    # ------------------------------------------------------------------
    # 4. Arrange into grid and plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(n, n, figsize=(n * 1.5, n * 1.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.suptitle(f"VAE Samples — {prior_type} prior, latent_dim={latent_dim}", y=1.01)
    plt.tight_layout()

    out_dir = "src/results/vae/samples"
    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(config["weights_path"]))[0]
    out_path = os.path.join(out_dir, f"{run_name}_{n}x{n}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")
