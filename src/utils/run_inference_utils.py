import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision

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


def run_inference_ddpm(args, config):
    # Imports
    from src.models.ddpm import DDPM, Unet

    device = config.get("device", "cpu")

    network = Unet()

    # Define model
    model = DDPM(network, T=config["T"]).to(device)

    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()

    n = args.grid_size**2

    with torch.no_grad():
        samples = model.sample((n, 28 * 28))  # sample n images
        samples = samples.view(n, 1, 28, 28)  # reshape correctly

    # ---- Build grid and save ----
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    imgs = ((samples * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()

    rows = [np.concatenate([imgs[r * args.grid_size + c, 0] for c in range(args.grid_size)], axis=1) for r in range(args.grid_size)]

    grid = np.concatenate(rows, axis=0)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ax.set_title(f"DDPM Samples — {n} images", fontsize=12)

    out_dir = f"src/results/{config['model']}/samples"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.grid_size}x{args.grid_size}_{config['name']}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)

    plt.close(fig)
    print(f"Saved to: {out_path}")


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
    from src.models.vae_coders import BernoulliFullDecoder, GaussianFullEncoder
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
    encoder = GaussianFullEncoder(input_dim=784, latent_dim=latent_dim, hidden_dims=hidden_dims)
    decoder = BernoulliFullDecoder(latent_dim=latent_dim, output_shape=(28, 28), hidden_dims=hidden_dims)

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


# ---------------------------------------------------------------------------
# Add to run_inference_utils.py
# ---------------------------------------------------------------------------


def run_inference_inr_vae(args, config):
    """
    Sample a grid of images from a trained INRVAE.
    Mirrors run_inference_vae — same grid_size interface.
    """
    import matplotlib.pyplot as plt
    import torch.nn as nn

    from src.models.inr_vae_hypernet import INR, VAEINR
    from src.models.prior import GaussianPrior, MoGPrior
    from src.models.vae_coders import GaussianEncoder

    latent_dim = config["latent_dim"]
    device = config.get("device", "cpu")

    # --- INR ---
    # Small MLP: (x,y) coords -> pixel value. Has NO nn.Parameters — weights come from decoder.
    inr = INR(coord_dim=2, hidden_dim=config["inr_hidden_dim"], n_hidden=config["inr_layers"], out_dim=config["inr_out_dim"])

    # --- Encoder ---
    # Takes flattened binary image (784,) -> outputs mean+logvar for q(z|x)
    encoder_net = nn.Sequential(
        nn.Linear(784, config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], latent_dim * 2),  # outputs [mean | log-var], split inside GaussianEncoder
    )
    encoder = GaussianEncoder(encoder_net)

    # --- Decoder (hypernetwork) ---
    # Takes z (M,) -> flat INR weight vector
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], inr.num_weights),  # output dim must match INR parameter count exactly
    )
    # ------------------------------------------------------------------
    # 2. Prior
    # ------------------------------------------------------------------
    if config["prior"] == "gaussian":
        prior = GaussianPrior(latent_dim=latent_dim)
    elif config["prior"] == "mog":
        prior = MoGPrior(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported prior for INRVAE: {config['prior']}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    # --- Full model ---
    model = VAEINR(prior, encoder, decoder_net, inr, beta=1.0, prior_type=config["prior"]).to(device)

    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()
    resolutions = [64, 128, 256]
    n_samples = args.grid_size  # n_samples x n_samples grid

    def make_coord_grid(size, device):
        """Build a (size*size, 2) coordinate grid normalized to [-1, 1]."""
        lin = torch.linspace(-1, 1, size)
        grid_r, grid_c = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1).to(device)

    with torch.no_grad():
        z = model.prior().sample(torch.Size([n_samples**2])).to(device)  # (n_samples^2, M)
        flat_weights = model.decode_to_weights(z)  # (n_samples^2, num_weights)

        all_grids = {}
        for res in resolutions:
            coords = make_coord_grid(res, device)  # (res^2, 2)
            coords_batch = coords.unsqueeze(0).expand(n_samples**2, -1, -1)  # (n_samples^2, res^2, 2)
            pixels = model.inr(coords_batch, flat_weights)  # (n_samples^2, res^2, 1)
            images = pixels.squeeze(-1).view(n_samples**2, 1, res, res).cpu()  # (n_samples^2, 1, res, res)
            all_grids[res] = torchvision.utils.make_grid(images, nrow=n_samples, normalize=True, value_range=(0, 1))

    # Plot all three resolutions as subplots in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, res in zip(axes, resolutions):  # noqa: B905
        grid_np = all_grids[res].permute(1, 2, 0).numpy()
        ax.imshow(grid_np, cmap="gray", interpolation="nearest")
        ax.set_title(f"{res}x{res}", fontsize=16)
        ax.axis("off")

    out_dir = "src/results/vae_inr_hypernet/samples"
    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(config["weights_path"]))[0]
    out_path = os.path.join(out_dir, f"{run_name}_{n_samples}x{n_samples}.png")

    fig.suptitle(f"INR-VAE samples at multiple resolutions — {n_samples}x{n_samples} grid (same z)", fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Samples saved to {out_path}")
