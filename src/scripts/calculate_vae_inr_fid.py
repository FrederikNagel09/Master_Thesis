import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm

sys.path.append(".")

from src.utils.parser_utils import parse_config_vars

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def make_coord_grid(size: int, device) -> torch.Tensor:
    lin = torch.linspace(-1, 1, size)
    grid_r, grid_c = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1).to(device)


def get_inception_features(images_uint8: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    """
    images_uint8 : (N, H, W) or (N, H, W, 3)  uint8  [0-255]
    Returns      : (N, 2048) float32 activation matrix
    """
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception = InceptionV3([block_idx]).to(device).eval()

    # Ensure RGB  (N, 3, 299, 299)  float in [0, 1]
    resize = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # → [0,1]
            transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        ]
    )

    all_feats = []
    for start in tqdm(range(0, len(images_uint8), batch_size), desc="  Inception features"):
        batch_np = images_uint8[start : start + batch_size]
        tensors = torch.stack(
            [resize(Image.fromarray(img if img.ndim == 3 else img[:, :, None].repeat(3, axis=2))) for img in batch_np]
        ).to(device)
        with torch.no_grad():
            feats = inception(tensors)[0].squeeze(-1).squeeze(-1).cpu().numpy()
        all_feats.append(feats)

    return np.concatenate(all_feats, axis=0)


def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    return calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


# ─────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────


def compute_fid_and_save_grid(config: dict, grid_size: int = 8):
    """
    1. Load full MNIST training set → real features
    2. Sample grid_size² images from the INR-VAE → fake features
    3. Compute FID
    4. Save an 8x8 sample grid with title "INR-VAE  |  FID = {fid:.2f}"
    """
    from src.models.inr_vae_hypernet import INR, VAEINR
    from src.models.prior import GaussianPrior, MoGPrior
    from src.models.vae_coders import GaussianEncoder

    device = config.get("device", "cpu")
    latent_dim = config["latent_dim"]
    out_dir = "src/results/vae_inr_hypernet/samples"
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Build & load model ────────────────────────────────
    inr = INR(
        coord_dim=2,
        hidden_dim=config["inr_hidden_dim"],
        n_hidden=config["inr_layers"],
        out_dim=config["inr_out_dim"],
    )

    encoder_net = nn.Sequential(
        nn.Linear(784, config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], latent_dim * 2),
    )
    encoder = GaussianEncoder(encoder_net)

    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], inr.num_weights),
    )

    if config["prior"] == "gaussian":
        prior = GaussianPrior(latent_dim=latent_dim)
    elif config["prior"] == "mog":
        prior = MoGPrior(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown prior: {config['prior']}")

    model = VAEINR(prior, encoder, decoder_net, inr, beta=1.0, prior_type=config["prior"]).to(device)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()

    # ── 2. Real MNIST features ───────────────────────────────
    print("Loading MNIST …")
    mnist = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    real_np = (mnist.data.numpy()).astype(np.uint8)  # (60000, 28, 28)
    print(f"  {len(real_np):,} real images")
    print("Computing real Inception features …")

    n_samples = n_samples = len(real_np)

    real_feats = get_inception_features(real_np, device)

    # ── 3. Generated samples ─────────────────────────────────
    coords = make_coord_grid(28, device)  # (784, 2)

    print(f"Sampling {n_samples:,} images from prior …")
    fake_images = []
    chunk = 256
    with torch.no_grad():
        for start in tqdm(range(0, n_samples, chunk), desc="  Sampling"):
            end = min(start + chunk, n_samples)
            z = model.prior().sample(torch.Size([end - start])).to(device)
            w = model.decode_to_weights(z)
            c = coords.unsqueeze(0).expand(end - start, -1, -1)
            px = model.inr(c, w).squeeze(-1)  # (chunk, 784)
            imgs = (px.clamp(0, 1) * 255).byte().view(-1, 28, 28).cpu().numpy()
            fake_images.append(imgs)

    fake_np = np.concatenate(fake_images, axis=0)  # (N, 28, 28)  uint8

    print("Computing fake Inception features …")
    fake_feats = get_inception_features(fake_np, device)

    # ── 4. FID ───────────────────────────────────────────────
    print("Computing FID …")
    fid = compute_fid(real_feats, fake_feats)
    print(f"\n  ► FID = {fid:.3f}\n")

    # ── 5. 8x8 sample grid ───────────────────────────────────
    grid_n = grid_size * grid_size  # 64
    pad = 3  # px spacing between images

    with torch.no_grad():
        z_grid = model.prior().sample(torch.Size([grid_n])).to(device)
        w_grid = model.decode_to_weights(z_grid)
        c_grid = coords.unsqueeze(0).expand(grid_n, -1, -1)
        px_grid = model.inr(c_grid, w_grid).squeeze(-1)  # (64, 784)

    imgs_grid = px_grid.clamp(0, 1).view(grid_n, 1, 28, 28).cpu()
    grid_tensor = torchvision.utils.make_grid(
        imgs_grid,
        nrow=grid_size,
        padding=pad,
        normalize=False,
        pad_value=1.0,
    )  # (1, H, W)
    grid_np = grid_tensor[0].numpy()

    fig = plt.figure(figsize=(10, 10.6))
    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[0.06, 0.94],
        hspace=0.02,
        left=0.02,
        right=0.98,
        top=0.97,
        bottom=0.01,
    )

    # title row
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        f"INR-VAE  |  FID = {fid:.2f}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        transform=ax_title.transAxes,
    )

    # image grid
    ax_img = fig.add_subplot(gs[1])
    ax_img.imshow(grid_np, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_img.axis("off")

    run_name = os.path.splitext(os.path.basename(config["weights_path"]))[0]
    out_path = os.path.join(out_dir, f"{run_name}_fid_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ↳ grid saved → {out_path}")

    return fid


if __name__ == "__main__":
    config_path = "src/results/vae_inr_hypernet/experiments/vae_inr_MoG_Final_20-03-15:15.json"
    GRID_SIZE = 8

    config = parse_config_vars(config_path)

    print(f"Device     : {config['device']}")
    print(f"Weights    : {config['weights_path']}")
    print(f"Prior      : {config['prior']}")
    print(f"Latent dim : {config['latent_dim']}")
    print()

    fid = compute_fid_and_save_grid(config, grid_size=GRID_SIZE)
    print(f"\nDone — FID = {fid:.3f}")
