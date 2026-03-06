"""
Generates a 2x2 latent space PCA plot for all 4 models.
Each subplot shows the prior KDE contour + encoded MNIST data points coloured by class.

Output: src/results/general/latent_space_comparison.png
"""

import sys

sys.path.append(".")

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# ============================================================
#  HARDCODE CONFIG PATHS HERE
# ============================================================
VAE_GAUSS_CONFIG = "src/results/vae/experiments/vae_gauss_06-03-13:34.json"
VAE_MOG_CONFIG = "src/results/vae/experiments/vae_MoG_06-03-14:02.json"
INR_GAUSS_CONFIG = "src/results/vae_inr_hypernet/experiments/inr_vae_gauss_06-03-13:58.json"
INR_MOG_CONFIG = "src/results/vae_inr_hypernet/experiments/inr_vae_mog_06-03-13:59.json"
# ============================================================

OUT_DIR = "src/results/general"
os.makedirs(OUT_DIR, exist_ok=True)

N_BATCHES = 20  # batches of MNIST to encode (controls how many points are plotted)
BATCH_SIZE = 32
N_PRIOR_SAMPLES = 10000


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_mnist_loader(batch_size: int):
    threshold = 0.5
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (threshold < x).float().squeeze()),
        ]
    )
    dataset = datasets.MNIST("data/", train=True, download=False, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_vae(config: dict, device: str):
    from src.models.prior import GaussianPrior, MoGPrior, VAMPPrior
    from src.models.vae import VAE
    from src.models.vae_coders import BernoulliFullDecoder, GaussianFullEncoder

    latent_dim = config["latent_dim"]
    hidden_dims = config["hidden_dims"]
    prior_type = config["prior"]

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
    model.eval().to(device)
    return model, prior


def build_inr_vae(config: dict, device: str):
    from src.models.inr_vae_hypernet import INR, VAEINR
    from src.models.prior import GaussianPrior, MoGPrior
    from src.models.vae_coders import GaussianEncoder

    latent_dim = config["latent_dim"]

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
        raise ValueError(f"Unsupported prior: {config['prior']}")

    model = VAEINR(prior, encoder, decoder_net, inr, beta=1.0, prior_type=config["prior"]).to(device)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()
    return model, prior


# ------------------------------------------------------------
# Core: encode MNIST + sample prior → PCA → plot
# ------------------------------------------------------------


def compute_latent_pca(model, prior, loader, device: str):
    """
    Encodes N_BATCHES of MNIST images through the model encoder,
    samples N_PRIOR_SAMPLES from the prior, fits PCA on the prior
    samples and transforms both sets.

    Returns:
        z_2d        : (N, 2) encoded points in PCA space
        prior_2d    : (N_PRIOR_SAMPLES, 2) prior samples in PCA space
        labels      : (N,) class labels
    """
    point_list, label_list = [], []
    data_iter = iter(loader)

    with torch.no_grad():
        for _ in range(N_BATCHES):
            x, y = next(data_iter)
            x = x.to(device)
            x = x.view(x.size(0), -1)  # flatten (batch, 28, 28) → (batch, 784)
            q = model.encoder(x)
            z = q.rsample().detach().cpu()
            point_list.append(z)
            label_list.append(y)

    z_matrix = torch.cat(point_list, dim=0).numpy()  # (N, latent_dim)
    labels = torch.cat(label_list, dim=0).numpy()  # (N,)

    # Sample from prior
    prior_dist = prior()
    prior_samples = prior_dist.sample((N_PRIOR_SAMPLES,)).detach().cpu().numpy()  # (N_PRIOR_SAMPLES, latent_dim)

    # PCA fitted on prior, applied to both
    pca = PCA(n_components=2)
    prior_2d = pca.fit_transform(prior_samples)
    z_2d = pca.transform(z_matrix)

    return z_2d, prior_2d, labels, pca


def draw_latent_plot(ax, z_2d, prior_2d, labels, pca, title: str):
    """Draws the KDE contour + scatter onto the given axes."""
    x, y = z_2d[:, 0], z_2d[:, 1]
    x_prior, y_prior = prior_2d[:, 0], prior_2d[:, 1]

    # KDE over prior samples
    x_min = min(x_prior.min(), x.min())
    x_max = max(x_prior.max(), x.max())
    y_min = min(y_prior.min(), y.min())
    y_max = max(y_prior.max(), y.max())

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    xy_sample = np.vstack([xx.ravel(), yy.ravel()])
    kde = gaussian_kde(np.vstack([x_prior, y_prior]))
    density = kde(xy_sample).reshape(xx.shape)

    ax.contourf(xx, yy, density, levels=8, cmap="summer", alpha=1.0)
    sc = ax.scatter(x, y, c=labels, cmap="tab10", alpha=0.9, s=8, vmin=0, vmax=9)
    pca_var = pca.explained_variance_ratio_
    full_title = f"{title}\nPC1 = {pca_var[0]:.1%}   PC2 = {pca_var[1]:.1%}"
    ax.set_title(full_title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")

    n = 0.5  # padding around the outermost points
    ax.set_xlim(x.min() - n, x.max() + n)
    ax.set_ylim(y.min() - n, y.max() + n)

    return sc


# ------------------------------------------------------------
# Main plot
# ------------------------------------------------------------


def plot_latent_space(device: str):
    loader = get_mnist_loader(BATCH_SIZE)

    configs = [
        (VAE_GAUSS_CONFIG, "vae", "VAE  —  Gaussian prior"),
        (VAE_MOG_CONFIG, "vae", "VAE  —  MoG prior"),
        (INR_GAUSS_CONFIG, "vae_inr", "VAE-INR  —  Gaussian prior"),
        (INR_MOG_CONFIG, "vae_inr", "VAE-INR  —  MoG prior"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    for ax, (cfg_path, model_type, label) in zip(axes.flat, configs):  # noqa: B905
        cfg = load_config(cfg_path)
        latent_dim = cfg["latent_dim"]
        full_title = f"{label}   |   z = {latent_dim}"

        if model_type == "vae":
            model, prior = build_vae(cfg, device)
        else:
            model, prior = build_inr_vae(cfg, device)

        z_2d, prior_2d, labels, pca = compute_latent_pca(model, prior, loader, device)
        sc = draw_latent_plot(ax, z_2d, prior_2d, labels, pca, title=full_title)
        del model

    # Single shared colorbar for all subplots
    cbar = fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Digit class", fontsize=9)
    cbar.set_ticks(range(10))

    out_path = os.path.join(OUT_DIR, "latent_space_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Latent space plot] Saved → {out_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    plot_latent_space(device)
