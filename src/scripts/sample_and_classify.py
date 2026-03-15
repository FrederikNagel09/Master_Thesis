"""
Sample 1000 images from VAE and DDPM, classify them, and plot best/worst 10.

Usage:
    python sample_and_classify.py
"""

import json
import os
import sys

sys.path.append(".")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torchvision import transforms

from src.models.mnist_classifier import UNetClassifier

# ─────────────────────────────────────────────────────────────
# CONFIG PATHS
# ─────────────────────────────────────────────────────────────

VAE_CONFIG_PATH = "src/results/vae_inr_hypernet/experiments/inr_vae_gauss_06-03-13:58.json"
DDPM_CONFIG_PATH = "src/results/ddpm/experiments/ddpm_full_run_09-03-17:00.json"
CLASSIFIER_WEIGHTS = "src/results/classifier/weights.pth"
CLASSIFIER_CONFIG = "src/results/classifier/config.json"
OUT_PATH = "src/results/general/vae_vs_ddpm.png"

UNKNOWN_THRESHOLD = 0.9
N_SAMPLES = 256  # how many to sample before picking best/worst
TOP_K = 10  # images per row

# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# Classifier
# ══════════════════════════════════════════════════════════════


def load_classifier(device):
    with open(CLASSIFIER_CONFIG) as f:
        cfg = json.load(f)
    model = UNetClassifier(num_classes=cfg["num_classes"], base_ch=cfg["base_ch"]).to(device)
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def classify(classifier, images_01, device, batch_size=256):
    """
    Args:
        images_01: (N, 1, 28, 28) float tensor in [0, 1]
    Returns:
        labels:      list of str  — digit or "?" if below threshold
        confidences: torch.Tensor of shape (N,)
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    all_confs, all_preds = [], []

    for i in range(0, len(images_01), batch_size):
        batch = normalize(images_01[i : i + batch_size].to(device))
        probs = F.softmax(classifier(batch), dim=-1)
        confs, preds = probs.max(dim=-1)
        all_confs.append(confs.cpu())
        all_preds.append(preds.cpu())

    confidences = torch.cat(all_confs)  # (N,)
    preds = torch.cat(all_preds)  # (N,)

    labels = [
        str(p.item()) if c.item() >= UNKNOWN_THRESHOLD else "?"
        for p, c in zip(preds, confidences)  # noqa: B905
    ]
    return labels, confidences


def top_and_bottom(images, labels, confidences, k=TOP_K):
    """Return (top_imgs, top_labels, top_confs, bot_imgs, bot_labels, bot_confs)"""
    sorted_idx = confidences.argsort(descending=True)
    top_idx = sorted_idx[:k]
    bot_idx = sorted_idx[-k:].flip(0)  # worst-first → ascending conf

    def pick(idx):
        imgs = images[idx]
        lbls = [labels[i] for i in idx.tolist()]
        confs = confidences[idx]
        return imgs, lbls, confs

    return (*pick(top_idx), *pick(bot_idx))


# ══════════════════════════════════════════════════════════════
# Samplers
# ══════════════════════════════════════════════════════════════


def sample_vae(config, n, device):
    from src.models.inr_vae_hypernet import INR, VAEINR
    from src.models.prior import GaussianPrior, MoGPrior
    from src.models.vae_coders import GaussianEncoder

    latent_dim = config["latent_dim"]
    inr = INR(coord_dim=2, hidden_dim=config["inr_hidden_dim"], n_hidden=config["inr_layers"], out_dim=config["inr_out_dim"])
    encoder_net = nn.Sequential(
        nn.Linear(784, config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], latent_dim * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], inr.num_weights),
    )
    prior = GaussianPrior(latent_dim=latent_dim) if config["prior"] == "gaussian" else MoGPrior(latent_dim=latent_dim)

    model = VAEINR(prior, GaussianEncoder(encoder_net), decoder_net, inr, beta=1.0, prior_type=config["prior"]).to(device)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()

    lin = torch.linspace(-1, 1, 28)
    gr, gc = torch.meshgrid(lin, lin, indexing="ij")
    coords = torch.stack([gr.flatten(), gc.flatten()], dim=-1).to(device)

    all_imgs = []
    batch = 256
    with torch.no_grad():
        for i in range(0, n, batch):
            k = min(batch, n - i)
            z = model.prior().sample(torch.Size([k])).to(device)
            flat_w = model.decode_to_weights(z)
            coords_b = coords.unsqueeze(0).expand(k, -1, -1)
            pixels = model.inr(coords_b, flat_w).squeeze(-1).view(k, 1, 28, 28)
            all_imgs.append(pixels.cpu())
    return torch.clamp(torch.cat(all_imgs), 0, 1)


def sample_ddpm(config, n, device):
    from src.models.ddpm import DDPM, Unet

    model = DDPM(Unet(), t=config["T"]).to(device)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()

    all_imgs = []
    batch = 256
    with torch.no_grad():
        for i in range(0, n, batch):
            k = min(batch, n - i)
            s = model.sample((k, 28 * 28)).view(k, 1, 28, 28)
            all_imgs.append((s * 0.5 + 0.5).clamp(0, 1).cpu())
    return torch.cat(all_imgs)


# ══════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════


def draw_image_row(fig, gs, row_idx, images, labels, confs, row_color):
    """Draw one row of TOP_K images into the given GridSpec row."""
    KNOWN = "#2a9d3a"  # noqa: N806
    UNKNOWN = "#cc2222"  # noqa: N806

    for col_idx in range(TOP_K):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.set_facecolor("white")
        ax.imshow(images[col_idx, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")

        label = labels[col_idx]
        conf = confs[col_idx].item()
        is_unknown = label == "?"

        ax.set_title(
            f"{'?' if is_unknown else label}\n{conf * 100:.0f}%",
            color=UNKNOWN if is_unknown else KNOWN,
            fontsize=10,
            fontweight="bold",
            pad=3,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(UNKNOWN if is_unknown else row_color)
            spine.set_linewidth(1.8)


def plot_results(vae_data, ddpm_data):
    """
    vae_data / ddpm_data: tuples of
        (top_imgs, top_labels, top_confs, bot_imgs, bot_labels, bot_confs)
    """
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    COL_VAE = "#000000"  # noqa: N806
    COL_DDPM = "#000000"  # noqa: N806
    TEXT = "#111111"  # noqa: N806

    fig = plt.figure(figsize=(TOP_K * 1.9, 13))
    fig.patch.set_facecolor("white")

    # 5 rows: VAE-best, VAE-worst, spacer, DDPM-best, DDPM-worst
    gs = gridspec.GridSpec(
        5,
        TOP_K,
        figure=fig,
        height_ratios=[1, 1, 0.18, 1, 1],
        hspace=0.75,
        wspace=0.08,
    )

    # ── VAE block ─────────────────────────────────────────────
    fig.text(0.5, 0.965, "INR-VAE Samples", ha="center", color=COL_VAE, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.910, f"Best {TOP_K}  —  highest classifier confidence", ha="center", color=TEXT, fontsize=10)
    draw_image_row(fig, gs, 0, *vae_data[:3], COL_VAE)

    fig.text(0.5, 0.720, f"Worst {TOP_K}  —  lowest classifier confidence", ha="center", color=TEXT, fontsize=10)
    draw_image_row(fig, gs, 1, *vae_data[3:], COL_VAE)

    # ── spacer row (gs row 2) — intentionally empty ───────────

    # ── DDPM block ────────────────────────────────────────────
    fig.text(0.5, 0.490, "DDPM Samples", ha="center", color=COL_DDPM, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.450, f"Best {TOP_K}  —  highest classifier confidence", ha="center", color=TEXT, fontsize=10)
    draw_image_row(fig, gs, 3, *ddpm_data[:3], COL_DDPM)

    fig.text(0.5, 0.255, f"Worst {TOP_K}  —  lowest classifier confidence", ha="center", color=TEXT, fontsize=10)
    draw_image_row(fig, gs, 4, *ddpm_data[3:], COL_DDPM)

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {OUT_PATH}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════


def main():
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"  Device: {device}\n")

    with open(VAE_CONFIG_PATH) as f:
        vae_cfg = json.load(f)
    with open(DDPM_CONFIG_PATH) as f:
        ddpm_cfg = json.load(f)
    vae_cfg["device"] = ddpm_cfg["device"] = device

    classifier = load_classifier(device)
    print("  Classifier ready\n")

    # ── VAE ───────────────────────────────────────────────────
    print(f"  Sampling {N_SAMPLES} images from VAE...")
    vae_images = sample_vae(vae_cfg, N_SAMPLES, device)
    vae_labels, vae_confs = classify(classifier, vae_images, device)
    vae_data = top_and_bottom(vae_images, vae_labels, vae_confs)
    print(f"  VAE  best conf: {vae_data[2].max() * 100:.1f}%  worst conf: {vae_data[5].min() * 100:.1f}%")

    # ── DDPM ──────────────────────────────────────────────────
    print(f"  Sampling {N_SAMPLES} images from DDPM...")
    ddpm_images = sample_ddpm(ddpm_cfg, N_SAMPLES, device)
    ddpm_labels, ddpm_confs = classify(classifier, ddpm_images, device)
    ddpm_data = top_and_bottom(ddpm_images, ddpm_labels, ddpm_confs)
    print(f"  DDPM best conf: {ddpm_data[2].max() * 100:.1f}%  worst conf: {ddpm_data[5].min() * 100:.1f}%")

    print("\n  Building plot...")
    plot_results(vae_data, ddpm_data)


if __name__ == "__main__":
    main()
