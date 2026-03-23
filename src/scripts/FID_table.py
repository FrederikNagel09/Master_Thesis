"""
results_fid.py
Computes MNIST-classifier FID, Inception FID, and class distribution
for all three models. Caches real MNIST features to avoid recomputation.

Usage
-----
python src/scripts/FID_table.py \
    --ndm     src/train_results/ndm_unet_mnist/metadata/config.json \
    --inr_vae src/train_results/vae_inr_mnist/metadata/config.json \
    --ndm_inr src/train_results/ndm_inr_mlp_mnist/metadata/config.json \
    --out     src/results/fid_comparison.png

All three model config paths are optional.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from scipy import linalg
from torchvision import datasets, transforms
from tqdm import tqdm

from src.utility.inference import sample as model_sample

# =============================================================================
# Constants
# =============================================================================

N_SAMPLES = 10_000
SAMPLE_BATCH = 2000
CLASSIFIER_WEIGHTS = "src/results/classifier/weights.pth"
CLASSIFIER_CONFIG = "src/results/classifier/config.json"
CACHE_DIR = "src/results/cache"
CACHE_PATH = os.path.join(CACHE_DIR, "real_mnist_features.npz")

MODEL_LABELS = {
    "ndm": "NDM",
    "inr_vae": "VAE-INR",
    "ndm_inr": "NDM-INR",
}
MODEL_COLORS = {
    "ndm": "#2a6fdb",
    "inr_vae": "#e07b39",
    "ndm_inr": "#2ca05a",
}


# =============================================================================
# Device
# =============================================================================


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# MNIST Classifier
# =============================================================================


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.res = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.res(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class UNetClassifier(nn.Module):
    def __init__(self, num_classes=10, base_ch=16, dropout=0.1):
        super().__init__()
        self.down1 = DownBlock(1, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8, dropout=dropout)
        self.up1 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up3 = UpBlock(base_ch * 2, base_ch)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, base_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 2, num_classes),
        )

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x = self.bottleneck(x)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        return self.head(x)

    def get_features(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        x, _ = self.down1(x)
        x, _ = self.down2(x)
        x, _ = self.down3(x)
        x = self.bottleneck(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def _load_classifier(device: str) -> UNetClassifier:
    with open(CLASSIFIER_CONFIG) as f:
        cfg = json.load(f)
    model = UNetClassifier(
        num_classes=cfg["num_classes"],
        base_ch=cfg["base_ch"],
    ).to(device)
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
    model.eval()
    return model


# =============================================================================
# Inception
# =============================================================================


def _get_inception(device: str):
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()
    return inception


@torch.no_grad()
def _inception_features(
    images_01: torch.Tensor,
    inception,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """images_01: (N, 1, H, W) or (N, 3, H, W) in [0,1]. Returns (N, 2048)."""
    from torchvision.transforms.functional import resize

    all_feats = []
    for i in tqdm(range(0, len(images_01), batch_size), desc="    Inception features", leave=False):
        batch = images_01[i : i + batch_size]  # keep on CPU for resize
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = resize(batch, [299, 299], antialias=True)  # resize on CPU
        batch = batch.to(device)  # move to device after resize
        feats = inception(batch)[0].squeeze(-1).squeeze(-1).cpu().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats)


@torch.no_grad()
def _mnist_features(
    images_01: torch.Tensor,
    classifier: UNetClassifier,
    device: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features (N, D), predicted_labels (N,))."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    all_feats, all_preds = [], []
    for i in range(0, len(images_01), batch_size):
        batch = normalize(images_01[i : i + batch_size].to(device))
        all_feats.append(classifier.get_features(batch).cpu().numpy())
        all_preds.append(classifier(batch).argmax(1).cpu().numpy())
    return np.concatenate(all_feats), np.concatenate(all_preds)


# =============================================================================
# FID computation
# =============================================================================


def _frechet(mu1, s1, mu2, s2) -> float:
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def _fid(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    mu1, s1 = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu2, s2 = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)
    return _frechet(mu1, s1, mu2, s2)


# =============================================================================
# Real MNIST feature cache
# =============================================================================


def _load_or_compute_real_features(
    classifier: UNetClassifier,
    inception,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mnist_features, inception_features) for the real MNIST train set.
    Computes once and caches to CACHE_PATH; loads from cache on subsequent runs.
    """
    if os.path.exists(CACHE_PATH):
        print("  Loading cached real MNIST features …")
        data = np.load(CACHE_PATH)
        return data["mnist_features"], data["inception_features"]

    print("  Computing real MNIST features (first run — will be cached) …")
    mnist = datasets.MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=512, shuffle=False)

    all_imgs = []
    for x, _ in tqdm(loader, desc="    Loading MNIST", leave=False):
        all_imgs.append(x)
    real_images = torch.cat(all_imgs)  # (60000, 1, 28, 28)

    print("    Extracting MNIST classifier features …")
    mnist_feats, _ = _mnist_features(real_images, classifier, device)

    print("    Extracting Inception features …")
    inception_feats = _inception_features(real_images, inception, device)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(CACHE_PATH, mnist_features=mnist_feats, inception_features=inception_feats)
    print(f"  Real features cached → {CACHE_PATH}")

    return mnist_feats, inception_feats


# =============================================================================
# Uniformity score
# =============================================================================


def _uniformity_score(dist: np.ndarray) -> float:
    """KL divergence from uniform * 1000 for readability. Lower = more uniform."""
    uniform = np.ones(10) / 10
    kl = float(np.sum(dist * np.log((dist + 1e-10) / uniform)))
    return kl * 1000


# =============================================================================
# Plotting
# =============================================================================


def _build_figure(
    metrics: dict,
    out_path: str,
) -> None:
    """
    metrics: dict keyed by model_key, each with:
        mnist_fid, inception_fid, uniformity, dist_gen (np array len 10)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model_keys = list(metrics.keys())
    n_models = len(model_keys)
    digits = np.arange(10)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5 * n_models, 9))
    fig.patch.set_facecolor("white")

    # Table takes top 30%, bar plots take bottom 60%, small gap in between
    ax_table = fig.add_axes([0.05, 0.68, 0.90, 0.28])
    ax_table.axis("off")

    bar_axes = []
    bar_w = 0.82 / n_models
    for i in range(n_models):
        ax = fig.add_axes([0.08 + i * (bar_w + 0.02), 0.08, bar_w, 0.52])
        bar_axes.append(ax)

    # ── Table ─────────────────────────────────────────────────────────────────
    col_labels = ["Model", "MNIST FID ↓", "Inception FID ↓", "Uniformity ↓"]
    table_data = []
    for key in model_keys:
        m = metrics[key]
        table_data.append(
            [
                MODEL_LABELS[key],
                f"{m['mnist_fid']:.2f}",
                f"{m['inception_fid']:.2f}",
                f"{m['uniformity']:.2f}",
            ]
        )

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    # Find best (lowest) value per metric column
    best_mnist = min(range(n_models), key=lambda i: metrics[model_keys[i]]["mnist_fid"])
    best_inception = min(range(n_models), key=lambda i: metrics[model_keys[i]]["inception_fid"])
    best_uniformity = min(range(n_models), key=lambda i: metrics[model_keys[i]]["uniformity"])
    best_cols = {1: best_mnist, 2: best_inception, 3: best_uniformity}

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
        cell.set_text_props(color="#111111")

        if row == 0:  # header
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold", color="#111111")

        if row > 0 and col == 0:  # model name — colour coded
            key = model_keys[row - 1]
            cell.set_text_props(color=MODEL_COLORS[key], fontweight="bold")

        if row > 0 and col in best_cols:  # best value — bold green  # noqa: SIM102
            if best_cols[col] == row - 1:
                cell.set_text_props(color="#2a9d3a", fontweight="bold")

    ax_table.set_title(
        "Model Comparison — MNIST Generation",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#111111",
    )

    # ── Bar plots ─────────────────────────────────────────────────────────────
    y_max = max(metrics[k]["dist_gen"].max() for k in model_keys) * 100 * 1.25

    for i, (ax, key) in enumerate(zip(bar_axes, model_keys, strict=False)):
        dist = metrics[key]["dist_gen"]
        color = MODEL_COLORS[key]

        ax.bar(digits, dist * 100, color=color, alpha=0.85, width=0.65)
        ax.axhline(10, color="#999999", linewidth=1.0, linestyle="--", label="Uniform (10%)")

        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits], fontsize=10)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Digit", fontsize=10)
        ax.set_title(MODEL_LABELS[key], fontsize=11, fontweight="bold", color=color, pad=6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_edgecolor("#cccccc")
        ax.spines["bottom"].set_edgecolor("#cccccc")
        ax.tick_params(colors="#555555")
        ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        if i == 0:
            ax.set_ylabel("% of samples", fontsize=10)
            ax.legend(fontsize=9, framealpha=0.8, loc="upper right")
        else:
            ax.set_yticklabels([])

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="FID and class distribution comparison.")
    parser.add_argument("--ndm", type=str, default=None)
    parser.add_argument("--inr_vae", type=str, default=None)
    parser.add_argument("--ndm_inr", type=str, default=None)
    parser.add_argument("--out", type=str, default="src/results/fid_comparison.png")
    args = parser.parse_args()

    requested = {}
    for key in ("ndm", "inr_vae", "ndm_inr"):
        path = getattr(args, key)
        if path is not None:
            requested[key] = path

    if not requested:
        print("No config paths provided. Pass at least one of --ndm, --inr_vae, --ndm_inr.")
        sys.exit(1)

    device = _get_device()
    print(f"\n{'='*55}")
    print(f"  FID Comparison  |  device={device}  |  n={N_SAMPLES:,}")
    print(f"{'='*55}\n")

    # ── Load classifier and inception ─────────────────────────────────────────
    print("  Loading MNIST classifier …")
    classifier = _load_classifier(device)
    print("  Loading Inception …")
    inception = _get_inception(device)

    # ── Real MNIST features (cached) ──────────────────────────────────────────
    real_mnist_feats, real_inception_feats = _load_or_compute_real_features(classifier, inception, device)

    # ── Per-model evaluation ──────────────────────────────────────────────────
    metrics = {}

    for model_key, config_path in requested.items():
        label = MODEL_LABELS[model_key]
        print(f"\n── {label} ──────────────────────────────────────────")

        # Sample
        print(f"  Sampling {N_SAMPLES:,} images …")
        t0 = time.time()
        images = model_sample(
            model_name=model_key,
            config_path=config_path,
            n_samples=N_SAMPLES,
            device=device,
            batch_size=SAMPLE_BATCH,
        )  # (N, C, H, W) in [0,1]
        print(f"  Sampling done in {time.time() - t0:.1f}s")

        # MNIST classifier features + predictions
        print("  Extracting MNIST classifier features …")
        gen_mnist_feats, gen_preds = _mnist_features(images, classifier, device)

        # Inception features
        print("  Extracting Inception features …")
        gen_inception_feats = _inception_features(images, inception, device)

        # FID scores
        mnist_fid = _fid(real_mnist_feats, gen_mnist_feats)
        inception_fid = _fid(real_inception_feats, gen_inception_feats)

        # Class distribution
        dist_gen = np.bincount(gen_preds, minlength=10) / len(gen_preds)
        uniformity = _uniformity_score(dist_gen)

        print(f"  MNIST FID     : {mnist_fid:.2f}")
        print(f"  Inception FID : {inception_fid:.2f}")
        print(f"  Uniformity    : {uniformity:.2f}")

        metrics[model_key] = {
            "mnist_fid": mnist_fid,
            "inception_fid": inception_fid,
            "uniformity": uniformity,
            "dist_gen": dist_gen,
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.splitext(args.out)[0] + ".json"
    json_out = {
        key: {
            "mnist_fid": float(m["mnist_fid"]),
            "inception_fid": float(m["inception_fid"]),
            "uniformity": float(m["uniformity"]),
            "class_distribution": m["dist_gen"].tolist(),
        }
        for key, m in metrics.items()
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  Results JSON saved → {json_path}")

    # ── Build figure ──────────────────────────────────────────────────────────
    print("  Building figure …")
    _build_figure(metrics, args.out)

    print(f"\n{'='*55}")
    for key, m in metrics.items():
        print(
            f"  {MODEL_LABELS[key]:<10} MNIST FID={m['mnist_fid']:.2f}  "
            f"Inception FID={m['inception_fid']:.2f}  "
            f"Uniformity={m['uniformity']:.2f}"
        )
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
