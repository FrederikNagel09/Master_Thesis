"""
Model Comparison: VAE vs DDPM
Computes FID scores, per-digit class distribution, and sample quality grids.

Usage:
    python calculate_FID.py

Outputs:
    src/results/comparison/fid_comparison.png   — full comparison report figure
    src/results/comparison/fid_results.json     — raw numbers
"""

import json
import os
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(".")
# ─────────────────────────────────────────────────────────────
# CONFIGURE YOUR MODEL CONFIG PATHS HERE
# ─────────────────────────────────────────────────────────────

VAE_CONFIG_PATH = "src/results/vae_inr_hypernet/experiments/inr_vae_gauss_06-03-13:58.json"
DDPM_CONFIG_PATH = "src/results/ddpm/experiments/ddpm_full_run_09-03-17:00.json"

# Number of samples for FID
N_SAMPLES = 2024
BATCH_SIZE = 1024
# Samples shown in the quality grid (n x n)
GRID_SIZE = 5

# Output paths
OUT_DIR = "src/results/general"
PLOT_PATH = os.path.join(OUT_DIR, "fid_comparison.png")
JSON_PATH = os.path.join(OUT_DIR, "fid_results.json")

# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# Classifier + FID (from mnist_classifier.py)
# ══════════════════════════════════════════════════════════════


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
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
    def __init__(self, num_classes=10, base_ch=32, dropout=0.0):
        super().__init__()
        self.down1 = DownBlock(1, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
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

    def _encode(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x = self.bottleneck(x)
        return x, s1, s2, s3

    def forward(self, x):
        x, s1, s2, s3 = self._encode(x)
        x = self.up3(self.up2(self.up1(x, s3), s2), s1)
        return self.head(x)

    def get_features(self, x):
        x, _, _, _ = self._encode(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class ModelEvaluator:
    """Loads the trained MNIST classifier and computes FID + digit distribution."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))

        # Load classifier
        clf_cfg_path = "src/results/classifier/config.json"
        clf_wts_path = "src/results/classifier/weights.pth"
        assert os.path.exists(clf_wts_path), f"Classifier weights not found at {clf_wts_path}. Run training first."

        with open(clf_cfg_path) as f:
            clf_cfg = json.load(f)

        self.classifier = UNetClassifier(
            num_classes=clf_cfg["num_classes"],
            base_ch=clf_cfg["base_ch"],
        ).to(self.device)
        self.classifier.load_state_dict(torch.load(clf_wts_path, map_location=self.device))
        self.classifier.eval()
        print("  ✓ Classifier loaded")

    # ── feature extraction ──────────────────────────────────

    @torch.no_grad()
    def _features_and_preds(self, images_01, batch_size=512):
        """
        Args:
            images_01: (N, 1, 28, 28) tensor in [0, 1]
        Returns:
            features: (N, 256) numpy array
            preds:    (N,)     numpy array  — predicted digit 0-9
        """
        feats, preds = [], []
        for i in range(0, len(images_01), batch_size):
            batch = self.normalize(images_01[i : i + batch_size].to(self.device))
            feats.append(self.classifier.get_features(batch).cpu().numpy())
            preds.append(self.classifier(batch).argmax(1).cpu().numpy())
        return np.concatenate(feats), np.concatenate(preds)

    # ── FID ─────────────────────────────────────────────────

    @staticmethod
    def _frechet(mu1, sig1, mu2, sig2):
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sig1 @ sig2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sig1 + sig2 - 2 * covmean))

    def compute_fid(self, real_images, gen_images):
        f_real, _ = self._features_and_preds(real_images)
        f_gen, _ = self._features_and_preds(gen_images)
        mu1, s1 = f_real.mean(0), np.cov(f_real, rowvar=False)
        mu2, s2 = f_gen.mean(0), np.cov(f_gen, rowvar=False)
        return self._frechet(mu1, s1, mu2, s2)

    def compute_all(self, real_images, gen_images):
        """Returns dict with fid, digit_dist_real, digit_dist_gen."""
        f_real, p_real = self._features_and_preds(real_images)
        f_gen, p_gen = self._features_and_preds(gen_images)
        mu1, s1 = f_real.mean(0), np.cov(f_real, rowvar=False)
        mu2, s2 = f_gen.mean(0), np.cov(f_gen, rowvar=False)
        fid = self._frechet(mu1, s1, mu2, s2)
        dist_real = np.bincount(p_real, minlength=10) / len(p_real)
        dist_gen = np.bincount(p_gen, minlength=10) / len(p_gen)
        return {"fid": fid, "dist_real": dist_real, "dist_gen": dist_gen}


# ══════════════════════════════════════════════════════════════
# Samplers
# ══════════════════════════════════════════════════════════════


def sample_vae(config, n, device) -> torch.Tensor:
    """Returns (n, 1, 28, 28) float tensor in [0, 1]."""
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
    prior = GaussianPrior(latent_dim=latent_dim) if config["prior"] == "gaussian" else MoGPrior(latent_dim=latent_dim)

    model = VAEINR(prior, encoder, decoder_net, inr, beta=1.0, prior_type=config["prior"]).to(device)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()

    # Sample at native 28x28
    lin = torch.linspace(-1, 1, 28)
    grid_r, grid_c = torch.meshgrid(lin, lin, indexing="ij")
    coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1).to(device)

    all_images = []
    batch = 256
    with torch.no_grad():
        for i in range(0, n, batch):
            k = min(batch, n - i)
            z = model.prior().sample(torch.Size([k])).to(device)
            flat_w = model.decode_to_weights(z)
            coords_b = coords.unsqueeze(0).expand(k, -1, -1)
            pixels = model.inr(coords_b, flat_w).squeeze(-1).view(k, 1, 28, 28)
            all_images.append(pixels.cpu())
    return torch.clamp(torch.cat(all_images), 0, 1)


def sample_ddpm(config, n, device) -> torch.Tensor:
    """Returns (n, 1, 28, 28) float tensor in [0, 1]."""
    from src.models.ddpm import DDPM, Unet

    network = Unet()
    model = DDPM(network, t=config["T"]).to(device)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()

    all_images = []
    batch = BATCH_SIZE
    with torch.no_grad():
        for i in range(0, n, batch):
            k = min(batch, n - i)
            s = model.sample((k, 28 * 28))
            s = s.view(k, 1, 28, 28)
            all_images.append(((s * 0.5 + 0.5).clamp(0, 1)).cpu())
    return torch.cat(all_images)


def get_real_images(n, device) -> torch.Tensor:  # noqa: ARG001
    """Returns n real MNIST images as (n, 1, 28, 28) float tensor in [0, 1]."""
    ds = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    imgs = []
    for x, _ in loader:
        imgs.append(x)
        if sum(i.shape[0] for i in imgs) >= n:
            break
    return torch.cat(imgs)[:n]


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════


def make_sample_grid(images_01, n=GRID_SIZE) -> np.ndarray:
    """Pick n*n random samples, return HWC numpy array."""
    idx = torch.randperm(len(images_01))[: n * n]
    grid = torchvision.utils.make_grid(images_01[idx], nrow=n, normalize=False, padding=2)
    return grid.permute(1, 2, 0).numpy()


def build_report(results: dict, vae_samples, ddpm_samples, real_samples):
    """
    Three-row figure:
      Row 1 — sample quality grids 5x5 (real / VAE / DDPM)
      Row 2 — three separate digit distribution bar charts
      Row 3 — summary metrics table (white, black text)
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    TEXT = "#111111"  # noqa: N806
    ACCENT_REAL = "#4a9eff"  # noqa: N806
    ACCENT_VAE = "#ff6b6b"  # noqa: N806
    ACCENT_DDPM = "#51cf66"  # noqa: N806

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[2.2, 1.5, 0.55],
        hspace=0.32,
        wspace=0.22,
    )

    # ── Row 1 : sample grids (5x5) ───────────────────────────
    grids = [
        ("Real MNIST", real_samples, ACCENT_REAL),
        ("VAE Samples", vae_samples, ACCENT_VAE),
        ("DDPM Samples", ddpm_samples, ACCENT_DDPM),
    ]
    for col, (title, imgs, color) in enumerate(grids):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(make_sample_grid(imgs), cmap="gray", interpolation="nearest")
        ax.set_title(title, color=color, fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # ── Row 2 : three separate digit distribution bar charts ──
    dist_real = results["vae"]["dist_real"]
    dist_vae = results["vae"]["dist_gen"]
    dist_ddpm = results["ddpm"]["dist_gen"]

    dists = [
        ("Real MNIST", dist_real, ACCENT_REAL),
        ("VAE", dist_vae, ACCENT_VAE),
        ("DDPM", dist_ddpm, ACCENT_DDPM),
    ]
    digits = np.arange(10)
    y_max = max(dist_real.max(), dist_vae.max(), dist_ddpm.max()) * 100 * 1.18

    for col, (title, dist, color) in enumerate(dists):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("white")
        ax.bar(digits, dist * 100, color=color, alpha=0.85, width=0.65)
        ax.axhline(10, color="#999999", linewidth=0.9, linestyle="--", label="Uniform (10%)")
        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits], fontsize=11, color=TEXT)
        ax.set_ylim(0, y_max)
        ax.set_title(title, color=color, fontsize=13, fontweight="bold", pad=6)
        ax.tick_params(colors=TEXT)
        ax.spines[:].set_color("#cccccc")
        for lbl in ax.get_yticklabels():
            lbl.set_color(TEXT)
        if col == 0:
            ax.set_ylabel("% of samples", fontsize=11, color=TEXT)
        else:
            ax.set_yticklabels([])
        if col == 2:
            ax.legend(facecolor="white", labelcolor=TEXT, fontsize=10, loc="upper right", framealpha=0.8)

    # ── Row 3 : summary table (white / black) ─────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.set_facecolor("white")
    ax_tbl.axis("off")

    vae_fid = results["vae"]["fid"]
    ddpm_fid = results["ddpm"]["fid"]
    vae_time = results["vae"]["sample_time"]
    ddpm_time = results["ddpm"]["sample_time"]

    better_fid = "VAE ✓" if vae_fid < ddpm_fid else "DDPM ✓"
    better_time = "VAE ✓" if vae_time < ddpm_time else "DDPM ✓"

    uniform = np.ones(10) / 10
    vae_coverage = float(np.sum((dist_vae - uniform) ** 2) * 1000)
    ddpm_coverage = float(np.sum((dist_ddpm - uniform) ** 2) * 1000)
    better_cov = "VAE ✓" if vae_coverage < ddpm_coverage else "DDPM ✓"

    col_labels = ["Metric", "VAE", "DDPM", "Better"]
    table_data = [
        ["FID Score ↓", f"{vae_fid:.2f}", f"{ddpm_fid:.2f}", better_fid],
        [f"Sample time ({N_SAMPLES:,} imgs) ↓", f"{vae_time:.1f}s", f"{ddpm_time:.1f}s", better_time],
        ["Digit coverage (↓ = more uniform)", f"{vae_coverage:.2f}", f"{ddpm_coverage:.2f}", better_cov],
    ]

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.7)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(color=TEXT, fontweight="bold")
        if col == 3 and row > 0:
            winner = table_data[row - 1][3]
            cell.set_text_props(
                color=ACCENT_DDPM if "DDPM" in winner else ACCENT_VAE,
                fontweight="bold",
            )

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved → {PLOT_PATH}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════


def main():
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"\n{'=' * 55}")
    print(f"  Model Comparison  |  device={device}  |  n={N_SAMPLES:,}")
    print(f"{'=' * 55}\n")

    # Load configs
    with open(VAE_CONFIG_PATH) as f:
        vae_cfg = json.load(f)
    with open(DDPM_CONFIG_PATH) as f:
        ddpm_cfg = json.load(f)
    vae_cfg["device"] = ddpm_cfg["device"] = device

    evaluator = ModelEvaluator(device=device)

    # Real images
    print("  Loading real MNIST images...")
    real = get_real_images(N_SAMPLES, device)

    # ── VAE ──────────────────────────────────────────────────
    print(f"\n  Sampling {N_SAMPLES:,} images from VAE...")
    t0 = time.time()
    vae_samples = sample_vae(vae_cfg, N_SAMPLES, device)
    vae_time = time.time() - t0
    print(f"  Done in {vae_time:.1f}s")

    print("  Computing VAE metrics...")
    vae_metrics = evaluator.compute_all(real, vae_samples)
    print(f"  VAE  FID = {vae_metrics['fid']:.2f}")

    # ── DDPM ─────────────────────────────────────────────────
    print(f"\n  Sampling {N_SAMPLES:,} images from DDPM...")
    t0 = time.time()
    ddpm_samples = sample_ddpm(ddpm_cfg, N_SAMPLES, device)
    ddpm_time = time.time() - t0
    print(f"  Done in {ddpm_time:.1f}s")

    print("  Computing DDPM metrics...")
    ddpm_metrics = evaluator.compute_all(real, ddpm_samples)
    print(f"  DDPM FID = {ddpm_metrics['fid']:.2f}")

    # ── Results ───────────────────────────────────────────────
    results = {
        "vae": {**vae_metrics, "sample_time": vae_time, "dist_real": vae_metrics["dist_real"], "dist_gen": vae_metrics["dist_gen"]},
        "ddpm": {**ddpm_metrics, "sample_time": ddpm_time, "dist_real": ddpm_metrics["dist_real"], "dist_gen": ddpm_metrics["dist_gen"]},
    }

    # Save JSON (convert numpy arrays)
    os.makedirs(OUT_DIR, exist_ok=True)
    json_results = {
        model: {
            "fid": float(v["fid"]),
            "sample_time": float(v["sample_time"]),
            "digit_distribution_generated": v["dist_gen"].tolist(),
            "digit_distribution_real": v["dist_real"].tolist(),
        }
        for model, v in results.items()
    }
    with open(JSON_PATH, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved → {JSON_PATH}")

    # Build report figure
    print("  Building comparison report...")
    build_report(results, vae_samples, ddpm_samples, real)

    print(f"\n{'=' * 55}")
    print(f"  VAE  FID : {results['vae']['fid']:.2f}")
    print(f"  DDPM FID : {results['ddpm']['fid']:.2f}")
    winner = "VAE" if results["vae"]["fid"] < results["ddpm"]["fid"] else "DDPM"
    print(f"  Winner   : {winner}  (lower FID is better)")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
