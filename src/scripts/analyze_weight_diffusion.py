"""
analyze_weight_diffusion.py

Standalone diagnostic script for comparing a broken vs. working WeightDiffusion model.

Does the following tests:
    1. Histogram, mean and std based on real images
        a) pre-normalized weights (raw that are used for reconstruction)
        b) normalized weights (used for diffusion)
        c) de-normalized weights
        d) post modulation weights 
    
    2. Histogram, mean and std of generated samples:
        a) raw samples
        b) denormalized weights 
        c) post modulation weights 

    3. Approximate Posterior collapse check:
        a) sends same image through N times and checks mu and logvar
        b) plots pca of mean of the N times and the others...

    4. Noise stability check
        a) adds differet levels of noise to weight vector...

        
TODO:
- Merge Histograms
- Do app pos collapse
- Do noise stability check


python src/scripts/analyze_weight_diffusion.py \
    --model_name Weight-Diffusion_working \
    --out_dir analysis_results/Weight-Diffusion_working \
    --config_path src/train_results/weight-diffusion/metadata/config.json \
    --weights_path src/train_results/weight-diffusion/weights/weights.pt

    
python src/scripts/analyze_weight_diffusion.py \
    --model_name Weight-Diffusion_broken \
    --out_dir analysis_results/Weight-Diffusion_broken \
    --config_path src/train_results/Weight-Diffusion-newMethod/metadata/config.json \
    --weights_path src/train_results/Weight-Diffusion-newMethod/weights/weights.pt


    --config_path src/train_results/Weight-Diffusion-newMethod_200/metadata/config.json \
    --weights_path src/train_results/Weight-Diffusion-newMethod_200/weights/weights.pt


    --config_path src/train_results/Weight-Diffusion-newMethod/metadata/config.json \
    --weights_path src/train_results/Weight-Diffusion-newMethod/weights/weights.pt

"""

import sys
import argparse
import json
import os
from types import SimpleNamespace

sys.path.append(".")
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
# Adjust these imports to match your project layout
# ---------------------------------------------------------------------------
from src.models.weight_diffusion import WeightDiffusion
from src.utility.model_builders.util.weight_diffusion_builder import (
    _build_weight_diffusion,
)

warnings.filterwarnings("ignore", message=".*aten::im2col.*")
# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------


def load_config(cfg_path: str) -> SimpleNamespace:
    """Load JSON config into a SimpleNamespace so attrs work like argparse."""
    with open(cfg_path) as f:
        raw = json.load(f)
    flat = {**raw.get("hparams", raw)}  # flatten hparams if nested
    return SimpleNamespace(**flat)


def load_model(cfg_path: str, ckpt_path: str, device: torch.device) -> WeightDiffusion:
    """
    Build model from config and load checkpoint weights.

    Args:
        cfg_path:  path to config.json
        ckpt_path: path to weights.pt
        device:    torch device
    Returns:
        model: WeightDiffusion — fully loaded, set to eval mode
    """
    args = load_config(cfg_path)
    data_config = {
        "channels": args.channels if hasattr(args, "channels") else 1,
        "img_size": args.img_size if hasattr(args, "img_size") else 28,
        "data_dim": args.data_dim if hasattr(args, "data_dim") else 784,
        "is_3d": getattr(args, "is_3d", False),
    }
    # Fall back to top-level data block if present
    if hasattr(args, "data"):
        data_config.update(args.data)

    model = _build_weight_diffusion(args, data_config)
    state = torch.load(ckpt_path, map_location=device)
    # Handle wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def get_mnist_loader(batch_size: int = 64, n_samples: int = 512) -> DataLoader:
    """Return a DataLoader with a small MNIST subset (flat tensors in [-1,1])."""
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1)),  # (784,)
        ]
    )
    ds = datasets.MNIST("data/", train=False, download=True, transform=tf)
    subset = torch.utils.data.Subset(ds, range(min(n_samples, len(ds))))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    # print(f"  saved → {path}")


def plot_histograms(
    arrays: dict[str, np.ndarray],
    title: str,
    out_dir: str,
    fname: str,
    n_bins: int = 80,
) -> None:
    """
    Overlay histograms for each entry in `arrays`.

    Args:
        arrays:  {label: 1-D numpy array}
        title:   figure title
        out_dir: output directory
        fname:   filename (without extension)
        n_bins:  histogram bin count
    Returns: None
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, arr in arrays.items():
        ax.hist(arr.flatten(), bins=n_bins, alpha=0.55, label=label, density=False)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend()
    _save(fig, out_dir, fname + ".png")


def print_stats(label: str, arr: np.ndarray) -> None:
    """Print mean/std/min/max for a flat array."""
    print(
        f"  [{label}] mean={arr.mean():.4f}  std={arr.std():.4f}"
        f"  min={arr.min():.4f}  max={arr.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_encoder_outputs(
    model: WeightDiffusion,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 4,
) -> dict[str, np.ndarray]:
    """
    Collect encoder outputs for a few batches.

    Args:
        model:    WeightDiffusion model
        loader:   DataLoader yielding (x, _) tuples
        device:   torch device
        n_batches: how many batches to accumulate
    Returns:
        dict with keys:
          "pre_norm"    — raw theta from encoder (B, mod_dim)
          "post_norm"   — after scaler.forward (B, mod_dim)
          "post_denorm" — scaler applied in reverse to post_norm (B, mod_dim)
          "pre_mod"     — base_params concatenated flat (B, weight_dim)
          "post_mod"    — decode_modulations output (B, weight_dim)
          "mu"          — encoder mean (B, mod_dim)
          "logvar"      — encoder logvar (B, mod_dim)
    """
    bufs: dict[str, list] = {
        k: []
        for k in [
            "theta_raw",
            "theta_norm",
            "theta_denorm",
            "theta_modulated",
            "mu",
            "logvar",
        ]
    }

    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)

        # Encoder
        mu, logvar = model.weight_encoder(x)
        theta_raw = model.weight_encoder._reparameterize(mu, logvar)

        bufs["mu"].append(mu.cpu().numpy())
        bufs["logvar"].append(logvar.cpu().numpy())
        bufs["theta_raw"].append(theta_raw.cpu().numpy())

        # Normalisation
        if model.normalize:
            norm = True
            theta_norm = model.scaler(theta_raw, reverse=False, training=False)
            theta_denorm = model.scaler(theta_norm, reverse=True, training=False)
        else:
            norm = False
            theta_norm = theta_raw
            theta_denorm = theta_raw

        bufs["theta_norm"].append(theta_norm.cpu().numpy())
        bufs["theta_denorm"].append(theta_denorm.cpu().numpy())

        # Pre- vs post-modulation (full weight vectors)
        # Pre-mod: just the base_params concatenated (no modulation applied)
        """
        NOT sure what this is?

        B = x.shape[0]
        pre_mod_parts = []
        for name in model.weight_encoder._param_names:
            bp = model.weight_encoder.base_params[name]          # (rows, cols)
            pre_mod_parts.append(
                bp.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
            )
        pre_mod = torch.cat(pre_mod_parts, dim=1)
        bufs["pre_mod"].append(pre_mod.cpu().numpy())
        """

        # Post-mod: full decoded weight vector
        post_mod = model.weight_encoder.decode_modulations(theta_raw)
        bufs["theta_modulated"].append(post_mod.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in bufs.items() if v}, norm


@torch.no_grad()
def get_generated_weights(
    model: WeightDiffusion,
    n_samples: int = 16,
) -> dict[str, np.ndarray]:
    """
    Run the reverse diffusion process and return weight statistics.

    Args:
        model:     WeightDiffusion model
        n_samples: number of samples to generate
    Returns:
        dict with keys:
          "pre_denorm"  — raw diffusion output before scaler.reverse (B, mod_dim)
          "post_denorm" — after scaler.reverse (B, mod_dim)
          "post_mod"    — full weight vector after decode_modulations (B, weight_dim)
          "x0_hat_traj" — dict {t: mean/std} tracked during sampling
    """
    # These have been de-normalized ???
    theta_raw = model.sample_weight(n_samples)

    theta_modulated = model.weight_encoder.decode_modulations(theta_raw)

    return {
        "theta_denormalized": theta_raw.cpu().numpy(),
        "theta_modulate": theta_modulated.cpu().numpy(),
    }


def _get_parser():
    parser = argparse.ArgumentParser(description="WeightDiffusion diagnostic analysis")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--weights_path", required=True)

    parser.add_argument("--out_dir", default="analysis_results/")
    parser.add_argument("--model_name", default="weight_diffusion")

    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="samples for generated weight analysis",
    )

    parser.add_argument(
        "--n_enc_batches",
        type=int,
        default=32,
        help="batches for encoder output collection",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "mps"
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    return device, args


@torch.no_grad()
def posterior_collapse_check_multiple(
    model: WeightDiffusion,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 64,
) -> dict[str, np.ndarray]:
    """
    Draw N reparameterized samples from each of 10 MNIST class images' posteriors.

    Args:
        model:     WeightDiffusion model
        loader:    DataLoader yielding (x, label) tuples, batch_size=1
        device:    torch device
        n_samples: number of reparameterization draws N per image
    Returns:
        dict with keys:
          "theta_raw"       — (10, N, mod_dim) reparameterized samples per class
          "theta_modulated" — (10, N, weight_dim) decoded modulations per class
          "mu"              — (10, mod_dim) encoder means per class
          "logvar"          — (10, mod_dim) encoder logvars per class
          "labels"          — (10,) class label for each of the 10 images
    """
    # Collect one image per MNIST class (0–9)
    class_images = {}
    for x, label in loader:
        lbl = label.item()
        if lbl not in class_images:
            class_images[lbl] = x.to(device)
        if len(class_images) == 10:
            break

    all_theta_raw = []
    all_theta_modulated = []
    all_mu = []
    all_logvar = []
    labels = []

    for lbl in sorted(class_images.keys()):
        x = class_images[lbl]  # (1, data_dim)

        mu, logvar = model.weight_encoder(x)  # each (1, mod_dim)

        # Draw N samples from this posterior
        mu_exp = mu.expand(n_samples, -1)  # (N, mod_dim)
        logvar_exp = logvar.expand(n_samples, -1)  # (N, mod_dim)
        theta_raw = model.weight_encoder._reparameterize(
            mu_exp, logvar_exp
        )  # (N, mod_dim)

        theta_modulated = model.weight_encoder.decode_modulations(
            theta_raw
        )  # (N, weight_dim)

        all_theta_raw.append(theta_raw.cpu().numpy())
        all_theta_modulated.append(theta_modulated.cpu().numpy())
        all_mu.append(mu.cpu().numpy())
        all_logvar.append(logvar.cpu().numpy())
        labels.append(lbl)

    all_theta_raw = np.stack(all_theta_raw, axis=0)  # (10, N, mod_dim)
    all_theta_modulated = np.stack(all_theta_modulated, axis=0)  # (10, N, weight_dim)
    all_mu = np.concatenate(all_mu, axis=0)  # (10, mod_dim)
    all_logvar = np.concatenate(all_logvar, axis=0)  # (10, mod_dim)

    return {
        "theta_raw": all_theta_raw,
        "theta_modulated": all_theta_modulated,
        "mu": all_mu,
        "logvar": all_logvar,
        "labels": labels,
    }


def plot_pca_posterior_multiple(
    collapse_outputs: dict[str, np.ndarray],
    out_dir: str,
    fname: str,
) -> None:
    """
    PCA scatter of N theta_raw and N theta_modulated samples for 10 MNIST classes.

    Args:
        collapse_outputs: output dict from posterior_collapse_check
        out_dir:          output directory
        fname:            filename without extension
    Returns: None
    """
    from sklearn.decomposition import PCA
    from scipy.stats import multivariate_normal

    labels = collapse_outputs["labels"]  # list of 10 ints
    n_class = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_class))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    datasets_to_plot = [
        ("theta_raw", collapse_outputs["mu"], "PCA — θ raw samples"),
        ("theta_modulated", collapse_outputs["mu"], "PCA — θ modulated samples"),
    ]

    for ax, (key, mu_all, title) in zip(axes, datasets_to_plot):
        data_all = collapse_outputs[key]  # (10, N, dim)

        # Fit PCA on all samples across all classes jointly
        flat = data_all.reshape(-1, data_all.shape[-1])  # (10*N, dim)
        pca = PCA(n_components=2)
        pca.fit(flat)

        # Fixed ±3 range, expanded if samples fall outside
        all_proj = pca.transform(flat)  # (10*N, 2)
        x_min = min(-3, all_proj[:, 0].min() * 1.1)
        x_max = max(3, all_proj[:, 0].max() * 1.1)
        y_min = min(-3, all_proj[:, 1].min() * 1.1)
        y_max = max(3, all_proj[:, 1].max() * 1.1)

        # Standard Gaussian contour background
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        rv = multivariate_normal(mean=[0, 0], cov=np.eye(2))
        zz = rv.pdf(np.dstack([xx, yy]))
        ax.contourf(xx, yy, zz, levels=8, cmap="summer")

        # Per-class scatter and mean star
        for i, lbl in enumerate(labels):
            samples_i = data_all[i]  # (N, dim)
            proj_i = pca.transform(samples_i)  # (N, 2)

            ax.scatter(
                proj_i[:, 0], proj_i[:, 1], c=[colors[i]], s=20, alpha=0.6, zorder=2
            )
            # ax.scatter(star[0], star[1],
            # c=[colors[i]], s=150, marker="*",
            # edgecolors="black", linewidths=0.5,
            # zorder=3, label=f"class {lbl}")

        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Posterior Collapse Check — PCA of N draws per MNIST class")
    fig.tight_layout()
    _save(fig, out_dir, fname + ".png")


@torch.no_grad()
def posterior_collapse_check(
    model: WeightDiffusion,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 64,
) -> dict[str, np.ndarray]:
    """
    Draw N reparameterized samples from a single image's posterior and decode them.

    Args:
        model:     WeightDiffusion model
        loader:    DataLoader yielding (x, _) tuples
        device:    torch device
        n_samples: number of reparameterization draws N
    Returns:
        dict with keys:
          "theta_raw"       — (N, mod_dim) reparameterized samples
          "theta_modulated" — (N, weight_dim) decoded modulations
          "mu"              — (1, mod_dim) encoder mean
          "logvar"          — (1, mod_dim) encoder logvar
    """
    # Get a single image and encode it once
    x, _ = next(iter(loader))
    x = x[:1].to(device)  # (1, data_dim)

    mu, logvar = model.weight_encoder(x)  # each (1, mod_dim)

    # Draw N samples by reparameterizing from the same (mu, logvar) N times
    mu_expanded = mu.expand(n_samples, -1)  # (N, mod_dim)
    logvar_expanded = logvar.expand(n_samples, -1)  # (N, mod_dim)
    theta_raw = model.weight_encoder._reparameterize(
        mu_expanded, logvar_expanded
    )  # (N, mod_dim)

    theta_modulated = model.weight_encoder.decode_modulations(
        theta_raw
    )  # (N, weight_dim)

    print("############### Posterior Collapse Check ###############")
    print_collapse_stats("theta_raw", theta_raw.cpu().numpy(), mu.cpu().numpy())
    print_collapse_stats(
        "theta_modulated", theta_modulated.cpu().numpy(), mu.cpu().numpy()
    )
    print("########################################################")

    return {
        "theta_raw": theta_raw.cpu().numpy(),
        "theta_modulated": theta_modulated.cpu().numpy(),
        "mu": mu.cpu().numpy(),
        "logvar": logvar.cpu().numpy(),
    }


def plot_pca_posterior(
    collapse_outputs: dict[str, np.ndarray],
    out_dir: str,
    fname: str,
) -> None:
    """
    PCA scatter of N theta_raw and N theta_modulated samples with Gaussian contour background.

    Args:
        collapse_outputs: output dict from posterior_collapse_check
        out_dir:          output directory
        fname:            filename without extension
    Returns: None
    """
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets_to_plot = [
        ("theta_raw", "PCA — θ raw samples"),
        ("theta_modulated", "PCA — θ modulated samples"),
    ]

    for ax, (key, title) in zip(axes, datasets_to_plot):
        data = collapse_outputs[key]  # (N, dim)
        mean = data.mean(axis=0, keepdims=True)  # (1, dim)

        # Fit PCA on the N samples and project both samples and mean
        pca = PCA(n_components=2)
        projected = pca.fit_transform(data)  # (N, 2)
        projected_mean = pca.transform(mean)  # (1, 2)

        # Gaussian KDE contour fitted to the projected samples
        # Fixed ±3 range, expanded if samples fall outside
        x_min = min(-3, projected[:, 0].min() * 1.1)
        x_max = max(3, projected[:, 0].max() * 1.1)
        y_min = min(-3, projected[:, 1].min() * 1.1)
        y_max = max(3, projected[:, 1].max() * 1.1)

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )

        rv = multivariate_normal(mean=[0, 0], cov=np.eye(2))
        zz = rv.pdf(np.dstack([xx, yy]))

        ax.contourf(xx, yy, zz, levels=8, cmap="summer")

        # Scatter samples and mean
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c="black",
            s=30,
            alpha=0.8,
            zorder=2,
            label="samples",
        )
        ax.scatter(
            projected_mean[:, 0],
            projected_mean[:, 1],
            c="red",
            s=120,
            marker="*",
            zorder=3,
            label="mean",
        )

        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()

    fig.suptitle(
        "Posterior Collapse Check — PCA of N draws from single image posterior"
    )
    fig.tight_layout()
    _save(fig, out_dir, fname + ".png")


def print_collapse_stats(
    label: str,
    samples: np.ndarray,
    mu: np.ndarray,
    collapse_threshold: float = 0.01,
) -> None:
    """
    Print posterior collapse diagnostics for a set of N samples.

    Args:
        label:               display name for the set
        samples:             (N, dim) array of samples
        mu:                  (1, dim) encoder mean
        collapse_threshold:  per-dim std below this is considered collapsed
    Returns: None
    """
    N, dim = samples.shape

    # Per-dim std — mean and min across dimensions
    per_dim_std = samples.std(axis=0)  # (dim,)
    collapsed_frac = (per_dim_std < collapse_threshold).mean()

    print(f"  [{label}]")
    print(
        f"    mean={samples.mean():.4f}  std={samples.std():.4f}  min={samples.min():.4f}  max={samples.max():.4f}"
    )
    print(
        f"    per-dim std  →  mean={per_dim_std.mean():.4f}  min={per_dim_std.min():.4f}  max={per_dim_std.max():.4f}"
    )
    print(f"    collapsed dims (<{collapse_threshold} std): {collapsed_frac*100:.1f}%")


@torch.no_grad()
def noise_stability_check(
    model: WeightDiffusion,
    loader: DataLoader,
    device: torch.device,
    noise_levels: list[float] = [0.0, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
    out_dir: str = "analysis_results/",
    fname: str = "noise_stability",
) -> None:
    """
    Add increasing noise to theta_raw and theta_modulated and decode to images.

    Row 1: noise added to theta_raw → modulate → decode
    Row 2: theta_raw → modulate → noise added to theta_modulated → decode

    Args:
        model:        WeightDiffusion model
        loader:       DataLoader yielding (x, _) tuples
        device:       torch device
        noise_levels: list of noise std values to apply
        out_dir:      output directory
        fname:        filename without extension
    Returns: None
    """
    # Get a single image and encode it
    x, _ = next(iter(loader))
    x = x[:1].to(device)  # (1, data_dim)

    mu, logvar = model.weight_encoder(x)
    theta_raw = model.weight_encoder._reparameterize(mu, logvar)  # (1, mod_dim)
    theta_mod = model.weight_encoder.decode_modulations(theta_raw)  # (1, weight_dim)

    n_cols = len(noise_levels)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.5, 6))

    for col, noise_std in enumerate(noise_levels):
        # Row 1: noise on theta_raw → modulate → decode
        noise = torch.randn_like(theta_raw) * noise_std
        noised_raw = theta_raw + noise
        modulated = model.weight_encoder.decode_modulations(
            noised_raw
        )  # (1, weight_dim)
        img_raw = model.decode_weights(modulated, coords=None)

        # Row 2: modulate first → noise on theta_modulated → decode
        noise_m = torch.randn_like(theta_mod) * noise_std
        noised_mod = theta_mod + noise_m
        img_mod = model.decode_weights(noised_mod, coords=None)  # (1, C, H, W)

        for row, img in enumerate([img_raw, img_mod]):
            ax = axes[row, col]
            img_np = img[0].cpu().float().numpy()  # (784,)

            # Reshape flat vector to (H, W) using config image size
            img_size = int(np.sqrt(img_np.shape[0]))
            img_np = img_np.reshape(img_size, img_size)

            ax.imshow(img_np, cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"σ={noise_std}", fontsize=9)

            axes[0, 0].set_ylabel("noise on θ_raw", fontsize=9)
            axes[1, 0].set_ylabel("noise on θ_modulated", fontsize=9)

    for row, lbl in enumerate(["Raw", "Modulated"]):
        axes[row, 0].text(
            -0.15,
            0.5,
            lbl,
            transform=axes[row, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
        )

    fig.suptitle("Noise Stability Check", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, fname + ".png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    device, args = _get_parser()

    # ------------------------------------------------------------------
    # Load Model and Data
    # ------------------------------------------------------------------
    print("Loading models...")
    model = load_model(args.config_path, args.weights_path, device)
    print("Successfully loaded broken model...")
    print("  done.\n")
    loader = get_mnist_loader(batch_size=64, n_samples=args.n_samples)

    print("=== [1-4] Encoder weight distributions ===")
    encoder_outputs, norm = get_encoder_outputs(
        model, loader, device, args.n_enc_batches
    )

    # ------------------------------------------------------------------
    # 1. Encoder Output Analysis
    # ------------------------------------------------------------------
    plot_histograms(
        {"theta_raw": encoder_outputs["theta_raw"]},
        "theta Raw from encoder",
        args.out_dir,
        "theta_raw",
    )
    plot_histograms(
        {"theta_modulated": encoder_outputs["theta_modulated"]},
        "theta Modulated",
        args.out_dir,
        "theta_modulated",
    )

    if norm:
        plot_histograms(
            {"theta_norm": encoder_outputs["theta_norm"]},
            "theta Normalized",
            args.out_dir,
            "theta_norm",
        )
        plot_histograms(
            {"theta_denorm": encoder_outputs["theta_denorm"]},
            "theta De-Normalized",
            args.out_dir,
            "theta_denorm",
        )
    # ------------------------------------------------------------------
    # 3. Approximate Posterior collapse check
    # ------------------------------------------------------------------
    # Single image — N draws from one posterior
    # loader_single = get_mnist_loader(batch_size=1, n_samples=1)
    # collapse_outputs = posterior_collapse_check(model, loader_single, device, n_samples=args.n_samples)
    # plot_pca_posterior(collapse_outputs, args.out_dir, "posterior_collapse_pca")

    # 10 images (one per MNIST class) — N draws from each posterior
    loader_multi = get_mnist_loader(batch_size=1, n_samples=512)
    collapse_outputs_multi = posterior_collapse_check_multiple(
        model, loader_multi, device, n_samples=args.n_samples
    )
    plot_pca_posterior_multiple(
        collapse_outputs_multi, args.out_dir, "posterior_collapse_pca_multiclass"
    )

    # ------------------------------------------------------------------
    # 4. Noise stability check
    # ------------------------------------------------------------------
    print("\n=== [4] Noise Stability Check ===")
    loader_single = get_mnist_loader(batch_size=1, n_samples=1)
    noise_stability_check(model, loader_single, device, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
