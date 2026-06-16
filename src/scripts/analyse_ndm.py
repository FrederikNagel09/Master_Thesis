"""
analyse_weight_ndm.py

Diagnostic script for a trained WeightNDMDiffusion model.

Compares:
  - Weight space : sampled weights vs real-image-encoded weights
                   (per-dimension mean, std, and histograms)
  - Image space  : visual grids of sampled images vs real images
                   vs reconstructed (encode → decode) images

Usage:
    python src/scripts/analyse_ndm.py \
        --config  src/train_results/Weight-NDM-Diffusion-Probabilistic/metadata/config.json \
        --weights src/train_results/Weight-NDM-Diffusion-Probabilistic/weights/weights.pt \
        --n_samples 1024 \
        --n_real    1024 \
        --out_dir   src/results/analyse_ndm/
"""

import argparse
import json
from datetime import datetime
import os
import sys
from types import SimpleNamespace
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
 
# ── make project root importable ──────────────────────────────────────────────
sys.path.append(".") 
from src.utility.dataset_builders import build_dataset
from src.utility.model_builders import _build_weight_ndm_diffusion
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def load_config(config_path: str) -> SimpleNamespace:
    """
    Load JSON run config into a flat SimpleNamespace so attribute access works
    exactly like argparse does in the training script.
 
    Args:
        config_path: path to run_config.json
    Returns:
        SimpleNamespace with all hparam fields as attributes
    """
    with open(config_path) as f:
        cfg = json.load(f)
    # hparams lives one level down in the JSON
    hparams = cfg.get("hparams", cfg)
    return SimpleNamespace(**hparams)
 
 
def load_model(args: SimpleNamespace, data_config: dict, weights_path: str, device: torch.device):
    """
    Build and load a WeightNDMDiffusion model from saved weights.
 
    Args:
        args:         hyperparameter namespace
        data_config:  dict with channels / img_size / data_dim
        weights_path: path to .pt checkpoint
        device:       torch device
    Returns:
        model in eval mode on device
    """
    model = _build_weight_ndm_diffusion(args, data_config)
    state = torch.load(weights_path, map_location=device)
    # support both raw state-dicts and checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model
 
 
@torch.no_grad()
def encode_real_images(
    model,
    loader: DataLoader,
    n_real: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode real images through the VAE encoder and return both the latent
    weight (theta_prime_raw) and the modulated weight (decode_modulations output),
    plus the raw pixel tensors.
 
    Args:
        model:  WeightNDMDiffusion model (eval mode)
        loader: DataLoader over the real dataset
        n_real: how many real images to process
        device: torch device
    Returns:
        theta_prime  : (n_real, modulation_dim)  latent weights (pre-modulation)
        theta_mod    : (n_real, modulation_dim)  modulated weights (post decode_modulations)
        real_images  : (n_real, C, H, W)         raw pixel tensors in [0, 1]
    """
    all_theta_prime = []
    all_theta_mod   = []
    all_images      = []
    collected       = 0
 
    for batch in loader:
        imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        b    = imgs.shape[0]
        remaining = n_real - collected
        if b > remaining:
            imgs = imgs[:remaining]
            b    = remaining
 
        if model.probablistic:
            mean, logvar = model.weight_encoder(imgs)
            w = model.weight_encoder._reparameterize(mean, logvar)
        else:
            w = model.weight_encoder(imgs)
 
        w_mod = model.weight_encoder.decode_modulations(w)
 
        all_theta_prime.append(w.cpu())
        all_theta_mod.append(w_mod.cpu())
        # reshape to (B, C, H, W) if dataset returns flat tensors
        if imgs.dim() == 2:
            imgs = imgs.view(imgs.shape[0], 1, int(imgs.shape[1] ** 0.5), int(imgs.shape[1] ** 0.5))
        all_images.append(imgs.cpu())
        collected += b
        if collected >= n_real:
            break
 
    return (
        torch.cat(all_theta_prime, dim=0),
        torch.cat(all_theta_mod,   dim=0),
        torch.cat(all_images,      dim=0),
    )
 
 
@torch.no_grad()
def decode_weights_to_images(
    model,
    weights: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Decode a batch of raw weight vectors (pre-decode_modulations) into pixel images.
 
    Args:
        model:      WeightNDMDiffusion model
        weights:    (N, modulation_dim) raw weight tensors
        device:     torch device
        batch_size: chunk size to avoid OOM
    Returns:
        (N, C, H, W) image tensors clamped to [0, 1]
    """
    C      = getattr(model, "channels", 1)
    H = W  = model.img_size
    images = []
    for start in range(0, len(weights), batch_size):
        chunk = weights[start : start + batch_size].to(device)
        theta = model.weight_encoder.decode_modulations(chunk)
        imgs  = model.decode_weights(theta, coords=None)   # (B, H*W) or (B, C, H, W)
        if imgs.dim() == 2:
            imgs = imgs.view(imgs.shape[0], C, H, W)
        images.append(imgs.cpu())
    return torch.cat(images, dim=0).clamp(0, 1)
 
 
 
 
def _mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    n_sub: int = 1000,
    bandwidth: float | None = None,
) -> float:
    """
    Unbiased RBF-kernel MMD² between two sets of vectors.
    Subsamples to n_sub points each for speed.
 
    Args:
        x:         (N, D) first sample set
        y:         (M, D) second sample set
        n_sub:     max samples per set to use
        bandwidth: RBF sigma² — defaults to median heuristic
    Returns:
        float  MMD² value (near 0 → distributions match)
    """
    x = x[:n_sub].float()
    y = y[:n_sub].float()
 
    # median heuristic for bandwidth
    if bandwidth is None:
        all_pts = torch.cat([x, y], dim=0)
        dists   = torch.cdist(all_pts, all_pts)
        bandwidth = dists.median().item() ** 2 + 1e-8
 
    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(a, b) ** 2
        return torch.exp(-d / (2.0 * bandwidth))
 
    kxx = rbf(x, x)
    kyy = rbf(y, y)
    kxy = rbf(x, y)
 
    n, m = x.shape[0], y.shape[0]
    # unbiased: zero out diagonal for same-set terms
    kxx.fill_diagonal_(0)
    kyy.fill_diagonal_(0)
    mmd2 = kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()
    return mmd2.item()
 
# ─────────────────────────────────────────────────────────────────────────────
# Weight-space analysis
# ─────────────────────────────────────────────────────────────────────────────
 
def analyse_weights(
    real_theta_prime:    torch.Tensor,
    real_theta_mod:      torch.Tensor,
    sampled_theta_prime: torch.Tensor,
    sampled_theta_mod:   torch.Tensor,
    out_dir:             str,
) -> dict:
    """
    Compute per-dimension statistics, MMD, and std-ratio metrics, save plots,
    and return all scalar statistics as a dict for JSON serialisation.
 
    Args:
        real_theta_prime:    (N_r, D) real latent weights (encoder output)
        real_theta_mod:      (N_r, D) real modulated weights (decode_modulations output)
        sampled_theta_prime: (N_s, D) sampled latent weights (sample_weight output)
        sampled_theta_mod:   (N_s, D) sampled modulated weights (decode_modulations output)
        out_dir:             directory to save plots
    Returns:
        dict of scalar statistics
    """
    os.makedirs(out_dir, exist_ok=True)
 
    rp = real_theta_prime.numpy()
    rm = real_theta_mod.numpy()
    sp = sampled_theta_prime.numpy()
    sm = sampled_theta_mod.numpy()
 
    # ── Per-dimension statistics (uses theta_prime for apples-to-apples) ──────
    rp_mean, rp_std = rp.mean(axis=0), rp.std(axis=0)
    sp_mean, sp_std = sp.mean(axis=0), sp.std(axis=0)
 
    D    = rp.shape[1]
    dims = np.arange(D)
 
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
 
    axes[0].plot(dims, rp_mean, label="Real θ′",    alpha=0.8, linewidth=0.8)
    axes[0].plot(dims, sp_mean, label="Sampled θ′", alpha=0.8, linewidth=0.8)
    axes[0].set_ylabel("Mean")
    axes[0].set_title("Per-dimension statistics  (θ′ = pre-modulation latent weights)")
    axes[0].legend()
 
    axes[1].plot(dims, rp_std, label="Real θ′",    alpha=0.8, linewidth=0.8)
    axes[1].plot(dims, sp_std, label="Sampled θ′", alpha=0.8, linewidth=0.8)
    axes[1].set_ylabel("Std")
    axes[1].set_xlabel("Dimension index")
    axes[1].legend()
 
    plt.tight_layout()
    path = os.path.join(out_dir, "weight_stats_per_dim.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] {path}")
 
    # ── 2×2 histogram grid ────────────────────────────────────────────────────
    panels = [
        (rp, "Real  —  θ′  (pre-modulation)",      "top-left"),
        (rm, "Real  —  θ   (post decode_modulations)",  "top-right"),
        (sp, "Sampled  —  θ′  (pre-modulation)",   "bot-left"),
        (sm, "Sampled  —  θ   (post decode_modulations)", "bot-right"),
    ]
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
 
    for ax, (data, title, _) in zip(axes, panels):
        ax.hist(data.flatten(), bins=300, density=True, color="steelblue", alpha=0.75)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        stats = (f"μ={data.mean():+.3f}  σ={data.std():.3f}\n"
                 f"min={data.min():.3f}  max={data.max():.3f}")
        ax.text(0.97, 0.95, stats, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
 
    fig.suptitle("Weight-value distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "weight_histogram_2x2.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] {path}")
 
    # ── Scalar metrics ────────────────────────────────────────────────────────
    mae_mean = float(np.abs(rp_mean - sp_mean).mean())
    mae_std  = float(np.abs(rp_std  - sp_std ).mean())
 
    print("  Computing MMD (this may take a moment) …")
    mmd_prime = _mmd_rbf(torch.from_numpy(rp), torch.from_numpy(sp))
    mmd_mod   = _mmd_rbf(torch.from_numpy(rm), torch.from_numpy(sm))
 
    # std ratio: how much decode_modulations scales spread — should match for real vs sampled
    std_ratio_real    = float(rm.std() / (rp.std() + 1e-8))
    std_ratio_sampled = float(sm.std() / (sp.std() + 1e-8))
 
    def _scalar_stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
        }
 
    stats = {
        "real_theta_prime":    _scalar_stats(rp),
        "real_theta_mod":      _scalar_stats(rm),
        "sampled_theta_prime": _scalar_stats(sp),
        "sampled_theta_mod":   _scalar_stats(sm),
        "mae_per_dim_mean":    mae_mean,
        "mae_per_dim_std":     mae_std,
        "mmd2_theta_prime":    mmd_prime,
        "mmd2_theta_mod":      mmd_mod,
        "std_ratio_real":      std_ratio_real,
        "std_ratio_sampled":   std_ratio_sampled,
    }
    return stats
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Image-space analysis
# ─────────────────────────────────────────────────────────────────────────────
 
def save_image_grid(
    images:   torch.Tensor,
    path:     str,
    n_side:   int = 8,
    title:    str = "",
) -> None:
    """
    Save an n_side × n_side grid of images using individual subplot axes.
 
    Args:
        images: (N, C, H, W) tensor in [0, 1]  — only first n_side² used
        path:   output file path
        n_side: grid is n_side × n_side
        title:  optional suptitle
    Returns:
        None
    """
    channels = images.shape[1]
    # (N, C, H, W) → (N, H, W, C) or (N, H, W) for grayscale
    imgs_np = images[:n_side * n_side].clamp(0, 1).permute(0, 2, 3, 1).numpy()
    if channels == 1:
        imgs_np = imgs_np.squeeze(-1)   # (N, H, W)
 
    fig, axes = plt.subplots(n_side, n_side, figsize=(n_side * 1.5, n_side * 1.5))
    for i, ax in enumerate(axes.flatten()):
        if channels == 1:
            ax.imshow(imgs_np[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(imgs_np[i], vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
 
    if title:
        fig.suptitle(title, fontsize=11)
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
 
 
def analyse_images(
    sampled_imgs: torch.Tensor,
    real_imgs:    torch.Tensor,
    recon_imgs:   torch.Tensor,
    out_dir:      str,
) -> None:
    """
    Save three 8×8 image grids (real / reconstructed / sampled) plus a
    combined 3-row comparison strip (8 images per row, one row per set).
 
    Args:
        sampled_imgs: (N, C, H, W) diffusion samples in [0, 1]
        real_imgs:    (N, C, H, W) real MNIST images
        recon_imgs:   (N, C, H, W) reconstructed images in [0, 1]
        out_dir:      directory to save plots
    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
 
    N_SIDE = 8   # 8×8 = 64 images per grid
 
    save_image_grid(sampled_imgs, os.path.join(out_dir, "grid_sampled.png"),
                    n_side=N_SIDE, title="Sampled images (diffusion in weight space)")
    save_image_grid(real_imgs,    os.path.join(out_dir, "grid_real.png"),
                    n_side=N_SIDE, title="Real MNIST images")
    save_image_grid(recon_imgs,   os.path.join(out_dir, "grid_reconstructed.png"),
                    n_side=N_SIDE, title="Reconstructed images (encode → decode)")
 
    # ── Combined strip: 3 rows × 8 images, labelled ──────────────────────────
    channels = real_imgs.shape[1]
    n_strip  = N_SIDE   # one row of 8 per set
 
    sets = [
        ("Real",     real_imgs[:n_strip]),
        ("Recon",    recon_imgs[:n_strip]),
        ("Sampled",  sampled_imgs[:n_strip]),
    ]
 
    fig, axes = plt.subplots(3, n_strip, figsize=(n_strip * 1.5, 3 * 1.5))
    for row_idx, (label, imgs) in enumerate(sets):
        imgs_np = imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()
        if channels == 1:
            imgs_np = imgs_np.squeeze(-1)
        for col_idx in range(n_strip):
            ax = axes[row_idx, col_idx]
            if channels == 1:
                ax.imshow(imgs_np[col_idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(imgs_np[col_idx], vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
        # row label on the leftmost cell
        axes[row_idx, 0].set_ylabel(label, fontsize=10, fontweight="bold", rotation=0,
                                    labelpad=40, va="center")
 
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    path = os.path.join(out_dir, "comparison_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
 
 
 
def save_summary(stats: dict, out_dir: str) -> None:
    """
    Save stats dict to JSON and print a formatted summary table.
 
    Args:
        stats:   dict returned by analyse_weights
        out_dir: directory to write summary.json into
    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
 
    stats["timestamp"] = datetime.now().isoformat(timespec="seconds")
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  [saved] {path}")
 
    # ── Pretty print ──────────────────────────────────────────────────────────
    W = 58  # box width
    def row(label, value, unit=""):
        line = f"  {label:<34}{value}{unit}"
        print(line)
 
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  WEIGHT-SPACE ANALYSIS SUMMARY".center(W) + "║")
    print("╠" + "═" * W + "╣")
 
    print("║" + "  Distribution statistics (global)".ljust(W) + "║")
    print("║" + "─" * W + "║")
    for key, label in [
        ("real_theta_prime",    "Real      θ′  (pre-mod) "),
        ("real_theta_mod",      "Real      θ   (post-mod)"),
        ("sampled_theta_prime", "Sampled   θ′  (pre-mod) "),
        ("sampled_theta_mod",   "Sampled   θ   (post-mod)"),
    ]:
        s = stats[key]
        print(f"║  {label}  μ={s['mean']:+.4f}  σ={s['std']:.4f}  [{s['min']:.3f}, {s['max']:.3f}]".ljust(W + 1) + "║")
 
    print("╠" + "═" * W + "╣")
    print("║" + "  Similarity metrics  (θ′: sampled vs real)".ljust(W) + "║")
    print("║" + "─" * W + "║")
    row("MAE per-dim mean (θ′):", f"{stats['mae_per_dim_mean']:.6f}")
    print("║" + "  " + "─" * (W - 2) + "║")
    row("MAE per-dim std  (θ′):", f"{stats['mae_per_dim_std']:.6f}")
    print("║" + "  " + "─" * (W - 2) + "║")
    mmd_p = stats["mmd2_theta_prime"]
    mmd_m = stats["mmd2_theta_mod"]
    row("MMD²  θ′  (pre-mod):", f"{mmd_p:.6f}",
        "  ◀ near 0 = distributions match")
    print("║" + "  " + "─" * (W - 2) + "║")
    row("MMD²  θ   (post-mod):", f"{mmd_m:.6f}",
        "  ◀ near 0 = distributions match")
    print("╠" + "═" * W + "╣")
    print("║" + "  decode_modulations scaling".ljust(W) + "║")
    print("║" + "─" * W + "║")
    sr  = stats["std_ratio_real"]
    ss  = stats["std_ratio_sampled"]
    row("std ratio θ/θ′  (real):   ", f"{sr:.4f}")
    row("std ratio θ/θ′  (sampled):", f"{ss:.4f}")
    delta = abs(sr - ss)
    note = "✓ consistent" if delta < 0.05 else "✗ mismatch — diffusion may be off-distribution"
    row("  Δ ratio:", f"{delta:.4f}  {note}")
    print("╚" + "═" * W + "╝")
    print()
 
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse a trained WeightNDMDiffusion model.")
    p.add_argument("--config",    required=True,  help="Path to run_config.json")
    p.add_argument("--weights",   required=True,  help="Path to weights .pt file")
    p.add_argument("--n_samples", type=int, default=512, help="Diffusion samples to draw")
    p.add_argument("--n_real",    type=int, default=512, help="Real images to encode/compare")
    p.add_argument("--out_dir",   default="analysis_out/", help="Output directory")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()
 
 
def main() -> None:
    args_cli = parse_args()
    device   = torch.device(args_cli.device)
 
    print(f"\n── Config  : {args_cli.config}")
    print(f"── Weights : {args_cli.weights}")
    print(f"── Device  : {device}\n")
 
    # ── Load hparams & build model ────────────────────────────────────────────
    print("[ 1 / 5 ]  Loading config & building model …")
    hparams = load_config(args_cli.config)
 
    data_config = {
        "channels": hparams.channels if hasattr(hparams, "channels") else 1,
        "img_size":  hparams.img_size  if hasattr(hparams, "img_size")  else 28,
        "data_dim":  hparams.data_dim  if hasattr(hparams, "data_dim")  else 784,
    }
 
    model = load_model(hparams, data_config, args_cli.weights, device)
    print(f"  Model loaded — modulation_dim = {model.weight_encoder.modulation_dim}")
 
    # ── Real data ─────────────────────────────────────────────────────────────
    print("[ 2 / 5 ]  Loading real MNIST images …")
    train_ds, _, _ = build_dataset(
        dataset_name=hparams.dataset,
        data_root=getattr(hparams, "data_root", "data/"),
        subset_frac=1.0,
        single_class=getattr(hparams, "single_class", False),
        single_class_label=getattr(hparams, "single_class_label", 1),
    )
    # subsample to exactly n_real images
    idx    = np.random.choice(len(train_ds), size=min(args_cli.n_real, len(train_ds)), replace=False)
    subset = Subset(train_ds, idx.tolist())
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)
 
    # ── Encode real images → weights ──────────────────────────────────────────
    print("[ 3 / 5 ]  Encoding real images through VAE encoder …")
    real_theta_prime, real_theta_mod, real_images = encode_real_images(
        model, loader, args_cli.n_real, device
    )
    print(f"  Encoded {len(real_theta_prime)} images  →  θ′ shape: {real_theta_prime.shape}")
 
    # Reconstruct real images via encode → decode (uses modulated weights)
    recon_images = decode_weights_to_images(model, real_theta_prime, device)
    print(f"  Reconstructed {len(recon_images)} images")
 
    # ── Sample from diffusion ─────────────────────────────────────────────────
    print(f"[ 4 / 5 ]  Drawing {args_cli.n_samples} diffusion samples …")
    with torch.no_grad():
        sampled_theta_prime = model.sample_weight(n_samples=args_cli.n_samples)
        # mirror model.sample(): decode_modulations then decode to pixels
        sampled_theta_mod = model.weight_encoder.decode_modulations(
            sampled_theta_prime.to(device)
        ).cpu()
    sampled_images = decode_weights_to_images(model, sampled_theta_prime, device)
    print(f"  Sampled {len(sampled_images)} images")
 
    # ── Analysis ──────────────────────────────────────────────────────────────
    print("[ 5 / 5 ]  Running analyses …")
 
    weight_out = os.path.join(args_cli.out_dir, "weight_space")
    image_out  = os.path.join(args_cli.out_dir, "image_space")
 
    print("\n▸ Weight-space analysis")
    stats = analyse_weights(
        real_theta_prime    = real_theta_prime.cpu(),
        real_theta_mod      = real_theta_mod.cpu(),
        sampled_theta_prime = sampled_theta_prime.cpu(),
        sampled_theta_mod   = sampled_theta_mod.cpu(),
        out_dir             = weight_out,
    )
 
    print("\n▸ Image-space analysis")
    analyse_images(
        sampled_imgs = sampled_images.cpu(),
        real_imgs    = real_images.cpu(),
        recon_imgs   = recon_images.cpu(),
        out_dir      = image_out,
    )
 
    print("\n▸ Saving summary …")
    save_summary(stats, out_dir=args_cli.out_dir)
 
    print(f"✓ All outputs saved to: {args_cli.out_dir}\n")
 
 
if __name__ == "__main__":
    main()