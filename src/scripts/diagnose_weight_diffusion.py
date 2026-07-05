"""
Diagnostic script for WeightDiffusion model.

Runs three diagnostics:
    1. Encoder distribution vs. prior  — are encoded weights N(0,1)?
    2. Direct substitution test        — do real encoded weights decode well vs. sampled?
    3. L2 distance test                — are sampled weights geometrically close to real ones?

Run from Master_Thesis/:
    python src/scripts/diagnose_weight_diffusion.py
"""

import json
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── Project root on path ──────────────────────────────────────────────────────

sys.path.append(".")

from src.models.weight_diffusion.WeightDiffusion import WeightDiffusion  # noqa: E402
from src.utility.dataset_builders import build_dataset  # noqa: E402
from src.scripts.get_all_plot_results import make_coord_grid  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH  = "src/train_results/weight-diffusion/metadata/config.json"
WEIGHTS_PATH = "src/train_results/weight-diffusion/weights/weights.pt"
OUT_DIR      = Path("src/train_results/weight-diffusion/diagnostics")


N_DIAG_SAMPLES = 512   # number of encoded samples for diagnostic 1 & 3
N_SUBST_PAIRS  = 8     # number of image pairs for diagnostic 2
N_DIFFUSION_SAMPLES = 8  # sampled weight vectors for diagnostics 2 & 3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> types.SimpleNamespace:
    """Load config.json and flatten hparams into a SimpleNamespace args object.

    Args:
        path: Path to config.json.
    Returns:
        args: SimpleNamespace with all hparams as attributes, plus top-level data fields.
    """
    with open(path) as f:
        cfg = json.load(f)
    hparams = cfg["hparams"]
    # Patch key that _build_weight_diffusion reads with wrong suffix
    hparams.setdefault("noise_predictor_t_embed", hparams.get("noise_predictor_t_embed_dim", 128))
    args = types.SimpleNamespace(**hparams)
    return args, cfg["data"]


def _build_model(args: types.SimpleNamespace, data_config: dict) -> WeightDiffusion:
    """Build WeightDiffusion from args and data_config using only transinr predictor.

    Args:
        args:        SimpleNamespace of hparams.
        data_config: Dict with channels, img_size, data_dim.
    Returns:
        model: Unloaded WeightDiffusion on CPU.
    """
    # Import here to avoid circular-import issues at module level
    from src.utility.model_builders.util.weight_diffusion_builder import _build_weight_diffusion  # adjust if path differs
    return _build_weight_diffusion(args, data_config)


def _load_model(args, data_config: dict, weights_path: Path, device: torch.device) -> WeightDiffusion:
    """Build and load trained weights into the model.

    Args:
        args:         SimpleNamespace of hparams.
        data_config:  Dict with channels, img_size, data_dim.
        weights_path: Path to weights.pt checkpoint.
        device:       Target device.
    Returns:
        model: WeightDiffusion in eval mode on device.
    """
    model = _build_model(args, data_config)
    state = torch.load(weights_path, map_location=device)
    # Handle wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _get_encoded_weights(
    model: WeightDiffusion,
    val_loader: torch.utils.data.DataLoader,
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode validation images and collect raw latents, means, logvars.

    Args:
        model:      WeightDiffusion in eval mode.
        val_loader: Validation DataLoader.
        n_samples:  Max samples to collect.
        device:     Target device.
    Returns:
        theta_raws: (N, weight_dim) raw reparameterized latents.
        means:      (N, weight_dim) encoder means.
        logvars:    (N, weight_dim) encoder log-variances.
    """
    theta_raws, means, logvars = [], [], []
    collected = 0
    with torch.no_grad():
        for x, _ in val_loader:
            if collected >= n_samples:
                break
            x = x.to(device)
            mean, logvar = model.weight_encoder(x)
            theta_raw = model.weight_encoder._reparameterize(mean, logvar)
            theta_raws.append(theta_raw.cpu())
            means.append(mean.cpu())
            logvars.append(logvar.cpu())
            collected += x.shape[0]
    return (
        torch.cat(theta_raws)[:n_samples],
        torch.cat(means)[:n_samples],
        torch.cat(logvars)[:n_samples],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 1 — Encoder distribution vs. N(0,1) prior
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_1_encoder_distribution(
    theta_raws: torch.Tensor,
    means: torch.Tensor,
    logvars: torch.Tensor,
    out_dir: Path,
) -> None:
    """Plot histograms of encoded means, stds, and raw latents vs. N(0,1).

    Args:
        theta_raws: (N, weight_dim) reparameterized latents.
        means:      (N, weight_dim) encoder means.
        logvars:    (N, weight_dim) encoder log-variances.
        out_dir:    Directory to save the figure.
    Returns:
        None
    """
    stds = (0.5 * logvars).exp()

    # Flatten across all dims for marginal distributions
    flat_mean   = means.flatten().numpy()
    flat_std    = stds.flatten().numpy()
    flat_logvar = logvars.flatten().numpy()
    flat_raw    = theta_raws.flatten().numpy()

    ref = np.random.randn(len(flat_raw))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Diagnostic 1: Encoder Distribution vs. N(0,1)", fontsize=13, fontweight="bold")

    bins = 80
    kw = dict(density=True, alpha=0.7, bins=bins)

    # μ distribution — should be ~N(0,1) if encoder isn't collapsed
    axes[0].hist(flat_mean,   **kw, label="encoder μ",      color="steelblue")
    axes[0].hist(ref,          **kw, label="N(0,1)",          color="orange", histtype="step", linewidth=1.5)
    axes[0].set_title(f"Encoder μ  (mean={flat_mean.mean():.3f}, std={flat_mean.std():.3f})")
    axes[0].legend(fontsize=8)

    # σ distribution — should be ~1.0 if no collapse
    axes[1].hist(flat_std,    **kw, label="encoder σ",      color="steelblue")
    axes[1].axvline(1.0, color="orange", linewidth=1.5, label="σ=1 target")
    axes[1].set_title(f"Encoder σ  (mean={flat_std.mean():.3f}, std={flat_std.std():.3f})")
    axes[1].legend(fontsize=8)

    # logvar — should be ~0 if no collapse
    axes[2].hist(flat_logvar, **kw, label="encoder logvar", color="steelblue")
    axes[2].axvline(0.0, color="orange", linewidth=1.5, label="logvar=0 target")
    axes[2].set_title(f"logvar  (mean={flat_logvar.mean():.3f}, std={flat_logvar.std():.3f})")
    axes[2].legend(fontsize=8)

    # Raw reparameterized latent vs N(0,1)
    axes[3].hist(flat_raw, **kw, label="θ_raw (encoded)", color="steelblue")
    axes[3].hist(ref,       **kw, label="N(0,1)",           color="orange", histtype="step", linewidth=1.5)
    axes[3].set_title(f"θ_raw  (mean={flat_raw.mean():.3f}, std={flat_raw.std():.3f})")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    plt.tight_layout()
    save_path = out_dir / "diag1_encoder_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 1] Saved → {save_path}")
    print(f"  μ:       mean={flat_mean.mean():.4f},   std={flat_mean.std():.4f}")
    print(f"  σ:       mean={flat_std.mean():.4f},   std={flat_std.std():.4f}  (target: 1.0)")
    print(f"  logvar:  mean={flat_logvar.mean():.4f}, std={flat_logvar.std():.4f}  (target: 0.0)")
    print(f"  θ_raw:   mean={flat_raw.mean():.4f},   std={flat_raw.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 2 — Direct substitution test
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_2_substitution_test(
    model: WeightDiffusion,
    val_loader: torch.utils.data.DataLoader,
    n_pairs: int,
    n_diffusion_samples: int,
    device: torch.device,
    out_dir: Path,
    img_size: int = 28,
    channels: int = 1,
) -> None:
    """Compare: original | recon from encoded θ | recon from diffusion-sampled θ.

    Args:
        model:                WeightDiffusion in eval mode.
        val_loader:           Validation DataLoader.
        n_pairs:              Number of images to test.
        n_diffusion_samples:  Number of diffusion samples to draw.
        device:               Target device.
        out_dir:              Directory to save the figure.
        img_size:             Spatial size of images.
        channels:             Number of image channels.
    Returns:
        None
    """
    # --- Collect real images ---
    x_batch = next(iter(val_loader))[0][:n_pairs].to(device)

    with torch.no_grad():
        # Path A: encode → decode (should be perfect reconstruction)
        mean, logvar = model.weight_encoder(x_batch)
        theta_encoded = model.weight_encoder._reparameterize(mean, logvar)
        theta_decoded_enc = model.weight_encoder.decode_modulations(theta_encoded)
        recon_encoded = model._inr_decode(theta_decoded_enc)   # (n_pairs, data_dim)

        # Path B: diffusion sample → decode
        theta_sampled = model.sample_weight(n_diffusion_samples, debug=False)  # (N, weight_dim)
        theta_decoded_samp = model.weight_encoder.decode_modulations(theta_sampled)
        recon_sampled = model._inr_decode(theta_decoded_samp)  # (N, data_dim)

    def _to_img(t: torch.Tensor) -> np.ndarray:
        """Flat tensor → HxW numpy in [0,1]."""
        arr = t.cpu().numpy().reshape(channels, img_size, img_size)
        arr = arr * 0.5 + 0.5   # [-1,1] → [0,1]
        return np.clip(arr[0] if channels == 1 else arr.transpose(1, 2, 0), 0, 1)

    n_cols   = max(n_pairs, n_diffusion_samples)
    n_rows   = 3  # originals / encoded reconstructions / diffusion reconstructions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))
    fig.suptitle("Diagnostic 2: Direct Substitution Test", fontsize=13, fontweight="bold")

    row_labels = ["Original", "Recon (encoded θ)", "Recon (sampled θ)"]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=4)

    for c in range(n_cols):
        # Row 0: originals (only n_pairs columns filled)
        if c < n_pairs:
            axes[0, c].imshow(_to_img(x_batch[c]), cmap="gray" if channels == 1 else None, vmin=0, vmax=1)
        else:
            axes[0, c].axis("off")

        # Row 1: encoded reconstructions
        if c < n_pairs:
            axes[1, c].imshow(_to_img(recon_encoded[c]), cmap="gray" if channels == 1 else None, vmin=0, vmax=1)
        else:
            axes[1, c].axis("off")

        # Row 2: diffusion-sampled reconstructions
        if c < n_diffusion_samples:
            axes[2, c].imshow(_to_img(recon_sampled[c]), cmap="gray" if channels == 1 else None, vmin=0, vmax=1)
        else:
            axes[2, c].axis("off")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = out_dir / "diag2_substitution_test.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 2] Saved → {save_path}")

    # Per-row MSE summary
    orig_flat = x_batch.reshape(n_pairs, -1).cpu()
    recon_enc_flat = recon_encoded[:n_pairs].cpu()
    mse_enc = F.mse_loss(recon_enc_flat, orig_flat * 0.5 + 0.5).item()  # encoded recon vs original
    print(f"  MSE (encoded recon vs original): {mse_enc:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 3 — L2 distance: sampled vs. encoded weights
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_3_l2_distance(
    model: WeightDiffusion,
    theta_raws: torch.Tensor,
    n_diffusion_samples: int,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Compare L2 distances: sampled-vs-encoded, encoded-vs-encoded, sampled-vs-N(0,1).

    Args:
        model:                WeightDiffusion in eval mode.
        theta_raws:           (N, weight_dim) encoded latents from validation set.
        n_diffusion_samples:  Number of diffusion samples to draw.
        device:               Target device.
        out_dir:              Directory to save figure.
    Returns:
        None
    """
    with torch.no_grad():
        theta_sampled = model.sample_weight(n_diffusion_samples, debug=False).cpu()  # (M, D)

    N, D = theta_raws.shape
    M    = theta_sampled.shape[0]

    # Pairwise L2: sampled vs encoded  (M x N matrix, take row means)
    # Done in chunks to avoid memory issues
    dist_samp_enc = torch.cdist(theta_sampled, theta_raws).mean(dim=1).numpy()  # (M,)

    # Pairwise L2: encoded vs encoded  (upper triangle)
    dist_enc_enc = torch.cdist(theta_raws, theta_raws)
    idx = torch.triu_indices(N, N, offset=1)
    dist_enc_enc = dist_enc_enc[idx[0], idx[1]].numpy()  # (N*(N-1)/2,)

    # Pairwise L2: sampled vs N(0,1) draws  (baseline — how far is sampled from pure noise)
    noise_ref = torch.randn(M, D)
    dist_samp_noise = torch.cdist(theta_sampled, noise_ref).mean(dim=1).numpy()  # (M,)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Diagnostic 3: L2 Distance Analysis", fontsize=13, fontweight="bold")

    bins = 50
    axes[0].hist(dist_samp_enc,  bins=bins, color="steelblue", alpha=0.8, density=True)
    axes[0].set_title(f"Sampled θ ↔ Encoded θ\nmean={dist_samp_enc.mean():.2f}")
    axes[0].set_xlabel("L2 distance")

    axes[1].hist(dist_enc_enc,   bins=bins, color="green",     alpha=0.8, density=True)
    axes[1].set_title(f"Encoded θ ↔ Encoded θ\nmean={dist_enc_enc.mean():.2f}")
    axes[1].set_xlabel("L2 distance")

    axes[2].hist(dist_samp_noise, bins=bins, color="orange",   alpha=0.8, density=True)
    axes[2].set_title(f"Sampled θ ↔ N(0,1)\nmean={dist_samp_noise.mean():.2f}")
    axes[2].set_xlabel("L2 distance")

    plt.tight_layout()
    save_path = out_dir / "diag3_l2_distances.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 3] Saved → {save_path}")
    print(f"  Sampled ↔ Encoded : mean={dist_samp_enc.mean():.4f},  std={dist_samp_enc.std():.4f}")
    print(f"  Encoded ↔ Encoded : mean={dist_enc_enc.mean():.4f},  std={dist_enc_enc.std():.4f}")
    print(f"  Sampled ↔ N(0,1)  : mean={dist_samp_noise.mean():.4f}, std={dist_samp_noise.std():.4f}")
    print()
    print("  KEY: If 'Sampled ↔ Encoded' >> 'Encoded ↔ Encoded', sampled weights land")
    print("       in a completely different region — the diffusion prior mismatch is confirmed.")


def diagnostic_4_noise_sensitivity(
    model: WeightDiffusion,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: Path,
    img_size: int = 28,
    channels: int = 1,
    n_images: int = 4,
    noise_scales: list = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
) -> None:
    """
    Encode real images, add increasing noise to theta, decode and visualize.
    
    Args:
        noise_scales: std of Gaussian noise added to encoded theta.
    Returns:
        None — saves figure to out_dir.
    """
    x_batch = next(iter(val_loader))[0][:n_images].to(device)

    def _to_img(t):
        arr = t.cpu().numpy().reshape(channels, img_size, img_size) * 0.5 + 0.5
        return np.clip(arr[0] if channels == 1 else arr.transpose(1, 2, 0), 0, 1)

    with torch.no_grad():
        mean, logvar = model.weight_encoder(x_batch)
        theta_encoded = model.weight_encoder._reparameterize(mean, logvar)

        fig, axes = plt.subplots(n_images, len(noise_scales), figsize=(len(noise_scales) * 1.5, n_images * 1.8))
        fig.suptitle("Diagnostic 4: INR Sensitivity to Noise on Encoded θ", fontsize=12, fontweight="bold")

        for j, scale in enumerate(noise_scales):
            theta_noisy = theta_encoded + scale * torch.randn_like(theta_encoded)
            theta_dec = model.weight_encoder.decode_modulations(theta_noisy)
            recons = model._inr_decode(theta_dec)
            axes[0, j].set_title(f"σ={scale}", fontsize=8)
            for i in range(n_images):
                axes[i, j].imshow(_to_img(recons[i]), cmap="gray" if channels == 1 else None, vmin=0, vmax=1)
                axes[i, j].axis("off")

    plt.tight_layout()
    save_path = out_dir / "diag4_noise_sensitivity.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 4] Saved → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load model and run all three diagnostics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load config & build model ─────────────────────────────────────────────
    args, data_cfg = _load_config(CONFIG_PATH)
    data_config = {
        "channels": data_cfg["channels"],
        "img_size":  data_cfg["img_size"],
        "data_dim":  data_cfg["data_dim"],
    }
    print("Building model...")
    model = _load_model(args, data_config, WEIGHTS_PATH, device)
    print("Model loaded.\n")

    # ── Build validation loader ───────────────────────────────────────────────
    _, val_dataset, _ = build_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        subset_frac=1.0,
        single_class=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0
    )

    # ── Encode validation set once (shared across diag 1 & 3) ────────────────
    print(f"Encoding {N_DIAG_SAMPLES} validation images...")
    theta_raws, means, logvars = _get_encoded_weights(model, val_loader, N_DIAG_SAMPLES, device)
    print("Done.\n")

    # ── Run diagnostics ───────────────────────────────────────────────────────
    print("=== Diagnostic 1: Encoder Distribution ===")
    diagnostic_1_encoder_distribution(theta_raws, means, logvars, OUT_DIR)
    print()

    print("=== Diagnostic 2: Substitution Test ===")
    diagnostic_2_substitution_test(
        model, val_loader,
        n_pairs=N_SUBST_PAIRS,
        n_diffusion_samples=N_DIFFUSION_SAMPLES,
        device=device,
        out_dir=OUT_DIR,
        img_size=data_config["img_size"],
        channels=data_config["channels"],
    )
    print()

    print("=== Diagnostic 3: L2 Distance ===")
    diagnostic_3_l2_distance(model, theta_raws, N_DIFFUSION_SAMPLES, device, OUT_DIR)
    print()

    diagnostic_4_noise_sensitivity(
        model, val_loader, device, OUT_DIR,
        img_size=data_config["img_size"],
        channels=data_config["channels"],
        n_images=4,
        noise_scales=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    print(f"All diagnostics complete. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()