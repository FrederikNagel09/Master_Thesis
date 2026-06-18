"""
eval_visual.py
Generates two evaluation plots for a trained INR-based model:
  1. A single-row of N upscaled samples at a custom resolution.
  2. Side-by-side comparison of upscaled reconstructions (top row)
     vs. bilinear/bicubic interpolated originals (bottom row).

Usage
-----
python src/scripts/get_results.py \
    --config_path src/train_results/Weight-Diffusion-Probabilistic/metadata/config.json \
    --sample_scale 128 \
    --n_samples 10 \
    --n_recon 8 \
    --interp_mode bicubic

python src/scripts/get_results.py \
    --config_path src/results/vae_baselinea10/vae_baselinea10_config.json \
    --checkpoint_path src/results/vae_baselinea10/vae_baselinea10_checkpoint.pt \
    --sample_scale 128 \
    --n_samples 10 \
    --n_recon 8 \
    --interp_mode bicubic  
"""

from __future__ import annotations
 
import argparse
import json
import os
import sys
from types import SimpleNamespace
 
sys.path.append(".")
 
import warnings
warnings.filterwarnings("ignore", message="The operator 'aten::im2col' is not currently supported on the MPS backend")
 
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
 
 
# ── Coord grid (mirrors make_coord_grid in the decoder) ───────────────────────
 
def make_coord_grid(shape: tuple[int, ...], range: list | tuple, device=None) -> torch.Tensor:
    """
    Build a coordinate grid matching the model's internal convention.
 
    Args:
        shape:  Spatial resolution, e.g. (H, W).
        range:  Coordinate range [minv, maxv] or per-dim [[min0,max0], ...].
        device: Target device.
    Returns:
        grid: (*shape, len(shape)) float tensor.
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], (list, tuple)):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid
 
 
# ── VAE model builder ─────────────────────────────────────────────────────────
 
def build_vae_model(vae_config: dict, channels: int, img_size: int, device: str):
    """
    Build and return a VAEWrapper from a saved VAE _config.json dict.
 
    Args:
        vae_config: Flat config dict loaded from <run_name>_config.json.
        channels:   Number of image channels (from build_dataset).
        img_size:   Spatial image size (from build_dataset).
        device:     Device string.
    Returns:
        model: VAEWrapper on device, weights NOT yet loaded.
    """
    import torch.nn as nn
    from src.models.LatentEncoder import ResNetLatentEncoder
    from src.models.trans_inr import TransInr
 
    class VAEWrapper(nn.Module):
        """Thin wrapper combining ResNetLatentEncoder + TransInr decoder."""
        def __init__(self, encoder, decoder, img_size, device):
            super().__init__()
            self.latent_encoder = encoder
            self.decoder = decoder
            self.img_size = img_size
            self.device = device
            self.register_buffer("coord_grid", make_coord_grid((img_size, img_size), (-1, 1)))
 
        def encode(self, x):
            """Returns (mu, logvar) — mirrors LDM encode() signature for reconstruct_at_scale."""
            mu, logvar = self.latent_encoder(x)
            # Return mu as the deterministic latent (no sampling during eval)
            return mu, None, None
 
    latent_dim  = vae_config["latent_dim"]
    latent_size = vae_config["latent_size"]
 
    encoder = ResNetLatentEncoder(
        in_channels=channels,
        latent_dim=latent_dim,
        latent_size=(latent_size, latent_size),
        hidden_dim=vae_config["latent_enc_hidden_dim"],
    )
    decoder = TransInr(
        tokenizer={
            "target": "src.models.trans_inr_helpers.LatentTokenizer",
            "params": {
                "latent_dim": latent_dim,
                "latent_size": latent_size,
                "patch_size": vae_config["latent_patch_size"],
                "dim": vae_config["dec_trans_dim"],
                "n_head": vae_config["dec_trans_n_head"],
                "head_dim": vae_config["dec_trans_head_dim"],
            },
        },
        inr={
            "target": "src.models.trans_inr_helpers.SIREN",
            "params": {
                "depth": vae_config["inr_layers"],
                "in_dim": 2,
                "out_dim": channels,
                "hidden_dim": vae_config["inr_hidden_dim"],
                "out_bias": 0.5,
            },
        },
        data_shape=(img_size, img_size),
        n_groups=vae_config["dec_trans_n_groups"],
        transformer={
            "target": "src.models.trans_inr_helpers.Transformer",
            "params": {
                "dim": vae_config["dec_trans_dim"],
                "encoder_depth": vae_config["dec_trans_enc_depth"],
                "decoder_depth": vae_config["dec_trans_dec_depth"],
                "n_head": vae_config["dec_trans_n_head"],
                "head_dim": vae_config["dec_trans_head_dim"],
                "ff_dim": vae_config["dec_trans_ff_dim"],
            },
        },
        update_strategy=vae_config["dec_trans_update_strategy"],
    )
    return VAEWrapper(encoder, decoder, img_size, device).to(device)
 
 
# ── Sampling at custom resolution ─────────────────────────────────────────────
 
def _to_numpy_images(x_hat: torch.Tensor, channels: int) -> np.ndarray:
    """
    Convert a (B, C, H, W) float tensor in [0,1] to a numpy image array.
 
    Args:
        x_hat:    (B, C, H, W) float tensor already clamped to [0,1].
        channels: Number of image channels.
    Returns:
        images: (B, H, W) for grayscale or (B, H, W, C) for RGB.
    """
    x_hat = x_hat.cpu().float()
    if channels == 1:
        return x_hat.squeeze(1).numpy()
    return x_hat.permute(0, 2, 3, 1).numpy()
 
 
@torch.no_grad()
def sample_at_scale(
    model,
    model_type: str,
    n_samples: int,
    scale: int,
    device: str,
    channels: int,
    hparams,
) -> np.ndarray:
    """
    Sample from the model and decode at a custom spatial resolution.
 
    Args:
        model:      Trained model (LDM, VAEWrapper, or WeightDiffusion).
        model_type: "ldm", "vae", or "weight_diffusion".
        n_samples:  Number of images to generate.
        scale:      Target H=W resolution.
        device:     Device string.
        channels:   Number of image channels.
        hparams:    Namespace/dict with latent shape info (used for VAE).
    Returns:
        images: (n_samples, scale, scale) or (n_samples, scale, scale, C) in [0,1].
    """
    coord = make_coord_grid((scale, scale), (-1, 1), device=device)  # (scale, scale, 2)
 
    if model_type == "ldm":
        # Use diffusion sampler then decode at custom scale via TransInr decoder
        z = model._sample_latent(n_samples, collect_snapshots=False, debug=False)
        if model._normalize:
            z = model._denormalize_z(z)
        x_hat = model.decoder(z, coord)                # (B, C, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
 
    elif model_type == "weight_diffusion":
        # Sample weight vectors via reverse diffusion, then decode via SIREN at custom scale
        # coord must be (B, H, W, 2) — _inr_decode handles the batch expand internally
        theta = model.sample_weight(n_samples)                                # (B, weight_dim)
        theta = model.weight_encoder.decode_modulations(theta)                # structured params
        # Pass custom coord grid; _inr_decode expects (B, H, W, 2) or (1, H, W, 2)
        coord_batched = coord.unsqueeze(0).expand(n_samples, -1, -1, -1)     # (B, scale, scale, 2)
        pixels = model._inr_decode(theta, coords=coord_batched)               # (B, scale*scale)
        x_hat = pixels.reshape(n_samples, channels, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
 
    else:
        # VAE: sample z ~ N(0, I) directly then decode via TransInr decoder
        latent_dim  = hparams["latent_dim"] if isinstance(hparams, dict) else hparams.latent_dim
        latent_size = hparams["latent_size"] if isinstance(hparams, dict) else hparams.latent_size
        z = torch.randn(n_samples, latent_dim, latent_size, latent_size, device=device)
        x_hat = model.decoder(z, coord)                # (B, C, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
 
    return _to_numpy_images(x_hat, channels)
 
 
# ── Reconstruction at custom resolution ───────────────────────────────────────
 
@torch.no_grad()
def reconstruct_at_scale(
    model,
    model_type: str,
    x: torch.Tensor,
    scale: int,
    device: str,
    channels: int,
) -> np.ndarray:
    """
    Encode a batch of images and decode at a custom spatial resolution.
 
    Args:
        model:      Trained model (LDM, VAEWrapper, or WeightDiffusion).
        model_type: "ldm", "vae", or "weight_diffusion".
        x:          (B, C, H, W) input images on device.
        scale:      Target H=W resolution for the reconstruction.
        device:     Device string.
        channels:   Number of image channels.
    Returns:
        recons: (B, scale, scale) or (B, scale, scale, C) float32 in [0,1].
    """
    coord = make_coord_grid((scale, scale), (-1, 1), device=device)  # (scale, scale, 2)
 
    # Reshape flat inputs to (B, C, H, W) for all paths
    if x.dim() == 2:
        img_size = int(round((x.shape[1] // channels) ** 0.5))
        x = x.view(x.shape[0], channels, img_size, img_size)
 
    if model_type == "weight_diffusion":
        # WeightDiffusion encodes to flat weight vectors, decodes via SIREN
        B = x.shape[0]  # noqa: N806
        theta_prime_raw, _, _ = model.encode(x)                              # (B, weight_dim)
        theta = model.weight_encoder.decode_modulations(theta_prime_raw)     # structured params
        coord_batched = coord.unsqueeze(0).expand(B, -1, -1, -1)            # (B, scale, scale, 2)
        pixels = model._inr_decode(theta, coords=coord_batched)              # (B, scale*scale)
        x_hat = pixels.reshape(B, channels, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
    else:
        # LDM and VAEWrapper both expose encode() → (z, ?, ?) and a TransInr decoder
        z, _, _ = model.encode(x)
        x_hat = model.decoder(z, coord)                                      # (B, C, scale, scale)
        x_hat = (x_hat * 0.5 + 0.5).clamp(0, 1)
 
    return _to_numpy_images(x_hat, channels)
 
 
# ── Path helper ───────────────────────────────────────────────────────────────
 
def _extract_run_name(config_path: str) -> str:
    """
    Extract run name from .../<run_name>/metadata/config.json.
 
    Args:
        config_path: Path to config JSON.
    Returns:
        run_name string.
    """
    parts = os.path.normpath(config_path).split(os.sep)
    try:
        idx = parts.index("metadata")
        return parts[idx - 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"Could not extract run name from: {config_path}\n"
            "Expected: .../<run_name>/metadata/config.json"
        )
 
 
# ── Plot 1: upscaled sample row ───────────────────────────────────────────────
 
def plot_sample_row(
    model,
    model_type: str,
    hparams,
    n_samples: int,
    scale: int,
    device: str,
    channels: int,
    epoch: int,
    run_dir: str,
) -> None:
    """
    Generate and save a single-row plot of upscaled model samples.
 
    Args:
        model:      Trained model.
        model_type: "ldm" or "vae".
        hparams:    Namespace/dict with latent shape info (used for VAE sampling).
        n_samples:  Number of samples in the row.
        scale:      Target resolution (scale x scale).
        device:     Device string.
        channels:   Number of image channels.
        epoch:      Epoch number (used in filename).
        run_dir:    Output directory.
    Returns:
        None
    """
    print(f"  Generating {n_samples} samples at {scale}x{scale} …")
    images = sample_at_scale(model, model_type, n_samples, scale, device, channels, hparams)
 
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 1.5, 1.5))
    for ax, img in zip(axes, images):
        if channels == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
 
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    save_path = os.path.join(run_dir, f"samples_upscaled_{scale}x{scale}_ep{epoch}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Upscaled samples saved → {save_path}")
 
 
# ── Plot 2: reconstruction vs interpolated original ───────────────────────────
 
def plot_recon_vs_interp(
    model,
    model_type: str,
    val_loader: torch.utils.data.DataLoader,
    n_images: int,
    scale: int,
    device: str,
    channels: int,
    interp_mode: str,
    epoch: int,
    run_dir: str,
) -> None:
    """
    Three-row comparison: reconstructions, interpolated originals, native originals.
 
    Args:
        model:       Trained model.
        model_type:  "ldm", "vae", or "weight_diffusion".
        val_loader:  Validation DataLoader for sourcing original images.
        n_images:    Number of images in each row.
        scale:       Target resolution for reconstruction and interpolation.
        device:      Device string.
        channels:    Number of image channels.
        interp_mode: Interpolation mode for upscaling originals, 'bilinear' or 'bicubic'.
        epoch:       Epoch number (used in filename).
        run_dir:     Output directory.
    Returns:
        None
    """
    print(f"  Fetching {n_images} validation images …")
 
    # Grab exactly n_images from the val loader
    x_batch = []
    for batch in val_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_batch.append(imgs)
        if sum(b.shape[0] for b in x_batch) >= n_images:
            break
    x = torch.cat(x_batch, dim=0)[:n_images]  # (N, C, H, W)
 
    # Reshape if flat — derive img_size from tensor shape, no config needed
    if x.dim() == 2:
        img_size = int(round((x.shape[1] // channels) ** 0.5))
        x = x.view(x.shape[0], channels, img_size, img_size)
 
    x = x.to(device)
 
    # ── Reconstructions at target scale ───────────────────────────────────────
    print(f"  Reconstructing at {scale}x{scale} …")
    recons = reconstruct_at_scale(model, model_type, x, scale, device, channels)
 
    # ── Interpolate originals to target scale ─────────────────────────────────
    print(f"  Interpolating originals to {scale}x{scale} ({interp_mode}) …")
    align = False if interp_mode == "nearest" else True
    originals_up = F.interpolate(
        x.cpu().float(),
        size=(scale, scale),
        mode=interp_mode,
        align_corners=align,
    )
    # Un-normalize: training data is in [-1,1], bring to [0,1]
    originals_up = (originals_up * 0.5 + 0.5).clamp(0, 1)
 
    if channels == 1:
        originals_up = originals_up.squeeze(1).numpy()   # (N, H, W)
    else:
        originals_up = originals_up.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
 
    # ── Unscaled originals (native resolution) ───────────────────────────────
    originals_native = (x.cpu().float() * 0.5 + 0.5).clamp(0, 1)
    if channels == 1:
        originals_native = originals_native.squeeze(1).numpy()   # (N, H, W)
    else:
        originals_native = originals_native.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
 
    # ── Build 3-row figure with a title text row above each image row ─────────
    # Layout: 6 rows total — alternating title row (small) + image row
    n_rows = 3
    title_h  = 0.18   # relative height for each title row
    image_h  = 1.0    # relative height for each image row
    row_heights = []
    for _ in range(n_rows):
        row_heights += [title_h, image_h]
 
    fig, axes = plt.subplots(
        n_rows * 2, n_images,
        figsize=(n_images * 1.5, n_rows * 1.5 + 0.2),
        gridspec_kw={"height_ratios": row_heights},
    )
 
    rows_data = [
        (f"Reconstruction ({scale}×{scale})", recons),
        (f"Original upscaled {scale}×{scale} ({interp_mode})", originals_up),
        ("Original (native resolution)", originals_native),
    ]
 
    for group_idx, (label, imgs) in enumerate(rows_data):
        title_row = group_idx * 2      # 0, 2, 4
        image_row = group_idx * 2 + 1  # 1, 3, 5
 
        # Title spans all columns — use leftmost ax, hide the rest
        for col in range(n_images):
            axes[title_row, col].axis("off")
        axes[title_row, 0].text(
            0.0, 0.5, label,
            transform=axes[title_row, 0].transAxes,
            fontsize=7, va="center", ha="left",
            fontweight="bold",
        )
 
        # Image row
        for col in range(n_images):
            ax = axes[image_row, col]
            img = imgs[col]
            if channels == 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
 
    plt.subplots_adjust(hspace=0.03, wspace=0.02)
    save_path = os.path.join(run_dir, f"recon_vs_interp_{scale}x{scale}_ep{epoch}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Recon vs interp saved → {save_path}")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Upscaled sample + reconstruction vs interpolation plots.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to LDM config.json or VAE _config.json.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to VAE checkpoint .pt (required for VAE models).")
    parser.add_argument("--sample_scale", type=int, default=32, help="Target resolution for upscaled samples and reconstructions.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples in the sample row.")
    parser.add_argument("--n_recon", type=int, default=8, help="Number of images in the reconstruction comparison.")
    parser.add_argument("--interp_mode", type=str, default="bicubic", choices=["bilinear", "bicubic"], help="Interpolation method for upscaling originals.")
    args = parser.parse_args()
 
    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
 
    with open(args.config_path) as f:
        config = json.load(f)
 
    device = _get_device()
 
    # ── Detect config format and build model accordingly ──────────────────────
    is_ldm = "hparams" in config
 
    if is_ldm:
        from src.utility.model_builders import build_model as build_ldm_model
 
        hparams = SimpleNamespace(**config["hparams"])
        data_cfg = config["data"]
        data_config = {
            "dataset": config["dataset"],
            "channels": data_cfg["channels"],
            "img_size": data_cfg["img_size"],
            "data_dim": data_cfg["data_dim"],
        }
        epoch     = config["epochs"]["end"]
        channels  = data_config["channels"]
        run_name  = _extract_run_name(args.config_path)
 
        # Distinguish WeightDiffusion from other LDM-style models by the model name
        model_name = hparams.model if hasattr(hparams, "model") else config.get("model", "")
        model_type = "weight_diffusion" if "weight" in model_name.lower() else "ldm"
 
        print(f"\n{'=' * 60}")
        print(f"  Eval Visual [{model_type.upper()}]  |  run={run_name}  |  device={device}")
        print(f"{'=' * 60}\n")
 
        print("  Building LDM model …")
        model = build_ldm_model(hparams, data_config).to(device)
 
        weights_path = config["paths"]["weights"]
        print(f"  Loading weights from {weights_path} …")
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
 
        # build_dataset args come from the saved hparams
        dataset_kwargs = dict(
            dataset_name=data_config["dataset"],
            data_root=hparams.data_root,
            subset_frac=hparams.subset_frac,
            single_class=hparams.single_class,
            single_class_label=hparams.single_class_label,
        )
        # hparams is a Namespace here — pass as-is to sample_at_scale
        hparams_for_sample = hparams
 
    else:
        # VAE flat config
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint_path is required for VAE models.")
 
        vae_config = config
        run_name   = vae_config.get("run_name", os.path.splitext(os.path.basename(args.config_path))[0])
        model_type = "vae"
 
        print(f"\n{'=' * 60}")
        print(f"  Eval Visual [VAE]  |  run={run_name}  |  device={device}")
        print(f"{'=' * 60}\n")
 
        # Need channels/img_size from the dataset before building the model
        print("  Building validation dataset to get data config …")
        _, val_dataset_tmp, data_config = build_dataset(
            dataset_name=vae_config["dataset"],
            data_root="data/",
            subset_frac=1.0,
            single_class=False,
        )
        channels = data_config["channels"]
        img_size = data_config["img_size"]
 
        print("  Building VAE model …")
        model = build_vae_model(vae_config, channels, img_size, device)
 
        print(f"  Loading checkpoint from {args.checkpoint_path} …")
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
 
        epoch = ckpt.get("epoch_reached", 0)
 
        dataset_kwargs = dict(
            dataset_name=vae_config["dataset"],
            data_root="data/",
            subset_frac=1.0,
            single_class=False,
        )
        # vae_config is a plain dict — sample_at_scale handles both dict and Namespace
        hparams_for_sample = vae_config
 
    model.eval()
 
    print(f"  Scale={args.sample_scale}  |  n_samples={args.n_samples}  |  n_recon={args.n_recon}")
 
    # ── Val loader ────────────────────────────────────────────────────────────
    print("  Building validation dataset …")
    _, val_dataset, _ = build_dataset(**dataset_kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.n_recon,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
 
    run_dir = os.path.join("src", "results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Output dir: {run_dir}\n")
 
    # ── Run plots ─────────────────────────────────────────────────────────────
    plot_sample_row(
        model=model,
        model_type=model_type,
        hparams=hparams_for_sample,
        n_samples=args.n_samples,
        scale=args.sample_scale,
        device=device,
        channels=channels,
        epoch=epoch,
        run_dir=run_dir,
    )
 
    plot_recon_vs_interp(
        model=model,
        model_type=model_type,
        val_loader=val_loader,
        n_images=args.n_recon,
        scale=args.sample_scale,
        device=device,
        channels=channels,
        interp_mode=args.interp_mode,
        epoch=epoch,
        run_dir=run_dir,
    )
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()