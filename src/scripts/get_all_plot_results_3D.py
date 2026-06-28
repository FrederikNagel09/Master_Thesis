"""
get_all_plots_3d.py
Evaluation visualizations for INR-based models trained on 3D voxel data.
Produces two plots per model family (VAE, Latent Diffusion, Weight Diffusion):

  1. Scale row: one sample decoded at 6 different resolutions
     [0.125, 0.25, 0.5, 1, 2, 4] × base_res (32), rendered as marching-cubes meshes.

  2. Reconstruction plot: 10 validation images (top row) vs their reconstructions
     (bottom row), rendered as marching-cubes meshes.

Model loading convention (same as get_all_plot_results.py):
  - First config path in each group  → one-stage model (nested hparams/data/paths config)
  - Second and third config paths    → two-stage models (flat config, checkpoint in same dir)

Usage
-----
CUDA_VISIBLE_DEVICES=0 python src/scripts/get_all_plot_results_3D.py \
    --vae_config_path src/train_results/Latent-two_stage_convergence-VOXEL/Latent-two_stage_convergence-VOXEL_vae_config.json \
    --vae_checkpoint_path src/train_results/Latent-two_stage_convergence-VOXEL/Latent-two_stage_convergence-VOXEL_vae_checkpoint.pt \
    --latent_config_paths \
        src/train_results/latent-probability-3D-data/metadata/config.json \
        src/train_results/Latent-two_stage_fixed-VOXEL/Latent-two_stage_fixed-VOXEL_ldm_config.json \
        src/train_results/Latent-two_stage_convergence-VOXEL/Latent-two_stage_convergence-VOXEL_ldm_config.json \
    --weight_config_paths \
        src/train_results/weight-probability-3D-data/metadata/config.json \
        src/train_results/wb_two_stage_fixed-VOXEL/wb_two_stage_fixed-VOXEL_wd_config.json \
        src/train_results/wd_two_stage_convergence-VOXEL/wd_two_stage_convergence-VOXEL_wd_config.json \
    --base_res 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.latent_diffusion.modules.trans_inr import make_coord_grid
from src.utility.plotting import _render_mesh_on_ax, _samples_to_voxel_grids


# Azimuth offsets per column so meshes are viewed from slightly different angles
_AZIM_OFFSETS = [-60, -30, 0, 30, 60, 90, 120, 150, 180, 210]

# Scale multipliers applied to base_res
SCALE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4]


# ── Path helpers ───────────────────────────────────────────────────────────────
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
        raise ValueError(  # noqa: B904
            f"Could not extract run name from: {config_path}\n"
            "Expected: .../<run_name>/metadata/config.json"
        )


def _safe_name(run_name: str) -> str:
    """
    Sanitize a run name for use in a filename.

    Args:
        run_name: Arbitrary run name string.
    Returns:
        Lowercased, filesystem-safe string.
    """
    return run_name.lower().replace(" ", "_").replace("-", "_")


# ── VAE model builder (identical to get_all_plot_results but with 3D INR) ─────
def build_vae_model_3d(vae_config: dict, channels: int, img_size: int, device: str):
    """
    Build and return a VAEWrapper for 3D voxel data from a saved config dict.

    Args:
        vae_config: Flat config dict loaded from <run_name>_config.json.
        channels:   Number of voxel channels.
        img_size:   Spatial size per dimension (D=H=W).
        device:     Device string.
    Returns:
        VAEWrapper on device, weights NOT yet loaded.
    """
    import torch.nn as nn

    from src.models.latent_diffusion.modules.LatentEncoder import ResNetLatentEncoder
    from src.models.latent_diffusion.modules.trans_inr import TransInr

    class VAEWrapper(nn.Module):
        """Thin wrapper combining ResNetLatentEncoder + TransInr decoder for 3D."""

        def __init__(self, encoder, decoder, img_size, device):
            super().__init__()
            self.latent_encoder = encoder
            self.decoder = decoder
            self.img_size = img_size
            self.device = device

        def encode(self, x):
            """Returns (mu, None, None) — deterministic latent during eval."""
            mu, _ = self.latent_encoder(x)
            return mu, None, None

    latent_dim = vae_config["latent_dim"]
    latent_size = vae_config["latent_size"]

    encoder = ResNetLatentEncoder(
        in_channels=channels,
        latent_dim=latent_dim,
        latent_size=(latent_size, latent_size),
        hidden_dim=vae_config["latent_enc_hidden_dim"],
    )
    decoder = TransInr(
        tokenizer={
            "target": "src.models.tokenizers.latent_tokenizer.LatentTokenizer",
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
            "target": "src.models.inr.siren.SIREN",
            "params": {
                "depth": vae_config["inr_layers"],
                "in_dim": 3,  # 3D coords
                "out_dim": channels,
                "hidden_dim": vae_config["inr_hidden_dim"],
                "out_bias": 0.5,
                "out_activation": "sigmoid",  # voxels are binary [0,1]
            },
        },
        data_shape=(img_size, img_size, img_size),
        n_groups=vae_config["dec_trans_n_groups"],
        transformer={
            "target": "src.models.utils.transformer.Transformer",
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


# ── 3D coord grid at a given resolution ───────────────────────────────────────
def _make_3d_coord_grid(res: int, device: str) -> torch.Tensor:
    """
    Build a (D, H, W, 3) coordinate grid at a given cubic resolution.

    Args:
        res:    Spatial resolution per dimension.
        device: Device string.
    Returns:
        (res, res, res, 3) float tensor.
    """
    return make_coord_grid((res, res, res), (-1, 1), device=device)


# ── Sample one volume from a model at a given resolution ──────────────────────
@torch.no_grad()
def _sample_single_3d(
    model,
    model_type: str,
    res: int,
    device: str,
    channels: int,
    vae_config: dict | None = None,
) -> np.ndarray:
    """
    Sample one volume from the model and decode at resolution res^3.
    For INR-based models the same latent/weight is decoded at arbitrary resolution.
    For weight diffusion models the SIREN is queried at the new coord grid.

    Args:
        model:      Trained model.
        model_type: "vae", "ldm", or "weight_diffusion".
        res:        Target resolution per spatial dimension.
        device:     Device string.
        channels:   Number of voxel channels.
        vae_config: Required when model_type == "vae".
    Returns:
        voxels: (D, H, W) numpy array from marching-cubes rendering.
    """
    coord = _make_3d_coord_grid(res, device)  # (res, res, res, 3)

    if model_type == "vae":
        latent_dim = vae_config["latent_dim"]
        latent_size = vae_config["latent_size"]
        z = torch.randn(1, latent_dim, latent_size, latent_size, device=device)
        x_hat = model.decoder(z, coord)  # (1, C, res, res, res) or flat
    elif model_type == "ldm":
        from src.models.latent_diffusion.TwoStageLDM import TwoStageLDM
        z = model._sample_latent(1) if isinstance(model, TwoStageLDM) \
            else model._sample_latent(1, collect_snapshots=False, debug=False)
        if hasattr(model, "_normalize") and model._normalize:
            z = model._denormalize_z(z)
        x_hat = model.decoder(z, coord)
    else:  # weight_diffusion
        theta_prime = model.sample_weight(1)
        theta = model.weight_encoder.decode_modulations(theta_prime)
        coord_batched = coord.unsqueeze(0)  # (1, res, res, res, 3)
        x_hat = model._inr_decode(theta, coords=coord_batched)

    # Reshape to (1, C, res, res, res) then squeeze to (D, H, W)
    x_hat = x_hat.reshape(1, channels, res, res, res)
    return _samples_to_voxel_grids(x_hat, channels, res)[0]  # (D, H, W)


# ── Plot 1: scale row ─────────────────────────────────────────────────────────
def plot_scale_row_3d(
    model,
    model_type: str,
    base_res: int,
    device: str,
    channels: int,
    title: str,
    save_path: str,
    vae_config: dict | None = None,
) -> None:
    """
    Sample ONCE per scale factor and render each as a marching-cubes mesh.
    Produces a single row of 6 subplots with scale labels below.

    Args:
        model:      Trained model.
        model_type: "vae", "ldm", or "weight_diffusion".
        base_res:   Native training resolution (e.g. 32).
        device:     Device string.
        channels:   Number of voxel channels.
        title:      Figure suptitle (model name).
        save_path:  Output PNG path.
        vae_config: Required when model_type == "vae".
    Returns:
        None
    """
    n_scales = len(SCALE_FACTORS)
    fig = plt.figure(figsize=(n_scales * 2.5, 3.0))

    for col, factor in enumerate(SCALE_FACTORS):
        res = max(4, round(base_res * factor))  # floor at 4 to avoid degenerate grids
        voxels = _sample_single_3d(model, model_type, res, device, channels, vae_config)

        ax = fig.add_subplot(1, n_scales, col + 1, projection="3d")
        _render_mesh_on_ax(ax, voxels, azim=_AZIM_OFFSETS[col % len(_AZIM_OFFSETS)])

        # Scale label below each panel
        fig.text(
            (col + 0.5) / n_scales,
            0.02,
            f"{factor}× ({res}³)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    fig.suptitle(f"3D Scale Row: {title}", fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Scale row saved -> {save_path}")


# ── Plot 2: reconstruction comparison ─────────────────────────────────────────
@torch.no_grad()
def plot_reconstruction_3d(
    model,
    model_type: str,
    val_loader: torch.utils.data.DataLoader,
    base_res: int,
    device: str,
    channels: int,
    title: str,
    save_path: str,
) -> None:
    """
    Pick 10 validation volumes, reconstruct them, render both rows as meshes.
    Top row: originals. Bottom row: reconstructions. 10 columns.

    Args:
        model:      Trained model.
        model_type: "vae", "ldm", or "weight_diffusion".
        val_loader: DataLoader yielding (volume, label) batches.
        base_res:   Native training resolution.
        device:     Device string.
        channels:   Number of voxel channels.
        title:      Figure suptitle (model name).
        save_path:  Output PNG path.
    Returns:
        None
    """
    N = 10  # noqa: N806
    x_list = []
    for batch in val_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_list.append(x)
        if sum(b.shape[0] for b in x_list) >= N:
            break
    x = torch.cat(x_list, dim=0)[:N].to(device)

    # Ensure (B, C, D, H, W)
    if x.dim() == 2:
        x = x.view(x.shape[0], channels, base_res, base_res, base_res)

    coord = _make_3d_coord_grid(base_res, device)

    # Reconstruct
    if model_type == "weight_diffusion":
        theta_prime, _, _ = model.encode(x)
        theta = model.weight_encoder.decode_modulations(theta_prime)
        coord_batched = coord.unsqueeze(0).expand(N, -1, -1, -1, -1)
        x_hat = model._inr_decode(theta, coords=coord_batched)
    else:
        # vae and ldm share encode() -> (z, _, _) and decoder(z, coord)
        z, _, _ = model.encode(x)
        x_hat = model.decoder(z, coord)

    x_hat = x_hat.reshape(N, channels, base_res, base_res, base_res)

    orig_grids = _samples_to_voxel_grids(x, channels, base_res)      # (N, D, H, W)
    recon_grids = _samples_to_voxel_grids(x_hat, channels, base_res)  # (N, D, H, W)

    fig = plt.figure(figsize=(N * 2.0, 5.0))

    for col in range(N):
        azim = _AZIM_OFFSETS[col % len(_AZIM_OFFSETS)]

        # Top row: original
        ax_orig = fig.add_subplot(2, N, col + 1, projection="3d")
        _render_mesh_on_ax(ax_orig, orig_grids[col], azim=azim)

        # Bottom row: reconstruction
        ax_recon = fig.add_subplot(2, N, N + col + 1, projection="3d")
        _render_mesh_on_ax(ax_recon, recon_grids[col], azim=azim)

    # Row labels on the left
    fig.text(0.01, 0.75, "Originals", va="center", ha="left", fontsize=9, fontweight="bold", rotation=90)
    fig.text(0.01, 0.25, "Reconstructions", va="center", ha="left", fontsize=9, fontweight="bold", rotation=90)

    fig.suptitle(f"3D Reconstructions: {title}", fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Reconstruction plot saved -> {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3D eval plots for INR-based voxel models.")

    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--latent_config_paths", type=str, nargs="+", default=[], help="One-stage then two-stage LDM configs (max 3).")
    parser.add_argument("--weight_config_paths", type=str, nargs="+", default=[], help="One-stage then two-stage WD configs (max 3).")
    parser.add_argument("--base_res", type=int, default=32, help="Native training resolution per spatial dimension.")

    args = parser.parse_args()

    if len(args.latent_config_paths) > 3:
        parser.error("Max 3 --latent_config_paths.")
    if len(args.weight_config_paths) > 3:
        parser.error("Max 3 --weight_config_paths.")

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device

    device = _get_device()
    output_dir = os.path.join("src", "results", "final_results_3d")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  3D Eval Plots  |  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # ── Load VAE config + dataset ─────────────────────────────────────────────
    with open(args.vae_config_path) as f:
        vae_config = json.load(f)

    print("  Building validation dataset ...")
    _, val_dataset, data_config = build_dataset(
        dataset_name=vae_config["dataset"],
        data_root="data/",
        subset_frac=1.0,
        single_class=False,
    )
    channels = data_config["channels"]
    img_size = data_config["img_size"]

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=0
    )

    # ── VAE ───────────────────────────────────────────────────────────────────
    print("--- Processing VAE Model ---")
    vae_model = build_vae_model_3d(vae_config, channels, img_size, device)
    vae_ckpt = torch.load(args.vae_checkpoint_path, map_location=device)
    vae_model.load_state_dict(vae_ckpt["model_state_dict"])
    vae_model.eval()

    plot_scale_row_3d(
        vae_model, "vae", args.base_res, device, channels,
        title="VAE-INR",
        save_path=os.path.join(output_dir, "vae_scale_row.png"),
        vae_config=vae_config,
    )
    plot_reconstruction_3d(
        vae_model, "vae", val_loader, args.base_res, device, channels,
        title="VAE-INR",
        save_path=os.path.join(output_dir, "vae_reconstructions.png"),
    )

    # ── Latent Diffusion models ────────────────────────────────────────────────
    if args.latent_config_paths:
        print(f"\n--- Processing Latent Diffusion Suite ({len(args.latent_config_paths)} variants) ---")
        from src.utility.model_builders import build_model as build_ldm_model
        from src.utility.model_builders.two_stage_builder import build_ldm as build_two_stage_ldm
        from src.models.latent_diffusion.TwoStageLDM import TwoStageLDM

        for idx, p in enumerate(args.latent_config_paths):
            with open(p) as f:
                l_cfg = json.load(f)

            if idx == 0:
                # One-stage: nested config
                l_hparams = SimpleNamespace(**l_cfg["hparams"])
                l_data_cfg = l_cfg["data"]
                l_data_config = {
                    "dataset": l_cfg["dataset"],
                    "channels": l_data_cfg["channels"],
                    "img_size": l_data_cfg["img_size"],
                    "data_dim": l_data_cfg["data_dim"],
                }
                run_name = _extract_run_name(p)
                print(f"  Building & loading (one-stage): {run_name} ...")
                l_model = build_ldm_model(l_hparams, l_data_config).to(device)
                l_ckpt = torch.load(l_cfg["paths"]["weights"], map_location=device)
                l_model.load_state_dict(l_ckpt["model_state_dict"])
            else:
                # Two-stage: flat config
                run_name = l_cfg["run_name"]
                ckpt_path = os.path.join(
                    os.path.dirname(os.path.abspath(p)),
                    f"{run_name}_ldm_checkpoint.pt"
                )
                ts_args = SimpleNamespace(T=l_cfg["T"], beta_1=l_cfg["beta_1"], beta_T=l_cfg["beta_T"])
                print(f"  Building & loading (two-stage): {run_name} ...")
                l_model = build_two_stage_ldm(
                    hparams=l_cfg, args=ts_args, channels=channels, img_size=img_size, device=device
                )
                l_ckpt = torch.load(ckpt_path, map_location=device)
                l_model.load_state_dict(l_ckpt["model_state_dict"])

            l_model.eval()
            safe = _safe_name(run_name)

            plot_scale_row_3d(
                l_model, "ldm", args.base_res, device, channels,
                title=run_name,
                save_path=os.path.join(output_dir, f"latent_scale_row_{safe}.png"),
            )
            plot_reconstruction_3d(
                l_model, "ldm", val_loader, args.base_res, device, channels,
                title=run_name,
                save_path=os.path.join(output_dir, f"latent_reconstructions_{safe}.png"),
            )

    # ── Weight Diffusion models ────────────────────────────────────────────────
    if args.weight_config_paths:
        print(f"\n--- Processing Weight Diffusion Suite ({len(args.weight_config_paths)} variants) ---")
        from src.utility.model_builders import build_model as build_ldm_model
        from src.scripts.two_stage_weight_training import build_full_wd_model

        for idx, p in enumerate(args.weight_config_paths):
            with open(p) as f:
                w_cfg = json.load(f)

            if idx == 0:
                # One-stage: nested config
                w_hparams = SimpleNamespace(**w_cfg["hparams"])
                w_data_cfg = w_cfg["data"]
                w_data_config = {
                    "dataset": w_cfg["dataset"],
                    "channels": w_data_cfg["channels"],
                    "img_size": w_data_cfg["img_size"],
                    "data_dim": w_data_cfg["data_dim"],
                }
                run_name = _extract_run_name(p)
                print(f"  Building & loading (one-stage): {run_name} ...")
                w_model = build_ldm_model(w_hparams, w_data_config).to(device)
                w_ckpt = torch.load(w_cfg["paths"]["weights"], map_location=device)
                state_dict = {k: v for k, v in w_ckpt["model_state_dict"].items() if k != "coords"}
                w_model.load_state_dict(state_dict, strict=False)
            else:
                # Two-stage: flat config
                run_name = w_cfg["run_name"]
                ckpt_path = os.path.join(
                    os.path.dirname(os.path.abspath(p)),
                    f"{run_name}_wd_weights.pt"
                )
                tsw_args = SimpleNamespace(T=w_cfg["T"], beta_1=w_cfg["beta_1"], beta_T=w_cfg["beta_T"])
                print(f"  Building & loading (two-stage): {run_name} ...")
                w_model = build_full_wd_model(
                    hparams=w_cfg,
                    args=tsw_args,
                    channels=channels,
                    img_size=img_size,
                    data_dim=data_config["data_dim"],
                    device=device,
                    is_3d=True,
                )
                w_ckpt = torch.load(ckpt_path, map_location=device)
                state_dict = {k: v for k, v in w_ckpt["full_model_state_dict"].items() if k != "coords"}
                w_model.load_state_dict(state_dict, strict=False)

            w_model.eval()
            safe = _safe_name(run_name)

            plot_scale_row_3d(
                w_model, "weight_diffusion", args.base_res, device, channels,
                title=run_name,
                save_path=os.path.join(output_dir, f"weight_scale_row_{safe}.png"),
            )
            plot_reconstruction_3d(
                w_model, "weight_diffusion", val_loader, args.base_res, device, channels,
                title=run_name,
                save_path=os.path.join(output_dir, f"weight_reconstructions_{safe}.png"),
            )

    print("\n3D Eval Plots Complete.")


if __name__ == "__main__":
    main()