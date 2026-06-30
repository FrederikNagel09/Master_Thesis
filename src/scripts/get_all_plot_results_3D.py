"""
get_all_plots_3d.py
Evaluation visualizations for INR-based models trained on 3D voxel data.
Produces two plots per model family (VAE, Latent Diffusion, Weight Diffusion):
  1. Scale row: the SAME sample decoded at 6 different resolutions
     [0.125, 0.25, 0.5, 1, 2, 4] × base_res (32), rendered as marching-cubes meshes.
  2. Reconstruction plot: the SAME 10 validation volumes across all models.
     Top row: originals. Bottom row: reconstructions. Row labels below each row.
Model loading convention (same as get_all_plot_results.py):
  - First config path in each group  → one-stage model (nested hparams/data/paths config)
  - Second and third config paths    → two-stage models (flat config, checkpoint in same dir)
Usage
-----
CUDA_VISIBLE_DEVICES=1 python src/scripts/get_all_plot_results_3D.py \
    --vae_config_path src/results/vae_3d_baseline/vae_3d_baseline_config.json \
    --vae_checkpoint_path src/results/vae_3d_baseline/vae_3d_baseline_checkpoint.pt \
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

_AZIM_OFFSETS = [120, 120, 120, 120, 120, 120, 120, 120, 120, 120]
_ELEV = 25

# Scale multipliers and their display labels
SCALE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4]
SCALE_LABELS  = ["0.125x", "0.25x", "0.5x", "1x", "2x", "4x"]

N_RECON = 10  # fixed validation volumes shared across all models


# -- Path helpers --------------------------------------------------------------
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


# -- VAE model builder ---------------------------------------------------------
def build_vae_model_3d(vae_config: dict, channels: int, img_size: int, device: str):
    """
    Build and return a VAEWrapper3D from a saved 3D VAE config dict.
    Args:
        vae_config: Flat config dict loaded from <run_name>_config.json.
        channels:   Number of voxel channels.
        img_size:   Spatial size per dimension (D=H=W).
        device:     Device string.
    Returns:
        VAEWrapper3D on device, weights NOT yet loaded.
    """
    from src.models.latent_diffusion.modules.LatentEncoder3D import Conv3DEncoder
    from src.models.latent_diffusion.modules.trans_inr import TransInr

    latent_dim = vae_config["latent_dim"]
    latent_size = vae_config["latent_size"]

    encoder = Conv3DEncoder(
        in_channels=channels,
        dim_z=latent_dim,
        base_channels=vae_config.get("enc_base_channels", 64),
        dropout=vae_config.get("enc_dropout", 0.0),
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
                "in_dim": 3,
                "out_dim": channels,
                "hidden_dim": vae_config["inr_hidden_dim"],
                "out_bias": 0.5,
                "out_activation": "sigmoid",
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
    from src.scripts.VAE_Baseline_Training_3D import VAEWrapper3D
    return VAEWrapper3D(encoder, decoder, img_size, device).to(device)


# -- 3D coord grid -------------------------------------------------------------
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


# -- Draw fixed validation batch (shared across all models) --------------------
def draw_fixed_val_batch(
    val_loader: torch.utils.data.DataLoader,
    n: int,
    channels: int,
    img_size: int,
    device: str,
) -> torch.Tensor:
    """
    Draw a fixed batch of N validation volumes, shared across all models.
    Args:
        val_loader: DataLoader yielding (volume, label) batches.
        n:          Number of volumes to collect.
        channels:   Number of voxel channels.
        img_size:   Spatial size per dimension.
        device:     Device string.
    Returns:
        x_fixed: (N, C, D, H, W) tensor on device.
    """
    x_list = []
    for batch in val_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_list.append(x)
        if sum(b.shape[0] for b in x_list) >= n:
            break
    x = torch.cat(x_list, dim=0)[:n]
    if x.dim() == 2:
        x = x.view(x.shape[0], channels, img_size, img_size, img_size)
    return x.to(device)


# -- Plot 1: scale row ---------------------------------------------------------
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
    Sample ONCE then decode the same latent/weight at each scale factor.
    Scale label centered below each panel via ax.set_title, no parenthetical.
    Args:
        model:      Trained model.
        model_type: "vae", "ldm", or "weight_diffusion".
        base_res:   Native training resolution (e.g. 32).
        device:     Device string.
        channels:   Number of voxel channels.
        title:      Figure suptitle.
        save_path:  Output PNG path.
        vae_config: Required when model_type == "vae".
    Returns:
        None
    """
    n_scales = len(SCALE_FACTORS)
    fig = plt.figure(figsize=(n_scales * 2.5, 3.0))

    # Sample latent/weight ONCE
    with torch.no_grad():
        if model_type == "vae":
            latent_dim = vae_config["latent_dim"]
            latent_size = vae_config["latent_size"]
            z = torch.randn(1, latent_dim, latent_size, latent_size, device=device)
        elif model_type == "ldm":
            from src.models.two_stage_models.latent_two_stage import TwoStageLDM
            z = model._sample_latent(1) if isinstance(model, TwoStageLDM) \
                else model._sample_latent(1, collect_snapshots=False, debug=False)
            if hasattr(model, "_normalize") and model._normalize:
                z = model._denormalize_z(z)
        else:  # weight_diffusion
            theta_prime = model.sample_weight(1)
            theta = model.weight_encoder.decode_modulations(theta_prime)

    # Decode the same latent at each scale
    for col, (factor, label) in enumerate(zip(SCALE_FACTORS, SCALE_LABELS, strict=False)):
        res = max(4, round(base_res * factor))

        with torch.no_grad():
            if model_type in ("vae", "ldm"):
                coord = _make_3d_coord_grid(res, device)
                slices = []
                for d_start in range(0, res, 8):
                    d_end = min(d_start + 8, res)
                    x_chunk = model.decoder(z, coord[d_start:d_end])
                    slices.append(x_chunk.cpu())
                x_hat = torch.cat(slices, dim=2).reshape(1, channels, res, res, res)
            else:
                coord = _make_3d_coord_grid(res, device)
                slices = []
                for d_start in range(0, res, 8):
                    d_end = min(d_start + 8, res)
                    coord_chunk = coord[d_start:d_end].unsqueeze(0)
                    x_chunk = model._inr_decode(theta, coords=coord_chunk)
                    x_chunk = x_chunk.reshape(1, channels, d_end - d_start, res, res)
                    slices.append(x_chunk.cpu())
                x_hat = torch.cat(slices, dim=2)

        voxels = _samples_to_voxel_grids(x_hat, channels, res)[0]
        voxels = voxels.transpose(2, 0, 1)

        ax = fig.add_subplot(1, n_scales, col + 1, projection="3d")
        _render_mesh_on_ax(ax, voxels, azim=_AZIM_OFFSETS[col % len(_AZIM_OFFSETS)], elev=_ELEV)

        # Centered label via ax.set_title — no parenthetical resolution
        ax.set_title(label, fontsize=9, fontweight="bold", pad=2)

    fig.suptitle(f"3D Scale Row: {title}", fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Scale row saved -> {save_path}")


# -- Plot 2: reconstruction comparison -----------------------------------------
@torch.no_grad()
def plot_reconstruction_3d(
    model,
    model_type: str,
    x_fixed: torch.Tensor,
    base_res: int,
    channels: int,
    title: str,
    save_path: str,
) -> None:
    """
    Reconstruct a fixed set of validation volumes shared across all models.
    Top row: originals. Bottom row: reconstructions. Labels centered below rows.
    Args:
        model:      Trained model.
        model_type: "vae", "ldm", or "weight_diffusion".
        x_fixed:    (N, C, D, H, W) fixed validation volumes on device.
        base_res:   Native training resolution.
        channels:   Number of voxel channels.
        title:      Figure suptitle.
        save_path:  Output PNG path.
    Returns:
        None
    """
    N = x_fixed.shape[0]  # noqa: N806
    coord = _make_3d_coord_grid(base_res, str(x_fixed.device))

    # Reconstruct each volume individually to stay within VRAM
    recon_list = []
    for i in range(N):
        xi = x_fixed[i : i + 1]
        if model_type == "weight_diffusion":
            theta_prime, _, _ = model.encode(xi)
            theta = model.weight_encoder.decode_modulations(theta_prime)
            x_hat = model._inr_decode(theta, coords=coord.unsqueeze(0))
            x_hat = x_hat.reshape(1, channels, base_res, base_res, base_res)
        else:
            z, _, _ = model.encode(xi)
            x_hat = model.decoder(z, coord)
            x_hat = x_hat.reshape(1, channels, base_res, base_res, base_res)
        recon_list.append(x_hat.cpu())

    x_hat_all = torch.cat(recon_list, dim=0)  # (N, C, D, H, W)

    orig_grids  = _samples_to_voxel_grids(x_fixed.cpu(), channels, base_res)  # (N, D, H, W)
    recon_grids = _samples_to_voxel_grids(x_hat_all,     channels, base_res)  # (N, D, H, W)

    orig_grids  = orig_grids.transpose(0, 3, 1, 2)   # (N, W, H, D)
    recon_grids = recon_grids.transpose(0, 3, 1, 2)  # (N, W, H, D)

    fig = plt.figure(figsize=(N * 2.0, 5.0))

    for col in range(N):
        azim = _AZIM_OFFSETS[col % len(_AZIM_OFFSETS)]

        ax_orig = fig.add_subplot(2, N, col + 1, projection="3d")
        _render_mesh_on_ax(ax_orig, orig_grids[col], azim=azim, elev=_ELEV)

        ax_recon = fig.add_subplot(2, N, N + col + 1, projection="3d")
        _render_mesh_on_ax(ax_recon, recon_grids[col], azim=azim, elev=_ELEV)

    # Row labels centered below each row
    fig.text(0.5, 0.52, "Originals",       ha="center", va="top",    fontsize=10, fontweight="bold")
    fig.text(0.5, 0.02, "Reconstructions", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle(f"3D Reconstructions: {title}", fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Reconstruction plot saved -> {save_path}")


# -- Entry point ---------------------------------------------------------------
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

    # Draw fixed validation batch ONCE — shared across all models
    print(f"  Drawing {N_RECON} fixed validation volumes (shared across all models) ...")
    x_fixed = draw_fixed_val_batch(val_loader, N_RECON, channels, img_size, device)

    # -- VAE -------------------------------------------------------------------
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
        vae_model, "vae", x_fixed, args.base_res, channels,
        title="VAE-INR",
        save_path=os.path.join(output_dir, "vae_reconstructions.png"),
    )

    # -- Latent Diffusion models -----------------------------------------------
    if args.latent_config_paths:
        print(f"\n--- Processing Latent Diffusion Suite ({len(args.latent_config_paths)} variants) ---")
        from src.utility.model_builders.model_builder import build_model as build_ldm_model
        from src.utility.model_builders.util.twostage_builder import build_ldm as build_two_stage_ldm
        from src.models.two_stage_models.latent_two_stage import TwoStageLDM

        for idx, p in enumerate(args.latent_config_paths):
            with open(p) as f:
                l_cfg = json.load(f)

            if idx == 0:
                l_hparams = SimpleNamespace(**l_cfg["hparams"])
                l_data_cfg = l_cfg["data"]
                l_data_config = {
                    "dataset": l_cfg["dataset"],
                    "channels": l_data_cfg["channels"],
                    "img_size": l_data_cfg["img_size"],
                    "data_dim": l_data_cfg["data_dim"],
                    "is_3d": True,
                }
                run_name = _extract_run_name(p)
                print(f"  Building & loading (one-stage): {run_name} ...")
                l_model = build_ldm_model(l_hparams, l_data_config).to(device)
                l_ckpt = torch.load(l_cfg["paths"]["weights"], map_location=device)
                l_model.load_state_dict(l_ckpt["model_state_dict"])
            else:
                run_name = l_cfg["run_name"]
                ckpt_path = os.path.join(
                    os.path.dirname(os.path.abspath(p)),
                    f"{run_name}_ldm_checkpoint.pt"
                )
                ts_args = SimpleNamespace(T=l_cfg["T"], beta_1=l_cfg["beta_1"], beta_T=l_cfg["beta_T"])
                print(f"  Building & loading (two-stage): {run_name} ...")
                l_model = build_two_stage_ldm(
                    hparams=l_cfg, args=ts_args, channels=channels, img_size=img_size,
                    device=device, is_3d=True,
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
                l_model, "ldm", x_fixed, args.base_res, channels,
                title=run_name,
                save_path=os.path.join(output_dir, f"latent_reconstructions_{safe}.png"),
            )

    # -- Weight Diffusion models -----------------------------------------------
    if args.weight_config_paths:
        print(f"\n--- Processing Weight Diffusion Suite ({len(args.weight_config_paths)} variants) ---")
        from src.utility.model_builders.model_builder import build_model as build_ldm_model
        from src.scripts.two_stage_weight_training import build_full_wd_model

        for idx, p in enumerate(args.weight_config_paths):
            with open(p) as f:
                w_cfg = json.load(f)

            if idx == 0:
                w_hparams = SimpleNamespace(**w_cfg["hparams"])
                w_data_cfg = w_cfg["data"]
                w_data_config = {
                    "dataset": w_cfg["dataset"],
                    "channels": w_data_cfg["channels"],
                    "img_size": w_data_cfg["img_size"],
                    "data_dim": w_data_cfg["data_dim"],
                    "is_3d": True,
                }
                run_name = _extract_run_name(p)
                print(f"  Building & loading (one-stage): {run_name} ...")
                w_model = build_ldm_model(w_hparams, w_data_config).to(device)
                w_ckpt = torch.load(w_cfg["paths"]["weights"], map_location=device)
                state_dict = {k: v for k, v in w_ckpt["model_state_dict"].items() if k != "coords"}
                w_model.load_state_dict(state_dict, strict=False)
            else:
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
                w_model, "weight_diffusion", x_fixed, args.base_res, channels,
                title=run_name,
                save_path=os.path.join(output_dir, f"weight_reconstructions_{safe}.png"),
            )

    print("\n3D Eval Plots Complete.")


if __name__ == "__main__":
    main()