"""
eval_single_model.py

Generates final samples and computes eval metrics for a single model.
Supports three model families:
  - Original one-stage models  (config at .../metadata/config.json)
  - Two-stage Latent LDM       (config at .../<run>_ldm_config.json)
  - Two-stage Weight Diffusion (config at .../<run>_wd_config.json)

All three support both 2D image and 3D voxel (ShapeNet) data.

Original model usage:
---------------------
## LATENT DIFFUSION - ONE-STAGE - READY
python src/scripts/eval_single_model.py \
    --config_path src/train_results/latent-probability-3D-data/metadata/config.json \
    --n_fid_samples 4096

## WEIGHT DIFFUSION - ONE-STAGE - READY
python src/scripts/eval_single_model.py \
    --config_path src/train_results/weight-probability-3D-data/metadata/config.json \
    --n_fid_samples 4096


Latent two-stage usage (2D):
-----------------------------

## LATENT TWO-STAGE - FIXED - TEST
python src/scripts/eval_single_model.py \
    --config_path src/train_results/latent_two_stage_fixed/latent_two_stage_fixed_ldm_config.json \
    --weights_path src/train_results/latent_two_stage_fixed/latent_two_stage_fixed_ldm_weights.pt \
    --n_fid_samples 16

## WEIGHT TWO-STAGE - FIXED - TEST
python src/scripts/eval_single_model.py \
    --config_path src/train_results/wd_two_stage_fixed/wd_two_stage_fixed_wd_config.json \
    --weights_path src/train_results/wd_two_stage_fixed/wd_two_stage_fixed_wd_weights.pt \
    --n_fid_samples 16

## LATENT TWO-STAGE - CONVERGED - TEST
python src/scripts/eval_single_model.py \
    --config_path src/train_results/two_stage_convergence/two_stage_convergence_ldm_config.json \
    --weights_path src/train_results/two_stage_convergence/two_stage_convergence_ldm_weights.pt \
    --n_fid_samples 16

## WEIGHT TWO-STAGE - CONVERGED - TEST
python src/scripts/eval_single_model.py \
    --config_path src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_wd_config.json \
    --weights_path src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_wd_weights.pt \
    --n_fid_samples 16





Latent two-stage usage (3D ShapeNet):
--------------------------------------
python src/scripts/eval_single_model.py \
    --config_path src/train_results/latent-probability-3D-data/metadata/config.json \
    --weights_path src/train_results/latent-probability-3D-data/weights/weights.pt \
    --n_fid_samples 12

Weight diffusion two-stage usage (2D):
---------------------------------------
python src/scripts/eval_single_model.py \
    --config_path src/train_results/wd_two_stage/wd_two_stage_wd_config.json \
    --weights_path src/train_results/wd_two_stage/wd_two_stage_wd_weights.pt \
    --n_fid_samples 4096

Weight diffusion two-stage usage (3D):
---------------------------------------
python src/scripts/eval_single_model.py \
    --config_path src/train_results/wd_two_stage_shapenet/wd_two_stage_shapenet_wd_config.json \
    --weights_path src/train_results/wd_two_stage_shapenet/wd_two_stage_shapenet_wd_weights.pt \
    --n_fid_samples 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(".")

import warnings

import torch

warnings.filterwarnings(
    "ignore",
    message="The operator 'aten::im2col' is not currently supported on the MPS backend",
)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG-TYPE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _config_type(config: dict) -> str:
    if "run_name" in config and "hparams" not in config:
        if "noise_predictor_type" in config:
            return "weight_two_stage"
        return "latent_two_stage"
    return "original"


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _extract_run_name_original(config_path: str) -> str:
    parts = os.path.normpath(config_path).split(os.sep)
    try:
        metadata_idx = parts.index("metadata")
        return parts[metadata_idx - 1]
    except (ValueError, IndexError):
        raise ValueError(  # noqa: B904
            f"Could not extract run name from config path: {config_path}\n"
            "Expected format: .../<run_name>/metadata/config.json"
        )


def _dataset_defaults(dataset: str, is_3d: bool) -> tuple[int, int]:
    if is_3d:
        return 1, 32
    mapping = {
        "mnist": (1, 28),
        "cifar10": (3, 32),
        "celeba": (3, 64),
    }
    if dataset.lower() not in mapping:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    return mapping[dataset.lower()]


# ──────────────────────────────────────────────────────────────────────────────
# TWO-STAGE LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_latent_two_stage(
    config: dict,
    weights_path: str,
    device: torch.device,
) -> tuple[object, dict]:
    from src.utility.model_builders.util.twostage_builder import build_ldm

    is_3d = config.get("is_3d", False)
    dataset = config["dataset"]
    channels, img_size = _dataset_defaults(dataset, is_3d)

    data_config = {
        "dataset": dataset,
        "channels": channels,
        "img_size": img_size,
        "data_dim": channels * img_size ** (3 if is_3d else 2),
        "is_3d": is_3d,
    }

    args = SimpleNamespace(T=config["T"], beta_1=config["beta_1"], beta_T=config["beta_T"])
    ldm = build_ldm(config, args, channels, img_size, device, is_3d=is_3d)

    ckpt = torch.load(weights_path, map_location=device)
    ldm.load_state_dict(ckpt["ldm_state_dict"])
    ldm.eval()
    print(f"  Loaded latent two-stage LDM weights from: {weights_path}")

    return ldm, data_config


def _load_weight_two_stage(
    config: dict,
    weights_path: str,
    device: torch.device,
) -> tuple[object, dict]:
    from src.scripts.two_stage_weight_training import build_full_wd_model

    is_3d = config.get("is_3d", False)
    dataset = config["dataset"]
    channels, img_size = _dataset_defaults(dataset, is_3d)
    data_dim = channels * img_size ** (3 if is_3d else 2)

    data_config = {
        "dataset": dataset,
        "channels": channels,
        "img_size": img_size,
        "data_dim": data_dim,
        "is_3d": is_3d,
    }

    args = SimpleNamespace(T=config["T"], beta_1=config["beta_1"], beta_T=config["beta_T"])
    model = build_full_wd_model(config, args, channels, img_size, data_dim, device, is_3d=is_3d)

    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["full_model_state_dict"])
    model.eval()
    print(f"  Loaded weight-diffusion two-stage weights from: {weights_path}")

    return model, data_config


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate samples and compute eval metrics.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to model config JSON")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to weights .pt file")
    parser.add_argument("--n_fid_samples", type=int, default=10000, help="Samples for FID/MMD/COV")
    parser.add_argument("--fid_batch_size", type=int, default=64, help="Generation batch size")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    from src.utility.general import _get_device
    from src.utility.dataset_builders import build_dataset
    from src.utility.plotting import plot_final_samples

    device = torch.device(_get_device())
    family = _config_type(config)
    subset_frac_temp = 1.0
    # 1. Initialize model, data_config, and params based on model family
    if family == "latent_two_stage":
        if args.weights_path is None:
            raise ValueError("--weights_path is required for latent two-stage models.")
        run_name = config["run_name"]
        model, data_config = _load_latent_two_stage(config, args.weights_path, device)
        
        model_type = "latent_two_stage"
        data_root = "data/"
        batch_size = args.fid_batch_size
        epoch = 0
        single_class, single_class_label, subset_frac = False, None, subset_frac_temp

    elif family == "weight_two_stage":
        if args.weights_path is None:
            raise ValueError("--weights_path is required for weight two-stage models.")
        run_name = config["run_name"]
        model, data_config = _load_weight_two_stage(config, args.weights_path, device)
        
        model_type = "weight_two_stage"
        data_root = "data/"
        batch_size = args.fid_batch_size
        epoch = 0
        single_class, single_class_label, subset_frac = False, None, subset_frac_temp

    else:
        from src.utility.model_builders.model_builder import build_model
        hparams = SimpleNamespace(**config["hparams"])
        data_cfg = config["data"]
        data_config = {
            "dataset": config["dataset"],
            "channels": data_cfg["channels"],
            "img_size": data_cfg["img_size"],
            "data_dim": data_cfg["data_dim"],
            "is_3d": data_cfg.get("is_3d", False),
        }
        run_name = _extract_run_name_original(args.config_path)
        
        model = build_model(hparams, data_config).to(device)
        weights_path = args.weights_path or config["paths"]["weights"]
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        # Remove the unexpected 'coords' key if it exists
        if "coords" in state_dict:
            del state_dict["coords"]
            
        model.load_state_dict(state_dict)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        model_type = hparams.model.lower()
        if "weight" in model_type:
            model_type = "weight_inr_diffusion"
        elif "latent" in model_type:
            model_type = "latent_inr_diffusion"

        data_root = getattr(hparams, "data_root", "data/")
        batch_size = getattr(hparams, "batch_size", args.fid_batch_size)
        epoch = config.get("epochs", {}).get("end", 0)
        single_class = getattr(hparams, "single_class", False)
        single_class_label = getattr(hparams, "single_class_label", None)
        subset_frac = subset_frac_temp

    # 2. Setup universal directories and val_loader
    run_dir = os.path.join("src", "results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f"  Eval ({family})  |  run={run_name}  |  device={device}")
    print(f"  Output dir: {run_dir}")
    print(f"{'=' * 55}\n")
    print("  Building validation dataset …")

    _, val_dataset, _ = build_dataset(
        dataset_name=data_config["dataset"],
        data_root=data_root,
        subset_frac=subset_frac,
        single_class=single_class,
        single_class_label=single_class_label,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=(family == "original"),
    )

    # 3. Route everything to plot_final_samples
    print("  Routing to plot_final_samples ...")
    plot_final_samples(
        model=model,
        model_type=model_type,
        epoch=epoch,
        run_dir=run_dir,
        device=device,
        data_config=data_config,
        n_fid_samples=args.n_fid_samples,
        val_loader=val_loader,
    )

if __name__ == "__main__":
    main()