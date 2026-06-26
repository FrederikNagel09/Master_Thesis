"""
eval_single_model.py

Generates final samples and computes eval metrics for a single model.
Supports both original one-stage models and two-stage VAE+LDM models.

Original model usage:
---------------------
python src/scripts/eval_single_model.py \
    --config_path src/train_results/latent-diffusion-ramp/metadata/config.json \
    --n_fid_samples 4096

Two-stage model usage (2D):
---------------------------
python src/scripts/eval_single_model.py \
    --config_path src/train_results/two_stage_fixed/two_stage_fixed_ldm_config.json \
    --weights_path src/train_results/two_stage_fixed/two_stage_fixed_ldm_weights.pt \
    --n_fid_samples 4096

Two-stage model usage (3D ShapeNet):
-------------------------------------
python src/scripts/eval_single_model.py \
    --config_path src/train_results/two_stage_shapenet/two_stage_shapenet_ldm_config.json \
    --weights_path src/train_results/two_stage_shapenet/two_stage_shapenet_ldm_weights.pt \
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
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _extract_run_name_original(config_path: str) -> str:
    """
    Extract run name from .../<run_name>/metadata/config.json.

    Args:
        config_path (str): path to config JSON
    Returns:
        str: run name
    """
    parts = os.path.normpath(config_path).split(os.sep)
    try:
        metadata_idx = parts.index("metadata")
        return parts[metadata_idx - 1]
    except (ValueError, IndexError):
        raise ValueError(  # noqa: B904
            f"Could not extract run name from config path: {config_path}\n"
            "Expected format: .../<run_name>/metadata/config.json"
        )


def _is_two_stage_config(config: dict) -> bool:
    """
    Detect whether a config JSON belongs to a two-stage model.
    Two-stage configs have 'run_name' at the top level and no 'hparams' key.

    Args:
        config (dict): loaded config JSON
    Returns:
        bool: True if two-stage model config
    """
    return "run_name" in config and "hparams" not in config


# ──────────────────────────────────────────────────────────────────────────────
# TWO-STAGE MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────


def _load_two_stage_model(
    config: dict,
    weights_path: str,
    device: torch.device,
) -> tuple[object, dict]:
    """
    Build and load a TwoStageLDM from a two-stage config and weights file.

    Args:
        config       (dict):          loaded _ldm_config.json
        weights_path (str):           path to _ldm_weights.pt
        device       (torch.device):  target device
    Returns:
        tuple: (TwoStageLDM on device in eval mode, data_config dict)
    """
    from src.scripts.two_stage_latent_training import build_ldm

    is_3d = config.get("is_3d", False)
    dataset = config["dataset"]

    # Resolve dataset properties from config
    if is_3d:
        channels, img_size = 1, 32
    elif dataset == "mnist":
        channels, img_size = 1, 28
    elif dataset in ("cifar10", "celeba"):
        channels, img_size = 3, 32
    else:
        raise ValueError(f"Unknown dataset '{dataset}' in two-stage config.")

    data_config = {
        "dataset": dataset,
        "channels": channels,
        "img_size": img_size,
        "data_dim": channels * img_size ** (3 if is_3d else 2),
        "is_3d": is_3d,
    }

    # Reconstruct the args namespace build_ldm expects
    args = SimpleNamespace(
        T=config["T"],
        beta_1=config["beta_1"],
        beta_T=config["beta_T"],
    )

    ldm = build_ldm(config, args, channels, img_size, device, is_3d=is_3d)

    ckpt = torch.load(weights_path, map_location=device)
    ldm.load_state_dict(ckpt["ldm_state_dict"])
    ldm.eval()
    print(f"  Loaded two-stage LDM weights from: {weights_path}")

    return ldm, data_config


# ──────────────────────────────────────────────────────────────────────────────
# TWO-STAGE EVAL
# ──────────────────────────────────────────────────────────────────────────────


def _eval_two_stage(
    ldm: object,
    config: dict,  # noqa: ARG001
    data_config: dict,
    run_name: str,
    run_dir: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Run eval for a two-stage model: FID for 2D, MMD/COV + voxel slices for 3D.

    Args:
        ldm         (TwoStageLDM):       loaded model in eval mode
        config      (dict):              two-stage config dict
        data_config (dict):              dataset config
        run_name    (str):               run identifier
        run_dir     (str):               output directory
        args        (argparse.Namespace): CLI args
        device      (torch.device):      target device
    Returns:
        None
    """
    import torchvision.utils as vutils

    from src.utility.dataset_builders import build_dataset

    is_3d = data_config.get("is_3d", False)
    dataset = data_config["dataset"]
    data_config["channels"]

    # Val loader needed for MMD/COV reference set
    _, val_dataset, _ = build_dataset(
        dataset_name=dataset,
        data_root="data/",
        subset_frac=1.0,
        single_class=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.fid_batch_size, shuffle=False
    )

    metrics = {"run_name": run_name, "is_3d": is_3d}

    if is_3d:
        # ── 3D: MMD/COV + voxel slice grid ───────────────────────────────────
        from src.utility.voxel_metrics import compute_mmd_cov

        print(f"  Generating {args.n_fid_samples} 3D samples …")
        all_samples = []
        remaining = args.n_fid_samples
        with torch.no_grad():
            while remaining > 0:
                n = min(args.fid_batch_size, remaining)
                all_samples.append(ldm.p_sample_loop(n).cpu())
                remaining -= n
        generated = torch.cat(all_samples, dim=0)  # (N, 1, D, H, W)

        # Collect full reference val set
        ref_batches = [batch[0] for batch in val_loader]
        reference = torch.cat(ref_batches, dim=0)  # (M, 1, D, H, W)

        print(
            f"  Computing MMD/COV ({generated.shape[0]} gen vs {reference.shape[0]} ref) …"
        )
        mmd, cov = compute_mmd_cov(generated, reference)
        print(f"  MMD: {mmd:.4f} | COV: {cov:.4f}")
        metrics.update({"mmd": mmd, "cov": cov})

        # Save mid-slice visualisation grid (axial slice at D//2)
        D = generated.shape[2]  # noqa: N806
        slices = generated[:64, 0, D // 2, :, :]  # (64, H, W)
        slices = slices.unsqueeze(1).clamp(0, 1)  # (64, 1, H, W)
        vutils.save_image(
            slices,
            os.path.join(run_dir, f"{run_name}_voxel_slices_8x8.png"),
            nrow=8,
            padding=2,
        )
        print(f"  Voxel slice grid saved → {run_dir}")

    else:
        # ── 2D: Inception FID + sample grid ──────────────────────────────────
        from src.utility.classifier_utils import (
            _get_inception,
            _inception_features,
            _load_classifier,
            _load_or_compute_real_features,
            _mnist_features,
        )
        from src.utility.metrics_util import _fid

        is_mnist = dataset == "mnist"
        inception = _get_inception(device)

        if is_mnist:
            classifier = _load_classifier(device)
            real_mnist_feats, real_inception_feats, _ = _load_or_compute_real_features(
                classifier, inception, device
            )
        else:
            _, real_inception_feats, _ = _load_or_compute_real_features(
                None, inception, device
            )
            classifier = None

        print(f"  Generating {args.n_fid_samples} samples for FID …")
        all_samples = []
        remaining = args.n_fid_samples
        with torch.no_grad():
            while remaining > 0:
                n = min(args.fid_batch_size, remaining)
                imgs = (ldm.p_sample_loop(n) * 0.5 + 0.5).clamp(0, 1)
                all_samples.append(imgs.cpu())
                remaining -= n
        fid_tensor = torch.cat(all_samples, dim=0)

        gen_inception_feats = _inception_features(fid_tensor, inception, device)
        inception_fid = float(_fid(real_inception_feats, gen_inception_feats))
        metrics["inception_fid"] = inception_fid
        print(f"  Inception FID: {inception_fid:.2f}")

        if is_mnist:
            gen_mnist_feats, _ = _mnist_features(fid_tensor, classifier, device)
            mnist_fid = float(_fid(real_mnist_feats, gen_mnist_feats))
            metrics["mnist_fid"] = mnist_fid
            print(f"  MNIST FID: {mnist_fid:.2f}")

        # Sample grid
        vutils.save_image(
            fid_tensor[:64],
            os.path.join(run_dir, f"{run_name}_samples_8x8.png"),
            nrow=8,
            padding=2,
        )
        print(f"  Sample grid saved → {run_dir}")

    # Save metrics JSON
    metrics_path = os.path.join(run_dir, f"{run_name}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate samples and compute eval metrics for a single model."
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to model config JSON"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to weights .pt file (required for two-stage models)",
    )
    parser.add_argument(
        "--n_fid_samples",
        type=int,
        default=10000,
        help="Number of samples to generate for FID/MMD/COV",
    )
    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=64,
        help="Batch size for sample generation",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    from src.utility.general import _get_device

    device = _get_device()

    if _is_two_stage_config(config):
        # ── Two-stage model path ───────────────────────────────────────────
        if args.weights_path is None:
            raise ValueError("--weights_path is required for two-stage models.")

        run_name = config["run_name"]
        run_dir = os.path.join("src", "results", run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'=' * 55}")
        print(f"  Eval (Two-Stage)  |  run={run_name}  |  device={device}")
        print(f"  Output dir: {run_dir}")
        print(f"{'=' * 55}\n")

        ldm, data_config = _load_two_stage_model(config, args.weights_path, device)
        _eval_two_stage(ldm, config, data_config, run_name, run_dir, args, device)

    else:
        # ── Original one-stage model path ──────────────────────────────────
        from src.utility.dataset_builders import build_dataset
        from src.utility.model_builders import build_model
        from src.utility.plotting import plot_final_samples

        hparams = SimpleNamespace(**config["hparams"])
        data_cfg = config["data"]
        data_config = {
            "dataset": config["dataset"],
            "channels": data_cfg["channels"],
            "img_size": data_cfg["img_size"],
            "data_dim": data_cfg["data_dim"],
        }
        epoch = config["epochs"]["end"]
        run_name = _extract_run_name_original(args.config_path)
        run_dir = os.path.join("src", "results", run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'=' * 55}")
        print(f"  Eval Samples  |  run={run_name}  |  device={device}")
        print(f"  Output dir: {run_dir}")
        print(f"{'=' * 55}\n")

        print("  Building model …")
        model = build_model(hparams, data_config).to(device)

        weights_path = args.weights_path or config["paths"]["weights"]
        print(f"  Loading weights from {weights_path} …")
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("  Building validation dataset …")
        _, val_dataset, _ = build_dataset(
            dataset_name=data_config["dataset"],
            data_root=hparams.data_root,
            subset_frac=hparams.subset_frac,
            single_class=hparams.single_class,
            single_class_label=hparams.single_class_label,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=hparams.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=hparams.num_workers,
        )

        plot_final_samples(
            model=model,
            model_type=hparams.model,
            epoch=epoch,
            run_dir=run_dir,
            device=device,
            data_config=data_config,
            val_loader=val_loader,
            debug=False,
            n_fid_samples=args.n_fid_samples,
        )


if __name__ == "__main__":
    main()
