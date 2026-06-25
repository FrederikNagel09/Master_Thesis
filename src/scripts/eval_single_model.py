"""
eval_samples.py
Generates final samples and computes FID + ELBO for a single model.

Usage
-----
python src/scripts/eval_single_model.py \
    --config_path src/train_results/latent-diffusion-ramp/metadata/config.json \
    --n_fid_samples 4096
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

warnings.filterwarnings("ignore", message="The operator 'aten::im2col' is not currently supported on the MPS backend")


def _extract_run_name(config_path: str) -> str:
    """
    Extracts run name from .../<run_name>/metadata/config.json.

    Args:
        config_path: Path to the model config JSON.
    Returns:
        run_name string extracted from the path.
    """
    parts = os.path.normpath(config_path).split(os.sep)
    # Structure is always <run_name>/metadata/config.json
    try:
        metadata_idx = parts.index("metadata")
        return parts[metadata_idx - 1]
    except (ValueError, IndexError):
        raise ValueError(  # noqa: B904
            f"Could not extract run name from config path: {config_path}\n" "Expected format: .../<run_name>/metadata/config.json"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate samples and compute FID/ELBO for a single model.")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--n_fid_samples", type=int, default=10000, help="Number of samples to generate for FID computation.")
    args = parser.parse_args()

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
    from src.utility.model_builders import build_model
    from src.utility.plotting import plot_final_samples

    with open(args.config_path) as f:
        config = json.load(f)

    hparams = SimpleNamespace(**config["hparams"])
    data_cfg = config["data"]
    data_config = {
        "dataset": config["dataset"],
        "channels": data_cfg["channels"],
        "img_size": data_cfg["img_size"],
        "data_dim": data_cfg["data_dim"],
    }
    epoch = config["epochs"]["end"]

    device = _get_device()
    run_name = _extract_run_name(args.config_path)
    run_dir = os.path.join("src", "results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f"  Eval Samples  |  run={run_name}  |  device={device}")
    print(f"  Output dir: {run_dir}")
    print(f"{'=' * 55}\n")

    # ── Build & load model ────────────────────────────────────────────────────
    print("  Building model …")
    model = build_model(hparams, data_config).to(device)

    weights_path = config["paths"]["weights"]
    print(f"  Loading weights from {weights_path} …")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Val loader (for ELBO) ─────────────────────────────────────────────────
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
    print("### val loader", len(val_loader.dataset), "samples, batch size", hparams.batch_size, "->", len(val_loader), "batches.")
    # ── Generate samples + FID + ELBO ─────────────────────────────────────────
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
