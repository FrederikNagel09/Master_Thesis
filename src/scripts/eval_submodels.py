"""
eval_submodels.py
Evaluates FID, Reconstruction Loss, and ELBO across all saved weight snapshots
and plots each metric as a function of training epoch.

Usage
-----
python src/scripts/eval_submodels.py \
    --config_path src/train_results/latent-diffusion-2/metadata/config.json \
    --weights_dir  src/train_results/latent-diffusion-2/weights \
    --n_fid_samples 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(".")
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore", message="The operator 'aten::im2col' is not currently supported on the MPS backend")


# ── Weight loading ─────────────────────────────────────────────────────────────

def _load_weights(model: torch.nn.Module, weights_path: str, device: str) -> None:
    """
    Load weights into model, handling both raw state dicts and wrapped checkpoints.

    Args:
        model:        Model to load weights into (modified in-place).
        weights_path: Path to the .pt file.
        device:       Device string for map_location.
    Returns:
        None
    """
    checkpoint = torch.load(weights_path, map_location=device)
    # Final weights.pt is wrapped; snapshots are raw state dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def _epoch_from_path(path: str, fallback_epoch: int | None = None) -> int:
    """
    Extract epoch number from a weight file path.

    Handles two formats:
      - weights_epoch_{epoch}.pt  →  parse from filename
      - weights.pt                →  read "epoch" key from checkpoint dict;
                                     use fallback_epoch if key absent

    Args:
        path:           Absolute or relative path to the .pt file.
        fallback_epoch: Epoch to use if the checkpoint has no "epoch" key.
    Returns:
        Integer epoch number.
    """
    basename = os.path.basename(path)

    if basename.startswith("weights_epoch_"):
        # e.g. weights_epoch_50.pt
        stem = os.path.splitext(basename)[0]          # weights_epoch_50
        return int(stem.split("_")[-1])

    # Final checkpoint — read epoch stored inside the file
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        return int(checkpoint["epoch"])

    if fallback_epoch is not None:
        return fallback_epoch

    raise ValueError(
        f"Could not determine epoch for '{path}'. "
        "The checkpoint has no 'epoch' key and no fallback was provided."
    )


def _collect_weight_paths(weights_dir: str) -> list[tuple[int, str]]:
    """
    Scan weights_dir for all .pt files and return them sorted by epoch.

    Args:
        weights_dir: Directory containing only relevant .pt weight files.
    Returns:
        List of (epoch, abs_path) tuples sorted ascending by epoch.
    """
    paths = [
        os.path.join(weights_dir, f)
        for f in os.listdir(weights_dir)
        if f.endswith(".pt")
    ]
    if not paths:
        raise FileNotFoundError(f"No .pt files found in: {weights_dir}")

    # Separate snapshots and final so we only torch.load weights.pt once
    snapshots = [p for p in paths if os.path.basename(p) != "weights.pt"]
    finals    = [p for p in paths if os.path.basename(p) == "weights.pt"]

    results: list[tuple[int, str]] = []

    for p in snapshots:
        results.append((_epoch_from_path(p), p))

    # Infer final epoch from checkpoint; fall back to max_snapshot + 50
    if finals:
        max_snapshot_epoch = max(e for e, _ in results) if results else 0
        epoch = _epoch_from_path(finals[0], fallback_epoch=max_snapshot_epoch + 50)
        results.append((epoch, finals[0]))

    results.sort(key=lambda x: x[0])
    return results


# ── Per-checkpoint evaluation ──────────────────────────────────────────────────

def _eval_checkpoint(
    model: torch.nn.Module,
    weights_path: str,
    model_type: str,
    device: str,
    data_config: dict,
    val_loader: torch.utils.data.DataLoader,
    n_fid_samples: int,
) -> dict:
    """
    Load one checkpoint and compute FID, rec loss, and ELBO.

    Args:
        model:          Freshly built model (weights will be overwritten in-place).
        weights_path:   Path to the .pt snapshot or final checkpoint.
        model_type:     Model type string (e.g. "ndm", "inr_vae").
        device:         Device string.
        data_config:    Dict with "channels", "img_size", "data_dim", "dataset".
        val_loader:     Validation DataLoader for rec loss / ELBO.
        n_fid_samples:  Number of generated samples used for FID.
    Returns:
        Dict with keys "mnist_fid", "inception_fid", "rec_loss", "elbo".
    """
    from src.utility.classifier_utils import (
        _get_inception,
        _inception_features,
        _load_classifier,
        _load_or_compute_real_features,
        _mnist_features,
    )
    from src.utility.metrics_util import _fid
    from src.utility.plotting import _model_to_grid

    _load_weights(model, weights_path, device)
    model.eval()

    channels = data_config["channels"]
    dataset  = data_config.get("dataset", "mnist").lower()

    if dataset != "mnist":
        raise NotImplementedError(
            "Only MNIST is currently supported. "
            "Add a CIFAR-10 equivalent of _load_or_compute_real_features to extend."
        )

    # ── Generate FID samples ──────────────────────────────────────────────────
    fid_batch_size = 1024
    fid_batches = []
    for start in range(0, n_fid_samples, fid_batch_size):
        batch_n = min(fid_batch_size, n_fid_samples - start)
        batch, _ = _model_to_grid(model, model_type, batch_n, device, data_config, debug=False)
        fid_batches.append(batch)

    fid_grid = np.concatenate(fid_batches, axis=0)

    # (N, C, H, W) float tensor in [0, 1]
    if channels == 1:
        fid_tensor = torch.from_numpy(fid_grid).unsqueeze(1).float()
    else:
        fid_tensor = torch.from_numpy(fid_grid).permute(0, 3, 1, 2).float()

    # ── FID ───────────────────────────────────────────────────────────────────
    classifier  = _load_classifier(device)
    inception   = _get_inception(device)
    real_mnist_feats, real_inception_feats, _ = _load_or_compute_real_features(classifier, inception, device)

    gen_mnist_feats, _   = _mnist_features(fid_tensor, classifier, device)
    gen_inception_feats  = _inception_features(fid_tensor, inception, device)
    mnist_fid            = _fid(real_mnist_feats, gen_mnist_feats)
    inception_fid        = _fid(real_inception_feats, gen_inception_feats)

    # ── Rec loss ──────────────────────────────────────────────────────────────
    if not hasattr(model, "compute_rec_loss"):
        raise AttributeError(
            f"{type(model).__name__} has no 'compute_rec_loss' method. "
            "All models must implement this for eval_submodels."
        )
    rec_loss = model.compute_rec_loss(val_loader)

    # ── ELBO ─────────────────────────────────────────────────────────────────
    if not hasattr(model, "compute_full_elbo"):
        raise AttributeError(
            f"{type(model).__name__} has no 'compute_full_elbo' method. "
            "All models must implement this for eval_submodels."
        )
    elbo = model.compute_full_elbo(val_loader)

    return {
        "mnist_fid":     mnist_fid,
        "inception_fid": inception_fid,
        "rec_loss":      rec_loss,
        "elbo":          elbo,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_metrics(
    epochs: list[int],
    metrics: list[dict],
    run_dir: str,
    run_name: str,
) -> None:
    """
    Save one figure per metric (FID, Inception FID, Rec Loss, ELBO) over epochs.

    Args:
        epochs:   List of epoch integers, same length as metrics.
        metrics:  List of metric dicts from _eval_checkpoint, same order as epochs.
        run_dir:  Directory where figures are saved.
        run_name: Used in plot titles for identification.
    Returns:
        None
    """
    keys_labels = [
        ("mnist_fid",     "MNIST FID"),
        ("inception_fid", "Inception FID"),
        ("rec_loss",      "Reconstruction Loss"),
        ("elbo",          "ELBO"),
    ]

    for key, label in keys_labels:
        values = [m[key] for m in metrics]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over Training — {run_name}")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(run_dir, f"curve_{key}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FID, Rec Loss, and ELBO across weight snapshots."
    )
    parser.add_argument("--config_path",    type=str, required=True)
    parser.add_argument("--weights_dir",    type=str, required=True,
                        help="Folder containing all .pt weight snapshots and final weights.pt.")
    parser.add_argument("--n_fid_samples",  type=int, default=10000,
                        help="Number of samples to generate for FID computation.")
    args = parser.parse_args()

    from src.utility.dataset_builders import build_dataset
    from src.utility.general import _get_device
    from src.utility.model_builders import build_model

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config_path) as f:
        config = json.load(f)

    hparams     = SimpleNamespace(**config["hparams"])
    data_cfg    = config["data"]
    data_config = {
        "dataset":   config["dataset"],
        "channels":  data_cfg["channels"],
        "img_size":  data_cfg["img_size"],
        "data_dim":  data_cfg["data_dim"],
    }

    device   = _get_device()
    run_name = os.path.basename(os.path.normpath(args.weights_dir + "/.."))
    run_dir  = os.path.join("src", "results", run_name, "curves")
    os.makedirs(run_dir, exist_ok=True)

    # ── Discover checkpoints ──────────────────────────────────────────────────
    weight_pairs = _collect_weight_paths(args.weights_dir)
    print(f"\n{'=' * 55}")
    print(f"  eval_submodels  |  run={run_name}  |  device={device}")
    print(f"  Found {len(weight_pairs)} checkpoints:")
    for epoch, path in weight_pairs:
        print(f"    epoch {epoch:>5} — {os.path.basename(path)}")
    print(f"{'=' * 55}\n")

    # ── Validation loader ─────────────────────────────────────────────────────
    print("  Building validation dataset …")
    _, val_dataset, _ = build_dataset(
        dataset_name=data_config["dataset"],
        data_root=hparams.data_root,
        subset_frac=0.1,
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
    print(f"  Val loader: {len(val_loader.dataset)} samples → {len(val_loader)} batches.\n")

    # ── Build model once, swap weights per checkpoint ─────────────────────────
    print("  Building model …")
    model = build_model(hparams, data_config).to(device)

    epochs_list: list[int]  = []
    metrics_list: list[dict] = []

    for i, (epoch, weights_path) in enumerate(weight_pairs):
        print(f"  [{i + 1}/{len(weight_pairs)}] Evaluating epoch {epoch} ({os.path.basename(weights_path)}) …")
        metrics = _eval_checkpoint(
            model=model,
            weights_path=weights_path,
            model_type=hparams.model,
            device=device,
            data_config=data_config,
            val_loader=val_loader,
            n_fid_samples=args.n_fid_samples,
        )
        epochs_list.append(epoch)
        metrics_list.append(metrics)

        print(
            f"    MNIST FID={metrics['mnist_fid']:.2f}  "
            f"Inception FID={metrics['inception_fid']:.2f}  "
            f"Rec Loss={metrics['rec_loss']:.4f}  "
            f"ELBO={metrics['elbo']:.4f}"
        )

    # ── Save raw numbers ──────────────────────────────────────────────────────
    results = [{"epoch": e, **m} for e, m in zip(epochs_list, metrics_list)]
    json_path = os.path.join(run_dir, "submodel_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics saved → {json_path}")

    # ── Plot curves ───────────────────────────────────────────────────────────
    print("  Plotting curves …")
    _plot_metrics(epochs_list, metrics_list, run_dir, run_name)

    print(f"\n  Done. All outputs in: {run_dir}\n")


if __name__ == "__main__":
    main()