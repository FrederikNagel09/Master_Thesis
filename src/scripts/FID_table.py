"""
results_fid.py
Computes MNIST-classifier FID, Inception FID, and class distribution
for all three models. Caches real MNIST features to avoid recomputation.

Usage
-----
python src/scripts/FID_table.py \
    --ndm     src/train_results/ndm_unet_mnist/metadata/config.json \
    --inr_vae src/train_results/vae_inr_mnist/metadata/config.json \
    --ndm_inr src/train_results/ndm_inr_mlp_mnist/metadata/config.json \
    --out     src/results/fid_comparison.png

All three model config paths are optional.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.append(".")

import numpy as np

from src.configs.results_config import (
    FID_SAMPLE_BATCH,
    FID_SCORE_SAMPLES,
    MODEL_LABELS,
)
from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
    _mnist_features,
)
from src.utility.general import _get_device
from src.utility.inference import sample as model_sample
from src.utility.metrics_util import _fid, _uniformity_score
from src.utility.plotting import _build_figure

# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="FID and class distribution comparison.")
    parser.add_argument("--ndm", type=str, default=None)
    parser.add_argument("--inr_vae", type=str, default=None)
    parser.add_argument("--ndm_inr", type=str, default=None)
    parser.add_argument("--out", type=str, default="src/results/fid_comparison.png")
    args = parser.parse_args()

    requested = {}
    for key in ("ndm", "inr_vae", "ndm_inr"):
        path = getattr(args, key)
        if path is not None:
            requested[key] = path

    if not requested:
        print("No config paths provided. Pass at least one of --ndm, --inr_vae, --ndm_inr.")
        sys.exit(1)

    device = _get_device()
    print(f"\n{'=' * 55}")
    print(f"  FID Comparison  |  device={device}  |  n={FID_SCORE_SAMPLES:,}")
    print(f"{'=' * 55}\n")

    # ── Load classifier and inception ─────────────────────────────────────────
    print("  Loading MNIST classifier …")
    classifier = _load_classifier(device)
    print("  Loading Inception …")
    inception = _get_inception(device)

    # ── Real MNIST features (cached) ──────────────────────────────────────────
    real_mnist_feats, real_inception_feats = _load_or_compute_real_features(classifier, inception, device)

    # ── Per-model evaluation ──────────────────────────────────────────────────
    metrics = {}

    for model_key, config_path in requested.items():
        label = MODEL_LABELS[model_key]
        print(f"\n── {label} ──────────────────────────────────────────")

        # Sample
        print(f"  Sampling {FID_SCORE_SAMPLES:,} images …")
        t0 = time.time()
        images = model_sample(
            model_name=model_key,
            config_path=config_path,
            n_samples=FID_SCORE_SAMPLES,
            device=device,
            batch_size=FID_SAMPLE_BATCH,
        )  # (N, C, H, W) in [0,1]
        print(f"  Sampling done in {time.time() - t0:.1f}s")

        # MNIST classifier features + predictions
        print("  Extracting MNIST classifier features …")
        gen_mnist_feats, gen_preds = _mnist_features(images, classifier, device)

        # Inception features
        print("  Extracting Inception features …")
        gen_inception_feats = _inception_features(images, inception, device)

        # FID scores
        mnist_fid = _fid(real_mnist_feats, gen_mnist_feats)
        inception_fid = _fid(real_inception_feats, gen_inception_feats)

        # Class distribution
        dist_gen = np.bincount(gen_preds, minlength=10) / len(gen_preds)
        uniformity = _uniformity_score(dist_gen)

        print(f"  MNIST FID     : {mnist_fid:.2f}")
        print(f"  Inception FID : {inception_fid:.2f}")
        print(f"  Uniformity    : {uniformity:.2f}")

        metrics[model_key] = {
            "mnist_fid": mnist_fid,
            "inception_fid": inception_fid,
            "uniformity": uniformity,
            "dist_gen": dist_gen,
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.splitext(args.out)[0] + ".json"
    json_out = {
        key: {
            "mnist_fid": float(m["mnist_fid"]),
            "inception_fid": float(m["inception_fid"]),
            "uniformity": float(m["uniformity"]),
            "class_distribution": m["dist_gen"].tolist(),
        }
        for key, m in metrics.items()
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  Results JSON saved → {json_path}")

    # ── Build figure ──────────────────────────────────────────────────────────
    print("  Building figure …")
    _build_figure(metrics, args.out)

    print(f"\n{'=' * 55}")
    for key, m in metrics.items():
        print(
            f"  {MODEL_LABELS[key]:<10} MNIST FID={m['mnist_fid']:.2f}  "
            f"Inception FID={m['inception_fid']:.2f}  "
            f"Uniformity={m['uniformity']:.2f}"
        )
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
