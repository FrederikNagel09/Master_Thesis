"""
compute_fid.py
Computes MNIST-classifier FID for a single model config.

Usage
-----
python src/scripts/compute_FID.py \
    --model ndm \
    --config src/trained_models/ndm_attention_mnist_scale/metadata/config.json \
    --n 2000

--n defaults to the FID_SCORE_SAMPLES constant if not provided.
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.append(".")

from src.configs.results_config import FID_SAMPLE_BATCH, FID_SCORE_SAMPLES
from src.utility.classifier_utils import (
    _get_inception,
    _load_classifier,
    _load_or_compute_real_features,
    _mnist_features,
)
from src.utility.general import _get_device
from src.utility.inference import sample as model_sample
from src.utility.metrics_util import _fid


def main():
    parser = argparse.ArgumentParser(description="Compute MNIST-classifier FID for a single model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config JSON.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ndm", "inr_vae", "ndm_inr"],
        help="Model type key (ndm | inr_vae | ndm_inr).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=FID_SCORE_SAMPLES,
        help=f"Number of samples to generate (default: {FID_SCORE_SAMPLES}).",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"\n{'=' * 50}")
    print(f"  MNIST FID  |  device={device}  |  n={args.n:,}")
    print(f"{'=' * 50}\n")

    # ── Load classifier ───────────────────────────────────────────────────────
    print("  Loading MNIST classifier …")
    classifier = _load_classifier(device)

    # ── Real MNIST features (cached) ──────────────────────────────────────────
    # Pass None for inception so only the MNIST features are loaded/cached.
    print("  Loading cached real MNIST features …")
    inception = _get_inception(device)
    real_mnist_feats, _ = _load_or_compute_real_features(classifier, inception, device)

    # ── Sample from model ─────────────────────────────────────────────────────
    print(f"  Sampling {args.n:,} images …")
    t0 = time.time()
    images = model_sample(
        model_name=args.model,
        config_path=args.config,
        n_samples=args.n,
        device=device,
        batch_size=FID_SAMPLE_BATCH,
    )  # (N, C, H, W) in [0, 1]
    print(f"  Sampling done in {time.time() - t0:.1f}s")

    # ── Extract generated features ────────────────────────────────────────────
    print("  Extracting MNIST classifier features …")
    gen_mnist_feats, _ = _mnist_features(images, classifier, device)

    # ── Compute FID ───────────────────────────────────────────────────────────
    mnist_fid = _fid(real_mnist_feats, gen_mnist_feats)

    print(f"\n{'=' * 50}")
    print(f"  MNIST FID : {mnist_fid:.4f}")
    print(f"{'=' * 50}\n")

    return mnist_fid


if __name__ == "__main__":
    main()
