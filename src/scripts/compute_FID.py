"""
compute_fid.py
Computes MNIST-classifier FID for a single model config.

Usage
-----
python src/scripts/compute_FID.py \
    --model inr_vae \
    --config src/trained_models/vae_inr_mnist_modulation/metadata/config.json \
    --n 50000
    --classifier_type inception

--n defaults to the FID_SCORE_SAMPLES constant if not provided.
"""

from __future__ import annotations

import argparse
import sys

sys.path.append(".")

from src.configs.results_config import FID_SAMPLE_BATCH, FID_SCORE_SAMPLES
from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
    _mnist_features,
)
from src.utility.general import _get_device
from src.utility.inference import sample as model_sample
from src.utility.metrics_util import _fid


def compute_fid_score(model_name: str, config_path: str, device: str, n: int = FID_SCORE_SAMPLES, classifier_type="mnist") -> float:
    # ── Load classifier ───────────────────────────────────────────────────────
    print("  Loading MNIST classifier …")
    classifier = _load_classifier(device)

    # ── Real MNIST features (cached) ──────────────────────────────────────────
    # Pass None for inception so only the MNIST features are loaded/cached.
    print("  Loading cached real MNIST features …")
    inception = _get_inception(device)
    real_mnist_feats, real_inception_feats = _load_or_compute_real_features(classifier, inception, device)

    # ── Sample from model ─────────────────────────────────────────────────────
    print(f"  Sampling {n:,} images …")
    images = model_sample(
        model_name=model_name,
        config_path=config_path,
        n_samples=n,
        device=device,
        batch_size=FID_SAMPLE_BATCH,
    )
    # ── Extract generated features and compute FID────────────────────────────────────────────
    if classifier_type == "inception":
        print("  Extracting Inception-v3 classifier features …")
        gen_features = _inception_features(images, inception, device)
        fid_score = _fid(real_inception_feats, gen_features)
    else:
        print("  Extracting MNIST classifier features …")
        gen_mnist_feats, _ = _mnist_features(images, classifier, device)
        fid_score = _fid(real_mnist_feats, gen_mnist_feats)

    print(f"\n{'=' * 50}")
    print(f"  {classifier_type} FID : {fid_score:.4f}")
    print(f"{'=' * 50}\n")

    return fid_score


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
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="mnist",
        choices=["mnist", "inception"],
        help="Which classifier's features to use for FID (default: mnist).",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"\n{'=' * 50}")
    print(f"  {args.classifier_type.upper()} FID  |  device={device}  |  n={args.n:,}")
    print(f"{'=' * 50}\n")

    fid_score = compute_fid_score(
        model_name=args.model, config_path=args.config, device=device, n=args.n, classifier_type=args.classifier_type
    )

    return fid_score


if __name__ == "__main__":
    main()
