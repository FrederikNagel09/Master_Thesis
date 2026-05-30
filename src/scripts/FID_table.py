"""
results_fid.py
Computes MNIST-classifier FID, Inception FID, and class distribution
for all three models. Caches real MNIST features to avoid recomputation.
Usage
-----
python src/scripts/FID_table.py \
    --model_1_name latent_inr_diffusion \
    --model_1_config src/trained_models/latent_inr_diffusion_probablistic_annealing_bigger/metadata/config.json \
    --model_2_name latent_inr_diffusion \
    --model_2_config src/trained_models/latent_inr_diffusion_probablistic_noNorm_noscaling/metadata/config.json \
    --model_3_name latent_inr_diffusion \
    --model_3_config src/trained_models/latent_inr_diffusion_probablistic_noscaling/metadata/config.json \
    --out src/results/fid_comparison.png

python src/scripts/FID_table.py \
    --model_1_name transinr_vae \
    --model_1_config src/results/vae-mnist-0.1_config.json \
    --model_2_name transinr_vae \
    --model_2_config src/results/vae-mnist-0.5_config.json \
    --model_3_name transinr_vae \
    --model_3_config src/results/vae-mnist-1.0_config.json \
    --out src/results/fid_comparison.png


All three model names and config paths are required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.append(".")
import warnings

import numpy as np

from src.configs.results_config import (
    FID_SAMPLE_BATCH,
    FID_SCORE_SAMPLES,
    MODEL_COLORS,
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

warnings.filterwarnings("ignore", message="The operator 'aten::im2col' is not currently supported on the MPS backend")

VALID_MODELS = {"latent_inr_diffusion", "weight_inr_diffusion", "transinr_vae"}


def main():
    parser = argparse.ArgumentParser(description="FID and class distribution comparison.")
    for i in (1, 2, 3):
        parser.add_argument(f"--model_{i}_name", type=str, required=True, choices=VALID_MODELS)
        parser.add_argument(f"--model_{i}_config", type=str, required=True)
    parser.add_argument("--out", type=str, default="src/results/fid_comparison.png")
    args = parser.parse_args()

    requested = {
        f"model_{i}": {
            "name": getattr(args, f"model_{i}_name"),
            "config": getattr(args, f"model_{i}_config"),
        }
        for i in (1, 2, 3)
    }

    device = _get_device()
    print(f"\n{'=' * 55}")
    print(f"  FID Comparison  |  device={device}  |  n={FID_SCORE_SAMPLES:,}")
    print(f"{'=' * 55}\n")

    print("  Loading MNIST classifier …")
    classifier = _load_classifier(device)
    print("  Loading Inception …")
    inception = _get_inception(device)

    real_mnist_feats, real_inception_feats, real_dist = _load_or_compute_real_features(classifier, inception, device)

    metrics = {}
    sample_images = {}  # populated across all iterations before _build_figure

    for slot, model_info in requested.items():
        model_key = model_info["name"]
        config_path = model_info["config"]
        label = MODEL_LABELS[model_key]
        print(f"\n── {label} ──────────────────────────────────────────")

        print(f"  Sampling {FID_SCORE_SAMPLES:,} images …")
        t0 = time.time()
        images = model_sample(
            model_name=model_key,
            config_path=config_path,
            n_samples=FID_SCORE_SAMPLES,
            device=device,
            batch_size=FID_SAMPLE_BATCH,
        )
        sample_images[slot] = images[:16].cpu().numpy()
        print(f"  Sampling done in {time.time() - t0:.1f}s")

        print("  Extracting MNIST classifier features …")
        gen_mnist_feats, gen_preds = _mnist_features(images, classifier, device)

        print("  Extracting Inception features …")
        gen_inception_feats = _inception_features(images, inception, device)

        mnist_fid = _fid(real_mnist_feats, gen_mnist_feats)
        inception_fid = _fid(real_inception_feats, gen_inception_feats)
        dist_gen = np.bincount(gen_preds, minlength=10) / len(gen_preds)
        uniformity = _uniformity_score(dist_gen)

        print(f"  MNIST FID     : {mnist_fid:.2f}")
        print(f"  Inception FID : {inception_fid:.2f}")
        print(f"  Uniformity    : {uniformity:.2f}")

        metrics[slot] = {
            "mnist_fid": mnist_fid,
            "inception_fid": inception_fid,
            "uniformity": uniformity,
            "dist_gen": dist_gen,
            "label": MODEL_LABELS[model_key],
            "color": MODEL_COLORS[model_key],
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
    _build_figure(metrics, sample_images, real_dist, args.out)

    print(f"\n{'=' * 55}")
    for _, m in metrics.items():
        print(
            f"  {m['label']:<25} MNIST FID={m['mnist_fid']:.2f}  "
            f"Inception FID={m['inception_fid']:.2f}  "
            f"Uniformity={m['uniformity']:.2f}"
        )
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
