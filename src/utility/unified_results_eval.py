"""
unified_results_eval.py
Single entry point for producing results plots + metrics for ONE model type
(vae | latent | weight), given its 2D-trained and 3D-trained checkpoints.

Outputs (written to <output_root>/<run_name>_<model_type>/):
  latent_pca.png, latent_interpolation.png        (2D only, skipped for model_type=weight)
  weight_pca.png, weight_interpolation.png        (2D only, all model types)
  sample_grid_2d.png, sample_grid_3d.png          (8x8, no headline/spacing)
  sample_comparison.png                           (4x 2D @128px + 4x 3D, one row)
  reconstruction_comparison_{both,originals,recons}.png
  metrics.json                                    ({"2d": {...}, "3d": {...}})

Usage
-----
############ VAE #######################################
python src/utility/unified_results_eval.py \
    --model_type vae \
    --config_path_2d src/results/vae_baseline_1.0/vae_baseline_1.0_config.json \
    --weights_path_2d src/results/vae_baseline_1.0/vae_baseline_1.0_checkpoint.pt \
    --config_path_3d src/results/vae_3d_baseline_1.0_newLoss/vae_3d_baseline_1.0_newLoss_config.json \
    --weights_path_3d src/results/vae_3d_baseline_1.0_newLoss/vae_3d_baseline_1.0_newLoss_checkpoint.pt \
    --run_name vae_baseline_suite \
    --n_metric_samples 5000 --metric_batch_size 512 --n_pca_samples 5000
############################################################


############ Latent one stage ##########################
python src/utility/unified_results_eval.py \
    --model_type latent \
    --config_path_2d src/train_results/latent-diffusion/metadata/config.json \
    --weights_path_2d src/train_results/latent-diffusion/weights/weights.pt \
    --config_path_3d src/train_results/latent-diffusion-VOXEL-newLoss/metadata/config.json \
    --weights_path_3d src/train_results/latent-diffusion-VOXEL-newLoss/weights/weights.pt \
    --run_name latent_one_stage_suite \
    --n_metric_samples 64 --metric_batch_size 64 --n_pca_samples 64
#########################################################


############ Latent Fixed #######################################
python src/utility/unified_results_eval.py \
    --model_type latent \
    --config_path_2d src/train_results/Latent-two_stage_fixed/Latent-two_stage_fixed_ldm_config.json \
    --weights_path_2d src/train_results/Latent-two_stage_fixed/Latent-two_stage_fixed_ldm_checkpoint.pt \
    --config_path_3d src/train_results/VOXEL-Latent-Fixed-TEST/VOXEL-Latent-Fixed-TEST_ldm_config.json \
    --weights_path_3d src/train_results/VOXEL-Latent-Fixed-TEST/VOXEL-Latent-Fixed-TEST_ldm_checkpoint.pt \
    --run_name latent_fixed_suite \
    --n_metric_samples 64 --metric_batch_size 64 --n_pca_samples 64
########################################################################


############ Latent Converged ####################################
python src/utility/unified_results_eval.py \
    --model_type latent \
    --config_path_2d src/train_results/Latent-two_stage_convergence/Latent-two_stage_convergence_ldm_config.json \
    --weights_path_2d src/train_results/Latent-two_stage_convergence/Latent-two_stage_convergence_ldm_checkpoint.pt \
    --config_path_3d src/train_results/VOXEL-Latent-Converge-TEST/VOXEL-Latent-Converge-TEST_ldm_config.json \
    --weights_path_3d src/train_results/VOXEL-Latent-Converge-TEST/VOXEL-Latent-Converge-TEST_ldm_checkpoint.pt \
    --run_name latent_converged_suite \
    --n_metric_samples 5120 --metric_batch_size 64 --n_pca_samples 2024
########################################################################


############ Weight one stage ###############################
python src/utility/unified_results_eval.py \
    --model_type weight \
    --config_path_2d src/train_results/Weight-Diffusion-newMethoda40/metadata/config.json \
    --weights_path_2d src/train_results/Weight-Diffusion-newMethoda40/weights/weights.pt \
    --config_path_3d src/train_results/VOXEL-Weight-Diffusion-TEST/metadata/config.json \
    --weights_path_3d src/train_results/VOXEL-Weight-Diffusion-TEST/weights/weights.pt \
    --run_name weight_one_stage_bad_version \
    --n_metric_samples 2024 --metric_batch_size 1024 --n_pca_samples 2024
############################################################


############ Weight Fixed ####################################
python src/utility/unified_results_eval.py \
    --model_type weight \
    --config_path_2d src/train_results/weight-two-stage-fixed/weight-two-stage-fixed_wd_config.json \
    --weights_path_2d src/train_results/weight-two-stage-fixed/weight-two-stage-fixed_wd_weights.pt \
    --config_path_3d src/train_results/VOXEL-Weight-Fixed-TEST/VOXEL-Weight-Fixed-TEST_wd_config.json \
    --weights_path_3d src/train_results/VOXEL-Weight-Fixed-TEST/VOXEL-Weight-Fixed-TEST_wd_weights.pt \
    --run_name weight_fixed_suite \
    --n_metric_samples 5120 --metric_batch_size 64 --n_pca_samples 2024
########################################################################


############ Weight Converged ########################################
python src/utility/unified_results_eval.py \
    --model_type weight \
    --config_path_2d src/train_results/weight-two-stage-convergence/weight-two-stage-convergence_wd_config.json \
    --weights_path_2d src/train_results/weight-two-stage-convergence/weight-two-stage-convergence_wd_weights.pt \
    --config_path_3d src/train_results/VOXEL-Weight-Converge-TEST/VOXEL-Weight-Converge-TEST_wd_config.json\
    --weights_path_3d src/train_results/VOXEL-Weight-Converge-TEST/VOXEL-Weight-Converge-TEST_wd_weights.pt \
    --run_name weight_converged_suite \
    --n_metric_samples 5120 --metric_batch_size 64 --n_pca_samples 2024
####################################################################################
"""

from __future__ import annotations
import math
import argparse
import json
import os
import random
import sys
from types import SimpleNamespace
from tqdm import tqdm

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from src.utility.dataset_builders import build_dataset
from src.utility.general import _get_device
from src.utility.model_builders.model_builder import (
    build_model as build_diffusion_model,
)
from src.utility.model_builders.util.twostage_builder import (
    build_ldm as build_two_stage_ldm,
)
from src.scripts.two_stage_weight_training import build_full_wd_model
from src.scripts.get_all_plot_results import build_vae_model, make_coord_grid
from src.scripts.get_all_plot_results_3D import build_vae_model_3d
from src.models.two_stage_models.latent_two_stage import TwoStageLDM
from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
)
from src.utility.metrics_util import _fid
from src.utility.voxel_metrics import compute_mmd_cov
from src.utility.plotting import _render_mesh_on_ax, _samples_to_voxel_grids


# ── Config / path helpers ──────────────────────────────────────────────────────
def _load_json(path: str) -> dict:
    """Loads a JSON config file.
    Args: path - filesystem path to .json file.
    Returns: parsed dict.
    """
    with open(path) as f:
        return json.load(f)


def _detect_stage(config: dict) -> str:
    """Detects training stage from config schema (not used for model_type=='vae').
    Args: config - loaded config dict.
    Returns: 'one_stage', 'two_stage_latent', or 'two_stage_weight'.
    """
    if "hparams" in config:
        return "one_stage"
    if "noise_predictor_type" in config:
        return "two_stage_weight"
    return "two_stage_latent"


# ── Model bundle construction ──────────────────────────────────────────────────
def prepare_model(
    model_type: str, config_path: str, weights_path: str, is_3d: bool, device: str
) -> dict:
    """Loads config, dataset, and a built+weight-loaded model for one data-dim slot.
    Args: model_type - 'vae'|'latent'|'weight'. config_path - path to config json.
          weights_path - explicit checkpoint path to load. is_3d - True for 3D/voxel data.
          device - torch device string.
    Returns: dict with keys: model, stage, val_dataset, channels, img_size, data_dim,
             dataset_name, base_coord_grid (unbatched coord grid at native resolution),
             latent_dim, latent_size (vae only; None otherwise).
    """
    config = _load_json(config_path)
    dataset_name = config["dataset"]
    stage = "vae" if model_type == "vae" else _detect_stage(config)
    _, val_dataset, data_config = build_dataset(
        dataset_name=dataset_name,
        data_root="data/",
        subset_frac=1.0,
        single_class=False,
    )
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]
    base_coord_grid = make_coord_grid(
        (img_size,) * (3 if is_3d else 2), (-1, 1), device=device
    )

    latent_dim, latent_size = None, None

    if model_type == "vae":
        builder = build_vae_model_3d if is_3d else build_vae_model
        model = builder(config, channels, img_size, device)
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        latent_dim, latent_size = config["latent_dim"], config["latent_size"]

    elif model_type == "latent":
        if stage == "one_stage":
            hparams = SimpleNamespace(**config["hparams"])
            d_cfg = config["data"]
            l_data_config = {
                "dataset": dataset_name,
                "channels": d_cfg["channels"],
                "img_size": d_cfg["img_size"],
                "data_dim": d_cfg["data_dim"],
                "is_3d": is_3d,
            }
            model = build_diffusion_model(hparams, l_data_config).to(device)
            ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            ts_args = SimpleNamespace(
                T=config["T"], beta_1=config["beta_1"], beta_T=config["beta_T"]
            )
            model = build_two_stage_ldm(
                hparams=config,
                args=ts_args,
                channels=channels,
                img_size=img_size,
                device=device,
                is_3d=is_3d,
            )
            ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

    else:  # weight
        if stage == "one_stage":
            hparams = SimpleNamespace(**config["hparams"])
            d_cfg = config["data"]
            w_data_config = {
                "dataset": dataset_name,
                "channels": d_cfg["channels"],
                "img_size": d_cfg["img_size"],
                "data_dim": d_cfg["data_dim"],
                "is_3d": is_3d,
            }
            model = build_diffusion_model(hparams, w_data_config).to(device)
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = {
                k: v for k, v in ckpt["model_state_dict"].items() if k != "coords"
            }
            model.load_state_dict(state_dict, strict=False)
        else:
            tsw_args = SimpleNamespace(
                T=config["T"], beta_1=config["beta_1"], beta_T=config["beta_T"]
            )
            model = build_full_wd_model(
                hparams=config,
                args=tsw_args,
                channels=channels,
                img_size=img_size,
                data_dim=data_dim,
                device=device,
                is_3d=is_3d,
            )
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = {
                k: v for k, v in ckpt["full_model_state_dict"].items() if k != "coords"
            }
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    return {
        "model": model,
        "stage": stage,
        "val_dataset": val_dataset,
        "channels": channels,
        "img_size": img_size,
        "data_dim": data_dim,
        "dataset_name": dataset_name,
        "base_coord_grid": base_coord_grid,
        "latent_dim": latent_dim,
        "latent_size": latent_size,
    }


# ── Generic sample / encode / decode dispatch (native space per model_type) ───
@torch.no_grad()
def sample_vectors(
    bundle: dict, model_type: str, n_samples: int, batch_size: int, device: str
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    """Draws vectors from the model's native generative process, batched.
    Args: bundle - model bundle from prepare_model (must have 'is_3d' set). model_type -
          'vae'|'latent'|'weight'. n_samples - total vectors to draw. batch_size - draw
          chunk size. device - torch device string.
    Returns: (vecs, latent_shape): vecs is (n_samples, D) flat CPU tensor; latent_shape
             is the per-sample (C,*spatial) shape needed to un-flatten latent vectors
             before decoding (None for weight vectors, already flat/decodable as-is).
    """
    model = bundle["model"]
    is_3d = bundle["is_3d"]
    vecs, latent_shape = [], None
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        if model_type == "vae":
            ls = bundle["latent_size"]
            z = torch.randn(
                b, bundle["latent_dim"], *([ls] * (2 if is_3d else 2)), device=device
            )
        elif model_type == "latent":
            z = (
                model._sample_latent(b)
                if isinstance(model, TwoStageLDM)
                else model._sample_latent(b, collect_snapshots=False, debug=False)
            )
        else:
            theta_prime = model.sample_weight(b)
            z = model.weight_encoder.decode_modulations(theta_prime)

        if model_type != "weight" and latent_shape is None:
            latent_shape = tuple(z.shape[1:])
        vecs.append(z.reshape(b, -1).cpu())
        remaining -= b
    return torch.cat(vecs, dim=0), latent_shape


@torch.no_grad()
def encode_vectors(
    bundle: dict, model_type: str, x: torch.Tensor, batch_size: int, device: str
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    """Encodes real data into flat native-space vectors (latent for vae/latent, weight
    for weight-diffusion), batched.
    Args: bundle - model bundle. model_type - 'vae'|'latent'|'weight'. x - (N,C,*spatial)
          real data on CPU. batch_size - sub-batch size. device - torch device string.
    Returns: (vecs, latent_shape) as in sample_vectors.
    """
    model = bundle["model"]
    vecs, latent_shape = [], None
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        if model_type == "weight":
            theta_prime_raw, _, _ = model.encode(xb)
            v = model.weight_encoder.decode_modulations(theta_prime_raw)
        else:
            z, _, _ = model.encode(xb)
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])
            v = z.reshape(z.shape[0], -1)
        vecs.append(v.reshape(v.shape[0], -1).cpu())
    return torch.cat(vecs, dim=0), latent_shape


@torch.no_grad()
def decode_latent_3d_chunked(
    model,
    z: torch.Tensor,
    coord_grid: torch.Tensor,
    channels: int,
    depth_chunk: int = 8,
) -> torch.Tensor:
    """Decodes a latent batch to a 3D voxel grid, chunked over depth to bound VRAM.
    Args: model - vae/ldm model. z - (B,C,*latent_spatial) latent batch. coord_grid -
          (D,H,W,3) unbatched coord grid. channels - voxel channels. depth_chunk - depth
          slice size per decode call.
    Returns: (B,channels,D,H,W) decoded tensor on CPU.
    """
    res = coord_grid.shape[0]
    B = z.shape[0]
    slices = []
    for d_start in range(0, res, depth_chunk):
        d_end = min(d_start + depth_chunk, res)
        slices.append(model.decoder(z, coord_grid[d_start:d_end]).cpu())
    return torch.cat(slices, dim=2).reshape(B, channels, res, res, res)


@torch.no_grad()
def decode_weight_3d_chunked(
    model,
    theta: torch.Tensor,
    coord_grid: torch.Tensor,
    channels: int,
    depth_chunk: int = 8,
) -> torch.Tensor:
    """Decodes a weight vector batch to a 3D voxel grid, chunked over depth.
    Args: model - weight-diffusion model. theta - (B,D) flat weight vectors. coord_grid -
          (D,H,W,3) unbatched coord grid. channels - voxel channels. depth_chunk - depth
          slice size per decode call.
    Returns: (B,channels,D,H,W) decoded tensor on CPU.
    """
    res = coord_grid.shape[0]
    B = theta.shape[0]
    slices = []
    for d_start in range(0, res, depth_chunk):
        d_end = min(d_start + depth_chunk, res)
        coord_chunk = coord_grid[d_start:d_end].unsqueeze(0).expand(B, -1, -1, -1, -1)
        x_chunk = model._inr_decode(theta, coords=coord_chunk).reshape(
            B, channels, d_end - d_start, res, res
        )
        slices.append(x_chunk.cpu())
    return torch.cat(slices, dim=2)


@torch.no_grad()
def decode_vectors(
    bundle: dict,
    model_type: str,
    vecs: torch.Tensor,
    latent_shape: tuple[int, ...] | None,
    coord_grid: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Decodes flat native-space vectors back to pixel/voxel space, chunked for memory.
    Args: bundle - model bundle. model_type - 'vae'|'latent'|'weight'. vecs - (N,D) flat
          vectors on CPU. latent_shape - per-sample spatial shape to un-flatten latent
          vectors (unused for weight). coord_grid - unbatched coord grid at target
          resolution. batch_size - decode sub-batch size. device - torch device string.
    Returns: (N,C,*spatial) decoded tensor on CPU, raw (not unnormalized) range.
    """
    model = bundle["model"]
    channels = bundle["channels"]
    is_3d = bundle["is_3d"]
    out = []
    for start in range(0, vecs.shape[0], batch_size):
        vb = vecs[start : start + batch_size].to(device)
        b = vb.shape[0]
        if model_type == "weight":
            if is_3d:
                out.append(decode_weight_3d_chunked(model, vb, coord_grid, channels))
            else:
                coord_b = coord_grid.unsqueeze(0).expand(b, -1, -1, -1)
                pixels_flat = model._inr_decode(vb, coords=coord_b)
                res = coord_grid.shape[0]
                out.append(pixels_flat.reshape(b, channels, res, res).cpu())
        else:
            z = vb.reshape(b, *latent_shape)
            if is_3d:
                out.append(decode_latent_3d_chunked(model, z, coord_grid, channels))
            else:
                out.append(model.decoder(z, coord_grid).cpu())
    return torch.cat(out, dim=0)


# ── Display conversion ──────────────────────────────────────────────────────────
def _to_display_list(x: torch.Tensor, channels: int, is_3d: bool) -> list:
    """Converts a decoded batch tensor into a list of per-sample numpy arrays for display.
    Args: x - (N,C,*spatial) tensor in [0,1], on CPU. channels - number of channels.
          is_3d - whether x holds 3D voxel data.
    Returns: list of N numpy arrays: (H,W)/(H,W,C) for 2D, (D,H,W) voxel grid for 3D.
    """
    if is_3d:
        grids = _samples_to_voxel_grids(x, channels, x.shape[-1])
        return [grids[i] for i in range(grids.shape[0])]
    x = x.float()
    if channels == 1:
        return [x[i, 0].numpy() for i in range(x.shape[0])]
    return [x[i].permute(1, 2, 0).numpy() for i in range(x.shape[0])]


# ── Weight-space extraction for vae/latent model types ─────────────────────────
@torch.no_grad()
def _weight_via_forward_with_weights(
    model, z_batch: torch.Tensor, coord_grid: torch.Tensor
) -> torch.Tensor:
    """Extracts flat SIREN weight vectors from latent codes via the decoder's
    forward_with_weights path (vae/latent model types only).
    Args: model - vae/ldm/two_stage model. z_batch - (B,*latent_spatial) latent batch.
          coord_grid - unbatched coord grid.
    Returns: (B,D) flat weight vectors on CPU.
    """
    coord_b = coord_grid.unsqueeze(0).expand(
        z_batch.shape[0], *([-1] * coord_grid.dim())
    )
    _, w = model.decoder.forward_with_weights(z_batch, coord_b)
    return w.cpu()


@torch.no_grad()
def decode_weight_path(
    bundle: dict,
    model_type: str,
    path: torch.Tensor,
    coord_grid: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Decodes a weight-space interpolation path to pixel space (any model_type, 2D only).
    Args: bundle - model bundle. model_type - 'vae'|'latent'|'weight'. path - (n,D) flat
          weight vectors. coord_grid - unbatched coord grid. device - torch device string.
    Returns: (n,C,H,W) decoded tensor on CPU.
    """
    model = bundle["model"]
    n = path.shape[0]
    channels = bundle["channels"]
    wv = path.to(device)
    if model_type == "weight":
        coord_b = coord_grid.unsqueeze(0).expand(n, -1, -1, -1)
        pixels_flat = model._inr_decode(wv, coords=coord_b)
        res = coord_grid.shape[0]
        return pixels_flat.reshape(n, channels, res, res).cpu()
    # vae/latent: unflatten into per-layer INR params, query INR directly
    decoder = model.decoder
    params, offset = {}, 0
    for name, shape in decoder.inr.param_shapes.items():
        numel = shape[0] * shape[1]
        params[name] = wv[:, offset : offset + numel].reshape(n, shape[0], shape[1])
        offset += numel
    decoder.inr.set_params(params)
    coord_b = coord_grid.unsqueeze(0).expand(n, -1, -1, -1)
    pred = decoder.inr(coord_b)
    return pred.permute(0, 3, 1, 2).contiguous().cpu()


@torch.no_grad()
def decode_latent_path(
    bundle: dict,
    path: torch.Tensor,
    latent_shape: tuple[int, ...],
    coord_grid: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Decodes a latent interpolation path to pixel space (vae/latent model_type only).
    Args: bundle - model bundle. path - (n,D) flat latent path. latent_shape - per-sample
          latent spatial shape. coord_grid - unbatched coord grid. device - torch device.
    Returns: (n,C,H,W) decoded tensor on CPU.
    """
    model = bundle["model"]
    z = path.reshape(path.shape[0], *latent_shape).to(device)
    return model.decoder(z, coord_grid).cpu()


@torch.no_grad()
def get_analysis_vectors(
    bundle: dict,
    model_type: str,
    x_real: torch.Tensor,
    gen_n: int,
    gen_batch: int,
    device: str,
) -> dict:
    """Produces latent-space (if applicable) and weight-space vectors (real + generated)
    for PCA/interpolation, reusing shared sampled/encoded latents wherever possible.
    Args: bundle - model bundle. model_type - 'vae'|'latent'|'weight'. x_real -
          (n_pca,C,H,W) real batch. gen_n - number of generated vectors. gen_batch -
          generation batch size. device - torch device string.
    Returns: dict with keys 'latent' (only for vae/latent) and 'weight' (always), each a
             dict with 'real_vecs'/'gen_vecs' (numpy), 'real_raw' (unflattened CPU
             tensor, for interpolation), 'latent_shape' (or None for weight space).
    """
    model = bundle["model"]
    coord = bundle["base_coord_grid"]
    out = {}

    if model_type == "weight":
        theta_r = []
        for start in range(0, x_real.shape[0], 256):
            xb = x_real[start : start + 256].to(device)
            tp, _, _ = model.encode(xb)
            theta_r.append(model.weight_encoder.decode_modulations(tp).cpu())
        real_theta = torch.cat(theta_r, dim=0)

        theta_g, remaining = [], gen_n
        while remaining > 0:
            b = min(gen_batch, remaining)
            tp = model.sample_weight(b)
            theta_g.append(model.weight_encoder.decode_modulations(tp).cpu())
            remaining -= b
        gen_theta = torch.cat(theta_g, dim=0)

        out["weight"] = {
            "real_vecs": real_theta.numpy(),
            "gen_vecs": gen_theta.numpy(),
            "real_raw": real_theta,
            "latent_shape": None,
        }
        return out

    # vae/latent: shared latent encode/sample, reused for both latent-space and weight-space
    z_r = []
    for start in range(0, x_real.shape[0], 256):
        xb = x_real[start : start + 256].to(device)
        z, _, _ = model.encode(xb)
        z_r.append(z.cpu())
    real_z = torch.cat(z_r, dim=0)

    z_g, remaining = [], gen_n
    while remaining > 0:
        b = min(gen_batch, remaining)
        z = (
            model._sample_latent(b)
            if isinstance(model, TwoStageLDM)
            else model._sample_latent(b, collect_snapshots=False, debug=False)
        )
        z_g.append(z.cpu())
        remaining -= b
    gen_z = torch.cat(z_g, dim=0)

    latent_shape = tuple(real_z.shape[1:])
    out["latent"] = {
        "real_vecs": real_z.reshape(real_z.shape[0], -1).numpy(),
        "gen_vecs": gen_z.reshape(gen_z.shape[0], -1).numpy(),
        "real_raw": real_z,
        "latent_shape": latent_shape,
    }

    real_w = torch.cat(
        [
            _weight_via_forward_with_weights(
                model, real_z[s : s + 256].to(device), coord
            )
            for s in range(0, real_z.shape[0], 256)
        ],
        dim=0,
    )
    gen_w = torch.cat(
        [
            _weight_via_forward_with_weights(
                model, gen_z[s : s + 256].to(device), coord
            )
            for s in range(0, gen_z.shape[0], 256)
        ],
        dim=0,
    )
    out["weight"] = {
        "real_vecs": real_w.numpy(),
        "gen_vecs": gen_w.numpy(),
        "real_raw": real_w,
        "latent_shape": None,
    }
    return out


def pick_interp_pair(labels: np.ndarray) -> tuple[int, int]:
    """Picks two indices with different class labels for interpolation endpoints.
    Args: labels - (N,) int class labels.
    Returns: (idx1, idx2) tuple of distinct-class indices.
    """
    idx1 = random.randrange(len(labels))
    candidates = np.where(labels != labels[idx1])[0]
    idx2 = int(random.choice(candidates))
    return idx1, idx2


# ── Plotting: PCA + interpolation (2D only) ─────────────────────────────────────


def _get_pca_titles(space_name: str, model_type: str) -> tuple[str, str]:
    if model_type == "vae" or model_type == "latent":
        if space_name.lower() == "latent":
            # Use r"..." (raw string) so python doesn't misinterpret backslashes
            left_title = r"Aggregate Posterior: $q_\varphi(\mathbf{z} \mid \mathbf{s})$"
            right_title = r"Prior: $p(\mathbf{z})$"

        if space_name.lower() == "weight":
            # Corrected to reflect that these are pushed-forward manifolds of the latent space
            left_title = r"$p_\phi(\theta \mid \mathbf{z_g})$"
            right_title = r"$p_\phi(\theta \mid \mathbf{z_s})$"

    elif model_type == "weight":
        left_title = r"Aggregate Posterior: $q_\varphi(\theta \mid \mathbf{s})$"
        right_title = r"Prior: $p(\theta)$"

    return left_title, right_title


def plot_vector_space_pca(
    real_vecs: np.ndarray,
    real_labels: np.ndarray,
    gen_vecs: np.ndarray,
    interp_path: np.ndarray,
    space_name: str,
    title: str,
    save_path: str,
    model_type: str,
) -> None:
    """Fits PCA(2) on real vectors, projects generated vectors + interp path into it,
    renders real (KDE + class labels + linear interp) vs generated scatter panels with tight layout.
    """
    pca = PCA(n_components=2)
    real_2d = pca.fit_transform(real_vecs)
    gen_2d = pca.transform(gen_vecs)
    interp_2d = pca.transform(interp_path)

    all_x = np.concatenate([real_2d[:, 0], gen_2d[:, 0], interp_2d[:, 0]])
    all_y = np.concatenate([real_2d[:, 1], gen_2d[:, 1], interp_2d[:, 1]])
    xlim = (all_x.min() - 0.05 * np.ptp(all_x), all_x.max() + 0.05 * np.ptp(all_x))
    ylim = (all_y.min() - 0.05 * np.ptp(all_y), all_y.max() + 0.05 * np.ptp(all_y))

    xx, yy = np.mgrid[xlim[0] : xlim[1] : 150j, ylim[0] : ylim[1] : 150j]
    density = gaussian_kde(real_2d.T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(
        xx.shape
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    n_classes = int(real_labels.max()) + 1

    # Define discrete boundaries for the colorbar mapping to align midpoints perfectly
    boundaries = np.arange(n_classes + 1) - 0.5

    axes[0].contourf(xx, yy, density, levels=8, cmap="summer")
    scatter = axes[0].scatter(
        real_2d[:, 0],
        real_2d[:, 1],
        c=real_labels,
        cmap="tab10",
        vmin=0,
        vmax=n_classes - 1,
        s=8,
        alpha=0.85,
        linewidths=0,
    )

    axes[1].contourf(xx, yy, density, levels=8, cmap="summer")
    axes[1].scatter(
        gen_2d[:, 0], gen_2d[:, 1], color="black", s=8, alpha=0.6, linewidths=0
    )

    left_title, right_title = _get_pca_titles(space_name, model_type)
    axes[0].set_title(left_title, fontweight="bold")
    axes[1].set_title(right_title, fontweight="bold")

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontweight="bold"
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight="bold"
        )

        # 1. Bold the tick labels on both axes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    # 2. Move the plots closer together
    fig.subplots_adjust(wspace=0.0)

    # 3. Create discrete colorbar with perfectly centered integer text labels
    cbar = fig.colorbar(
        scatter,
        ax=axes,
        location="bottom",
        shrink=0.5,
        pad=0.12,
        boundaries=boundaries,
        values=range(n_classes),
    )
    cbar.set_ticks(range(n_classes))

    # Bold the colorbar labels as well to match the axes style
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight("bold")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA saved -> {save_path}")


def plot_vector_interpolation_row(
    images: np.ndarray, channels: int, space_name: str, title: str, save_path: str
) -> None:
    """Plots a single row of decoded images along a linear interpolation path.
    Args: images - (n_steps,H,W) or (n_steps,H,W,C) images in [0,1]. channels - n channels.
          space_name - 'Latent'|'Weight'. title - run name. save_path - output PNG.
    Returns: None.
    """
    n = images.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 1.8), gridspec_kw={"wspace": 0.0})
    for i, ax in enumerate(axes):
        ax.imshow(
            images[i],
            cmap="gray" if channels == 1 else None,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        ax.axis("off")
    # fig.suptitle(f"{space_name} Interpolation: {title}", fontweight="bold", y=1.05)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Interpolation row saved -> {save_path}")


def run_space_analysis(
    bundle: dict,
    model_type: str,
    x_real: torch.Tensor,
    y_real: np.ndarray,
    args: argparse.Namespace,
    run_name: str,
    output_dir: str,
    device: str,
) -> None:
    """Runs latent-space (if applicable) and weight-space PCA + linear interpolation
    analysis for the 2D slot, reusing the samples already drawn for metrics.
    Args: bundle - model bundle. model_type - 'vae'|'latent'|'weight'. x_real -
          (n_pca,C,H,W) real batch (subset of the metrics batch). y_real - (n_pca,)
          class labels. args - parsed CLI args. run_name - model run name. output_dir -
          output directory. device - torch device string.
    Returns: None.
    """
    vecs = get_analysis_vectors(
        bundle, model_type, x_real, args.n_pca_samples, args.metric_batch_size, device
    )

    for space, d in vecs.items():
        idx1, idx2 = pick_interp_pair(y_real)
        alphas = torch.linspace(0, 1, args.n_interp_steps).view(
            -1, *([1] * (d["real_raw"].dim() - 1))
        )
        w1, w2 = d["real_raw"][idx1 : idx1 + 1], d["real_raw"][idx2 : idx2 + 1]
        path = (1 - alphas) * w1 + alphas * w2
        path_flat = path.reshape(path.shape[0], -1)

        plot_vector_space_pca(
            d["real_vecs"],
            y_real,
            d["gen_vecs"],
            path_flat.numpy(),
            space.capitalize(),
            run_name,
            os.path.join(output_dir, f"{space}_pca.png"),
            model_type,
        )

        if space == "latent":
            decoded = decode_latent_path(
                bundle, path_flat, d["latent_shape"], bundle["base_coord_grid"], device
            )
        else:
            decoded = decode_weight_path(
                bundle, model_type, path_flat, bundle["base_coord_grid"], device
            )
        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
        images = np.stack(_to_display_list(decoded, bundle["channels"], False))
        plot_vector_interpolation_row(
            images,
            bundle["channels"],
            space.capitalize(),
            run_name,
            os.path.join(output_dir, f"{space}_interpolation.png"),
        )


# ── Plotting: sample grid / sample comparison / reconstruction comparison ──────


def plot_sample_grid(images: list, is_3d: bool, channels: int, save_path: str) -> None:
    """Renders a compact 8x8 grid of samples by stitching 2D images into a single matrix
    to completely eliminate sub-pixel rendering gaps.
    """
    fig = plt.figure(figsize=(12, 12))

    if is_3d:
        for idx, img in enumerate(images):
            ax = fig.add_subplot(8, 8, idx + 1, projection="3d")
            _render_mesh_on_ax(ax, img.transpose(2, 0, 1), azim=120, elev=25)
            ax.axis("off")
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
    else:
        # 1. Reshape the list of 64 images into an 8x8 block grid structure
        # images are assumed to be a list of 64 arrays
        grid_2d = [images[i * 8 : (i + 1) * 8] for i in range(8)]

        # 2. Physically stitch them into one massive 2D array
        stitched_image = np.block(grid_2d)

        # 3. Plot it as a single subplot filling the entire figure
        ax = fig.add_subplot(111)
        ax.imshow(
            stitched_image,
            cmap="gray" if channels == 1 else None,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        ax.axis("off")

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.savefig(
        save_path, dpi=150, bbox_inches="tight", pad_inches=0.0, facecolor="white"
    )
    plt.close(fig)
    print(f"  Sample grid saved -> {save_path}")


def plot_sample_comparison_row(
    imgs_2d: list, imgs_3d: list, channels_2d: int, save_path: str
) -> None:
    """Renders one row: 4 upscaled 2D samples followed by 4 3D mesh samples.
    Args: imgs_2d - list of 4 (128,128)/(128,128,C) arrays in [0,1]. imgs_3d - list of
          4 (D,H,W) voxel grids. channels_2d - channel count for 2D imshow. save_path -
          output PNG path.
    Returns: None.
    """
    fig = plt.figure(figsize=(12, 1.8))
    for i, img in enumerate(imgs_2d):
        ax = fig.add_subplot(1, 8, i + 1)
        ax.imshow(
            img,
            cmap="gray" if channels_2d == 1 else None,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        ax.axis("off")
    for i, vox in enumerate(imgs_3d):
        ax = fig.add_subplot(1, 8, 5 + i, projection="3d")
        _render_mesh_on_ax(ax, vox.transpose(2, 0, 1), azim=120, elev=25)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.0)
    fig.savefig(
        save_path, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white"
    )
    plt.close(fig)
    print(f"  Sample comparison saved -> {save_path}")


def plot_recon_comparison(
    originals_2d: list,
    originals_3d: list,
    recons_2d: list,
    recons_3d: list,
    channels_2d: int,
    mode: str,
    save_path: str,
) -> None:
    """Renders reconstruction comparison plots for 4 2D + 4 3D samples with row labels.
    Args: originals_2d/3d - 4 real originals each. recons_2d/3d - 4 reconstructions each.
          channels_2d - channel count for 2D imshow. mode - 'both'|'originals'|'recons'.
          save_path - output PNG path.
    Returns: None.
    """
    # 1. Determine rows and their corresponding text labels
    rows = []
    labels = []
    if mode in ("both", "originals"):
        rows.append((originals_2d, originals_3d))
        labels.append("Originals")
    if mode in ("both", "recons"):
        rows.append((recons_2d, recons_3d))
        labels.append("Reconstructions")

    num_rows = len(rows)
    fig = plt.figure(
        figsize=(12.5, 1.9 * num_rows)
    )  # Slightly widened to accommodate the text safely

    for r, (row_2d, row_3d) in enumerate(rows):
        for i, img in enumerate(row_2d):
            ax = fig.add_subplot(num_rows, 8, r * 8 + i + 1)
            ax.imshow(
                img,
                cmap="gray" if channels_2d == 1 else None,
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="auto",
            )
            ax.axis("off")
        for i, vox in enumerate(row_3d):
            ax = fig.add_subplot(num_rows, 8, r * 8 + 5 + i, projection="3d")
            _render_mesh_on_ax(ax, vox.transpose(2, 0, 1), azim=120, elev=25)
            ax.axis("off")

    # 2. Adjust subplots to leave a clear margin on the left for the text labels
    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.0, hspace=0.1)

    # 3. Add the vertical row headlines safely using figure coordinates
    if num_rows == 1:
        # If only one row, we can perfectly center it vertically using supylabel
        fig.supylabel(labels[0], fontsize=12, fontweight="bold", x=0.04)
    else:
        # If two rows, we calculate the exact vertical midpoint of each row
        # Row 0 (top) is centered at ~0.73, Row 1 (bottom) is centered at ~0.27
        y_positions = [0.7, 0.30]
        for label, y_pos in zip(labels, y_positions):
            fig.text(
                0.04,
                y_pos,
                label,
                rotation="vertical",
                va="center",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )

    fig.savefig(
        save_path, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white"
    )
    plt.close(fig)
    print(f"  Reconstruction comparison ({mode}) saved -> {save_path}")


# ── Metrics ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_fid_metric(
    gen_images: torch.Tensor, dataset_name: str, device: str
) -> float:
    """Computes FID between generated and real image features.
    Args: gen_images - (N,C,H,W) generated images in [0,1] on CPU. dataset_name -
          dataset name string (used to select MNIST classifier features vs plain
          inception, matching the existing eval convention). device - torch device string.
    Returns: FID score as float.
    """
    inception = _get_inception(device)
    if dataset_name.lower() == "mnist":
        classifier = _load_classifier(device)
        _, real_feats, _ = _load_or_compute_real_features(classifier, inception, device)
    else:
        _, real_feats, _ = _load_or_compute_real_features(None, inception, device)
    gen_feats = _inception_features(gen_images, inception, device)
    return float(_fid(real_feats, gen_feats))


def compute_recon_mse(recons: torch.Tensor, originals: torch.Tensor) -> float:
    """Computes mean-squared reconstruction error.
    Args: recons - (N,C,*spatial) reconstructed data. originals - (N,C,*spatial) targets.
    Returns: scalar MSE.
    """
    return float(F.mse_loss(recons, originals, reduction="mean"))


def compute_mmd_cov_metric(
    gen_voxels: torch.Tensor, real_voxels: torch.Tensor
) -> tuple[float, float]:
    """Computes MMD and coverage between generated and real voxel sets. Note: the real
    reference set here is the fixed --n_metric_samples batch, not the full val set.
    Args: gen_voxels - (N,C,D,H,W) generated volumes. real_voxels - (M,C,D,H,W) real
          reference volumes.
    Returns: (mmd, cov) tuple of floats.
    """
    mmd, cov = compute_mmd_cov(gen_voxels, real_voxels)
    return float(mmd), float(cov)


# ── Per-slot orchestration ──────────────────────────────────────────────────────
def process_slot(
    model_type: str,
    bundle: dict,
    is_3d: bool,
    args: argparse.Namespace,
    output_dir: str,
    run_name: str,
    device: str,
) -> dict:
    """Runs the full analysis pipeline for one (model_type, dimensionality) slot:
    draws fixed real + generated batches (reused across metrics/plots), computes
    metrics, sample grid, and (2D only) PCA/interpolation analysis.
    Args: model_type - 'vae'|'latent'|'weight'. bundle - prepare_model() output.
          is_3d - whether this slot is 3D data. args - parsed CLI args. output_dir -
          directory to save plots into. run_name - model run name for titles.
          device - torch device string.
    Returns: dict with 'metrics', 'gen_first4', 'real_first4', 'recon_first4' (lists of
             4 display-ready arrays), used to build cross-dataset comparison plots.
    """
    bundle["is_3d"] = is_3d
    dim_tag = "3d" if is_3d else "2d"
    ndim = 3 if is_3d else 2

    val_loader = torch.utils.data.DataLoader(
        bundle["val_dataset"], batch_size=256, shuffle=True, drop_last=False
    )
    x_list, y_list, n_collected = [], [], 0
    for x, y in val_loader:
        if x.dim() == 2:
            side = round((x.shape[1] // bundle["channels"]) ** (1.0 / ndim))
            x = x.view(x.shape[0], bundle["channels"], *([side] * ndim))
        x_list.append(x)
        y_list.append(y)
        n_collected += x.shape[0]
        if n_collected >= args.n_metric_samples:
            break
    x_real = torch.cat(x_list, dim=0)[: args.n_metric_samples]
    y_real = torch.cat(y_list, dim=0)[: args.n_metric_samples].numpy()

    print(f"  Sampling {args.n_metric_samples} generated vectors ...")
    gen_vecs, latent_shape = sample_vectors(
        bundle, model_type, args.n_metric_samples, args.metric_batch_size, device
    )
    gen_decoded = decode_vectors(
        bundle,
        model_type,
        gen_vecs,
        latent_shape,
        bundle["base_coord_grid"],
        args.metric_batch_size,
        device,
    )
    if not is_3d:
        gen_decoded = (gen_decoded * 0.5 + 0.5).clamp(0, 1)

    print(f"  Encoding {args.n_metric_samples} real samples for reconstruction ...")
    real_vecs, real_shape = encode_vectors(
        bundle, model_type, x_real, args.metric_batch_size, device
    )
    recon_decoded = decode_vectors(
        bundle,
        model_type,
        real_vecs,
        real_shape,
        bundle["base_coord_grid"],
        args.metric_batch_size,
        device,
    )
    if not is_3d:
        recon_decoded = (recon_decoded * 0.5 + 0.5).clamp(0, 1)
        x_real_unnorm = (x_real.float() * 0.5 + 0.5).clamp(0, 1)
    else:
        x_real_unnorm = x_real.float()

    print("  Computing metrics ...")
    metrics = {}
    if is_3d:
        mmd, cov = compute_mmd_cov_metric(gen_decoded, x_real_unnorm)
        metrics["mmd"], metrics["cov"] = mmd, cov
        metrics["voxel_acc"] = compute_reconstruction_loss(
            bundle,
            model_type,
            bundle["val_dataset"],
            args.n_recon_samples,
            args.metric_batch_size,
            device,
        )
    else:
        metrics["fid"] = compute_fid_metric(gen_decoded, bundle["dataset_name"], device)
        metrics["psnr"] = compute_reconstruction_loss(
            bundle,
            model_type,
            bundle["val_dataset"],
            args.n_recon_samples,
            args.metric_batch_size,
            device,
        )
    #metrics["elbo"] = compute_elbo(
        #bundle, model_type, bundle["val_dataset"], args.metric_batch_size, device
    #)

    grid_imgs = _to_display_list(gen_decoded[:64], bundle["channels"], is_3d)
    plot_sample_grid(
        grid_imgs,
        is_3d,
        bundle["channels"],
        os.path.join(output_dir, f"sample_grid_{dim_tag}.png"),
    )

    if not is_3d:
        n_pca = min(args.n_pca_samples, args.n_metric_samples)
        print(f"  Running latent/weight-space analysis ({n_pca} PCA points) ...")
        run_space_analysis(
            bundle,
            model_type,
            x_real[:n_pca],
            y_real[:n_pca],
            args,
            run_name,
            output_dir,
            device,
        )

        comp_coord = make_coord_grid((128, 128), (-1, 1), device=device)
        gen_hi = decode_vectors(
            bundle, model_type, gen_vecs[:4], latent_shape, comp_coord, 4, device
        )
        gen_hi = (gen_hi * 0.5 + 0.5).clamp(0, 1)
        gen_first4 = _to_display_list(gen_hi, bundle["channels"], False)
    else:
        gen_first4 = _to_display_list(gen_decoded[:4], bundle["channels"], True)

    real_first4 = _to_display_list(x_real_unnorm[:4], bundle["channels"], is_3d)
    recon_first4 = _to_display_list(recon_decoded[:4], bundle["channels"], is_3d)

    return {
        "metrics": metrics,
        "gen_first4": gen_first4,
        "real_first4": real_first4,
        "recon_first4": recon_first4,
    }


# ── ELBO and Reconstruction metric functions ──────────────────────────────────


def _get_mu_logvar(
    bundle: dict, model_type: str, is_3d: bool, x: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatches to the correct encode() output slots for mu/logvar across model types.
    2D VAE returns (mu, logvar, None); all other model/dim combos return (sample, mu, logvar).
    Args: bundle - model bundle. model_type - 'vae'|'latent'|'weight'. is_3d - whether 3D slot.
          x - (B,C,*spatial) input batch, already on device. device - torch device string.
    Returns: (mu, logvar) matching shape, on device.
    """
    model = bundle["model"]

    out = model.encode(x)
    if model_type == "vae" and not is_3d:
        return out[0], out[1]
    return out[1], out[2]


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Draws one reparameterized sample z = mu + std * eps, eps ~ N(0,I).
    Args: mu - mean tensor, any shape. logvar - log-variance tensor, same shape as mu.
    Returns: sampled tensor, same shape as mu.
    """
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def compute_recon_term(
    x: torch.Tensor, x_hat: torch.Tensor, is_3d: bool
) -> torch.Tensor:
    """Per-sample reconstruction loss matching the training convention: BCE for binary 3D
    occupancy, scaled MSE for continuous 2D data in [-1,1]. No renormalization — x and
    x_hat are used in their raw decoder/dataset range, matching _l_rec exactly.
    Args: x - (B,*) target, raw scale. x_hat - (B,*) decoded output, same raw scale.
          is_3d - whether data is binary voxel occupancy.
    Returns: (B,) per-sample summed reconstruction loss.
    """
    b = x.shape[0]
    x_flat = x.reshape(b, -1)
    x_hat_flat = x_hat.reshape(b, -1)
    if is_3d:
        eps = 1e-7
        x_hat_clamped = x_hat_flat.clamp(eps, 1 - eps)
        return F.binary_cross_entropy(x_hat_clamped, x_flat, reduction="none").sum(
            dim=-1
        )
    x_flat = x_flat.clamp(-1, 1)
    return 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1)


def compute_eval_recon_term(
    x: torch.Tensor, x_hat: torch.Tensor, is_3d: bool
) -> torch.Tensor:
    """Per-sample evaluation reconstruction metric.
    PSNR (dB) for continuous 2D data in [-1,1], voxel accuracy (%) for binary 3D occupancy.
    Args: x     - (B, *) target, raw scale.
          x_hat - (B, *) decoded output, same raw scale.
          is_3d - whether data is binary voxel occupancy.
    Returns: (B,) per-sample metric. Higher is better for both.
    """
    b = x.shape[0]
    x_flat = x.reshape(b, -1)
    x_hat_flat = x_hat.reshape(b, -1)
    n_elements = x_flat.shape[1]

    if is_3d:
        # voxel accuracy: fraction of correctly predicted binary voxels
        predicted = (x_hat_flat >= 0.5).float()
        return (predicted == x_flat).float().sum(dim=-1) / n_elements * 100

    # PSNR: data range is 2.0 for [-1,1], clamp target to valid range
    x_flat = x_flat.clamp(-1, 1)
    mse = ((x_flat - x_hat_flat) ** 2).mean(dim=-1)
    # clamp mse to avoid log(0) for perfect reconstructions
    return 10 * torch.log10(4.0 / mse.clamp(min=1e-10))


@torch.no_grad()
def compute_reconstruction_loss(
    bundle: dict,
    model_type: str,
    val_dataset,
    n_samples: int,
    batch_size: int,
    device: str,
) -> float:
    """Average validation reconstruction loss. Per image: encode once for (mu, logvar),
    draw n_samples reparameterized latents/weights, decode each, apply compute_recon_term,
    average the n_samples per-image losses, then average over the full validation set.
    Args: bundle - model bundle (must have 'is_3d' set). model_type - 'vae'|'latent'|'weight'.
          val_dataset - validation dataset. n_samples - reparam draws per image (use 10).
          batch_size - encode/decode sub-batch size. device - torch device string.
    Returns: scalar reconstruction loss (sum-over-elements per sample, mean over dataset).
    """
    is_3d = bundle["is_3d"]
    coord_grid = bundle["base_coord_grid"]
    channels = bundle["channels"]
    ndim = 3 if is_3d else 2

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    total_loss, n_total = 0.0, 0

    for x, _ in tqdm(loader, desc="Reconstruction loss"):
        if x.dim() == 2:
            side = round((x.shape[1] // channels) ** (1.0 / ndim))
            x = x.view(x.shape[0], channels, *([side] * ndim))
        b = x.shape[0]
        x_dev = x.to(device)

        mu, logvar = _get_mu_logvar(bundle, model_type, is_3d, x_dev, device)
        mu_flat, logvar_flat = mu.reshape(b, -1), logvar.reshape(b, -1)
        latent_shape = None if model_type == "weight" else tuple(mu.shape[1:])

        sample_losses = torch.zeros(b, device=device)
        for _ in range(n_samples):
            z = _reparameterize(mu_flat, logvar_flat)
            if model_type == "weight":
                z = bundle["model"].weight_encoder.decode_modulations(
                    z
                )  # code -> full flat weight vector
            x_hat = decode_vectors(
                bundle, model_type, z.cpu(), latent_shape, coord_grid, b, device
            ).to(device)
            sample_losses += compute_eval_recon_term(x_dev, x_hat, is_3d)

        sample_losses /= n_samples
        total_loss += sample_losses.sum().item()
        n_total += b

    return total_loss / n_total


def _entropy_term(logvar: torch.Tensor) -> torch.Tensor:
    """Total differential entropy of diagonal Gaussian q(z|x)=N(mu,exp(logvar)), summed
    over latent dims (valid since dims are independent under the diagonal assumption).
    Args: logvar - (B,*latent_dims) log-variance.
    Returns: (B,) per-sample entropy in nats.
    """
    entropy_per_dim = 0.5 * (1.0 + math.log(2.0 * math.pi) + logvar)
    return entropy_per_dim.reshape(logvar.shape[0], -1).sum(dim=-1)


def _kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu,exp(logvar)) || N(0,I)), summed over latent dims.
    Args: mu - (B,*latent_dims) mean. logvar - (B,*latent_dims) log-variance.
    Returns: (B,) per-sample KL divergence in nats.
    """
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl_per_dim.reshape(mu.shape[0], -1).sum(dim=-1)


@torch.no_grad()
def compute_elbo(
    bundle: dict, model_type: str, val_dataset, batch_size: int, device: str
) -> float:
    """Average validation ELBO (positive, higher-is-better). VAE: -L_rec - KL(q||N(0,I)).
    Diffusion (latent/weight): -(L_rec + sum_{t=0}^{T-1} L_diff(t) - H[q(z|x)]), with
    L_diff(t) reweighted to the true VLB term per timestep (one eps draw per t).
    Args: bundle - model bundle (must have 'is_3d' set). model_type - 'vae'|'latent'|'weight'.
          val_dataset - validation dataset. batch_size - sub-batch size. device - torch device.
    Returns: scalar average ELBO in nats.
    """
    model = bundle["model"]
    is_3d = bundle["is_3d"]
    channels = bundle["channels"]
    ndim = 3 if is_3d else 2

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    total_elbo, n_total = 0.0, 0
    n_elements = None
    for x, _ in tqdm(loader, desc="ELBO"):
        if x.dim() == 2:
            side = round((x.shape[1] // channels) ** (1.0 / ndim))
            x = x.view(x.shape[0], channels, *([side] * ndim))
        b = x.shape[0]
        x_dev = x.to(device)
        if n_elements is None:
            n_elements = x_dev[0].numel()
        mu, logvar = _get_mu_logvar(bundle, model_type, is_3d, x_dev, device)
        x0 = _reparameterize(
            mu, logvar
        )  # single sample, reused for L_rec and diffusion x0

        if model_type == "vae":
            x_hat = (
                model.decoder(x0, bundle["base_coord_grid"])
                if not is_3d
                else decode_latent_3d_chunked(
                    model, x0, bundle["base_coord_grid"], bundle["channels"]
                )
            )
            l_rec = compute_recon_term(x_dev, x_hat.to(device), is_3d)
            kl = _kl_to_standard_normal(mu, logvar)
            elbo = l_rec + kl
            if n_elements is None:
                n_elements = x_dev[0].numel()
            elbo = elbo  # / n_elements  # nats per element
        else:
            if model_type == "weight":
                theta = model.weight_encoder.decode_modulations(
                    x0
                )  # code -> full flat weight vector
                x_hat = model._inr_decode(theta)
                l_rec = compute_recon_term(x_dev, x_hat, is_3d)

            else:
                x_hat = model.decoder(x0, bundle["base_coord_grid"])
                l_rec = compute_recon_term(x_dev, x_hat.to(device), is_3d)

            T = model.beta.shape[0]
            l_diff_sum = torch.zeros(b, device=device)
            for t in tqdm(range(T), desc="  diffusion sum", leave=False):
                t_idx = torch.full((b,), t, device=device, dtype=torch.long)
                t_norm = t_idx.float().unsqueeze(-1) / (T - 1)
                eps = torch.randn_like(x0)
                sqrt_ab = (
                    model.alpha_cumprod[t_idx].sqrt().view(b, *([1] * (x0.dim() - 1)))
                )
                sqrt_1mab = (
                    (1 - model.alpha_cumprod[t_idx])
                    .sqrt()
                    .view(b, *([1] * (x0.dim() - 1)))
                )
                x_t = sqrt_ab * x0 + sqrt_1mab * eps
                scaling = model.beta[t_idx] / (
                    2 * model.alpha[t_idx] * (1 - model.alpha_cumprod[t_idx])
                )

                if model_type == "latent":
                    eps_hat = model.noise_predictor(x_t, t_norm)
                    mse_sum = (eps_hat - eps).pow(2).reshape(b, -1).sum(dim=-1)
                    l_diff_sum +=  mse_sum
                else:
                    v_target = sqrt_ab.view(b, -1) * eps - sqrt_1mab.view(b, -1) * x0
                    v_hat = model.denoiser(x_t, t_norm)
                    mse_sum = (v_hat - v_target).pow(2).sum(dim=-1)
                    l_diff_sum += mse_sum

            h_q = _entropy_term(logvar)
            if n_elements is None:
                n_elements = x_dev[0].numel()
            elbo = l_diff_sum 
            elbo = elbo  # / n_elements  # nats per element

        total_elbo += elbo.sum().item()
        n_total += b

    final_elbo = total_elbo / n_total
    return final_elbo


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified results/eval script across VAE, Latent, and Weight diffusion models, 2D+3D."
    )
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["vae", "latent", "weight"]
    )
    parser.add_argument("--config_path_2d", type=str, required=True)
    parser.add_argument("--weights_path_2d", type=str, required=True)
    parser.add_argument("--config_path_3d", type=str, required=True)
    parser.add_argument("--weights_path_3d", type=str, required=True)
    parser.add_argument("--n_metric_samples", type=int, default=1024)
    parser.add_argument("--metric_batch_size", type=int, default=128)
    parser.add_argument("--n_pca_samples", type=int, default=512)
    parser.add_argument("--n_recon_samples", type=int, default=10)
    parser.add_argument("--n_interp_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--output_root", type=str, default=os.path.join("src", "results")
    )
    args = parser.parse_args()

    if args.n_pca_samples > args.n_metric_samples:
        parser.error(
            "--n_pca_samples cannot exceed --n_metric_samples (PCA points are drawn from the metric batch)."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _get_device()

    output_dir = os.path.join(args.output_root, f"{args.run_name}_{args.model_type}")
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"\n{'=' * 60}\n  Unified Results  |  {args.model_type}  |  {args.run_name}\n  Output: {output_dir}\n{'=' * 60}\n"
    )

    print("--- Loading 2D model ---")
    bundle_2d = prepare_model(
        args.model_type,
        args.config_path_2d,
        args.weights_path_2d,
        is_3d=False,
        device=device,
    )
    print("--- Loading 3D model ---")
    bundle_3d = prepare_model(
        args.model_type,
        args.config_path_3d,
        args.weights_path_3d,
        is_3d=True,
        device=device,
    )

    print("\n--- Processing 2D slot ---")
    res_2d = process_slot(
        args.model_type, bundle_2d, False, args, output_dir, args.run_name, device
    )
    print("\n--- Processing 3D slot ---")
    res_3d = process_slot(
        args.model_type, bundle_3d, True, args, output_dir, args.run_name, device
    )

    print("\n--- Building cross-dataset comparison plots ---")
    plot_sample_comparison_row(
        res_2d["gen_first4"],
        res_3d["gen_first4"],
        bundle_2d["channels"],
        os.path.join(output_dir, "sample_comparison.png"),
    )
    for mode in ("both", "originals", "recons"):
        plot_recon_comparison(
            res_2d["real_first4"],
            res_3d["real_first4"],
            res_2d["recon_first4"],
            res_3d["recon_first4"],
            bundle_2d["channels"],
            mode,
            os.path.join(output_dir, f"reconstruction_comparison_{mode}.png"),
        )

    metrics = {"2d": res_2d["metrics"], "3d": res_3d["metrics"]}
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")
    print("\nUnified Results Complete.")


if __name__ == "__main__":
    main()
