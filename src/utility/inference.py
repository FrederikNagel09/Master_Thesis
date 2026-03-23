"""
inference.py
Universal inference / sampling for all models.

Public API
----------
    from src.utils.inference import sample

    images = sample(
        model_name  = "ndm",
        config_path = "src/train_results/my_run/config.json",
        n_samples   = 16,
    )
    # returns: torch.Tensor (N, C, H, W) in [0, 1] on CPU

For models that support multi-resolution (inr_vae, ndm_inr) pass:
    images = sample(..., resolution=128)
"""

import sys

import numpy as np
import torch

sys.path.append(".")

from src.utility.general import (
    _config_to_namespace,
    _flat_to_image,
    _load_config,
    _make_coord_grid,
)
from src.utility.model_builders import build_model

# =============================================================================
# Per-model sampling
# =============================================================================


def _sample_ndm(
    model: torch.nn.Module,
    n_samples: int,
    data_config: dict,
    **kwargs,  # noqa: ARG001
) -> torch.Tensor:
    """
    Sample from NeuralDiffusionModel.
    Returns (N, C, H, W) in [0, 1].
    """
    channels = data_config["channels"]
    img_size = data_config["img_size"]

    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples)  # (N, data_dim) in [-1, 1]

    samples = samples * 0.5 + 0.5  # [-1, 1] → [0, 1]
    return _flat_to_image(samples, n_samples, channels, img_size)


def _sample_inr_vae(
    model: torch.nn.Module,
    n_samples: int,
    device: str,
    data_config: dict,
    resolution: int | None = None,
) -> torch.Tensor:
    """
    Sample from VAEINR.
    Supports arbitrary resolution via INR re-rendering.
    Returns (N, C, H, W) in [0, 1].
    """
    channels = data_config["channels"]
    img_size = resolution or data_config["img_size"]
    dev = torch.device(device)

    coords = _make_coord_grid(img_size, dev)  # (img_size^2, 2)
    coords_batch = coords.unsqueeze(0).expand(n_samples, -1, -1)  # (N, img_size^2, 2)

    model.eval()
    with torch.no_grad():
        z = model.prior().sample(torch.Size([n_samples])).to(dev)  # (N, latent_dim)
        flat_weights = model.decode_to_weights(z)  # (N, num_weights)
        pixels = model.inr(coords_batch, flat_weights)  # (N, img_size^2, C)

    pixels = pixels.permute(0, 2, 1)  # (N, C, img_size^2)
    return _flat_to_image(pixels, n_samples, channels, img_size)


def _sample_ndm_inr(
    model: torch.nn.Module,
    n_samples: int,
    device: str,
    data_config: dict,
    resolution: int | None = None,
) -> torch.Tensor:
    """
    Sample from NeuralDiffusionModel with INR decoder.
    Supports arbitrary resolution via coord override.
    Returns (N, C, H, W) in [0, 1].
    """
    channels = data_config["channels"]
    img_size = resolution or data_config["img_size"]
    dev = torch.device(device)

    # Build custom coord grid only if a non-native resolution is requested
    coords = _make_coord_grid(img_size, dev) if resolution else None

    model.eval()
    with torch.no_grad():
        pixels = model.sample(n_samples, coords=coords)  # (N, img_size^2)

    return _flat_to_image(pixels, n_samples, channels, img_size)


def _sample_at_resolutions_inr_vae(
    model,
    n_rows: int,
    resolutions: list[int],
    channels: int,
    device: str,
) -> np.ndarray:
    """
    Sample n_rows latent codes and render each at all resolutions.
    Returns array of shape (n_rows, n_res, H_max, W_max, C) — images are
    stored at their native resolution, returned as a list of lists.
    Returns: list of lists — rows[i][j] is numpy array for row i, resolution j.
    """
    dev = torch.device(device)
    rows = []

    with torch.no_grad():
        # Sample n_rows latent codes once
        z = model.prior().sample(torch.Size([n_rows])).to(dev)  # (n_rows, latent_dim)
        flat_weights = model.decode_to_weights(z)  # (n_rows, num_weights)

        for i in range(n_rows):
            row_imgs = []
            w = flat_weights[i : i + 1]  # (1, num_weights)
            for res in resolutions:
                coords = _make_coord_grid(res, dev)  # (res^2, 2)
                coords_batch = coords.unsqueeze(0)  # (1, res^2, 2)
                pixels = model.inr(coords_batch, w)  # (1, res^2, C)
                pixels = pixels.permute(0, 2, 1).reshape(1, channels, res, res)
                pixels = pixels.clamp(0, 1).squeeze(0).cpu()  # (C, res, res)
                if channels == 1:  # noqa: SIM108
                    img = pixels.squeeze(0).numpy()  # (res, res)
                else:
                    img = pixels.permute(1, 2, 0).numpy()  # (res, res, C)
                row_imgs.append(img)
            rows.append(row_imgs)

    return rows


def _sample_at_resolutions_ndm_inr(
    model,
    n_rows: int,
    resolutions: list[int],
    channels: int,
    device: str,
) -> list:
    """
    Run full diffusion sampling n_rows times, then decode each at all resolutions.
    Returns list of lists — rows[i][j] is numpy array for row i, resolution j.
    """
    dev = torch.device(device)
    rows = []

    with torch.no_grad():
        for i in range(n_rows):
            print(f"    Sampling row {i + 1}/{n_rows} …")
            row_imgs = []
            for res in resolutions:
                coords = _make_coord_grid(res, dev) if res != resolutions[0] else None
                pixels = model.sample(1, coords=coords)  # (1, res^2)
                pixels = pixels.clamp(0, 1).reshape(1, channels, res, res)
                pixels = pixels.squeeze(0).cpu()  # (C, res, res)
                img = pixels.squeeze(0).numpy() if channels == 1 else pixels.permute(1, 2, 0).numpy()
                row_imgs.append(img)
            rows.append(row_imgs)

    return rows


# =============================================================================
# Dispatch table
# =============================================================================


_SAMPLE_FN = {
    "ndm": _sample_ndm,
    "inr_vae": _sample_inr_vae,
    "ndm_inr": _sample_ndm_inr,
}


# =============================================================================
# Public API
# =============================================================================


def sample(
    model_name: str,
    config_path: str,
    n_samples: int = 16,
    device: str | None = None,
    resolution: int | None = None,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Load a trained model from a config file and sample from it.

    Parameters
    ----------
    model_name  : One of "ndm", "inr_vae", "ndm_inr".
    config_path : Path to the config.json saved during training.
    n_samples   : Number of images to sample.
    device      : Device to run on. Defaults to the best available.
    resolution  : Optional output resolution (only used by inr_vae and ndm_inr).
                  If None, uses the native training resolution.

    Returns
    -------
    images : torch.Tensor of shape (N, C, H, W) in [0, 1] on CPU.
    """
    # ── Load config ───────────────────────────────────────────────────────────
    config = _load_config(config_path)
    args = _config_to_namespace(config)
    data_config = config["data"]
    weights_path = config["paths"]["weights"]

    # ── Device ────────────────────────────────────────────────────────────────

    print(f"  Inference on : {device}")

    # ── Build & load model ────────────────────────────────────────────────────
    print(f"  Building model : {model_name}")
    model = build_model(args, data_config).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"  Weights loaded : {weights_path}")

    # ── Sample ────────────────────────────────────────────────────────────────
    sample_fn = _SAMPLE_FN.get(model_name)
    if sample_fn is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_SAMPLE_FN.keys())}")

    # ── Batched sampling ──────────────────────────────────────────────────────
    print(f"  Sampling {n_samples} images in batches of {batch_size} …")
    all_images = []
    remaining = n_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        batch = sample_fn(
            model=model,
            n_samples=n,
            device=device,
            data_config=data_config,
            resolution=resolution,
        )
        all_images.append(batch.cpu())  # move to CPU immediately to free VRAM
        remaining -= n

    images = torch.cat(all_images, dim=0)
    print(f"  Done. Output shape: {tuple(images.shape)}  range: [{images.min():.3f}, {images.max():.3f}]")
    return images
