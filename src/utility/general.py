import argparse
import json
import os
import types
from datetime import datetime

import torch
from torch import nn

from src.utility.model_builders.model_builder import build_model

_RESULTS_ROOT = "src/train_results"

# =============================================================================
# Scheduler
# =============================================================================


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # if torch.backends.mps.is_available():
    # return "mps"
    return "cpu"


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _config_to_namespace(config: dict) -> types.SimpleNamespace:
    flat = {}
    for key in ("model", "dataset", "run_name"):
        if key in config:
            flat[key] = config[key]
    for value in config.values():
        if isinstance(value, dict):
            flat.update(value)
    return types.SimpleNamespace(**flat)


def _make_coord_grid(resolution: int, device: torch.device) -> torch.Tensor:
    """Build (resolution*resolution, 2) coordinate grid in [-1, 1]."""
    lin = torch.linspace(-1, 1, resolution, device=device)
    gr, gc = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([gr.flatten(), gc.flatten()], dim=-1)


def _load_model(config_path: str, device: str):
    """Load model from config, return (model, data_config)."""
    config = _load_config(config_path)
    args = _config_to_namespace(config)
    data_config = config["data"]
    weights_path = config["paths"]["weights"]

    model = build_model(args, data_config).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, data_config


def _draw_grid(
    axes: list,
    images: torch.Tensor,
    channels: int,
) -> None:
    """
    Draw a GRID_SIZE x GRID_SIZE image grid onto a list of axes.
    axes  : flat list of GRID_SIZE*GRID_SIZE axes
    images: (N, C, H, W) tensor in [0, 1]
    """
    for i, ax in enumerate(axes):
        img = images[i]
        if channels == 1:
            ax.imshow(
                img.squeeze(0).numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        else:
            ax.imshow(
                img.permute(1, 2, 0).numpy(), vmin=0, vmax=1, interpolation="nearest"
            )
        ax.axis("off")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then linear decay to near-zero."""

    def lr_lambda(current_step: int) -> float:
        floor = 1e-8 / peak_lr
        if current_step < warmup_steps:
            return max(floor, current_step / max(warmup_steps, 1))
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(floor, 1.0 - progress * (1.0 - floor))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _run_dir(run_name: str) -> str:
    path = os.path.join(_RESULTS_ROOT, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def _save_checkpoint(model: nn.Module, optimizer, epoch: int, run_dir: str) -> str:
    path = os.path.join(run_dir, "weights.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    return path


def _load_checkpoint(path: str, model: nn.Module, optimizer) -> int:
    """Load checkpoint into model and optimizer in-place. Returns epoch."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    print(f"  Resumed from '{path}'  (epoch {start_epoch})")
    return start_epoch


def _load_graph_data(run_dir: str) -> dict:
    """Load existing training_graph_data.json or return an empty history dict."""
    metadata_path = os.path.join(run_dir, "metadata")
    path = os.path.join(metadata_path, "training_graph_data.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"steps": [], "total": [], "diff": [], "prior": [], "rec": [], "lr": []}


def _save_graph_data(history: dict, run_dir: str) -> None:
    metadata_path = os.path.join(run_dir, "metadata")
    path = os.path.join(metadata_path, "training_graph_data.json")
    with open(path, "w") as f:
        json.dump(history, f)


def _save_config(
    args: argparse.Namespace,
    data_config: dict,
    run_dir: str,
    weights_path: str,
    start_epoch: int,
    end_epoch: int,
    start_time: datetime,
    end_time: datetime,
) -> None:
    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # ── Build hparam sections ─────────────────────────────────────────────────
    # Unknown models fall back to saving all sections

    config = {
        "run_name": args.run_name,
        "model": args.model,
        "dataset": args.dataset,
        "timing": {
            "start": start_time.isoformat(timespec="seconds"),
            "end": end_time.isoformat(timespec="seconds"),
            "duration": f"{hours:02d}h {minutes:02d}m {seconds:02d}s",
        },
        "epochs": {
            "start": start_epoch,
            "end": end_epoch,
            "total": end_epoch - start_epoch,
        },
        "paths": {
            "weights": weights_path,
            "run_dir": run_dir,
        },
        "data": data_config,
        "hparams": vars(args),
    }
    metadata_path = os.path.join(run_dir, "metadata")
    path = os.path.join(metadata_path, "config.json")
    # Merge with existing config if resuming (preserve original start time/epoch)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        config["timing"]["start"] = old["timing"].get(
            "start", config["timing"]["start"]
        )
        config["epochs"]["start"] = old["epochs"].get("start", start_epoch)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config  saved → {path}")


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _config_to_namespace(config: dict) -> types.SimpleNamespace:
    """
    Flatten all section dicts in the config into a single SimpleNamespace
    so build_model() can access everything via dot notation, just like argparse.
    """
    flat = {}

    # Top-level scalar fields
    for key in ("model", "dataset", "run_name"):
        if key in config:
            flat[key] = config[key]

    # Flatten every section dict (training, inr, vae, diffusion, etc.)
    for value in config.values():
        if isinstance(value, dict):
            flat.update(value)

    return types.SimpleNamespace(**flat)


def _make_coord_grid(resolution: int, device: torch.device) -> torch.Tensor:
    """Build a (resolution*resolution, 2) coordinate grid in [-1, 1]."""
    lin = torch.linspace(-1, 1, resolution, device=device)
    grid_r, grid_c = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)


def _flat_to_image(
    pixels: torch.Tensor,
    n_samples: int,
    channels: int,
    resolution: int,
) -> torch.Tensor:
    """
    Reshape flat pixel tensor to (N, C, H, W) and clip to [0, 1].
    pixels: (N, resolution*resolution*channels) or (N, resolution*resolution, channels)
    """
    pixels = pixels.reshape(n_samples, channels, resolution, resolution)
    return pixels.clamp(0.0, 1.0).cpu()


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────


def load_ldm_config(path: str) -> dict:
    """
    Load hparams from a trained LDM config JSON.

    Args:
        path (str): path to config .json
    Returns:
        dict: hparams block from config
    """
    with open(path, "r") as f:  # noqa: UP015
        config = json.load(f)
    required_keys = [
        "latent_dim",
        "latent_size",
        "latent_patch_size",
        "latent_enc_hidden_dim",
        "dec_trans_dim",
        "dec_trans_n_head",
        "dec_trans_head_dim",
        "dec_trans_ff_dim",
        "dec_trans_enc_depth",
        "dec_trans_dec_depth",
        "dec_trans_n_groups",
        "dec_trans_update_strategy",
        "inr_hidden_dim",
        "inr_layers",
        "dataset",
        "pred_d_model",
        "pred_n_heads",
        "pred_n_layers",
        "pred_d_ff",
        "pred_t_embed_dim",
        "noise_predictor_dropout",
    ]
    hparams = config["hparams"]
    missing = [k for k in required_keys if k not in hparams]
    if missing:
        raise ValueError(f"LDM config missing required keys: {missing}")
    return hparams


_DDPM_ONLY_FILES = [
    "_ldm_checkpoint.pt",
    "_ldm_weights.pt",
    "_ldm_config.json",
    "_ddpm_training_curves.png",
    "_eval_metrics.json",
    "_ldm_samples_8x8.png",
]


def _clear_ddpm_files(results_dir: str, run_name: str) -> None:
    """
    Delete only Stage-2 output files, leaving VAE files intact.

    Args:
        results_dir (str): results directory
        run_name    (str): run identifier prefix
    Returns:
        None
    """
    for suffix in _DDPM_ONLY_FILES:
        path = os.path.join(results_dir, f"{run_name}{suffix}")
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale DDPM file: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CONVERGENCE DETECTION
# ──────────────────────────────────────────────────────────────────────────────


class SmoothedPlateauDetector:
    """
    Stops training when a smoothed validation signal stops improving.
    Compares mean of first vs second half of a rolling window of check values.
    """

    def __init__(self, patience: int, delta: float) -> None:
        """
        Args:
            patience (int):   number of checks in the rolling window
            delta    (float): minimum improvement to count as progress
        """
        self.patience = patience
        self.delta = delta
        self._window: list[float] = []

    def step(self, value: float) -> bool:
        """
        Record a new check value and return True if training should stop.

        Args:
            value (float): latest validation metric (lower is better)
        Returns:
            bool: True if plateau detected
        """
        self._window.append(value)
        if len(self._window) < self.patience:
            return False

        window = self._window[-self.patience :]
        mid = len(window) // 2
        first_half_avg = sum(window[:mid]) / mid
        second_half_avg = sum(window[mid:]) / (len(window) - mid)
        improvement = first_half_avg - second_half_avg
        return improvement < self.delta
