import argparse
import json
import os
import types
from datetime import datetime

import torch
from torch import nn

_RESULTS_ROOT = "src/train_results"

# =============================================================================
# Scheduler
# =============================================================================


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


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    from src.configs.hyperparameter_config import MODEL_SECTIONS, SECTIONS

    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # ── Build hparam sections ─────────────────────────────────────────────────
    # Unknown models fall back to saving all sections
    active_sections = MODEL_SECTIONS.get(args.model, list(SECTIONS.keys()))
    args_dict = vars(args)
    hparams = {section: {k: args_dict[k] for k in SECTIONS[section] if k in args_dict} for section in active_sections}

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
        **hparams,  # each section becomes a top-level key in the JSON
    }
    metadata_path = os.path.join(run_dir, "metadata")
    path = os.path.join(metadata_path, "config.json")
    # Merge with existing config if resuming (preserve original start time/epoch)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        config["timing"]["start"] = old["timing"].get("start", config["timing"]["start"])
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
