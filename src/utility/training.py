"""
training.py
Universal training loop for all models.

All models must return: (total_loss, l_diff, l_prior, l_rec)
For components that are not applicable, return torch.tensor(0.0).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(".")

from typing import TYPE_CHECKING

from src.utility.general import _build_scheduler
from src.utility.plotting import print_training_summary

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Universal training loop
# =============================================================================


def train(
    model: nn.Module,
    model_type: str,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: str,
    name: str,
    # ── Optimiser ────────────────────────────────────────────────────────────
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    # ── Scheduler ────────────────────────────────────────────────────────────
    use_scheduler: bool = True,
    warmup_steps: int = 5_000,
    peak_lr: float | None = None,
    # ── Logging ──────────────────────────────────────────────────────────────
    log_every_n_steps: int = 20,
    save_dir: str = "results",
    # ── Callbacks ────────────────────────────────────────────────────────────
    sample_fn: Callable[[nn.Module, int, str], None] | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
    # ── Resuming ─────────────────────────────────────────────────────────────
    start_epoch: int = 0,
    history: dict | None = None,
    data_config: dict | None = None,
    deactivate_progress_bar=False,
) -> nn.Module:
    """
    Train *model* for *epochs* epochs and return the trained model.

    Parameters
    ----------
    model       : nn.Module to train.
    model_type  : String tag that controls the forward/loss call.
    data_loader : DataLoader; see module docstring for expected batch format.
    epochs      : Number of epochs to train (not counting start_epoch).
    device      : torch device string, e.g. "cuda" or "cpu".
    name        : Run name used for file names.
    lr          : Base learning rate for Adam.
    weight_decay: L2 regularisation for Adam.
    grad_clip   : Max-norm gradient clipping; set to 0 to disable.
    use_scheduler: Whether to attach the warmup+decay LR scheduler.
    warmup_steps: Number of linear-warmup steps.
    peak_lr     : LR at the top of the warmup (defaults to *lr*).
    log_every_n_steps: How often to append to the running-average history.
    save_dir    : Root directory; graphs saved to <save_dir>/graphs/,
                  samples to <save_dir>/samples/.
    sample_fn   : Optional callable(model, step, device) -> None.
                  Called at 5 evenly-spaced checkpoints and once at the end.
    start_epoch : Resume offset; adjusts scheduler and step counter.
    """

    # ── Directories ──────────────────────────────────────────────────────────
    weights_dir = os.path.join(save_dir, "weights")
    metadata_dir = os.path.join(save_dir, "metadata")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    _peak_lr = peak_lr if peak_lr is not None else lr

    steps_per_epoch = len(data_loader)
    total_steps = steps_per_epoch * (epochs + start_epoch)
    completed_steps = steps_per_epoch * start_epoch

    scheduler = None
    if use_scheduler:
        if completed_steps > 0:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        scheduler = _build_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=_peak_lr,
        )
        # Fast-forward scheduler to the correct step when resuming
        if completed_steps > 0:
            scheduler.last_epoch = completed_steps - 1

    # ── Sampling checkpoints (5 evenly-spaced + final) ───────────────────────
    _sample_steps: set[int] = set()
    if sample_fn is not None:
        interval = max(1, (steps_per_epoch * epochs) // 5)
        for i in range(1, 6):
            _sample_steps.add(completed_steps + i * interval)

    # ── History ──────────────────────────────────────────────────────────────
    if history is None:
        history: dict[str, list] = {
            "steps": [],
            "total": [],
            "diff": [],
            "prior": [],
            "rec": [],
            "lr": [],
        }

    # Running accumulators (reset every log_every_n_steps)
    running: dict[str, float] = {"total": 0.0, "diff": 0.0, "prior": 0.0, "rec": 0.0}
    running_count = 0

    # ── Progress bar ─────────────────────────────────────────────────────────
    if deactivate_progress_bar:
        tqdm_file = open(os.path.join(save_dir, "tqdm.log"), "w")  # noqa: SIM115
        progress_bar = tqdm(
            total=steps_per_epoch * epochs,
            desc=f"Training {name}",
            unit="step",
            file=tqdm_file,
            dynamic_ncols=True,
        )
    else:
        progress_bar = tqdm(
            total=steps_per_epoch * epochs,
            desc=f"Training {name}",
            unit="step",
            dynamic_ncols=True,
        )

    global_step = completed_steps
    model.train()

    # ── Precompute coord grid for inr_vae ─────────────────────────────────────
    _coords = None
    if model_type == "inr_vae":
        img_size = data_config["img_size"]
        lin = torch.linspace(-1, 1, img_size, device=device)
        gr, gc = torch.meshgrid(lin, lin, indexing="ij")
        _coords = torch.stack([gr.flatten(), gc.flatten()], dim=-1)  # (img_size^2, 2)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        for batch in data_loader:
            # ── Forward pass (model-type dispatch) ───────────────────────────
            if model_type == "inr_vae":
                image_flat, _ = batch
                image_flat = image_flat.to(device)  # (B, data_dim)
                b = image_flat.shape[0]
                coords = _coords.unsqueeze(0).expand(b, -1, -1)  # (B, img_size^2, 2)

                if data_config["channels"] == 1:
                    pixels = ((image_flat * 0.5 + 0.5).clamp(0, 1)).unsqueeze(-1)  # (B, H*W, 1)
                else:
                    pixels = ((image_flat * 0.5 + 0.5).clamp(0, 1)).reshape(b, -1, data_config["channels"])  # (B, H*W, C)

                loss, l_diff, l_prior, l_rec = model(image_flat, coords, pixels)
            elif model_type == "ndm_transinr":
                x = batch[0] if isinstance(batch, list | tuple) else batch
                x = x.to(device)
                # TransInrEncoder expects spatial (B, C, H, W), but the dataloader
                # yields flat (B, C*H*W). Reshape here so model internals stay clean.
                C = data_config["channels"]  # noqa: N806
                H = W = data_config["img_size"]  # noqa: N806
                x = x.view(x.shape[0], C, H, W)
                loss, l_diff, l_prior, l_rec = model.loss(x)
            else:
                x = batch[0] if isinstance(batch, list | tuple) else batch
                x = x.to(device)
                loss, l_diff, l_prior, l_rec = model.loss(x)

            # ── Backward pass ────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # ── Accumulate ───────────────────────────────────────────────────
            global_step += 1
            running_count += 1
            running["total"] += loss.item()
            running["diff"] += l_diff.item()
            running["prior"] += l_prior.item()
            running["rec"] += l_rec.item()

            current_lr = scheduler.get_last_lr()[0] if scheduler else lr

            # ── Progress bar postfix ─────────────────────────────────────────
            progress_bar.set_postfix(
                epoch=f"{epoch}/{start_epoch + epochs}",
                loss=f"{loss.item():.4f}",
                diff=f"{l_diff.item():.4f}",
                prior=f"{l_prior.item():.4f}",
                rec=f"{l_rec.item():.4f}",
                lr=f"{current_lr:.2e}",
            )
            progress_bar.update()

            # ── Periodic history append ───────────────────────────────────────
            if global_step % log_every_n_steps == 0:
                fractional_epoch = global_step / steps_per_epoch
                history["steps"].append(fractional_epoch)
                history["total"].append(running["total"] / running_count)
                history["diff"].append(running["diff"] / running_count)
                history["prior"].append(running["prior"] / running_count)
                history["rec"].append(running["rec"] / running_count)
                history["lr"].append(current_lr)
                running = dict.fromkeys(running, 0.0)
                running_count = 0

            # ── Sampling checkpoints ─────────────────────────────────────────
            if sample_fn is not None and global_step in _sample_steps:
                model.eval()
                with torch.no_grad():
                    if model_type in ("ndm", "ndm_inr", "ndm_transinr", "ndm_temporal_transinr"):
                        sample_fn(model, global_step, device, batch=batch)
                    else:
                        sample_fn(model, global_step, device)
                model.train()

        # ── End of epoch: update training plot ───────────────────────────────
        if epoch_callback is not None:
            epoch_callback(history)

    progress_bar.close()
    if deactivate_progress_bar:
        tqdm_file.close()
    # ── End-of-training summary (visible in LSF email) ───────────────────────────
    print_training_summary(name, history, global_step, completed_steps, start_epoch, epochs, lr)

    return model
