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

from src.configs.general_config import GLOBAL_DEBUG_BOOL

sys.path.append(".")

from typing import TYPE_CHECKING

from src.utility.general import _build_scheduler, _save_checkpoint
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
    warmup_steps: int = 5_000,  # noqa: ARG001
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
) -> nn.Module:
    """
    Train *model* for *epochs* epochs and return the trained model.

    Parameters
    ----------
    model       : nn.Module to train.
    model_type  : String tag that controls the forward/loss call.
    data_loader : DataLoader; see module docstring for expected batch format.
    epochs      : Number of epochs to train (not counting start_epoch). In
                  two_stage fixed mode this must equal stage_one_epochs +
                  stage_two_epochs. In two_stage convergence mode (both stage
                  epoch values are 0) this instead acts as a safety cap on
                  total epochs in case convergence never triggers.
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
    two_stage   : If True, train VAE (stage 1) then DDPM (stage 2) sequentially
                  instead of the joint end-to-end loss.
    stage_one_epochs : Epochs for stage 1. Set both stage values to 0 to run
                  each stage until plateau instead of a fixed epoch count.
    stage_two_epochs : Epochs for stage 2. See stage_one_epochs.
    stage1_plateau_window  : Window size (in logged points) for stage-1 flatness check.
    stage1_rel_threshold   : Relative flatness threshold for stage-1 rec/kl plateau.
    stage2_plateau_window  : Window size (in logged points) for stage-2 flatness check.
    stage2_rel_threshold   : Relative flatness threshold for stage-2 diff plateau
                  (looser than stage 1 by default — DDPM loss is noisier).
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
        else:
            scheduler = _build_scheduler(
                optimizer,
                warmup_steps=0.1 * total_steps,
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

    progress_bar = tqdm(
        total=steps_per_epoch * epochs,
        desc=f"Training {name}",
        unit="step",
        dynamic_ncols=True,
        file=sys.stderr,
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
    epoch = start_epoch
    stop_training = False
    while not stop_training:
        epoch += 1

        if GLOBAL_DEBUG_BOOL:
            print(f"\n############## EPOCH: {epoch} ##############\n")

        for batch in data_loader:
            x = batch[0] if isinstance(batch, list | tuple) else batch
            x = x.to(device)

            # ── Forward pass (model-type / stage dispatch) ───────────────────

            loss, l_diff, l_prior, l_rec = model.loss(x)

            # ── Backward pass ────────────────────────────────────────────────
            optimizer.zero_grad()

            if torch.isnan(loss):
                print(f"CRITICAL: Loss is NaN at step {global_step}. Skipping...")
                continue

            loss.backward()

            nan_found = False
            for pname, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of: {pname}")
                    nan_found = True
                    break
            if nan_found:
                optimizer.zero_grad()
                continue

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

            progress_bar.set_postfix(
                rec=f"{l_rec.item():.4f}",
                diff=f"{l_diff.item():.4f}",
                loss=f"{loss.item():.4f}",
                prior=f"{l_prior.item():.4f}",
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

            # ── Sampling checkpoints (fixed mode only — see note above) ──────
            if sample_fn is not None and global_step in _sample_steps:
                model.eval()
                with torch.no_grad():
                    if model_type in (
                        "ndm",
                        "ndm_inr",
                        "ndm_static_mlpinr",
                        "ndm_transinr",
                        "ndm_temporal_transinr",
                        "weight_inr_diffusion",
                        "latent_inr_diffusion",
                        "latent_ndm_inr_diffusion",
                        "weight_inr_ndm_diffusion",
                    ):
                        sample_fn(model, global_step, device, batch=batch)
                    else:
                        sample_fn(model, global_step, device)
                model.train()

        _save_checkpoint(model, optimizer, epoch, weights_dir)

        if epoch_callback is not None:
            epoch_callback(history)

        # Safety cap — always checked, fixed mode included as a backstop
        if epoch - start_epoch >= epochs:
            stop_training = True

    progress_bar.close()

    print_training_summary(name, history, global_step, completed_steps, start_epoch, epoch - start_epoch, lr)

    return model
