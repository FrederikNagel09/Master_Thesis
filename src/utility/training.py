"""
training.py
Universal training loop for all models.

All models must return: (total_loss, l_diff, l_prior, l_rec)
For components that are not applicable, return torch.tensor(0.0).
"""

from __future__ import annotations

import os
import sys

import numpy as np
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
def _get_beta(global_step: int, beta_final: float, warmup_steps: int) -> float:
    """
    Beta stays 0 for burnin_steps, then linearly ramps to beta_final over warmup_steps.
    
    Args:
        global_step: current training step
        beta_final: target beta value
        warmup_steps: steps to ramp from 0 to beta_final after burnin
        burnin_steps: steps to hold beta at 0 before ramping
    Returns:
        float: current beta value
    """
    return beta_final * min(1.0, (global_step) / warmup_steps)


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
    deactivate_progress_bar=False,
    freeze_encoder: float | None = None,  # noqa: ARG001
    two_stage: bool = False,
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

    # two-stage training control variables (for applicable models)
    stage2_triggered = False
    plateau_window = 40
    rel_threshold = 0.02
    min_stage1_steps = 50000
    kl_warmup_steps = 30000
    beta = 0.0

    if two_stage:
        print("[Training] Two-stage mode: freezing denoiser for stage 1.")
        if model_type == "weight_inr_diffusion":
            for p in model.denoiser.parameters():
                p.requires_grad = False
        elif model_type == "latent_inr_diffusion":
            for p in model.noise_predictor.parameters():
                p.requires_grad = False

    lambda_kl = model.lambda_kl if hasattr(model, "lambda_kl") else 1.0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        if GLOBAL_DEBUG_BOOL:
            print(f"\n############## EPOCH: {epoch} ##############\n")
        for batch in data_loader:
            # ── Forward pass (model-type dispatch) ───────────────────────────

            x = batch[0] if isinstance(batch, list | tuple) else batch
            x = x.to(device)

            if two_stage and not stage2_triggered:
                beta = _get_beta(global_step, lambda_kl, kl_warmup_steps)
                loss, l_diff, l_prior, l_rec = model.loss_vae(x, beta)
            else:
                loss, l_diff, l_prior, l_rec = model.loss(x, lambda_kl)

            # ── NaN/divergence diagnostics ──────────────────────────────────

            # ── Backward pass ────────────────────────────────────────────────
            optimizer.zero_grad()
            # 2. Check for NaN in Loss
            if torch.isnan(loss):
                print(f"CRITICAL: Loss is NaN at step {global_step}. Skipping...")
                continue  # Safe to skip here as we haven't done backward() yet
            loss.backward()
            # 4. Check for NaN in Gradients (Crucial Step)
            # This finds exactly which layer is exploding before you clip or step
            nan_found = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of: {name}")
                    nan_found = True
                    break

            if nan_found:
                optimizer.zero_grad()  # Clear the bad gradients
                continue  # Skip this step entirely

            # 5. Gradient Clipping and Step
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
                # epoch=f"{epoch}/{start_epoch + epochs}",
                loss=f"{loss.item():.4f}",
                diff=f"{l_diff.item():.4f}",
                prior=f"{l_prior.item():.4f}",
                rec=f"{l_rec.item():.4f}",
                beta=f"{beta:.2e}",
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

                # Two-stage: check for rec loss plateau → switch to stage 2
                if two_stage and not stage2_triggered and len(history["rec"]) >= plateau_window:
                    rec_window = history["rec"][-plateau_window:]
                    kl_window = history["prior"][-plateau_window:]
                    rec_flat = (max(rec_window) - min(rec_window)) / (abs(np.mean(rec_window)) + 1e-8) < rel_threshold
                    kl_flat = (max(kl_window) - min(kl_window)) / (abs(np.mean(kl_window)) + 1e-8) < rel_threshold
                    if rec_flat and kl_flat and global_step > min_stage1_steps:
                        stage2_triggered = True
                        print(f"[Step {global_step}] Rec plateaued — switching to stage 2.")
                        if model_type in ("weight_inr_diffusion", "weight_inr_ndm_diffusion"):
                            for p in model.weight_encoder.parameters():
                                p.requires_grad = False
                            for p in model.denoiser.parameters():
                                p.requires_grad = True
                            remaining_steps = total_steps - global_step
                            optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=lr, weight_decay=weight_decay)
                        elif model_type == "latent_inr_diffusion":
                            for p in model.latent_encoder.parameters():
                                p.requires_grad = False
                            for p in model.decoder.parameters():
                                p.requires_grad = False
                            for p in model.noise_predictor.parameters():
                                p.requires_grad = True
                            remaining_steps = total_steps - global_step
                            optimizer = torch.optim.Adam(model.noise_predictor.parameters(), lr=lr, weight_decay=weight_decay)
                        if use_scheduler:
                            scheduler = _build_scheduler(
                                optimizer,
                                warmup_steps=0.1 * remaining_steps,
                                total_steps=remaining_steps,
                                peak_lr=_peak_lr,
                            )

            # ── Sampling checkpoints ─────────────────────────────────────────
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

        # ── End of epoch: update training plot ───────────────────────────────
        _save_checkpoint(model, optimizer, epoch, weights_dir)

        if epoch_callback is not None:
            epoch_callback(history)

    progress_bar.close()
    if deactivate_progress_bar:
        tqdm_file.close()
    # ── End-of-training summary (visible in LSF email) ───────────────────────────
    print_training_summary(name, history, global_step, completed_steps, start_epoch, epochs, lr)

    return model
