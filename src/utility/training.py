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

def _get_beta(
    global_step: int,
    beta_final: float,
    warmup_steps: int,
    burnin_steps: int = 0,
) -> float:
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
    print("total steps:", warmup_steps)
    if global_step < burnin_steps:
        return 0.0
    return beta_final * min(1.0, (global_step - burnin_steps) / warmup_steps)




def _is_plateaued(window_vals: list[float], rel_threshold: float) -> bool:
    """
    Checks whether a window of loss values has gone relatively flat.

    Args:
        window_vals: recent loss values to check for flatness
        rel_threshold: max allowed (max-min)/|mean| to count as flat
    Returns:
        bool: True if the window is flat under rel_threshold
    """
    return (max(window_vals) - min(window_vals)) / (abs(np.mean(window_vals)) + 1e-8) < rel_threshold


def _build_constant_after_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    peak_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then hold constant at peak_lr — for use when total_steps is unknown ahead of time."""

    def lr_lambda(current_step: int) -> float:
        floor = 1e-8 / peak_lr
        if current_step < warmup_steps:
            return max(floor, current_step / max(warmup_steps, 1))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    # ── Two-stage training ───────────────────────────────────────────────────
    two_stage: bool = False,
    stage_one_epochs: int = 150,
    stage_two_epochs: int = 250,
    stage1_plateau_window: int = 40,
    stage1_rel_threshold: float = 0.02,
    stage2_plateau_window: int = 40,
    stage2_rel_threshold: float = 0.05,
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

    # ── Two-stage epoch-budget validation ───────────────────────────────────
    convergence_mode = two_stage and stage_one_epochs == 0 and stage_two_epochs == 0
    if two_stage and not convergence_mode:
        if stage_one_epochs <= 0 or stage_two_epochs <= 0:
            raise ValueError(
                "stage_one_epochs and stage_two_epochs must both be 0 (convergence mode) "
                "or both be positive (fixed mode); got "
                f"stage_one_epochs={stage_one_epochs}, stage_two_epochs={stage_two_epochs}."
            )
        if epochs != stage_one_epochs + stage_two_epochs:
            raise ValueError(
                f"epochs ({epochs}) must equal stage_one_epochs + stage_two_epochs "
                f"({stage_one_epochs} + {stage_two_epochs} = {stage_one_epochs + stage_two_epochs})."
            )
    if convergence_mode:
        print(f"[Training] Two-stage convergence mode: stages run until plateau, " f"capped at {epochs} total epochs as a safety net.")

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
        if convergence_mode:
            # Total step count unknown ahead of time — warm up then hold flat.
            scheduler = _build_constant_after_warmup_scheduler(
                optimizer,
                warmup_steps=0.1 * (steps_per_epoch * stage_one_epochs or steps_per_epoch * 10),
                peak_lr=_peak_lr,
            )
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
    if sample_fn is not None and not convergence_mode:
        interval = max(1, (steps_per_epoch * epochs) // 5)
        for i in range(1, 6):
            _sample_steps.add(completed_steps + i * interval)
    # In convergence mode, sampling checkpoints can't be pre-scheduled since the
    # total step count is unknown; sample_fn is instead called once per stage
    # transition and once at the very end (see end-of-loop / stage-switch code).

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

    # ── Two-stage training control variables ─────────────────────────────────
    current_stage = 1 if two_stage else None
    stage1_epoch_count = 0
    stage2_epoch_count = 0
    kl_burnin_steps = 0 
    kl_warmup_steps = 0.4 * total_steps  
    beta = 0.0
    print("total steps:", total_steps)
    print("kl warmup steps:", kl_warmup_steps)
    beta = 0.0
    global_step = 0

    if two_stage:
        print("[Training] Two-stage mode: starting stage 1 (VAE).")
        if model_type == "weight_inr_diffusion":
            for p in model.denoiser.parameters():
                p.requires_grad = False
        elif model_type == "latent_inr_diffusion":
            for p in model.noise_predictor.parameters():
                p.requires_grad = False

    lambda_kl = model.lambda_kl if hasattr(model, "lambda_kl") else 1.0

    def _switch_to_stage_two() -> None:
        """Freeze the VAE, unfreeze the denoiser/noise predictor, rebuild the optimiser+scheduler."""
        nonlocal optimizer, scheduler, current_stage
        print(f"[Step {global_step}] Stage 1 complete — switching to stage 2 (DDPM).")
        if model_type in ("weight_inr_diffusion", "weight_inr_ndm_diffusion"):
            for p in model.weight_encoder.parameters():
                p.requires_grad = False
            for p in model.denoiser.parameters():
                p.requires_grad = True
            trainable_params = model.denoiser.parameters()
        elif model_type == "latent_inr_diffusion":
            for p in model.latent_encoder.parameters():
                p.requires_grad = False
            for p in model.decoder.parameters():
                p.requires_grad = False
            for p in model.noise_predictor.parameters():
                p.requires_grad = True
            trainable_params = model.noise_predictor.parameters()
        else:
            trainable_params = model.parameters()

        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        if use_scheduler:
            if convergence_mode:
                scheduler = _build_constant_after_warmup_scheduler(
                    optimizer,
                    warmup_steps=0.1 * steps_per_epoch * 10,
                    peak_lr=_peak_lr,
                )
            else:
                remaining_steps = total_steps - global_step
                scheduler = _build_scheduler(
                    optimizer,
                    warmup_steps=0.1 * remaining_steps,
                    total_steps=remaining_steps,
                    peak_lr=_peak_lr,
                )
        current_stage = 2
        if sample_fn is not None:
            model.eval()
            with torch.no_grad():
                sample_fn(model, global_step, device, batch=batch)
            model.train()

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
            if two_stage and current_stage == 1:
                print("total steps:", total_steps)
                print("kl warmup steps:", kl_warmup_steps)
                beta = _get_beta(global_step, lambda_kl, kl_warmup_steps, kl_burnin_steps)
                loss, l_diff, l_prior, l_rec = model.loss_vae(x, beta)
            elif two_stage and current_stage == 2:
                loss, l_diff, l_prior, l_rec = model.loss_ddpm(x, lambda_kl)
            else:
                loss, l_diff, l_prior, l_rec = model.loss(x, lambda_kl)

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
                stage=f"{current_stage}" if two_stage else "-",
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

                # Convergence-mode stage-1 plateau check (rec + kl both flat)
                if convergence_mode and current_stage == 1 and len(history["rec"]) >= stage1_plateau_window:
                    rec_flat = _is_plateaued(history["rec"][-stage1_plateau_window:], stage1_rel_threshold)
                    kl_flat = _is_plateaued(history["prior"][-stage1_plateau_window:], stage1_rel_threshold)
                    if rec_flat and kl_flat:
                        _switch_to_stage_two()

                # Convergence-mode stage-2 plateau check (diff flat) → stop training
                if convergence_mode and current_stage == 2 and len(history["diff"]) >= stage2_plateau_window:
                    diff_flat = _is_plateaued(history["diff"][-stage2_plateau_window:], stage2_rel_threshold)
                    if diff_flat:
                        print(f"[Step {global_step}] Stage 2 plateaued — stopping training.")
                        stop_training = True
                        break

            # ── Sampling checkpoints (fixed mode only — see note above) ──────
            if sample_fn is not None and not convergence_mode and global_step in _sample_steps:
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

        # ── End of epoch: fixed-mode stage transition / stopping ─────────────
        if two_stage and not convergence_mode:
            if current_stage == 1:
                stage1_epoch_count += 1
                if stage1_epoch_count >= stage_one_epochs:
                    _switch_to_stage_two()
            elif current_stage == 2:
                stage2_epoch_count += 1
                if stage2_epoch_count >= stage_two_epochs:
                    stop_training = True

        _save_checkpoint(model, optimizer, epoch, weights_dir)

        if epoch_callback is not None:
            epoch_callback(history)

        # Safety cap — always checked, fixed mode included as a backstop
        if epoch - start_epoch >= epochs:
            if convergence_mode and not stop_training:
                print(f"[Training] WARNING: hit epoch cap ({epochs}) before stage 2 converged. Stopping anyway.")
            stop_training = True

    progress_bar.close()
    if deactivate_progress_bar:
        tqdm_file.close()
    print_training_summary(name, history, global_step, completed_steps, start_epoch, epoch - start_epoch, lr)

    return model
