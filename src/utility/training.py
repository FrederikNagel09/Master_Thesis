"""
training.py
Universal training loop for all models.

All models must return: (total_loss, l_diff, l_prior, l_rec)
For components that are not applicable, return torch.tensor(0.0).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.models.two_stage_models.latent_two_stage import TwoStageLDM
from src.models.vae.vae_wrapper import VAEWrapper, _get_beta
from src.configs.general_config import GLOBAL_DEBUG_BOOL
from src.utility.model_builders.util.vae_builder import build_vae
from src.utility.evaluation import (
    _build_val_noise_cache,
    compute_ddpm_val_loss,
    compute_fid,
)
from src.utility.model_builders.util.twostage_builder import build_ldm
from src.utility.save_weigths_checkpoints import (
    save_ldm_checkpoint,
    save_ldm_weights,
    save_vae_checkpoint,
    save_vae_weights,
)

sys.path.append(".")

from typing import TYPE_CHECKING

from src.utility.general import (
    SmoothedPlateauDetector,
    _build_scheduler,
    _save_checkpoint,
)
from src.utility.plotting import (
    print_training_summary,
    save_ddpm_training_graph,
    save_vae_training_graph,
)

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

    print_training_summary(
        name,
        history,
        global_step,
        completed_steps,
        start_epoch,
        epoch - start_epoch,
        lr,
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# VAE TRAINING STAGE
# ──────────────────────────────────────────────────────────────────────────────


def train_vae(
    args: argparse.Namespace,
    hparams: dict,
    dataloader: DataLoader,
    val_loader: DataLoader,
    channels: int,
    img_size: int,
    is_3d: bool,
    results_dir: str,
    device: torch.device,
) -> VAEWrapper:
    """
    Train the VAE stage and save weights. Returns the trained VAEWrapper.
    Args:
        args        (argparse.Namespace): CLI args
        hparams     (dict):               LDM arch hparams
        dataloader  (DataLoader):         training data loader
        val_loader  (DataLoader):         validation data loader
        channels    (int):                image channels
        img_size    (int):                spatial size per dimension
        is_3d       (bool):               whether input is volumetric
        results_dir (str):                output directory
        device      (torch.device):       target device
    Returns:
        VAEWrapper: trained model on device
    """
    print("\n" + "=" * 60)
    print("  STAGE 1 — VAE TRAINING")
    print("=" * 60)
    model = build_vae(hparams, channels, img_size, device, is_3d=is_3d)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = {"elbo": [], "recon": [], "kl": []}
    graph_path = os.path.join(results_dir, f"{args.run_name}_vae_training_curves.png")
    if args.mode == "fixed":
        max_epochs = args.vae_epochs
        detector = None
    else:
        max_epochs = args.vae_max_epochs
        detector = SmoothedPlateauDetector(
            patience=args.vae_patience, delta=args.vae_delta
        )
    steps_per_epoch = len(dataloader)
    kl_warmup_steps = max(1, int(args.kl_warmup_frac * max_epochs)) * steps_per_epoch
    global_step = 0
    progress = tqdm(
        total=max_epochs * steps_per_epoch, desc="VAE Training", unit="step"
    )
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_recon = 0.0
        running_kl = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            if not is_3d and x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)
            optimizer.zero_grad()
            beta_kl = _get_beta(global_step, args.lambda_kl_max, kl_warmup_steps)
            x_recon, mu, logvar = model(x)
            x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
            x_flat = x.reshape(x.shape[0], -1)

            if is_3d:
                # Binary voxels: BCE against sigmoid output
                eps = 1e-7
                x_hat_flat = x_hat_flat.clamp(eps, 1 - eps)
                loss_recon = F.binary_cross_entropy(
                    x_hat_flat, x_flat, reduction="none"
                ).sum(dim=-1).mean()
            else:
                # Continuous pixels in [-1,1]: Gaussian MSE
                x_flat = x_flat.clamp(-1, 1)
                loss_recon = 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1).mean()

            loss_kl = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
            )
            total_loss = loss_recon + beta_kl * loss_kl
            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            history["elbo"].append(total_loss.item())
            history["recon"].append(loss_recon.item())
            history["kl"].append(loss_kl.item())
            running_recon += loss_recon.item()
            running_kl += loss_kl.item()
            progress.set_postfix(
                {
                    "epoch": f"{epoch}/{max_epochs}",
                    "Recon": f"{loss_recon.item():.4f}",
                    "KL": f"{loss_kl.item():.2f}",
                    "β": f"{beta_kl:.3f}",
                }
            )
            progress.update(1)
            global_step += 1
        epoch_recon = running_recon / steps_per_epoch
        epoch_kl = running_kl / steps_per_epoch
        print(
            f"  [VAE epoch {epoch}] Recon: {epoch_recon:.5f} | KL: {epoch_kl:.3f} | β: {beta_kl:.4f}"
        )
        save_vae_checkpoint(
            model, optimizer, epoch, history, results_dir, args.run_name
        )
        save_vae_training_graph(history, steps_per_epoch, epoch, graph_path)
        if detector is not None and epoch % args.vae_check_every == 0:
            model.eval()
            val_elbo = 0.0
            n_seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    if not is_3d and x.dim() == 2:
                        x = x.view(x.shape[0], channels, img_size, img_size)
                    x_recon, mu, logvar = model(x)
                    x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
                    x_flat = x.reshape(x.shape[0], -1)

                    if is_3d:
                        eps = 1e-7
                        x_hat_flat = x_hat_flat.clamp(eps, 1 - eps)
                        recon = F.binary_cross_entropy(
                            x_hat_flat, x_flat, reduction="none"
                        ).sum(dim=-1).mean()
                    else:
                        x_flat = x_flat.clamp(-1, 1)
                        recon = 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1).mean()

                    kl = -0.5 * torch.mean(
                        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
                    )
                    val_elbo += (recon + args.lambda_kl_max * kl).item() * x.shape[0]
                    n_seen += x.shape[0]
            val_elbo /= n_seen
            print(f"  [VAE convergence check @ epoch {epoch}] Val ELBO: {val_elbo:.5f}")
            if detector.step(val_elbo):
                print(f"  VAE converged at epoch {epoch} — switching to DDPM stage.")
                break
    progress.close()
    save_vae_weights(model, hparams, results_dir, args.run_name)
    return model

# ──────────────────────────────────────────────────────────────────────────────
# DDPM TRAINING STAGE
# ──────────────────────────────────────────────────────────────────────────────


def train_ddpm(
    args: argparse.Namespace,
    hparams: dict,
    vae: VAEWrapper,
    dataloader: DataLoader,
    val_loader: DataLoader,
    channels: int,
    img_size: int,
    is_3d: bool,
    data_config: dict,
    results_dir: str,
    device: torch.device,
    vae_epochs_done: int,
) -> TwoStageLDM:
    """
    Train the DDPM stage on top of the frozen pre-trained VAE.

    Args:
        args           (argparse.Namespace): CLI args
        hparams        (dict):               LDM arch hparams
        vae            (VAEWrapper):         trained VAE
        dataloader     (DataLoader):         training data loader
        val_loader     (DataLoader):         validation data loader
        channels       (int):                image channels
        img_size       (int):                spatial size per dimension
        is_3d          (bool):               whether input is volumetric
        data_config    (dict):               dataset config
        results_dir    (str):                output directory
        device         (torch.device):       target device
        vae_epochs_done(int):                VAE epochs trained (for logging)
    Returns:
        TwoStageLDM: trained model on device
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 — DDPM TRAINING")
    print("=" * 60)

    ldm = build_ldm(hparams, args, channels, img_size, device, is_3d=is_3d)

    ldm.latent_encoder.load_state_dict(vae.latent_encoder.state_dict())
    ldm.decoder.load_state_dict(vae.decoder.state_dict())
    print("  Loaded pre-trained VAE weights into LDM (encoder + decoder frozen).")

    optimizer = optim.AdamW(
        ldm.noise_predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_epochs": [],
        "fid_scores": [],
        "fid_epochs": [],
    }
    graph_path = os.path.join(results_dir, f"{args.run_name}_ddpm_training_curves.png")

    if args.mode == "fixed":
        max_epochs = args.total_epochs - vae_epochs_done
        detector = None
        print(
            f"  Fixed mode: {max_epochs} DDPM epochs "
            f"({args.total_epochs} total - {vae_epochs_done} VAE)."
        )
    else:
        max_epochs = args.ddpm_max_epochs
        detector = SmoothedPlateauDetector(
            patience=args.ddpm_patience, delta=args.ddpm_delta
        )

    steps_per_epoch = len(dataloader)
    val_cache = _build_val_noise_cache(val_loader, args.T, device)

    fid_check_epochs = set(max(1, int(f * max_epochs)) for f in args.fid_fractions)  # noqa: C401
    warmup_cutoff = int(0.25 * max_epochs)
    fid_check_epochs = {e for e in fid_check_epochs if e > warmup_cutoff}

    progress = tqdm(
        total=max_epochs * steps_per_epoch, desc="DDPM Training", unit="step"
    )

    for epoch in range(1, max_epochs + 1):
        ldm.train()
        running_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)
            if not is_3d and x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)
            B = x.shape[0]  # noqa: N806
            optimizer.zero_grad()

            with torch.no_grad():
                mu, logvar = ldm.latent_encoder(x)
                z0 = ldm.latent_encoder.reparameterize(mu, logvar)

            t = torch.randint(0, args.T, (B,), device=device)
            noise = torch.randn_like(z0)
            z_t = ldm.q_sample(z0, t, noise)

            t_norm = (t.float() / (args.T - 1)).unsqueeze(1)
            eps_pred = ldm.noise_predictor(z_t, t_norm)
            loss = ((eps_pred - noise) ** 2).mean()

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    ldm.noise_predictor.parameters(), args.grad_clip
                )
            optimizer.step()

            history["train_loss"].append(loss.item())
            running_loss += loss.item()

            progress.set_postfix(
                {
                    "epoch": f"{epoch}/{max_epochs}",
                    "train_MSE": f"{loss.item():.5f}",
                }
            )
            progress.update(1)

        epoch_loss = running_loss / steps_per_epoch
        print(f"  [DDPM epoch {epoch}] Train MSE: {epoch_loss:.5f}")

        if epoch % args.ddpm_check_every == 0:
            val_loss = compute_ddpm_val_loss(ldm, val_cache, device, channels=channels, img_size=img_size, is_3d=is_3d)
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            print(f"  [DDPM val @ epoch {epoch}] Val MSE: {val_loss:.5f}")

            if detector is not None and detector.step(val_loss):
                print(f"  DDPM converged at epoch {epoch} — stopping DDPM training.")
                save_ldm_checkpoint(
                    ldm, optimizer, epoch, history, results_dir, args.run_name
                )
                save_ddpm_training_graph(history, steps_per_epoch, epoch, graph_path)
                break

        if epoch in fid_check_epochs:
            fid_score = compute_fid(
                ldm, data_config, args.n_fid_samples, args.fid_batch_size, device
            )
            history["fid_scores"].append(fid_score)
            history["fid_epochs"].append(epoch)
            print(f"  [FID @ epoch {epoch}] Inception FID: {fid_score:.2f}")

        save_ldm_checkpoint(ldm, optimizer, epoch, history, results_dir, args.run_name)
        save_ddpm_training_graph(history, steps_per_epoch, epoch, graph_path)

    progress.close()
    save_ldm_weights(ldm, hparams, args, results_dir, args.run_name, is_3d=is_3d)
    return ldm
