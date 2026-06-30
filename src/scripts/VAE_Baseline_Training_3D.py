"""
VAE_Baseline_Training_3D.py
Trains a TransINR-VAE baseline on 3D voxel data (ShapeNet).
Uses Conv3DEncoder instead of ResNetLatentEncoder, MSE reconstruction loss
without clamping (voxels are binary [0,1]), and MMD/COV instead of FID.

Usage
-----
python src/scripts/VAE_Baseline_Training_3D.py \
    --run_name vae_3d_baseline \
    --ldm_config src/train_results/latent-probability-3D-data/metadata/config.json \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 1.0 \
    --lambda_kl_max 1.0 \
    --kl_warmup_frac 0.4 \
    --n_eval_samples 128 \
    --eval_batch_size 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(".")

from src.models.latent_diffusion.modules.LatentEncoder3D import Conv3DEncoder
from src.models.latent_diffusion.modules.trans_inr import TransInr, make_coord_grid
from src.utility.dataset_builders import build_dataset


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments. Arch hyperparams are loaded from --ldm_config.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(description="Train a 3D TransINR-VAE baseline model")

    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--ldm_config", type=str, required=True, help="Path to LDM config .json")
    p.add_argument("--results_dir", type=str, default="src/results")

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--subset_frac", type=float, default=1.0)

    p.add_argument("--lambda_kl_max", type=float, default=0.1)
    p.add_argument("--kl_warmup_frac", type=float, default=0.4)

    p.add_argument("--n_eval_samples", type=int, default=256, help="Samples to generate for MMD/COV eval")
    p.add_argument("--eval_batch_size", type=int, default=16)

    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")

    # Conv3DEncoder-specific hparams (can override config defaults)
    p.add_argument("--enc_base_channels", type=int, default=64)
    p.add_argument("--enc_dropout", type=float, default=0.0)

    return p.parse_args()


# ── Config loading ─────────────────────────────────────────────────────────────
def load_ldm_config(path: str) -> dict:
    """
    Load and return the hparams block from a trained LDM config JSON.

    Args:
        path: Path to the LDM config .json file.
    Returns:
        hparams dict extracted from config["hparams"].
    """
    with open(path) as f:
        config = json.load(f)

    required_keys = [
        "latent_dim",
        "latent_size",
        "latent_patch_size",
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
    ]
    hparams = config["hparams"]
    missing = [k for k in required_keys if k not in hparams]
    if missing:
        raise ValueError(f"LDM config is missing required keys: {missing}")
    return hparams


# ── VAE wrapper ────────────────────────────────────────────────────────────────
class VAEWrapper3D(nn.Module):
    """
    Wraps Conv3DEncoder + TransInr decoder into a 3D VAE.
    The coord grid is (D, H, W, 3) and is stored as a non-persistent buffer
    since it depends on resolution and is rebuilt at decode time.
    """

    def __init__(
        self,
        encoder: Conv3DEncoder,
        decoder: TransInr,
        img_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.latent_encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.device = device
        coord_grid = make_coord_grid((img_size, img_size, img_size), (-1, 1))
        self.register_buffer("coord_grid", coord_grid)  # (D, H, W, 3)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z at native resolution using the stored coord grid.

        Args:
            z: (B, latent_dim, latent_size, latent_size) latent tensor.
        Returns:
            x_recon: (B, C, D, H, W) reconstructed voxels.
        """
        B = z.shape[0]  # noqa: N806
        coords = self.coord_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        return self.decoder(z, coords)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space (deterministic during eval via reparameterize).

        Args:
            x: (B, C, D, H, W) input voxels.
        Returns:
            z:      (B, latent_dim, latent_size, latent_size) latent sample.
            mu:     Posterior mean, same shape as z.
            logvar: Posterior log-variance, same shape as z.
        """
        mu, logvar = self.latent_encoder(x)
        z = self.latent_encoder.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward: encode → reparameterize → decode.

        Args:
            x: (B, C, D, H, W) input voxels.
        Returns:
            x_recon: (B, C, D, H, W) reconstructed voxels.
            mu:      Posterior mean.
            logvar:  Posterior log-variance.
        """
        z, mu, logvar = self.encode(x)
        x_recon = self._decode_latent(z)
        return x_recon, mu, logvar


# ── Model builder ──────────────────────────────────────────────────────────────
def build_model(
    hparams: dict,
    channels: int,
    img_size: int,
    device: torch.device,
    enc_base_channels: int = 64,
    enc_dropout: float = 0.0,
) -> VAEWrapper3D:
    """
    Build VAEWrapper3D (Conv3DEncoder + TransInr decoder) from LDM hparams.

    Args:
        hparams:          LDM hparams dict.
        channels:         Number of voxel channels.
        img_size:         Spatial size per dimension (D=H=W).
        device:           Target device.
        enc_base_channels: Base channel count for Conv3DEncoder.
        enc_dropout:      Dropout rate for Conv3DEncoder.
    Returns:
        VAEWrapper3D on device.
    """
    latent_dim = hparams["latent_dim"]
    latent_size = hparams["latent_size"]

    encoder = Conv3DEncoder(
        in_channels=channels,
        dim_z=latent_dim,
        base_channels=enc_base_channels,
        dropout=enc_dropout,
    )

    decoder = TransInr(
        tokenizer={
            "target": "src.models.tokenizers.latent_tokenizer.LatentTokenizer",
            "params": {
                "latent_dim": latent_dim,
                "latent_size": latent_size,
                "patch_size": hparams["latent_patch_size"],
                "dim": hparams["dec_trans_dim"],
                "n_head": hparams["dec_trans_n_head"],
                "head_dim": hparams["dec_trans_head_dim"],
            },
        },
        inr={
            "target": "src.models.inr.siren.SIREN",
            "params": {
                "depth": hparams["inr_layers"],
                "in_dim": 3,                 # 3D coords
                "out_dim": channels,
                "hidden_dim": hparams["inr_hidden_dim"],
                "out_bias": 0.5,
                "out_activation": "sigmoid", # voxels are binary [0,1]
            },
        },
        data_shape=(img_size, img_size, img_size),
        n_groups=hparams["dec_trans_n_groups"],
        transformer={
            "target": "src.models.utils.transformer.Transformer",
            "params": {
                "dim": hparams["dec_trans_dim"],
                "encoder_depth": hparams["dec_trans_enc_depth"],
                "decoder_depth": hparams["dec_trans_dec_depth"],
                "n_head": hparams["dec_trans_n_head"],
                "head_dim": hparams["dec_trans_head_dim"],
                "ff_dim": hparams["dec_trans_ff_dim"],
            },
        },
        update_strategy=hparams["dec_trans_update_strategy"],
    )

    model = VAEWrapper3D(encoder, decoder, img_size, device).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Conv3DEncoder params : {enc_params:,}")
    print(f"  TransInr decoder params : {dec_params:,}")
    print(f"  Total params : {enc_params + dec_params:,}")

    return model


# ── KL annealing ───────────────────────────────────────────────────────────────
def _get_beta(global_step: int, beta_final: float, warmup_steps: int) -> float:
    """
    Linearly ramp beta from 0 to beta_final over warmup_steps.

    Args:
        global_step:  Current training step.
        beta_final:   Target KL weight.
        warmup_steps: Steps to reach beta_final.
    Returns:
        float: Current beta value.
    """
    return beta_final * min(1.0, global_step / max(1, warmup_steps))


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch_reached: int,
    history: dict,
    run_name: str,
    results_dir: str,
) -> None:
    """
    Save full training checkpoint (weights, optimizer state, epoch, history).

    Args:
        model:          VAEWrapper3D.
        optimizer:      Optimizer at current state.
        epoch_reached:  Last fully completed epoch.
        history:        Per-step loss history dict.
        run_name:       Run identifier.
        results_dir:    Output directory.
    Returns:
        None
    """
    path = os.path.join(results_dir, f"{run_name}_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_reached": epoch_reached,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict]:
    """
    Load a checkpoint into model and optimizer in-place.

    Args:
        path:      Path to checkpoint .pt file.
        model:     VAEWrapper3D (architecture must match).
        optimizer: Optimizer instance.
        device:    Device to map tensors onto.
    Returns:
        (epoch_reached, history): Last completed epoch and loss history.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch_reached = ckpt["epoch_reached"]
    history = ckpt["history"]
    print(f"  Resumed from checkpoint: {path} (epoch {epoch_reached})")
    return epoch_reached, history


def save_model(
    model: nn.Module,
    hparams: dict,
    run_name: str,
    results_dir: str,
    enc_base_channels: int,
    enc_dropout: float,
) -> None:
    """
    Save final model weights and config JSON.

    Args:
        model:            Trained VAEWrapper3D.
        hparams:          LDM hparams dict.
        run_name:         Run identifier.
        results_dir:      Output directory.
        enc_base_channels: Saved to config for reproducibility.
        enc_dropout:      Saved to config for reproducibility.
    Returns:
        None
    """
    weights_path = os.path.join(results_dir, f"{run_name}_weights.pt")
    config_path = os.path.join(results_dir, f"{run_name}_config.json")

    torch.save({"model_state_dict": model.state_dict()}, weights_path)

    arch_keys = [
        "dataset",
        "latent_dim",
        "latent_size",
        "latent_patch_size",
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
    ]
    config = {k: hparams[k] for k in arch_keys if k in hparams}
    config["run_name"] = run_name
    config["enc_base_channels"] = enc_base_channels
    config["enc_dropout"] = enc_dropout
    config["is_3d"] = True

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Model weights saved → {weights_path}")
    print(f"  Model config  saved → {config_path}")


# ── Training curves ────────────────────────────────────────────────────────────
def save_training_graph(
    history: dict,
    steps_per_epoch: int,
    total_epochs_so_far: int,
    save_path: str,
    plot_every_n: int = 50,
) -> None:
    """
    Save a 3-panel training graph (ELBO, recon loss, KL loss).

    Args:
        history:             Dict with keys "elbo", "recon", "kl".
        steps_per_epoch:     Optimizer steps per epoch.
        total_epochs_so_far: Total epochs completed.
        save_path:           Output .png path.
        plot_every_n:        Downsample factor for plotting.
    Returns:
        None
    """
    max_ticks = 10
    step = max(1, total_epochs_so_far // max_ticks)
    tick_positions = [i * steps_per_epoch // plot_every_n for i in range(0, total_epochs_so_far + 1, step)]
    tick_labels = [str(i) for i in range(0, total_epochs_so_far + 1, step)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("elbo", "Total ELBO", "tab:blue"),
        ("recon", "Reconstruction Loss (MSE)", "tab:orange"),
        ("kl", "KL Loss", "tab:green"),
    ]
    for ax, (key, title, color) in zip(axes, panels, strict=False):
        downsampled = history[key][::plot_every_n]
        ax.plot(range(len(downsampled)), downsampled, color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("3D VAE Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Final evaluation: recon MSE + MMD/COV ─────────────────────────────────────
def compute_eval_metrics(
    model: VAEWrapper3D,
    hparams: dict,
    val_loader: DataLoader,
    data_config: dict,
    n_eval_samples: int,
    eval_batch_size: int,
    device: torch.device,
    results_dir: str,
    run_name: str,
    total_epochs: int,
) -> None:
    """
    Compute and save reconstruction MSE + MMD/COV for 3D voxel generation.

    Args:
        model:           Trained VAEWrapper3D in eval mode.
        hparams:         LDM hparams dict (for latent shape).
        val_loader:      DataLoader over the validation set.
        data_config:     Dict with "channels", "img_size".
        n_eval_samples:  Number of prior samples for MMD/COV.
        eval_batch_size: Generation batch size.
        device:          Target device.
        results_dir:     Output directory.
        run_name:        Run identifier.
        total_epochs:    Total epochs trained.
    Returns:
        None
    """
    from src.utility.voxel_metrics import compute_mmd_cov

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    latent_dim = hparams["latent_dim"]
    latent_size = hparams["latent_size"]

    model.eval()

    # ── Reconstruction MSE on validation set ──────────────────────────────────
    print("  Computing validation reconstruction MSE ...")
    total_mse, n_seen = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size, img_size)
            x_recon, _, _ = model(x)
            total_mse += ((x.reshape(x.shape[0], -1) - x_recon.reshape(x.shape[0], -1)) ** 2).sum(dim=-1).sum().item()
            n_seen += x.shape[0]
    avg_val_mse = total_mse / n_seen

    # ── Generate samples for MMD/COV ──────────────────────────────────────────
    print(f"  Generating {n_eval_samples} prior samples for MMD/COV ...")
    all_samples = []
    remaining = n_eval_samples
    with torch.no_grad():
        while remaining > 0:
            n = min(eval_batch_size, remaining)
            z = torch.randn(n, latent_dim, latent_size, latent_size, device=device)
            x_hat = model._decode_latent(z)
            x_hat = x_hat.reshape(n, channels, img_size, img_size, img_size)
            all_samples.append(x_hat.cpu())
            remaining -= n
    generated = torch.cat(all_samples, dim=0)

    # Reference: full validation set
    ref_batches = [batch[0] for batch in val_loader]
    reference = torch.cat(ref_batches, dim=0)
    if reference.dim() == 2:
        reference = reference.view(reference.shape[0], channels, img_size, img_size, img_size)

    print(f"  Computing MMD/COV ({generated.shape[0]} gen vs {reference.shape[0]} ref) ...")
    mmd, cov = compute_mmd_cov(generated, reference)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 45}")
    print(f"  Eval Summary  —  {run_name} ({total_epochs} epochs)")
    print(f"{'=' * 45}")
    print(f"  Avg Val MSE : {avg_val_mse:.6f}  (n={n_seen})")
    print(f"  MMD         : {mmd:.4f}")
    print(f"  COV         : {cov:.4f}")
    print(f"{'=' * 45}\n")

    metrics = {
        "run_name": run_name,
        "total_epochs": total_epochs,
        "n_val_samples": n_seen,
        "avg_val_recon_mse": avg_val_mse,
        "mmd": mmd,
        "cov": cov,
    }
    metrics_path = os.path.join(results_dir, f"{run_name}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Eval metrics saved → {metrics_path}")


# ── Main training loop ─────────────────────────────────────────────────────────
def run_training(args: argparse.Namespace) -> None:
    """
    Full training orchestration: build → train → checkpoint → eval.

    Args:
        args: Parsed CLI arguments.
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 3D VAE Training: {args.run_name} | device={device} ---")

    hparams = load_ldm_config(args.ldm_config)
    print(f"Dataset: {hparams['dataset']}")

    dataset, val_dataset, data_config = build_dataset(
        dataset_name=hparams["dataset"],
        data_root="data/",
        subset_frac=args.subset_frac,
        single_class=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    channels = data_config["channels"]
    img_size = data_config["img_size"]

    model = build_model(hparams, channels, img_size, device, args.enc_base_channels, args.enc_dropout)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        epoch_offset, history = load_checkpoint(args.resume, model, optimizer, device)
    else:
        epoch_offset = 0
        history = {"elbo": [], "recon": [], "kl": []}

    total_epochs_planned = epoch_offset + args.epochs
    kl_warmup_steps = max(1, int(args.kl_warmup_frac * total_epochs_planned)) * len(dataloader)

    results_dir = os.path.join(args.results_dir, args.run_name)
    if not args.resume and os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    graph_path = os.path.join(results_dir, f"{args.run_name}_training_curves.png")

    print(f"  {args.epochs} new epochs (global {epoch_offset + 1} → {total_epochs_planned})")
    print(f"  KL warmup over {kl_warmup_steps} steps")

    global_step = 0
    steps_per_epoch = len(dataloader)
    progress_bar = tqdm(total=args.epochs * steps_per_epoch, desc="Training", unit="step")

    for epoch in range(1, args.epochs + 1):
        global_epoch = epoch_offset + epoch
        model.train()
        running_mse = 0.0
        running_kl = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            # Ensure (B, C, D, H, W)
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size, img_size)

            optimizer.zero_grad()
            beta = _get_beta(global_step, args.lambda_kl_max, kl_warmup_steps)

            x_recon, mu, logvar = model(x)

            # BCE loss for binary voxel occupancy
            x_flat = x.reshape(x.shape[0], -1)
            x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
            eps = 1e-7
            x_hat_flat = x_hat_flat.clamp(eps, 1 - eps)
            loss_recon = F.binary_cross_entropy(x_hat_flat, x_flat, reduction="none").sum(dim=-1).mean()

            loss_kl = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
            )
            total_loss = loss_recon + beta * loss_kl
            total_loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            history["elbo"].append(total_loss.item())
            history["recon"].append(loss_recon.item())
            history["kl"].append(loss_kl.item())
            running_mse += loss_recon.item()
            running_kl += loss_kl.item()
            global_step += 1

            progress_bar.set_postfix({
                "epoch": f"{global_epoch}/{total_epochs_planned}",
                "MSE": f"{loss_recon.item():.4f}",
                "KL": f"{loss_kl.item():.2f}",
                "β": f"{beta:.3f}",
            })
            progress_bar.update(1)

        epoch_mse = running_mse / steps_per_epoch
        epoch_kl = running_kl / steps_per_epoch
        print(f"  [epoch {global_epoch}] MSE: {epoch_mse:.5f} | KL: {epoch_kl:.3f} | β: {beta:.4f}")

        save_checkpoint(model, optimizer, global_epoch, history, args.run_name, results_dir)
        save_training_graph(history, steps_per_epoch, global_epoch, graph_path)

    progress_bar.close()

    save_model(model, hparams, args.run_name, results_dir, args.enc_base_channels, args.enc_dropout)

    compute_eval_metrics(
        model=model,
        hparams=hparams,
        val_loader=val_loader,
        data_config=data_config,
        n_eval_samples=args.n_eval_samples,
        eval_batch_size=args.eval_batch_size,
        device=device,
        results_dir=results_dir,
        run_name=args.run_name,
        total_epochs=total_epochs_planned,
    )


def main() -> None:
    args = parse_args()

    os.makedirs("src/logs", exist_ok=True)
    log_path = f"src/logs/{args.run_name}.log"
    log_file = open(log_path, "w")  # noqa: SIM115
    sys.stdout = log_file
    try:
        run_training(args)
    finally:
        log_file.close()
        sys.stdout = sys.__stdout__
        print(f"Training complete. Log saved to {log_path}")


if __name__ == "__main__":
    main()