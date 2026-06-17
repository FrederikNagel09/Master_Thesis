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

sys.path.append(".")
import warnings

from src.models.LatentEncoder import ResNetLatentEncoder
from src.models.trans_inr import TransInr, make_coord_grid
from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
    _mnist_features,
)
from src.utility.dataset_builders import build_dataset
from src.utility.metrics_util import _fid

warnings.filterwarnings("ignore", message="The operator 'aten::im2col'")

"""
python src/scripts/VAE_Baseline_Training.py \
    --run_name vae_testing \
    --ldm_config src/train_results/Latent-Diffusion-Probabilistic-1616/metadata/config.json \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 1.0 \
    --lambda_kl_max 1.0 \
    --n_fid_samples 8 \
    --fid_batch_size 8 

Resume:
python src/scripts/VAE_Baseline_Training.py \
    --run_name vae-cifar10-baseline \
    --ldm_config src/train_results/Latent-Diffusion-Deterministic/config.json \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --resume src/results/vae-cifar10-baseline/vae-cifar10-baseline_checkpoint.pt
"""

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────


def _print_decoder_info(decoder: TransInr) -> int:
    """
    Prints a parameter count summary for a TransInr decoder.
    Args:
        decoder : instantiated TransInr decoder
    Returns:
        total : total parameter count (int)
    """
    # ── Tokenizer breakdown ───────────────────────────────────────────────
    prefc_params = sum(p.numel() for p in decoder.tokenizer.prefc.parameters())
    posemb_params = decoder.tokenizer.posemb.numel()
    local_params = sum(p.numel() for p in decoder.tokenizer.local_attn.parameters())
    global_params = sum(p.numel() for p in decoder.tokenizer.global_attn.parameters())
    tok_params = sum(p.numel() for p in decoder.tokenizer.parameters())
    n_patches = decoder.tokenizer.posemb.shape[1]
    tok_dim = decoder.tokenizer.posemb.shape[2]

    # ── Transformer breakdown ─────────────────────────────────────────────
    cls_name = decoder.transformer.__class__.__name__
    trans_params = sum(p.numel() for p in decoder.transformer.parameters())
    if cls_name == "Transformer":
        enc_params = sum(p.numel() for p in decoder.transformer.encoder.parameters())
        dec_params = sum(p.numel() for p in decoder.transformer.decoder.parameters())
    else:
        enc_params = trans_params
        dec_params = None

    # ── Wtoken / INR breakdown ────────────────────────────────────────────
    n_wtokens = decoder.wtokens.shape[0]
    wtoken_dim = decoder.wtokens.shape[1]
    wtoken_params = decoder.wtokens.numel()
    postfc_params = sum(p.numel() for p in decoder.wtoken_postfc.parameters())
    base_params = sum(p.numel() for p in decoder.base_params.values())
    inr_params = sum(p.numel() for p in decoder.inr.parameters())
    total = sum(p.numel() for p in decoder.parameters())

    # ── Print: architecture stats ─────────────────────────────────────────
    print("############## Latent Decoder Summary: ##############")
    print("---- Architecture Stats ------------------------------")
    print(f"  Data tokens               : {n_patches:>6}   (dim={tok_dim})")
    print(f"  Weight tokens             : {n_wtokens:>6}   (dim={wtoken_dim})")

    # ── Print: parameter counts ───────────────────────────────────────────
    print("---- Parameters --------------------------------------")
    print(f"Tokenizer                   : {tok_params:>12,}")
    print(f"  Pre-FC                    : {prefc_params:>12,}")
    print(f"  Positional embedding      : {posemb_params:>12,}")
    print(f"  Local attention           : {local_params:>12,}")
    print(f"  Global attention          : {global_params:>12,}")
    print(f"Transformer                 : {trans_params:>12,}")
    if dec_params is not None:
        print(f"  Encoder                   : {enc_params:>12,}")
        print(f"  Decoder                   : {dec_params:>12,}")
    print(f"Weight tokens               : {wtoken_params:>12,}")
    print(f"Wtoken post-FC              : {postfc_params:>12,}")
    print(f"Base INR params             : {base_params:>12,}")
    print(f"SIREN (INR module)          : {inr_params:>12,}")
    print("--------------------------------------------------------------")
    print(f"Total                       : {total:>12,}")
    print("--------------------------------------------------------------")

    return total


def _print_latent_encoder_info(encoder: ResNetLatentEncoder) -> int:
    """
    Prints a parameter count summary for a ResNetLatentEncoder.
    Args:
        encoder : instantiated ResNetLatentEncoder
    Returns:
        total : total parameter count (int)
    """
    stem_params = sum(p.numel() for p in encoder.stem.parameters())
    layer1_params = sum(p.numel() for p in encoder.layer1.parameters())
    layer2_params = sum(p.numel() for p in encoder.layer2.parameters())
    layer3_params = sum(p.numel() for p in encoder.layer3.parameters())
    layer4_params = sum(p.numel() for p in encoder.layer4.parameters())
    backbone_params = layer1_params + layer2_params + layer3_params + layer4_params
    upsample_mu = sum(p.numel() for p in encoder.upsample_mu.parameters())
    upsample_logvar = sum(p.numel() for p in encoder.upsample_logvar.parameters())
    upsample_params = upsample_mu + upsample_logvar
    total = sum(p.numel() for p in encoder.parameters())

    print("############## Latent Encoder Summary: #############")
    print(f"Stem                   : {stem_params:>12,}")
    print(f"ResNet backbone        : {backbone_params:>12,}")
    print(f"  layer1               : {layer1_params:>12,}")
    print(f"  layer2               : {layer2_params:>12,}")
    print(f"  layer3               : {layer3_params:>12,}")
    print(f"  layer4               : {layer4_params:>12,}")
    print(f"Learnable upsample     : {upsample_params:>12,}")
    print("-------------------------------------------------------------")
    print(f"Total                  : {total:>12,}")
    print("-------------------------------------------------------------")

    return total


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments. Arch hyperparams are loaded from --ldm_config;
    only training-specific and run-specific args live here.

    Returns:
        argparse.Namespace: parsed arguments
    """
    p = argparse.ArgumentParser(description="Train a TransINR-VAE baseline model")

    # Run
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--ldm_config", type=str, required=True, help="Path to trained LDM config .json")
    p.add_argument("--results_dir", type=str, default="src/results")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--subset_frac", type=float, default=1.0)

    # KL
    p.add_argument("--lambda_kl_max", type=float, default=0.1)
    p.add_argument("--kl_warmup_frac", type=float, default=0.4)
    p.add_argument("--n_fid_samples", type=int, default=1000, help="Number of samples to generate for FID evaluation")
    p.add_argument("--fid_batch_size", type=int, default=64, help="Batch size to use during FID sample generation and feature extraction")

    # Resume
    p.add_argument("--resume", type=str, default=None, help="Path to VAE checkpoint .pt to resume from")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────


def load_ldm_config(path: str) -> dict:
    """
    Loads and returns the hparams block from a trained LDM config JSON.

    Args:
        path: path to the LDM config .json file
    Returns:
        hparams dict extracted from config["hparams"]
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
    ]
    hparams = config["hparams"]
    missing = [k for k in required_keys if k not in hparams]
    if missing:
        raise ValueError(f"LDM config is missing required keys: {missing}")

    return hparams


# ──────────────────────────────────────────────────────────────────────────────
# VAE SYSTEM WRAPPER
# ──────────────────────────────────────────────────────────────────────────────


class VAEWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, img_size: int, device: torch.device):
        super().__init__()
        self.latent_encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.device = device

        coord_grid = make_coord_grid((img_size, img_size), (-1, 1))
        self.register_buffer("coord_grid", coord_grid)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a latent tensor through the TransInr decoder."""
        batch_size = z.shape[0]
        coords = self.coord_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
        return self.decoder(z, coords)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.latent_encoder(x)
        z = self.latent_encoder.reparameterize(mu, logvar)
        x_recon = self._decode_latent(z)
        return x_recon, mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT SAVE / LOAD
# ──────────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch_reached: int,
    history: dict[str, list[float]],
    run_name: str,
    results_dir: str,
) -> None:
    """
    Saves a full training checkpoint (weights, optimizer, epoch, history).

    Args:
        model:          trained VAEWrapper
        optimizer:      optimizer at current state
        epoch_reached:  last fully completed global epoch index
        history:        per-step loss history dict with keys "elbo", "recon", "kl"
        run_name:       name of the run
        results_dir:    directory to save into
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
) -> tuple[int, dict[str, list[float]]]:
    """
    Loads a checkpoint into model and optimizer in-place.

    Args:
        path:      path to checkpoint .pt file
        model:     VAEWrapper instance (architecture must match checkpoint)
        optimizer: optimizer instance
        device:    device to map tensors onto
    Returns:
        (epoch_reached, history) — last completed global epoch and loss history
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch_reached = ckpt["epoch_reached"]
    history = ckpt["history"]
    print(f"Resumed from checkpoint: {path} (epoch {epoch_reached})")
    return epoch_reached, history


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────


def save_training_graph(
    history: dict[str, list[float]],
    steps_per_epoch: int,
    total_epochs_so_far: int,
    save_path: str,
    plot_every_n: int = 100,
) -> None:
    """
    Saves a 3-panel training graph (total ELBO, recon loss, KL loss).
    X-axis is in global epochs across the full training history.

    Args:
        history:              dict with keys "elbo", "recon", "kl", per-step values
        steps_per_epoch:      optimizer steps per epoch
        total_epochs_so_far:  total global epochs completed (across all runs)
        save_path:            full file path to save the .png
        plot_every_n:         plot every nth step to reduce noise and save time
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
        ("recon", "Reconstruction Loss", "tab:orange"),
        ("kl", "KL Loss", "tab:green"),
    ]

    for ax, (key, title, color) in zip(axes, panels):  # noqa: B905
        downsampled = history[key][::plot_every_n]
        ax.plot(range(len(downsampled)), downsampled, color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SAVING (weights-only export, separate from checkpoint)
# ──────────────────────────────────────────────────────────────────────────────


def save_model(
    model: nn.Module,
    hparams: dict,
    run_name: str,
    results_dir: str,
) -> None:
    """
    Saves final model weights (state_dict) and config (JSON) to results_dir.

    Args:
        model:       trained VAEWrapper
        hparams:     LDM hparams dict used to build the model
        run_name:    name of the run
        results_dir: directory to save into
    Returns:
        None
    """
    weights_path = os.path.join(results_dir, f"{run_name}_weights.pt")
    config_path = os.path.join(results_dir, f"{run_name}_config.json")

    torch.save(model.state_dict(), weights_path)

    # Save only the arch-relevant subset so it's self-contained
    arch_keys = [
        "dataset",
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
    ]
    config = {k: hparams[k] for k in arch_keys}
    config["run_name"] = run_name

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model weights saved to {weights_path}")
    print(f"Model config  saved to {config_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ──────────────────────────────────────────────────────────────────────────────


def build_model(
    hparams: dict,
    channels: int,
    img_size: int,
    device: torch.device,
) -> nn.Module:
    """
    Builds VAEWrapper (encoder + TransInr decoder) from LDM hparams.

    Args:
        hparams:   LDM hparams dict (from config["hparams"])
        channels:  number of image channels from dataset config
        img_size:  spatial image size from dataset config
        device:    torch device
    Returns:
        VAEWrapper: assembled and device-placed model
    """
    latent_dim = hparams["latent_dim"]
    latent_size = hparams["latent_size"]
    latent_size_tuple = (latent_size, latent_size)

    encoder = ResNetLatentEncoder(
        in_channels=channels,
        latent_dim=latent_dim,
        latent_size=latent_size_tuple,
        hidden_dim=hparams["latent_enc_hidden_dim"],
    )

    decoder = TransInr(
        tokenizer={
            "target": "src.models.trans_inr_helpers.LatentTokenizer",
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
            "target": "src.models.trans_inr_helpers.SIREN",
            "params": {
                "depth": hparams["inr_layers"],
                "in_dim": 2,
                "out_dim": channels,
                "hidden_dim": hparams["inr_hidden_dim"],
                "out_bias": 0.5,
            },
        },
        data_shape=(img_size, img_size),
        n_groups=hparams["dec_trans_n_groups"],
        transformer={
            "target": "src.models.trans_inr_helpers.Transformer",
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

    model = VAEWrapper(encoder, decoder, img_size, device).to(device)

    # Print model stats (captured by log file via stdout redirect)
    print("\n########## Decoder INR Parameter Breakdown: ##############")
    print(f"  {'Layer':<10} | {'Shape':>16}   {'Total':>8}")
    print(f"  {'─'*10}-+-{'─'*16}---{'─'*8}")
    inr_total = 0
    for name, shape in decoder.inr.param_shapes.items():
        total_els = shape[0] * shape[1]
        shape_str = f"{shape[0]}x{shape[1]}"
        print(f"  {name:<10} | {shape_str:>16}   {total_els:>8,}")
        inr_total += total_els
    print(f"  {'─'*10}-+-{'─'*16}---{'─'*8}")
    print(f"  {'TOTAL':<10} | {'':>16}   {inr_total:>8,}")
    print("############## Latent Space & INR Summary: #############")
    print(f"Latent variable (diffusion) : ({latent_dim}, {latent_size_tuple[0]}, {latent_size_tuple[1]})")
    print("________________________________________________________")
    print(f"latent dim: {latent_dim * latent_size_tuple[0] * latent_size_tuple[1]}")
    print(f"INR dim.  : {inr_total}")
    print("########################################################")
    print("\n########## Encoder Info: ##############")
    _print_latent_encoder_info(encoder)
    print("\n########## Decoder Info: ##############")
    _print_decoder_info(decoder)

    return model


def compute_eval_metrics(
    model: nn.Module,
    hparams: dict,
    val_loader: DataLoader,
    data_config: dict,
    n_fid_samples: int,
    device: torch.device,
    results_dir: str,
    run_name: str,
    total_epochs: int,
    fid_batch_size: int,
) -> None:
    """
    Computes and saves FID, class uniformity, and avg reconstruction MSE.
    MNIST: computes both MNIST-classifier FID and Inception FID.
    Other datasets: Inception FID only.

    Args:
        model:          trained VAEWrapper in eval mode
        hparams:        LDM hparams dict (for latent shape)
        val_loader:     DataLoader over the validation set
        data_config:    dict with "channels", "img_size", "dataset"
        n_fid_samples:  number of generated samples for FID
        device:         torch device
        results_dir:    directory to save metrics JSON into
        run_name:       run name used in output filename
        total_epochs:   total global epochs trained (for JSON metadata)
    Returns:
        None
    """
    import numpy as np

    channels = data_config["channels"]
    dataset_name = data_config.get("dataset", "mnist").lower()
    latent_dim = hparams["latent_dim"]
    latent_size = hparams["latent_size"]
    is_mnist = dataset_name == "mnist"

    # ── Batched generation of FID samples ─────────────────────────────────────
    print(f"  Generating {n_fid_samples} samples for FID …")
    all_samples = []
    batch_size = fid_batch_size
    remaining = n_fid_samples
    with torch.no_grad():
        while remaining > 0:
            n = min(batch_size, remaining)
            z = torch.randn(n, latent_dim, latent_size, latent_size).to(device)
            imgs = (model._decode_latent(z) * 0.5 + 0.5).clamp(0, 1)  # (N, C, H, W)
            all_samples.append(imgs.cpu())
            remaining -= n
    fid_tensor = torch.cat(all_samples, dim=0)  # (n_fid_samples, C, H, W)

    # ── FID ───────────────────────────────────────────────────────────────────
    inception = _get_inception(device)
    mnist_fid = None

    if is_mnist:
        print("  Computing MNIST classifier FID …")
        classifier = _load_classifier(device)
        real_mnist_feats, real_inception_feats, _ = _load_or_compute_real_features(classifier, inception, device)
        gen_mnist_feats, gen_preds = _mnist_features(fid_tensor, classifier, device)
        mnist_fid = float(_fid(real_mnist_feats, gen_mnist_feats))
    else:
        # Reuse inception real features; caller must ensure they exist for non-MNIST
        _, real_inception_feats, _ = _load_or_compute_real_features(None, inception, device)
        gen_preds = None  # no classifier for non-MNIST

    print("  Computing Inception FID …")
    gen_inception_feats = _inception_features(fid_tensor, inception, device)
    inception_fid = float(_fid(real_inception_feats, gen_inception_feats))

    # ── Uniformity (normalized entropy over predicted classes) ────────────────
    uniformity_score = None
    class_breakdown = None
    if gen_preds is not None:
        print("  Computing class uniformity …")
        n_classes = 10
        predicted_classes = torch.from_numpy(gen_preds)
        class_counts = torch.bincount(predicted_classes, minlength=n_classes).float()
        class_probs = class_counts / class_counts.sum()
        entropy = -(class_probs * (class_probs + 1e-8).log()).sum()
        uniformity_score = float(entropy / np.log(n_classes))
        class_breakdown = {str(i): int(class_counts[i].item()) for i in range(n_classes)}

    # ── Avg reconstruction MSE over validation set ────────────────────────────
    print("  Computing validation reconstruction MSE …")
    total_mse = 0.0
    n_seen = 0
    img_size = data_config["img_size"]
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)
            x_recon, _, _ = model(x)
            x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)
            x_hat_flat = x_recon.reshape(x.shape[0], -1)
            total_mse += ((x_flat - x_hat_flat) ** 2).sum(dim=-1).sum().item()
            n_seen += x.shape[0]
    avg_val_mse = total_mse / n_seen

    # ── Print + save ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 45}")
    print(f"  Eval Summary  —  {run_name} ({total_epochs} epochs)")
    print(f"{'=' * 45}")
    print(f"  Avg Val MSE   : {avg_val_mse:.6f}  (n={n_seen})")
    if mnist_fid is not None:
        print(f"  MNIST FID     : {mnist_fid:.2f}")
    print(f"  Inception FID : {inception_fid:.2f}")
    if uniformity_score is not None:
        print(f"  Uniformity    : {uniformity_score:.4f}  (0=collapsed, 1=uniform)")
    print(f"{'=' * 45}\n")

    metrics = {
        "run_name": run_name,
        "total_epochs": total_epochs,
        "n_val_samples": n_seen,
        "avg_val_recon_mse": avg_val_mse,
        "mnist_fid": mnist_fid,
        "inception_fid": inception_fid,
        "uniformity_score": uniformity_score,
        "class_breakdown": class_breakdown,
    }
    metrics_path = os.path.join(results_dir, f"{run_name}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Eval metrics saved → {metrics_path}")

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
# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────


def run_training(args: argparse.Namespace) -> None:
    """
    Main training loop.

    Args:
        args: parsed CLI arguments
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(f"--- Initialization Process Started: {args.run_name} ---")

    # 1. Load LDM config and derive dataset name
    hparams = load_ldm_config(args.ldm_config)
    print(f"Loaded LDM config from: {args.ldm_config}")
    print(f"Dataset from config: {hparams['dataset']}")

    # 2. Dataset
    dataset, val_dataset, data_config = build_dataset(
        dataset_name=hparams["dataset"],
        data_root="data/",
        subset_frac=args.subset_frac,
        single_class=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    channels = data_config["channels"]
    img_size = data_config["img_size"]

    # 3. Build model
    model = build_model(hparams, channels, img_size, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. Resume or fresh start
    if args.resume:
        epoch_offset, history = load_checkpoint(args.resume, model, optimizer, device)
    else:
        epoch_offset = 0
        history = {"elbo": [], "recon": [], "kl": []}

    total_epochs_planned = epoch_offset + args.epochs
    # KL warmup is relative to the total training budget across all runs
    kl_warmup_epochs = max(1, int(args.kl_warmup_frac * (total_epochs_planned)))

    results_dir = os.path.join(args.results_dir, args.run_name)
    # Clear existing run folder if starting fresh (not resuming)
    if not args.resume and os.path.exists(results_dir):
        import shutil

        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)
    graph_path = os.path.join(results_dir, f"{args.run_name}_training_curves.png")

    print(
        f"Training on {device} | {args.epochs} new epochs "
        f"(global {epoch_offset + 1} → {total_epochs_planned}) | "
        f"KL warm-up over first {kl_warmup_epochs} global epochs"
    )

    total_steps = args.epochs * len(dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training", unit="step")
    
    # two-stage training control variables (for applicable models)
    lambda_kl = args.lambda_kl_max
    min_stage1_steps = 50000
    kl_warmup_steps = 30000
    beta = 0.0
    global_step = 0

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        global_epoch = epoch_offset + epoch
        model.train()

        running_mse = 0.0
        running_kl = 0.0

        for batch in dataloader:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)

            optimizer.zero_grad()

            beta = _get_beta(global_step, lambda_kl, kl_warmup_steps)

            x_recon, mu, logvar = model(x)

            x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
            x_flat = x.reshape(x.shape[0], -1).clamp(-1, 1)

            loss_recon = 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1).mean()
            loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]))
            total_loss = loss_recon + beta * loss_kl

            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            progress_bar.set_postfix(
                {
                    "epoch": f"{global_epoch}/{total_epochs_planned}",
                    "MSE": f"{loss_recon.item():.4f}",
                    "KL": f"{loss_kl.item():.2f}",
                    "β": f"{beta:.3f}",
                }
            )
            progress_bar.update(1)

            history["elbo"].append(total_loss.item())
            history["recon"].append(loss_recon.item())
            history["kl"].append(loss_kl.item())

            running_mse += loss_recon.item()
            running_kl += loss_kl.item()
            global_step += 1

        epoch_mse = running_mse / len(dataloader)
        epoch_kl = running_kl / len(dataloader)
        print(f"      ↳ [Summary] Avg MSE: {epoch_mse:.5f} | Avg KL: {epoch_kl:.3f} | β: {beta:.4f}")

        save_checkpoint(model, optimizer, global_epoch, history, args.run_name, results_dir)
        save_training_graph(history, len(dataloader), global_epoch, graph_path)
    progress_bar.close()
    # 6. Final artefacts
    save_model(model, hparams, args.run_name, results_dir)

    import torchvision.utils as vutils

    model.eval()
    with torch.no_grad():
        # ── Sample grid (8x8 = 64 samples) ───────────────────────────────────
        z = torch.randn(64, hparams["latent_dim"], hparams["latent_size"], hparams["latent_size"]).to(device)
        samples = (model._decode_latent(z) * 0.5 + 0.5).clamp(0, 1)
        vutils.save_image(samples, os.path.join(results_dir, f"{args.run_name}_samples_8x8.png"), nrow=8, padding=2)

        # ── Row plots (10, 8, 6 samples) ─────────────────────────────────────
        for n_samples in [10, 8, 6]:
            z = torch.randn(n_samples, hparams["latent_dim"], hparams["latent_size"], hparams["latent_size"]).to(device)
            row = (model._decode_latent(z) * 0.5 + 0.5).clamp(0, 1)
            vutils.save_image(row, os.path.join(results_dir, f"{args.run_name}_samples_row{n_samples}.png"), nrow=n_samples, padding=2)

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        compute_eval_metrics(
            model,
            hparams,
            val_loader,
            data_config,
            n_fid_samples=args.n_fid_samples,
            fid_batch_size=args.fid_batch_size,
            device=device,
            results_dir=results_dir,
            run_name=args.run_name,
            total_epochs=total_epochs_planned,
        )


def main() -> None:
    args = parse_args()

    os.makedirs("src/logs", exist_ok=True)
    log_file_path = f"src/logs/{args.run_name}.log"
    log_file = open(log_file_path, "w")  # noqa: SIM115
    sys.stdout = log_file

    try:
        run_training(args)
    finally:
        log_file.close()
        sys.stdout = sys.__stdout__
        print(f"Training complete. Log saved to {log_file_path}")


if __name__ == "__main__":
    main()
