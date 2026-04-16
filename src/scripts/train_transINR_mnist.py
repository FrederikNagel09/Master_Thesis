"""
train_mnist.py
==============
Training script for TransInr on MNIST.

All hyperparameters are declared at the top of this file under
the CONFIG section — no command-line parsing needed.
"""

import math
import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T  # noqa: N812
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore", message="The operator 'aten::im2col'")

sys.path.append(".")

# Local modules (must be in the same directory / on PYTHONPATH)
from src.models.trans_inr import TransInr  # noqa: E402

# =============================================================================
#  CONFIG — edit everything here
# =============================================================================

# ---------- Data -------------------------------------------------------------
DATA_ROOT = "./data"  # Where MNIST will be downloaded
SUBSET_SIZE = None  # int → use only N training samples;
# None → use the full 60 000

# ---------- Training ---------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 2000  # linear warm-up before cosine decay
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0  # max grad norm; 0 to disable
SEED = 42

# ---------- Logging / Checkpointing ------------------------------------------
LOG_EVERY = 50  # log loss every N batches
EVAL_EVERY = 1  # run validation every N epochs
SAVE_EVERY = 5  # save checkpoint every N epochs
CHECKPOINT_DIR = "./src"
RESUME_CKPT = None  # path to checkpoint to resume from, or None

# ---------- Model — Tokenizer ------------------------------------------------
IMAGE_SIZE = 28  # MNIST is 28x28
IN_CHANNELS = 1  # grayscale
PATCH_SIZE = 4  # 4x4 patches → (28/4)² = 49 tokens
# NOTE: IMAGE_SIZE must be divisible by PATCH_SIZE

# ---------- Model — VAE ------------------------------------------------------
LATENT_DIM = 256  # Size of the bottleneck z
KL_WEIGHT = 0.0001  # Hyperparameter to balance MSE and KL divergence

# ---------- Model — Transformer ----------------------------------------------
DIM = 256
N_HEAD = 8
HEAD_DIM = 32
FF_DIM = 512
ENCODER_DEPTH = 4
DECODER_DEPTH = 4
DROPOUT = 0.0
UPDATE_STRATEGY = "scale"  # "normalize" | "scale" | "identity"

# ---------- Model — SIREN INR ------------------------------------------------
SIREN_HIDDEN_DIM = 256
SIREN_DEPTH = 5
SIREN_IN_DIM = 2  # (x, y) coordinates
SIREN_OUT_DIM = 1  # predict grayscale pixel intensity
SIREN_OMEGA = 30.0

# ---------- Weight-token groups ----------------------------------------------
N_GROUPS = 8  # number of wtokens per INR layer parameter
# New Config for the Latent Tokenizer
LATENT_CHAN = 16  # Dimension of the latent space (z channels)
LATENT_RES = 16  # Resolution of the latent grid (16x16)

# =============================================================================
#  END CONFIG
# =============================================================================
import torch.nn.functional as F  # noqa: E402, N812


class MNISTToLatentEncoder(nn.Module):
    def __init__(self, latent_chan=16, latent_res=16):
        super().__init__()
        self.latent_res = latent_res
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(64,128, 3, stride=1, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # 14x14
            nn.ReLU(),
        )
        self.fc_mu = nn.Conv2d(64, latent_chan, 1)
        self.fc_logvar = nn.Conv2d(64, latent_chan, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.enc(x)
        # Force the resolution to match LATENT_RES (16x16)
        features = F.interpolate(features, size=(self.latent_res, self.latent_res), mode="bilinear")

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class TransInrVAE(nn.Module):
    def __init__(self, encoder, trans_inr, latent_dim, transformer_dim):
        super().__init__()
        self.encoder = encoder
        self.trans_inr = trans_inr
        self.latent_to_tokens = nn.Linear(latent_dim, transformer_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        # Map z to the embedding space of your Transformer
        # We treat z as a "global context token" or a seed for the INR
        z_embed = self.latent_to_tokens(z).unsqueeze(1)

        # Pass to your existing model
        # Note: You may need to adjust your TransInr.forward to accept
        # these embeddings instead of raw images.
        return self.trans_inr(z_embed), mu, logvar


@torch.no_grad()
def plot_reconstructions(model, encoder, dataset, device, n=5):
    encoder.eval()
    model.eval()
    indices = random.sample(range(len(dataset)), n * n)

    # Keep on device for the forward pass
    raw_images = torch.stack([dataset[i][0] for i in indices]).to(device)

    # Pass through VAE
    z, _, _ = encoder(raw_images)
    recons = model(z).cpu()  # Move result to CPU
    images = raw_images.cpu()  # Move originals to CPU for plotting

    _fig, axes = plt.subplots(n, n * 2 + 1, figsize=(n * 2 * 1.5 + 1, n * 1.5))

    for row in range(n):
        for col in range(n):
            i = row * n + col
            # images[i, 0] is now a CPU tensor, so .numpy() (used by imshow) will work
            axes[row, col].imshow(images[i, 0], cmap="gray", vmin=-1, vmax=1)
            axes[row, col].axis("off")

            axes[row, col + n + 1].imshow(recons[i, 0], cmap="gray", vmin=-1, vmax=1)
            axes[row, col + n + 1].axis("off")

    # Divider column
    for row in range(n):
        axes[row, n].axis("off")

    # Titles centered over each grid
    axes[0, n // 2].set_title("Originals", fontsize=12, fontweight="bold")
    axes[0, n + 1 + n // 2].set_title("Reconstructions", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("src/results/trans_inr_reconstructions.png", dpi=150)
    print("[plot] Saved → reconstructions.png")
    model.train()


def print_model_stats(model):
    def count(params):
        return sum(p.numel() for p in params)

    tokenizer_p = count(model.tokenizer.parameters())
    transformer_p = count(model.transformer.parameters())
    base_p = count(model.base_params.values())
    wtoken_p = model.wtokens.numel()
    postfc_p = count(model.wtoken_postfc.parameters())
    inr_layers = list(model.inr.param_shapes.items())

    print("\n" + "=" * 50)
    print("  Model Statistics")
    print("=" * 50)
    print(f"  Tokenizer:          {tokenizer_p:>10,} params")
    print(f"  Transformer:        {transformer_p:>10,} params")
    print(f"  Base INR params:    {base_p:>10,} params")
    print(f"  Weight tokens:      {wtoken_p:>10,} params")
    print(f"  Wtoken post-fc:     {postfc_p:>10,} params")
    print(f"  {'─'*38}")
    print(f"  Total:              {tokenizer_p+transformer_p+base_p+wtoken_p+postfc_p:>10,} params")
    print("\n  INR architecture (SIREN):")
    for name, shape in inr_layers:
        print(f"    {name}: {shape[0]-1} → {shape[1]}  ({(shape[0])*shape[1]:,} params incl. bias)")
    print(f"  N weight-token groups: {model.wtokens.shape[0]}")
    print("=" * 50 + "\n")


def build_model(device):
    """Construct TransInr from the CONFIG constants above."""

    tokenizer_cfg = {
        "target": "src.models.trans_inr_helpers.LatentTokenizer",  # Update this path
        "params": {
            "latent_dim": LATENT_CHAN,
            "latent_size": LATENT_RES,
            "patch_size": 2,  # 14/2 = 7x7 tokens
            "n_head": N_HEAD,
            "head_dim": HEAD_DIM,
        },
    }

    inr_cfg = {
        "target": "src.models.trans_inr_helpers.SIREN",
        "params": {
            "depth": SIREN_DEPTH,
            "in_dim": SIREN_IN_DIM,
            "out_dim": SIREN_OUT_DIM,
            "hidden_dim": SIREN_HIDDEN_DIM,
            "omega": SIREN_OMEGA,
            "out_bias": 0,
        },
    }

    transformer_cfg = {
        "target": "src.models.trans_inr_helpers.Transformer",
        "params": {
            "dim": DIM,
            "encoder_depth": ENCODER_DEPTH,
            "decoder_depth": DECODER_DEPTH,
            "n_head": N_HEAD,
            "head_dim": HEAD_DIM,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
        },
    }

    model = TransInr(
        tokenizer=tokenizer_cfg,
        inr=inr_cfg,
        n_groups=N_GROUPS,
        data_shape=(IMAGE_SIZE, IMAGE_SIZE),
        transformer=transformer_cfg,
        update_strategy=UPDATE_STRATEGY,
    ).to(device)

    return model


def build_dataloaders():
    """Return train and validation DataLoaders for MNIST."""

    # Normalise to [-1, 1] to match the coordinate range the SIREN will output
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),  # → [-1, 1]
        ]
    )

    train_full = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    val_full = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    # --- optional subset ---
    if SUBSET_SIZE is not None:
        indices = torch.randperm(len(train_full))[:SUBSET_SIZE].tolist()
        train_ds = Subset(train_full, indices)
        print(f"[data] Using subset of {SUBSET_SIZE}/{len(train_full)} training samples")
    else:
        train_ds = train_full
        print(f"[data] Using full training set ({len(train_full)} samples)")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_full,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, val_full


class WarmupCosineScheduler:
    """Linear warm-up followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            if s <= self.warmup_steps:
                lr = base_lr * s / max(1, self.warmup_steps)
            else:
                progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_step": scheduler._step,
            "loss": loss,
        },
        path,
    )
    print(f"[ckpt] Saved → {path}")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler._step = ckpt.get("scheduler_step", 0)
    start_epoch = ckpt["epoch"] + 1
    print(f"[ckpt] Resumed from epoch {ckpt['epoch']} (loss={ckpt['loss']:.6f})")
    return start_epoch


@torch.no_grad()
def evaluate(encoder, model, loader, criterion, device):
    encoder.eval()
    model.eval()
    total_loss, n = 0.0, 0
    for images, _ in loader:
        images = images.to(device)

        # New: Get latent from encoder first
        z, mu, logvar = encoder(images)
        pred = model(z)

        loss = criterion(pred, images)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


def train():
    # ---- reproducibility --------------------------------------------------
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}")

    # ---- data -------------------------------------------------------------
    train_loader, val_loader, val_full = build_dataloaders()
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS

    # ---- model ------------------------------------------------------------
    encoder = MNISTToLatentEncoder(latent_chan=LATENT_CHAN, latent_res=LATENT_RES).to(device)
    trans_inr = build_model(device)
    # Optimization
    params = list(encoder.parameters()) + list(trans_inr.parameters())
    optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = WarmupCosineScheduler(optimizer, LR_WARMUP_STEPS, total_steps)

    start_epoch = 0
    if RESUME_CKPT is not None:
        start_epoch = load_checkpoint(RESUME_CKPT, encoder, optimizer, scheduler)

    print(f"\n[train] Starting — {NUM_EPOCHS} epochs, " f"{steps_per_epoch} steps/epoch, batch={BATCH_SIZE}\n")

    # ---- training loop ----------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        encoder.train()
        trans_inr.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False) as pbar:
            for step, (images, _) in enumerate(pbar):  # noqa: B007
                images = images.to(device)  # (B, 1, 28, 28)

                # Forward
                # Inside step loop:
                # 1. Encode image to latent grid
                z, mu, logvar = encoder(images)

                # 2. Decode latent grid back to image via TransInr
                pred = trans_inr(z)

                # 3. VAE Loss
                recon_loss = criterion(pred, images)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / images.size(0)

                # Adjust KL_WEIGHT (start small, e.g., 1e-4)
                loss = recon_loss + (1e-4 * kl_loss)

                pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.current_lr:.2e}")

                # Backward
                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
        # ---- epoch summary ------------------------------------------------
        avg_train = epoch_loss / steps_per_epoch

        # ---- validation ---------------------------------------------------
        if (epoch + 1) % EVAL_EVERY == 0:
            val_loss = evaluate(encoder, trans_inr, val_loader, criterion, device)
            print(f"[epoch {epoch+1:>3}] val_loss={val_loss:.6f}")

    # ---- final checkpoint -------------------------------------------------
    final_path = os.path.join(CHECKPOINT_DIR, "train_results/transinr_final.pt")
    save_checkpoint(trans_inr, optimizer, scheduler, NUM_EPOCHS - 1, avg_train, final_path)
    print("\n[done] Training complete.")

    plot_reconstructions(trans_inr, encoder, val_full, device)


if __name__ == "__main__":
    train()
