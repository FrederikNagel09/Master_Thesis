import math
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim
import torchvision
import torchvision.transforms as T  # noqa: N812
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore", message="The operator 'aten::im2col'")

sys.path.append(".")


from src.models.trans_inr import TransInr  # noqa: E402

# =============================================================================
#  CONFIG
# =============================================================================
DATA_ROOT = "./data"
SUBSET_SIZE = None
BATCH_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 2000
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEED = 42

# VAE Specifics
LATENT_CHAN = 16
LATENT_RES = 16
KL_WEIGHT_TARGET = 0.01
KL_ANNEAL_STEPS = 5000  # Steps to reach full KL weight

# Model Arch
IMAGE_SIZE = 28
PATCH_SIZE = 2
DIM = 256
N_HEAD = 8
HEAD_DIM = 32
FF_DIM = 512
ENCODER_DEPTH = 4
DECODER_DEPTH = 4
SIREN_HIDDEN_DIM = 256
SIREN_DEPTH = 5
N_GROUPS = 8

CHECKPOINT_DIR = "./src/train_results"
# =============================================================================


class MNISTToLatentEncoder(nn.Module):
    def __init__(self, latent_chan=16, latent_res=16):
        super().__init__()
        self.latent_res = latent_res
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
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
        features = F.interpolate(features, size=(self.latent_res, self.latent_res), mode="bilinear")
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAE(nn.Module):
    """Wrapper to hold both models and the config for easy saving/loading."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = MNISTToLatentEncoder(latent_chan=config["latent_chan"], latent_res=config["latent_res"])

        tokenizer_cfg = {
            "target": "src.models.trans_inr_helpers.LatentTokenizer",
            "params": {
                "latent_dim": config["latent_chan"],
                "latent_size": config["latent_res"],
                "patch_size": config["patch_size"],
                "dim": config["dim"],
                "n_head": config["n_head"],
                "head_dim": config["head_dim"],
            },
        }
        # ... (rest of the inr_cfg and transformer_cfg using config dict keys)
        # Assuming build_model logic is inside here for brevity
        self.trans_inr = TransInr(
            tokenizer=tokenizer_cfg,
            inr={
                "target": "src.models.trans_inr_helpers.SIREN",
                "params": {
                    "depth": config["siren_depth"],
                    "in_dim": 2,
                    "out_dim": 1,
                    "hidden_dim": config["siren_hidden"],
                    "omega": 30.0,
                    "out_bias": 0,
                },
            },
            n_groups=config["n_groups"],
            data_shape=(config["img_size"], config["img_size"]),
            transformer={
                "target": "src.models.trans_inr_helpers.Transformer",
                "params": {
                    "dim": config["dim"],
                    "encoder_depth": config["enc_depth"],
                    "decoder_depth": config["dec_depth"],
                    "n_head": config["n_head"],
                    "head_dim": config["head_dim"],
                    "ff_dim": config["ff_dim"],
                    "dropout": 0.0,
                },
            },
            update_strategy="scale",
        )

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        pred = self.trans_inr(z)
        return pred, mu, logvar


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
        z, _, _ = encoder(images)
        pred = model(z)

        loss = criterion(pred, images)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


def train():
    # 1. Setup Device & Data
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}")

    # ---- data -------------------------------------------------------------
    train_loader, _, _ = build_dataloaders()
    steps_per_epoch = len(train_loader)

    # 2. Pack Config
    cfg = {
        "latent_chan": LATENT_CHAN,
        "latent_res": LATENT_RES,
        "img_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "dim": DIM,
        "n_head": N_HEAD,
        "head_dim": HEAD_DIM,
        "ff_dim": FF_DIM,
        "enc_depth": ENCODER_DEPTH,
        "dec_depth": DECODER_DEPTH,
        "siren_hidden": SIREN_HIDDEN_DIM,
        "siren_depth": SIREN_DEPTH,
        "n_groups": N_GROUPS,
    }

    model = VAE(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\n[train] Starting — {NUM_EPOCHS} epochs, " f"{steps_per_epoch} steps/epoch, batch={BATCH_SIZE}\n")
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_recon, total_kl = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, _ in pbar:
            images = images.to(device)

            # KL Annealing logic
            kl_weight = min(KL_WEIGHT_TARGET, KL_WEIGHT_TARGET * (global_step / KL_ANNEAL_STEPS))

            pred, mu, logvar = model(images)

            recon_loss = F.mse_loss(pred, images)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]))
            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            global_step += 1
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", kl=f"{kl_loss.item():.2f}", kl_w=f"{kl_weight:.2e}")

        # Epoch Summary
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        print(f"Epoch {epoch+1} Summary: Recon={avg_recon:.6f} | KL={avg_kl:.6f}")

        # Save everything
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, "vae_transINR_weights_v4.pt")
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch}, save_path)


if __name__ == "__main__":
    train()
