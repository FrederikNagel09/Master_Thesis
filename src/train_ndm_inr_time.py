"""
Training script for NeuralDiffusionModel with INR reconstruction.
All hyperparameters are defined at the top of this file.
"""

import json
import os
import sys
from datetime import datetime

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(".")

from src.models.ndm__inr_time import INR, NeuralDiffusionModel, NoisePredictor, WeightEncoder

# =============================================================================
# Run identity & output paths
# =============================================================================

run_name = "ndm_inr_rec_scaling"
LOSS_PLOT_PATH = f"src/results/ndm_inr/training_graphs/loss_{run_name}.png"
SAMPLES_PATH = f"src/results/ndm_inr/samples/samples_{run_name}.png"
SCALED_SAMPLES_PATH = f"src/results/ndm_inr/samples/scaled_samples_{run_name}.png"
WEIGHTS_PATH = f"src/results/ndm_inr/weights/weights_{run_name}.pt"

# =============================================================================
# Hyperparameters
# =============================================================================

# --- Data ---
SINGLE_CLASS = False  # True  → train on digit "1" only
TARGET_CLASS = 1  # which digit to keep when SINGLE_CLASS=True
SUBSET_FRAC = 1.0  # fraction of (filtered) dataset to use; 1.0 = all
BATCH_SIZE = 128
REC_WARMUP = True  # If True, l_rec is weighted by a warmup factor that starts at 0 and ramps up to 1 over the first 33% of training. This can help early training stability when the untrained model's reconstructions are very poor.
REC_WARMUP_FRAC = 0.40  # Fraction of epochs before l_rec starts decaying

# --- Diffusion schedule ---
BETA_1 = 1e-4
BETA_T = 2e-2
T = 1000
SIGMA_TILDE_FACTOR = 1.0  # 1.0 = stochastic DDPM; 0.0 = deterministic DDIM

# --- INR architecture ---
INR_COORD_DIM = 2
INR_HIDDEN_DIM = 32
INR_N_HIDDEN = 3
INR_OUT_DIM = 1

# --- WeightEncoder  F_phi ---
F_PHI_HIDDEN_DIMS = [512, 512, 512]
F_PHI_T_EMBED_DIM = 64

# --- NoisePredictor  epsilon_theta ---
NOISE_HIDDEN_DIM = 512
NOISE_N_BLOCKS = 4
NOISE_T_EMBED_DIM = 128

# --- Optimiser ---
LR = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0  # max gradient norm; set to 0 to disable

# --- Training ---
EPOCHS = 300
IMG_SIZE = 28
DATA_DIM = IMG_SIZE * IMG_SIZE  # 784

# --- Visualisation ---
N_PLOT_ORIGINALS = 8  # original images shown in diagnostic figure
N_PLOT_SAMPLES = 8  # model samples shown in diagnostic figure
# Diagnostic figures are saved at 5 evenly-spaced epochs (including the last)
PLOT_EPOCHS = sorted(
    set(  # noqa: C401
        round(EPOCHS * k / 5) for k in range(1, 6)
    )
)

# =============================================================================
# Device
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# =============================================================================
# Ensure output directories exist
# =============================================================================

for path in [LOSS_PLOT_PATH, SAMPLES_PATH, SCALED_SAMPLES_PATH, WEIGHTS_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# =============================================================================
# Dataset
# =============================================================================

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),  # scale to roughly [-1, 1]
        transforms.Lambda(lambda x: x.flatten()),  # (784,)
    ]
)

train_data = datasets.MNIST("data/", train=True, download=True, transform=transform)
print("##### DATASET: MNIST #####")

if SINGLE_CLASS:
    print(f"##### Using only class {TARGET_CLASS} for training #####")
    indices = [i for i, (_, label) in enumerate(train_data) if label == TARGET_CLASS]
    train_data = Subset(train_data, indices)

n = int(len(train_data) * SUBSET_FRAC)
train_data = Subset(train_data, range(n))
print(f"Training on {len(train_data)} samples ({SUBSET_FRAC * 100:.1f}% of dataset)")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Grab a fixed batch of originals for the diagnostic plots (same images every time)
_fixed_iter = iter(train_loader)
fixed_originals, _ = next(_fixed_iter)  # (batch, 784)
fixed_originals = fixed_originals[:N_PLOT_ORIGINALS].to(device)

# =============================================================================
# Model
# =============================================================================

inr = INR(
    coord_dim=INR_COORD_DIM,
    hidden_dim=INR_HIDDEN_DIM,
    n_hidden=INR_N_HIDDEN,
    out_dim=INR_OUT_DIM,
)
weight_dim = inr.num_weights
print(f"INR weight_dim = {weight_dim}")

f_phi = WeightEncoder(
    data_dim=DATA_DIM,
    weight_dim=weight_dim,
    hidden_dims=F_PHI_HIDDEN_DIMS,
    t_embed_dim=F_PHI_T_EMBED_DIM,
)

network = NoisePredictor(
    weight_dim=weight_dim,
    hidden_dim=NOISE_HIDDEN_DIM,
    n_blocks=NOISE_N_BLOCKS,
    t_embed_dim=NOISE_T_EMBED_DIM,
)

model = NeuralDiffusionModel(
    network=network,
    F_phi=f_phi,
    inr=inr,
    beta_1=BETA_1,
    beta_T=BETA_T,
    T=T,
    sigma_tilde_factor=SIGMA_TILDE_FACTOR,
    data_dim=DATA_DIM,
    img_size=IMG_SIZE,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {n_params:,}")

# =============================================================================
# Optimiser
# =============================================================================

optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =============================================================================
# Logging buffers
# =============================================================================

# Per-step losses (for smooth plots)
log_loss_total = []
log_loss_diff = []
log_loss_prior = []
log_loss_rec = []
# Per-epoch mean losses
epoch_loss_total = []
epoch_loss_diff = []
epoch_loss_prior = []
epoch_loss_rec = []

# =============================================================================
# Helper: tensor -> numpy image (28x28, clipped to [0,1])
# =============================================================================


def to_img(flat: torch.Tensor) -> np.ndarray:
    """
    Convert a flat (784,) tensor from model space to a displayable (28,28) array.
    The training transform scales pixels to ~[-1, 1], so we invert that.
    """
    img = flat.detach().cpu().float()
    img = (img / 2.0 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    return img.reshape(IMG_SIZE, IMG_SIZE).numpy()


def to_img_scaled(flat: torch.Tensor) -> np.ndarray:
    """Min-max normalise to [0,1] — useful when samples have unknown range."""
    img = flat.detach().cpu().float().reshape(IMG_SIZE, IMG_SIZE)
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    return img.clamp(0, 1).numpy()


# =============================================================================
# Helper: save loss plot (4 subplots, one per loss term)
# =============================================================================


def save_loss_plot(step_indices):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training losses — {run_name}", fontsize=13)

    pairs = [
        (axes[0, 0], log_loss_total, "Total loss"),
        (axes[0, 1], log_loss_diff, "L_diff"),
        (axes[1, 0], log_loss_prior, "L_prior"),
        (axes[1, 1], log_loss_rec, "L_rec"),
    ]
    for ax, values, title in pairs:
        ax.plot(step_indices, values, linewidth=0.7, alpha=0.6, label="per step")
        # Overlay a simple running mean (window = ~1 epoch worth of steps)
        window = max(1, len(train_loader) // 4)
        if len(values) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(values, kernel, mode="valid")
            ax.plot(step_indices[window - 1 :], smoothed, linewidth=1.8, color="crimson", label="smoothed")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=120)
    plt.close(fig)


def save_experiment(model, optimiser, epoch):  # noqa: ARG001
    exp_dir = f"src/results/ndm_inr/experiments/{run_name}"
    os.makedirs(exp_dir, exist_ok=True)

    # --- Config ---
    config = {
        "meta": {
            "run_name": run_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "epochs_trained": epoch,
        },
        "paths": {
            "weights": WEIGHTS_PATH,
            "loss_plot": LOSS_PLOT_PATH,
            "samples": SAMPLES_PATH,
            "experiment_dir": exp_dir,
        },
        "data": {
            "single_class": SINGLE_CLASS,
            "target_class": TARGET_CLASS,
            "subset_frac": SUBSET_FRAC,
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "data_dim": DATA_DIM,
        },
        "diffusion": {
            "T": T,
            "beta_1": BETA_1,
            "beta_T": BETA_T,
            "sigma_tilde_factor": SIGMA_TILDE_FACTOR,
        },
        "inr": {
            "coord_dim": INR_COORD_DIM,
            "hidden_dim": INR_HIDDEN_DIM,
            "n_hidden": INR_N_HIDDEN,
            "out_dim": INR_OUT_DIM,
            "num_weights": inr.num_weights,
        },
        "f_phi": {
            "hidden_dims": F_PHI_HIDDEN_DIMS,
            "t_embed_dim": F_PHI_T_EMBED_DIM,
        },
        "noise_predictor": {
            "hidden_dim": NOISE_HIDDEN_DIM,
            "n_blocks": NOISE_N_BLOCKS,
            "t_embed_dim": NOISE_T_EMBED_DIM,
        },
        "optimiser": {
            "type": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "training": {
            "epochs": EPOCHS,
            "plot_epochs": PLOT_EPOCHS,
        },
    }

    config_path = f"{exp_dir}/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Experiment saved to {exp_dir}/")
    print("  weights → weights.pt")
    print("  config  → config.json")


# =============================================================================
# Helper: save diagnostic figure  (originals | reconstructions | samples)
# =============================================================================


def save_diagnostic(epoch: int):
    model.eval()
    with torch.no_grad():
        t0 = torch.zeros(N_PLOT_ORIGINALS, 1, device=device)
        weights = model.F_phi(fixed_originals, t0)
        recons = model._inr_decode(weights)
        samples = model.sample(n_samples=N_PLOT_SAMPLES)

    n_cols = max(N_PLOT_ORIGINALS, N_PLOT_SAMPLES)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 1.4, 5))
    fig.suptitle(f"Epoch {epoch} / {EPOCHS}  [{run_name}]", fontsize=12)

    row_labels = ["Originals", "Reconstructions", "Samples"]
    for row, (label, imgs) in enumerate(zip(row_labels, [fixed_originals, recons, samples])):  # noqa: B905
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < imgs.shape[0]:
                ax.imshow(to_img(imgs[col]), cmap="gray", vmin=0, vmax=1)
        # Row title centred above the row using the leftmost axis
        axes[row, 0].set_title(label, fontsize=9, loc="left", pad=3)

    plt.subplots_adjust(top=0.92, hspace=0.05, wspace=0.0)
    base, ext = os.path.splitext(SAMPLES_PATH)
    plt.savefig(f"{base}_ep{epoch:04d}{ext}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    model.train()


def save_final_samples():
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=25)  # (25, 784)

    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    fig.suptitle(f"Final samples — {run_name}", fontsize=11)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(to_img(samples[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    base, ext = os.path.splitext(SAMPLES_PATH)
    plt.savefig(f"{base}_final_grid{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Final sample grid saved.")
    model.train()


# =============================================================================
# Training loop
# =============================================================================

total_steps = EPOCHS * len(train_loader)
global_step = 0
step_indices = []

pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

for epoch in range(1, EPOCHS + 1):
    epoch_totals = {"total": 0.0, "diff": 0.0, "prior": 0.0, "rec": 0.0}
    n_batches = 0

    for x, _ in train_loader:
        x = x.to(device)

        optimiser.zero_grad()

        if REC_WARMUP:
            warmup_epochs = int(EPOCHS * REC_WARMUP_FRAC)
            if epoch <= warmup_epochs:
                rec_scale = 1.0
            else:
                rec_scale = 1.0 - (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)

        loss, l_diff, l_prior, l_rec = model.loss(x, rec_scale=rec_scale)
        loss.backward()

        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimiser.step()

        # --- Log ---
        global_step += 1
        step_indices.append(global_step)
        log_loss_total.append(loss.item())
        log_loss_diff.append(l_diff.item())
        log_loss_prior.append(l_prior.item())
        log_loss_rec.append(l_rec.item())

        epoch_totals["total"] += loss.item()
        epoch_totals["diff"] += l_diff.item()
        epoch_totals["prior"] += l_prior.item()
        epoch_totals["rec"] += l_rec.item()
        n_batches += 1

        pbar.set_postfix(
            {
                "ep": epoch,
                "loss": f"{loss.item():.3f}",
                "diff": f"{l_diff.item():.3f}",
                "rec": f"{l_rec.item():.3f}",
            }
        )
        pbar.update(1)

    # --- End of epoch ---
    epoch_loss_total.append(epoch_totals["total"] / n_batches)
    epoch_loss_diff.append(epoch_totals["diff"] / n_batches)
    epoch_loss_prior.append(epoch_totals["prior"] / n_batches)
    epoch_loss_rec.append(epoch_totals["rec"] / n_batches)

    # Update loss plot every epoch
    save_loss_plot(step_indices)

    # Diagnostic figure at the 5 scheduled epochs
    if epoch in PLOT_EPOCHS:
        print(f"\n→ Saving diagnostic figures at epoch {epoch} …")
        save_diagnostic(epoch)

pbar.close()

# =============================================================================
# Final saves
# =============================================================================

print("\nSaving model weights …")
torch.save(
    {
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "hparams": {
            "T": T,
            "beta_1": BETA_1,
            "beta_T": BETA_T,
            "sigma_tilde_factor": SIGMA_TILDE_FACTOR,
            "inr_hidden_dim": INR_HIDDEN_DIM,
            "inr_n_hidden": INR_N_HIDDEN,
            "noise_hidden_dim": NOISE_HIDDEN_DIM,
            "noise_n_blocks": NOISE_N_BLOCKS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
    },
    WEIGHTS_PATH,
)
print(f"Weights saved to  {WEIGHTS_PATH}")
print(f"Loss plot saved to {LOSS_PLOT_PATH}")
print("Done.")

print("\nGenerating final sample grid …")
save_final_samples()


save_experiment(model, optimiser, EPOCHS)


"""
with open("src/results/ndm_inr/experiments/<run_name>/config.json") as f:
    cfg = json.load(f)

inr     = INR(coord_dim=cfg["inr"]["coord_dim"], hidden_dim=cfg["inr"]["hidden_dim"],
              n_hidden=cfg["inr"]["n_hidden"], out_dim=cfg["inr"]["out_dim"])
f_phi   = WeightEncoder(data_dim=cfg["data"]["data_dim"], weight_dim=cfg["inr"]["num_weights"],
                        hidden_dims=cfg["f_phi"]["hidden_dims"], t_embed_dim=cfg["f_phi"]["t_embed_dim"])
network = NoisePredictor(weight_dim=cfg["inr"]["num_weights"], hidden_dim=cfg["noise_predictor"]["hidden_dim"],
                         n_blocks=cfg["noise_predictor"]["n_blocks"], t_embed_dim=cfg["noise_predictor"]["t_embed_dim"])
model   = NeuralDiffusionModel(network=network, F_phi=f_phi, inr=inr,
                                beta_1=cfg["diffusion"]["beta_1"], beta_T=cfg["diffusion"]["beta_T"],
                                T=cfg["diffusion"]["T"], sigma_tilde_factor=cfg["diffusion"]["sigma_tilde_factor"],
                                data_dim=cfg["data"]["data_dim"], img_size=cfg["data"]["img_size"]).to(device)

ckpt = torch.load(cfg["paths"]["weights"], map_location=device)
model.load_state_dict(ckpt["model_state"])
"""
