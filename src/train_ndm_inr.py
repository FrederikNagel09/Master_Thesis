"""
train_ndm_inr.py
Training script for the NDM-INR model on MNIST.
"""

import gzip
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")

from src.dataloaders.MNISTHyper import MNISTHyperDataset  # adjust import to your project layout
from src.models.ndm_inr import INR, NDMInr, NoisePredictorMLP, WeightEncoder

# ── Hyperparameters ───────────────────────────────────────────────────────────

# Data
MNIST_RAW_DIR = "data/MNIST/raw"
CLASSES = [1]  # which digit classes to train on (list of ints 0-9)
SUBSET_FRACTION = 1.0  # fraction of filtered data to use (0.0, 1.0]

# Model
INR_HIDDEN_DIM = 32
INR_N_HIDDEN = 3
LAYER_WIDTH = 128
DROPOUT = 0.2

# Training
BATCH_SIZE = 64
EPOCHS = 50
LR = 3e-4
WARM_UP_STEPS = 50
REC_WEIGHT = 1.0
T_DIFFUSION = 200
SIGMA_TILDE_FACTOR = 1.0

# Logging / plotting
PLOT_EVERY_N_EPOCHS = 1  # update the live loss plot every N epochs
DEVICE = torch.device("mps")

# Output paths
LOSS_PLOT_PATH = "src/results/ndm_inr/training_graphs/loss_ndm_inr.png"
SAMPLES_PATH = "src/results/ndm_inr/samples/samples_ndm_inr.png"
SCALED_SAMPLES_PATH = "src/results/ndm_inr/samples/scaled_samples_ndm_inr.png"
WEIGHTS_PATH = "src/results/ndm_inr/weights/weights_ndm_inr.pt"


# ── Dataset ───────────────────────────────────────────────────────────────────


class FilteredMNISTDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around MNISTHyperDataset that:
      - filters to a subset of digit classes
      - takes an optional fraction of the filtered data
      - reshapes image_flat (784,) → (1, 28, 28) for WeightEncoder

    MNISTHyperDataset is left completely untouched.
    """

    def __init__(
        self,
        raw_dir: str,
        split: str,
        classes: list[int],
        subset_fraction: float = 1.0,
    ):
        self.base = MNISTHyperDataset(mnist_raw_dir=raw_dir, split=split)

        # Load labels separately (same raw dir, same split)
        labels = self._load_labels(raw_dir, split)  # (N,) numpy int array

        # Filter indices to requested classes
        indices = [i for i, lbl in enumerate(labels) if lbl in classes]

        # Optionally take a subset
        if subset_fraction < 1.0:
            n = max(1, int(len(indices) * subset_fraction))
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(indices, size=n, replace=False).tolist()

        self.indices = indices
        print(f"  [{split}] classes={classes} | " f"subset_fraction={subset_fraction} | " f"n_samples={len(self.indices)}")

    @staticmethod
    def _load_labels(raw_dir: str, split: str) -> np.ndarray:
        candidates = (
            [Path(raw_dir) / "train-labels-idx1-ubyte", Path(raw_dir) / "train-labels-idx1-ubyte.gz"]
            if split == "train"
            else [Path(raw_dir) / "t10k-labels-idx1-ubyte", Path(raw_dir) / "t10k-labels-idx1-ubyte.gz"]
        )
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No MNIST label file found in {raw_dir} for split='{split}'.")
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            assert magic == 2049, f"Not an MNIST label file (magic={magic})"
            labels = np.frombuffer(f.read(n), dtype=np.uint8)
        return labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image_flat, coords, pixels = self.base[self.indices[idx]]
        # Reshape (784,) → (1, 28, 28) for WeightEncoder
        image_28 = image_flat.view(1, 28, 28)
        return image_28, coords, pixels


# ── Model setup ───────────────────────────────────────────────────────────────


def build_model(device: torch.device):
    inr = INR(coord_dim=2, hidden_dim=INR_HIDDEN_DIM, n_hidden=INR_N_HIDDEN, out_dim=1)
    weight_encoder = WeightEncoder(weight_dim=inr.num_weights, layer_width=LAYER_WIDTH, dropout=DROPOUT)
    noise_net = NoisePredictorMLP(num_weights=inr.num_weights, layer_width=LAYER_WIDTH, dropout=DROPOUT)
    model = NDMInr(
        weight_encoder=weight_encoder,
        noise_net=noise_net,
        inr=inr,
        T=T_DIFFUSION,
        sigma_tilde_factor=SIGMA_TILDE_FACTOR,
        rec_weight=REC_WEIGHT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in weight_encoder.parameters())
    noise_params = sum(p.numel() for p in noise_net.parameters())
    inr_params = sum(p.numel() for p in inr.parameters())

    print("=" * 70)
    print("NDMInr — model summary")
    print("=" * 70)
    print(f"  inr.num_weights  = {inr.num_weights}")
    print(f"  layer_width      = {LAYER_WIDTH}")
    print()
    print(f"  params — WeightEncoder : {enc_params:>10,}")
    print(f"  params — NoiseMLP      : {noise_params:>10,}")
    print(f"  params — INR           : {inr_params:>10,}")
    print(f"  params — total         : {total_params:>10,}")
    print("=" * 70)

    return model.to(device)


# ── Plotting ──────────────────────────────────────────────────────────────────


def update_loss_plot(
    steps_log: list[int],
    loss_log: list[float],
    diff_log: list[float],
    prior_log: list[float],
    rec_log: list[float],
    total_steps: int,
    save_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("NDM-INR training losses", fontsize=13)

    pairs = [
        (axes[0, 0], loss_log, "Total loss", "tab:blue"),
        (axes[0, 1], diff_log, "L_diff", "tab:orange"),
        (axes[1, 0], prior_log, "L_prior", "tab:green"),
        (axes[1, 1], rec_log, "L_rec", "tab:red"),
    ]
    for ax, data, title, color in pairs:
        ax.plot(steps_log, data, color=color, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_xlim(0, total_steps)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def save_sample_grid(
    model: NDMInr,
    device: torch.device,
    n_rows: int = 4,
    n_cols: int = 4,
    save_path: str = SAMPLES_PATH,
):
    """Sample n_rows*n_cols images at native 28x28 resolution."""
    n_samples = n_rows * n_cols
    coords_28 = _make_coords(28, device)  # (784, 2)

    model.eval()
    with torch.no_grad():
        pixels = model.sample(coords_28, n_samples=n_samples)  # (N, 784, 1)
    model.train()

    pixels = pixels.squeeze(-1).cpu().numpy()  # (N, 784)
    pixels = pixels.reshape(n_samples, 28, 28)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pixels[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved sample grid → {save_path}")


def save_scaled_samples(
    model: NDMInr,
    device: torch.device,
    resolutions: list[int] = (256, 512),
    n_samples: int = 4,
    save_path: str = SCALED_SAMPLES_PATH,
):
    """
    Render the same n_samples at multiple resolutions side by side.
    Each row is one sample; columns are the different resolutions.
    Because the INR is a continuous function we can query it at any grid.
    We use the same z_T seed across resolutions by sampling once and
    reusing the reverse-chain output weights.
    """
    # First run the reverse chain at native res to get theta_hat vectors
    # coords_native = _make_coords(28, device)

    model.eval()
    with torch.no_grad():
        # Tap into the reverse chain to get theta_hat instead of pixel output
        num_weights = model.num_weights
        z_t = torch.randn(n_samples, num_weights, device=device)

        for t in range(model.T - 1, -1, -1):
            t_norm = torch.full((n_samples, 1), t / max(model.T - 1, 1), device=device)
            t_idx = torch.full((n_samples,), t, dtype=torch.long, device=device)
            eps_hat = model.noise_net(z_t, t_norm)
            alpha_t = model.sqrt_alpha_cumprod[t].view(1, 1)
            sigma_t = model.sigma[t].view(1, 1)
            theta_hat = (z_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

            if t == 0:
                break

            s = t - 1
            s_idx = torch.full((n_samples,), s, dtype=torch.long, device=device)
            alpha_s = model.sqrt_alpha_cumprod[s].view(1, 1)
            sigma_s_sq = model.sigma_sq[s].view(1, 1)
            sigma_tilde_sq = model._sigma_tilde_sq(s_idx, t_idx)[0].view(1, 1)
            coeff = (sigma_s_sq - sigma_tilde_sq).clamp(min=0).sqrt() / sigma_t.clamp(min=1e-6)
            mu = alpha_s * theta_hat + coeff * (z_t - alpha_t * theta_hat)
            noise = torch.randn_like(z_t) if sigma_tilde_sq.item() > 0 else torch.zeros_like(z_t)
            z_t = mu + sigma_tilde_sq.clamp(min=0).sqrt() * noise

        # theta_hat: (n_samples, num_weights) — now render at each resolution
        all_renders = []
        for res in resolutions:
            coords_res = _make_coords(res, device)  # (res*res, 2)
            coords_res = coords_res.unsqueeze(0).expand(n_samples, -1, -1)  # (N, res*res, 2)
            pix = model.inr(coords_res, theta_hat)  # (N, res*res, 1)
            pix = pix.squeeze(-1).cpu().numpy().reshape(n_samples, res, res)
            all_renders.append((res, pix))

    model.train()

    n_cols = len(resolutions)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 3, n_samples * 3))
    # ensure axes is always 2-D
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, (res, renders) in enumerate(all_renders):
        for row in range(n_samples):
            axes[row, col].imshow(renders[row], cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"{res}x{res}", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved scaled samples → {save_path}")


def _make_coords(res: int, device: torch.device) -> torch.Tensor:
    """Build a (res*res, 2) coordinate grid normalised to [-1, 1]."""
    xs = torch.linspace(-1, 1, res, device=device)
    grid_r, grid_c = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)


# ── Training loop ─────────────────────────────────────────────────────────────


def train():
    # ── data ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Loading data")
    print("=" * 70)
    dataset = FilteredMNISTDataset(
        raw_dir=MNIST_RAW_DIR,
        split="train",
        classes=CLASSES,
        subset_fraction=SUBSET_FRACTION,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # ── model ─────────────────────────────────────────────────────────────
    print()
    model = build_model(DEVICE)

    # ── optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_steps = EPOCHS * len(loader)

    def lr_lambda(current_step: int) -> float:
        if current_step < WARM_UP_STEPS:
            return float(current_step + 1) / float(max(1, WARM_UP_STEPS))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── logging buffers ───────────────────────────────────────────────────
    # Per-step accumulators (reset each epoch for per-epoch averages)
    epoch_loss = []
    epoch_diff = []
    epoch_prior = []
    epoch_rec = []

    # Per-epoch averages stored for plotting
    steps_log = []  # x-axis value = global step at end of epoch
    loss_log = []
    diff_log = []
    prior_log = []
    rec_log = []

    # ── training ──────────────────────────────────────────────────────────
    progress_bar = tqdm(range(total_steps), desc="Training")
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss.clear()
        epoch_diff.clear()
        epoch_prior.clear()
        epoch_rec.clear()

        for image, coords, pixels in loader:
            image = image.to(DEVICE)  # (B, 1, 28, 28)
            coords = coords.to(DEVICE)  # (B, 784, 2)
            pixels = pixels.to(DEVICE)  # (B, 784, 1)

            optimizer.zero_grad()
            loss, l_diff, l_prior, l_rec = model(image, coords, pixels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss.append(loss.item())
            epoch_diff.append(l_diff.item())
            epoch_prior.append(l_prior.item())
            epoch_rec.append(l_rec.item())

            global_step += 1
            progress_bar.set_postfix(
                loss=f"{loss.item():10.4f}",
                diff=f"{l_diff.item():10.4f}",
                prior=f"{l_prior.item():10.4f}",
                rec=f"{l_rec.item():10.4f}",
                epoch=f"{epoch + 1}/{EPOCHS}",
            )
            progress_bar.update()

        # ── end of epoch: record averages ─────────────────────────────────
        steps_log.append(global_step)
        loss_log.append(np.mean(epoch_loss))
        diff_log.append(np.mean(epoch_diff))
        prior_log.append(np.mean(epoch_prior))
        rec_log.append(np.mean(epoch_rec))

        # ── periodic plot update ───────────────────────────────────────────
        if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0:
            update_loss_plot(
                steps_log,
                loss_log,
                diff_log,
                prior_log,
                rec_log,
                total_steps=total_steps,
                save_path=LOSS_PLOT_PATH,
            )

    progress_bar.close()

    # ── final loss plot (catches any trailing epochs not hit by modulo) ───
    update_loss_plot(
        steps_log,
        loss_log,
        diff_log,
        prior_log,
        rec_log,
        total_steps=total_steps,
        save_path=LOSS_PLOT_PATH,
    )
    print(f"\n  saved loss plot → {LOSS_PLOT_PATH}")

    # ── save weights ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"  saved model weights → {WEIGHTS_PATH}")

    # ── sample plots ──────────────────────────────────────────────────────
    print("\nGenerating samples...")
    save_sample_grid(model, DEVICE, n_rows=4, n_cols=4, save_path=SAMPLES_PATH)
    save_scaled_samples(
        model,
        DEVICE,
        resolutions=[256, 512],
        n_samples=4,
        save_path=SCALED_SAMPLES_PATH,
    )


if __name__ == "__main__":
    train()
