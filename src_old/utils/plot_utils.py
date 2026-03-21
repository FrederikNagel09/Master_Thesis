import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


def plot_training_and_reconstruction(
    history: dict, name: str, graph_dir: str, current_epoch: int, total_epochs: int, model, dataset, device
):
    """
    Plots the training curve and the final reconstruction side by side in a single figure.
    Saves the figure to the specified graph_dir with the given name.

    used by basic_inr training.
    """
    epochs = range(1, current_epoch + 1)

    height, width = dataset.image_shape
    model.eval().to(device)
    with torch.no_grad():
        coords = dataset.coords.to(device)
        preds = model(coords).cpu().numpy().reshape(height, width)
    original = dataset.image

    fig = plt.figure(figsize=(8, 10))

    # Top: training curve
    ax_top = fig.add_subplot(2, 1, 1)
    ax_top.plot(epochs, history["train_mse"], label="Train MSE", linewidth=2)
    ax_top.set_xlabel("Epoch")
    ax_top.set_ylabel("MSE Loss")
    ax_top.set_title(f"INR Training — {name}")
    ax_top.legend()
    ax_top.set_xlim(1, total_epochs)
    ax_top.grid(True, alpha=0.3)

    # Bottom: original vs reconstruction
    ax_orig = fig.add_subplot(2, 2, 3)
    ax_orig.imshow(original, cmap="gray", vmin=0, vmax=1)
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    ax_pred = fig.add_subplot(2, 2, 4)
    ax_pred.imshow(preds, cmap="gray", vmin=0, vmax=1)
    ax_pred.set_title("Reconstruction")
    ax_pred.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{name}.png"), dpi=150)
    plt.close(fig)


def plot_ndm_training(history: dict, name: str, graph_dir: str):
    os.makedirs(graph_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"NDM Training — {name}", fontsize=14, fontweight="bold", y=1.02)

    # Shared style
    spine_color = "#cccccc"
    grid_kw = dict(color="#eeeeee", linewidth=0.8, zorder=0)  # noqa: C408

    def style_ax(ax, title, ylabel, color):
        ax.set_title(title, fontsize=12, fontweight="medium", pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
        ax.tick_params(colors="#555555")
        ax.yaxis.grid(True, **grid_kw)
        ax.set_axisbelow(True)
        ax.plot(
            history["steps"],
            history[ylabel.lower().replace(" ", "_")],
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )
        # Smoothed overlay
        if len(history["steps"]) >= 10:
            kernel = max(1, len(history["steps"]) // 20)
            smoothed = np.convolve(
                history[ylabel.lower().replace(" ", "_")],
                np.ones(kernel) / kernel,
                mode="valid",
            )
            steps_trimmed = history["steps"][kernel - 1 :]
            ax.plot(steps_trimmed, smoothed, color=color, linewidth=2.5, alpha=0.5)

    # Panel 1 — Total loss (black)
    axes[0].set_title("Total Loss", fontsize=12, fontweight="medium", pad=8)
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    for spine in axes[0].spines.values():
        spine.set_edgecolor(spine_color)
    axes[0].tick_params(colors="#555555")
    axes[0].yaxis.grid(True, **grid_kw)
    axes[0].set_axisbelow(True)
    axes[0].plot(history["steps"], history["train_elbo"], color="black", linewidth=1.2, alpha=0.4)
    if len(history["steps"]) >= 10:
        kernel = max(1, len(history["steps"]) // 20)
        smoothed = np.convolve(history["train_elbo"], np.ones(kernel) / kernel, mode="valid")
        axes[0].plot(history["steps"][kernel - 1 :], smoothed, color="black", linewidth=2.2)

    # Panel 2 — Diffusion loss (blue)
    axes[1].set_title("Diffusion Loss", fontsize=12, fontweight="medium", pad=8)
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Diff Loss", fontsize=10)
    for spine in axes[1].spines.values():
        spine.set_edgecolor(spine_color)
    axes[1].tick_params(colors="#555555")
    axes[1].yaxis.grid(True, **grid_kw)
    axes[1].set_axisbelow(True)
    axes[1].plot(history["steps"], history["diff"], color="#2a6fdb", linewidth=1.2, alpha=0.4)
    if len(history["steps"]) >= 10:
        kernel = max(1, len(history["steps"]) // 20)
        smoothed = np.convolve(history["diff"], np.ones(kernel) / kernel, mode="valid")
        axes[1].plot(history["steps"][kernel - 1 :], smoothed, color="#2a6fdb", linewidth=2.2)

    # Panel 3 — KL / prior loss (green)
    axes[2].set_title("KL Prior Loss", fontsize=12, fontweight="medium", pad=8)
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("KL Loss", fontsize=10)
    for spine in axes[2].spines.values():
        spine.set_edgecolor(spine_color)
    axes[2].tick_params(colors="#555555")
    axes[2].yaxis.grid(True, **grid_kw)
    axes[2].set_axisbelow(True)
    axes[2].plot(history["steps"], history["prior"], color="#2ca05a", linewidth=1.2, alpha=0.4)
    if len(history["steps"]) >= 10:
        kernel = max(1, len(history["steps"]) // 20)
        smoothed = np.convolve(history["prior"], np.ones(kernel) / kernel, mode="valid")
        axes[2].plot(history["steps"][kernel - 1 :], smoothed, color="#2ca05a", linewidth=2.2)

    fig.tight_layout()
    save_path = os.path.join(graph_dir, f"{name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training graph saved to: {save_path}")


def moving_average(values: list[float], window: int) -> list[float]:
    """Simple centred-ish moving average (causal  uses past `window` points)."""
    out = []
    for i, _ in enumerate(values):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def plot_vae_training(history: dict, name: str, graph_dir: str, steps_per_epoch: int):
    os.makedirs(graph_dir, exist_ok=True)

    steps = history["steps"]
    window = max(1, len(steps) // 50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"VAE Training — {name}", fontsize=13)

    panels = [
        ("total_loss", "Total Loss", "Total Loss", "tab:blue"),
        ("recon_loss", "Reconstruction", "Reconstruction", "tab:orange"),
        ("kl_loss", "Prior", "KL Divergence", "tab:green"),
    ]

    for ax, (key, title, ylabel, colour) in zip(axes, panels, strict=False):
        raw = history[key]
        ma = moving_average(raw, window)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.plot(steps, raw, alpha=0.25, linewidth=0.8, color=colour, label="per-step")
        ax.plot(steps, ma, linewidth=1.8, color=colour, label=f"moving avg (w={window})")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Replace step-index ticks with epoch numbers
        total_steps = len(steps)
        n_epochs = max(1, total_steps // steps_per_epoch)
        tick_steps = [i * steps_per_epoch for i in range(n_epochs + 1)]
        tick_labels = [str(i) for i in range(n_epochs + 1)]
        ax.set_xticks(tick_steps)
        ax.set_xticklabels(tick_labels)

    fig.tight_layout()
    save_path = os.path.join(graph_dir, f"{name}.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  ↳ training graph saved → {save_path}")


def save_vae_inr_sample_grid(model, coords_sample, samples_dir: str, name: str, device, grid: int = 4):  # noqa: ARG001
    """
    Draw gridxgrid samples from the VAE prior, render them, and save a PNG.
    coords_sample : (784, 2) canonical pixel-coordinate grid on the correct device.
    """
    os.makedirs(samples_dir, exist_ok=True)
    model.eval()
    n = grid * grid

    with torch.no_grad():
        pixels = model.sample(coords_sample, n_samples=n)  # (n, 784, 1)

    pixels = pixels.squeeze(-1).cpu()  # (n, 784)
    side = int(math.isqrt(pixels.shape[1]))  # 28 for MNIST

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.5, grid * 1.5))
    fig.suptitle(f"Prior samples — {name}", fontsize=10)

    for idx, ax in enumerate(axes.flat):
        img = pixels[idx].reshape(side, side).numpy()
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    fig.tight_layout()
    save_path = os.path.join(samples_dir, f"{name}_samples.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ↳ sample grid saved    → {save_path}")


def _save_plot_vae_hypernet(history: dict, name: str, graph_dir: str, epoch: int, num_epochs: int):  # noqa: ARG001
    """
    Plot train_loss, bce_loss, and kl_loss across epochs.
    Saves to graph_dir/<name>.png. Called every epoch; overwrites previous.

    Args:
        history:    dict with keys 'train_loss', 'bce_loss', 'kl_loss'
        name:       run name, used as plot title and filename
        graph_dir:  directory to save the plot
        epoch:      current epoch (1-indexed), used for the x-axis label
        num_epochs: total epochs, used to fix x-axis range
    """
    epochs_so_far = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(name, fontsize=13)

    panels = [
        ("train_loss", "Total Loss"),
        ("bce_loss", "Reconstruction (BCE)"),
        ("kl_loss", "KL Divergence"),
    ]

    for ax, (key, title) in zip(axes, panels):  # noqa: B905
        ax.plot(epochs_so_far, history[key])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, max(num_epochs, len(epochs_so_far)))
        ax.grid(True)

    plt.tight_layout()
    os.makedirs(graph_dir, exist_ok=True)
    fig.savefig(os.path.join(graph_dir, f"{name}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


class VisualCheckpointer:
    """
    Saves growing side-by-side strips of F_net and sampler outputs
    at evenly spaced checkpoints throughout training.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        name: str,
        graph_dir: str,
        total_steps: int,
        n_checkpoints: int = 5,
        image_size: int = 28,
        dataset: str = "mnist",  # <-- add
        channels: int = 1,  # <-- add
        start_step: int = 0,  # <-- add
    ):
        self.dataset = dataset
        self.channels = channels
        self.model = model
        self.device = device
        self.name = name
        self.graph_dir = graph_dir
        self.image_size = image_size
        self.n_pixels = image_size * image_size

        self.checkpoint_steps = set(  # noqa: C401
            int(round(start_step + (total_steps - start_step) * i / n_checkpoints)) for i in range(1, n_checkpoints + 1)
        )

        # Fix a single image for F_net evaluation throughout training
        self.fixed_image = self._get_fixed_image()

        # Accumulated panels — each entry is a (H, W) numpy array
        self.f_net_panels: list[tuple[int, np.ndarray]] = []  # (step, img)
        self.sample_panels: list[tuple[int, np.ndarray]] = []

        os.makedirs(graph_dir, exist_ok=True)

    def _get_fixed_image(self) -> torch.Tensor:
        """Grab one MNIST image and keep it fixed for the whole run."""
        from torchvision import datasets, transforms

        if self.dataset == "cifar10":
            dataset = datasets.CIFAR10(root="data", train=True, download=False, transform=transforms.ToTensor())
        else:
            dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
        img, _ = dataset[0]
        img = img.view(1, -1).to(self.device)
        img = (img - 0.5) * 2.0
        return img

    def maybe_checkpoint(self, global_step: int):
        if global_step not in self.checkpoint_steps:
            return

        self.model.eval()
        with torch.no_grad():
            t_T = torch.ones(1, 1, device=self.device)  # noqa: N806
            transformed = self.model.F_phi(self.fixed_image, t_T)
            f_img = self._to_img(transformed)

            sample = self.model.sample(1)
            s_img = self._to_img(sample)

        self.model.train()

        self.f_net_panels.append((global_step, f_img))
        self.sample_panels.append((global_step, s_img))

        self._save_f_net_strip()
        self._save_strip(self.sample_panels, f"{self.name}_samples.png", "NDM samples")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _to_img(self, tensor: torch.Tensor) -> np.ndarray:
        arr = (tensor * 0.5 + 0.5).clamp(0, 1)
        if self.channels == 3:
            return np.transpose(arr.squeeze(0).cpu().numpy().reshape(3, self.image_size, self.image_size), (1, 2, 0))
        else:
            return arr.squeeze(0).cpu().numpy().reshape(self.image_size, self.image_size)

    def _save_f_net_strip(self):
        n = len(self.f_net_panels)
        # +1 column for the fixed original on the left
        fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.4))

        spine_color = "#cccccc"

        # ── Column 0: original image (shown once, static) ──────────────────────
        orig_img = self._to_img(self.fixed_image)
        if self.dataset == "cifar10":
            axes[0].imshow(orig_img, vmin=0, vmax=1, interpolation="nearest")
        else:
            axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_title("original", fontsize=9, color="#444444")
        axes[0].axis("off")
        # Subtle box to visually separate original from checkpoints
        for spine in axes[0].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(spine_color)

        # ── Columns 1..n: F_net outputs at each checkpoint ─────────────────────
        for ax, (step, img) in zip(axes[1:], self.f_net_panels, strict=False):
            if self.dataset == "cifar10":
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"step {step:,}", fontsize=9, color="#444444")
            ax.axis("off")

        fig.suptitle("F_net output (t=T)", fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()

        save_path = os.path.join(self.graph_dir, f"{self.name}_F_net.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_strip(
        self,
        panels: list[tuple[int, np.ndarray]],
        filename: str,
        suptitle: str,
    ):
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
        if n == 1:
            axes = [axes]

        for ax, (step, img) in zip(axes, panels, strict=False):
            if self.dataset == "cifar10":
                ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"step {step:,}", fontsize=9, color="#444444")
            ax.axis("off")

        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()

        save_path = os.path.join(self.graph_dir, filename)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
