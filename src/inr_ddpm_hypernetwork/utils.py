import math
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(".")


def make_coord_grid(height: int, width: int) -> torch.Tensor:
    """Build a (H*W, 2) coordinate grid normalized to [-1, 1]."""
    rows = torch.linspace(-1, 1, height)
    cols = torch.linspace(-1, 1, width)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)


def mse_to_psnr(mse: float) -> float:
    """Convert MSE (on [0,1] pixel values) to PSNR in dB."""
    if mse == 0:
        return float("inf")
    return -10 * math.log10(mse)


def _save_plot(history: dict, name: str, graph_dir: str, current_epoch: int, total_epochs: int):
    """
    Save a multi-curve training loss plot.
    Handles both the old single-loss format and the new multi-loss format.
    """
    epochs = range(1, current_epoch + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    if "total_loss" in history:
        ax.plot(epochs, history["total_loss"], label="Total Loss", linewidth=2)
        if "recon_loss" in history:
            ax.plot(epochs, history["recon_loss"], label="Reconstruction MSE", linewidth=1.5, linestyle="--")
        if "denoise_loss" in history:
            ax.plot(epochs, history["denoise_loss"], label="Denoising MSE", linewidth=1.5, linestyle=":")
    else:
        # Legacy single-loss format
        ax.plot(epochs, history["train_mse"], label="Train MSE", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"DiffusionHyperINR Training — {name}")
    ax.legend()
    ax.set_xlim(1, total_epochs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{name}.png"), dpi=150)
    plt.close(fig)


def _save_reconstruction_diffusion(
    model,
    dataset,
    name: str,
    graph_dir: str,
    device: str = "cpu",
    num_examples: int = 4,
):
    """
    Save a 3-row comparison grid:
        Row 1: Original 28x28 MNIST images
        Row 2: UNet's denoised reconstruction (from noised version of original)
        Row 3: INR reconstruction from UNet output

    Uses the dataset samples rather than pure generation, so we can compare
    to ground truth at the end of training.

    Args:
        model:        DiffusionHyperINR instance (on CPU after training)
        dataset:      MNISTHyperDataset
        name:         Run name for filename
        graph_dir:    Output directory
        device:       Device string
        num_examples: How many images to show
    """
    model.eval()
    underlying = dataset.dataset if hasattr(dataset, "dataset") else dataset
    h, w = underlying.image_shape  # (28, 28)

    indices = list(range(min(num_examples, len(dataset))))
    fig, axes = plt.subplots(3, len(indices), figsize=(3 * len(indices), 9))

    diffusion = model._get_diffusion(device)

    with torch.no_grad():
        for col, idx in enumerate(indices):
            image_32, coords, _ = dataset[idx]

            image_32_in = image_32.unsqueeze(0).to(device)  # (1, 1, 32, 32)
            coords_in = coords.unsqueeze(0).to(device)  # (1, 784, 2)

            # Use a mid-level timestep for a visible but still-recoverable noise level
            t_val = diffusion.T // 4
            t = torch.tensor([t_val], device=device)

            x_t, _ = diffusion.q_sample(image_32_in, t)
            x0_hat = model.unet(x_t, t)  # (1, 1, 32, 32)

            img_flat = x0_hat.view(1, -1)  # (1, 1024)
            flat_weights = model.hypernet(img_flat)
            pixel_preds = model.inr(coords_in, flat_weights)  # (1, 784, 1)

            # Original 28x28
            original_28 = image_32_in[0, 0, 2:30, 2:30].cpu().numpy()  # crop back to 28x28

            # UNet output cropped to 28x28
            unet_out_28 = x0_hat[0, 0, 2:30, 2:30].cpu().numpy()

            # INR reconstruction
            recon_28 = pixel_preds.squeeze(0).squeeze(-1).cpu().numpy().reshape(h, w)

            axes[0, col].imshow(original_28, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"Original #{idx}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(unet_out_28, cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title("UNet Denoise", fontsize=9)
            axes[1, col].axis("off")

            axes[2, col].imshow(recon_28, cmap="gray", vmin=0, vmax=1)
            axes[2, col].set_title("INR Recon", fontsize=9)
            axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("UNet Output", fontsize=10)
    axes[2, 0].set_ylabel("INR Recon", fontsize=10)

    plt.suptitle(f"DiffusionHyperINR Reconstructions — {name}")
    plt.tight_layout()

    out_path = os.path.join(graph_dir, f"{name}_reconstruction.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Reconstruction grid saved to {out_path}")


# ---------------------------------------------------------------------------
# Legacy helper — kept for backward compatibility
# ---------------------------------------------------------------------------


def _save_reconstruction_hyper(model, dataset, name: str, graph_dir: str, device: str = "cpu", num_examples: int = 4):
    """
    Legacy reconstruction saver for the original HyperINR (no diffusion).
    Kept for backward compatibility.
    """
    underlying = dataset.dataset if hasattr(dataset, "dataset") else dataset
    h, w = underlying.image_shape

    indices = list(range(min(num_examples, len(dataset))))
    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))

    with torch.no_grad():
        for col, idx in enumerate(indices):
            image_flat, coords, _ = dataset[idx]

            image_flat = image_flat.unsqueeze(0).to(device)
            coords_in = coords.unsqueeze(0).to(device)

            preds = model(image_flat, coords_in)
            recon = preds.squeeze(0).squeeze(-1).cpu().numpy().reshape(h, w)
            original = image_flat.squeeze(0).cpu().numpy().reshape(h, w)

            axes[0, col].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"Original {idx}")
            axes[0, col].axis("off")

            axes[1, col].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title(f"Recon {idx}")
            axes[1, col].axis("off")

    plt.suptitle(f"HyperINR Reconstructions — {name}")
    plt.tight_layout()
    out_path = os.path.join(graph_dir, f"{name}_reconstruction.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Reconstruction grid saved to {out_path}")
