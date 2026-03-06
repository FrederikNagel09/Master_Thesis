import os

import torch
from matplotlib import pyplot as plt


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
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: total loss
    axes[0].plot(epochs, history["loss"], label="Total loss")
    axes[0].set_title("Total ELBO Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Right: the three components
    axes[1].plot(epochs, history["ldiff"], label="L_diff")
    axes[1].plot(epochs, history["lprior"], label="L_prior")
    axes[1].plot(epochs, history["lrec"], label="L_rec")
    axes[1].set_title("Loss Components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(f"NDM Training — {name}", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(graph_dir, f"{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved to: {out_path}")


def plot_vae_training(history: dict, name: str, graph_dir: str):
    os.makedirs(graph_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["steps"], history["train_elbo"], label="Train ELBO loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"VAE Training — {name}")
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(graph_dir, f"{name}.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Training graph saved to: {save_path}")


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
