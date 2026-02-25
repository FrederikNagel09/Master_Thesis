import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# Add this function:
def _save_reconstruction(model, dataset, name, graph_dir, device):
    name = name.split("_")[0]

    height, width = dataset.image_shape
    model.eval().to(device)
    with torch.no_grad():
        coords = dataset.coords.to(device)  # all (H*W, 2) coords
        preds = model(coords).cpu().numpy().reshape(height, width)
    original = dataset.image  # already (H, W) float32 in [0,1]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(preds, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{name}_reconstruction.png"), dpi=150)
    plt.close(fig)
    print(f"Reconstruction saved to {graph_dir}/{name}_reconstruction.png")


# Add this function near the top of train.py:
def mse_to_psnr(mse: float) -> float:
    """Convert MSE (on [0,1] pixel values) to PSNR in dB."""
    if mse == 0:
        return float("inf")
    return -10 * math.log10(mse)  # add: import math at top


def train(
    model: nn.Module,
    dataset,
    name: str,
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    graph_dir: str = "src/INR/graphs",
):
    """
    Train an INR MLP on a coordinate → pixel value dataset.

    Args:
        model:       The neural network to train.
        dataset:     A MNISTCoordDataset (or any Dataset yielding (coords, pixels)).
        name:        Run name — used for the saved plot filename.
        num_epochs:  Number of training epochs.
        batch_size:  Batch size for the DataLoader.
        lr:          Learning rate for Adam.
        val_split:   Fraction of data to use as validation (e.g. 0.1 = 10%).
        device:      'cuda', 'mps', or 'cpu'. Auto-detected if None.
        graph_dir:   Directory where the training plot is saved.

    Returns:
        model: The trained model (on CPU).
        history: dict with keys 'train_loss' and 'val_loss' (lists, one value per epoch).
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(graph_dir, exist_ok=True)
    model = model.to(device)

    # Replace with:
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # MSE is the standard loss for INR regression on pixel values
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict = {"train_mse": []}

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        # --- Training ---
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / n_train

        epoch_bar.set_postfix({"train_loss": f"{train_loss:.5f}"})
        history["train_mse"].append(train_loss)

        # Save plot after every epoch so you can watch progress live
        _save_plot(history, name, graph_dir, epoch, num_epochs)

    model.cpu()
    print(f"\nTraining complete. Plot saved to {graph_dir}/{name}.png")
    print("Saving final reconstruction...")
    _save_reconstruction(model, dataset, name, graph_dir, device="cpu")
    print("Done.")
    return model


# ------------------------------------------------------------------
# Plotting helper
# ------------------------------------------------------------------


def _save_plot(history: dict, name: str, graph_dir: str, current_epoch: int, total_epochs: int):
    epochs = range(1, current_epoch + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_mse"], label="Train MSE", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"INR Training — {name}")
    ax.legend()
    ax.set_xlim(1, total_epochs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{name}.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Example entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.append(".")  # make sure local modules are importable

    from src.INR.dataloader import MNISTCoordDataset
    from src.INR.model import INRMLP

    dataset = MNISTCoordDataset(mnist_raw_dir="data/MNIST/raw", image_index=0)
    model = INRMLP(h1=256, h2=256, h3=256)

    train(
        model=model,
        dataset=dataset,
        name="inr_mnist_run1",
        num_epochs=200,
        batch_size=64,
        lr=1e-3,
    )
