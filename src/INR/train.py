import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.INR.utils import _save_plot, _save_reconstruction


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
