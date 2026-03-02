import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_old.inr_mlp_hypernetwork.utils import _save_plot, _save_reconstruction_hyper


def train_inr_mlp_hypernet(
    model: nn.Module,
    dataset,
    name: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str | None = None,
    graph_dir: str = "src/inr_hypernetwork/graphs",
):
    """
    Train the HyperINR model.

    Each batch:
        1. Feed full images to the hypernetwork -> flat INR weights
        2. Feed all pixel coords through the INR using those weights -> pixel predictions
        3. Compute MSE loss against ground truth pixels
        4. Backpropagate through hypernetwork only (INR has no parameters)

    Args:
        model:      HyperINR instance (only hypernet weights are trainable)
        dataset:    MNISTHyperDataset
        name:       Run name for saving plots/weights
        num_epochs: Number of training epochs
        batch_size: Number of images per batch
        lr:         Learning rate for Adam
        device:     'cuda', 'mps', or 'cpu' (auto-detected if None)
        graph_dir:  Directory to save training plots
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(graph_dir, exist_ok=True)
    model = model.to(device)

    # Verify only hypernetwork parameters are being optimised
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameter groups: {len(trainable)} tensors")
    assert all("hypernet" in n for n in trainable), "Found trainable params outside the hypernetwork — check model definition!"

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    n_batches = len(train_loader)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.hypernet.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history: dict = {"train_mse": []}

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        for image, coords, pixels in train_loader:
            # image:  (B, 784)
            # coords: (B, N, 2)
            # pixels: (B, N, 1)
            image = image.to(device)
            coords = coords.to(device)
            pixels = pixels.to(device)

            optimizer.zero_grad()

            # Forward: hypernetwork produces weights, INR uses them
            preds = model(image, coords)  # (B, N, 1)

            loss = criterion(preds, pixels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # scheduler.step()

        epoch_loss = running_loss / n_batches
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix({"train_mse": f"{epoch_loss:.5f}", "lr": f"{current_lr:.2e}"})
        history["train_mse"].append(epoch_loss)

        _save_plot(history, name, graph_dir, epoch, num_epochs)

    model.cpu()
    print(f"\nTraining complete. Plot saved to {graph_dir}/{name}.png")
    print("Saving final reconstruction...")
    _save_reconstruction_hyper(model, dataset, name, graph_dir, device="cpu")
    print("Done.")
    return model
