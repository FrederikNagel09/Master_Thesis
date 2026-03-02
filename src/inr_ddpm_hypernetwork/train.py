import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.inr_ddpm_hypernetwork.utils import _save_plot, _save_reconstruction_diffusion


def train(
    model: nn.Module,
    dataset,
    name: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    lambda_denoise: float = 1.0,
    device: str | None = None,
    graph_dir: str = "src/inr_ddpm_hypernetwork/graphs",
):
    """
    End-to-end training of DiffusionHyperINR.

    Each training step:
        1. Sample random timestep t for each image in the batch
        2. Add noise to the 32x32 MNIST image -> x_t
        3. UNet(x_t, t) -> x_0_hat  (predicted clean image)
        4. x_0_hat -> HyperNet -> flat INR weights
        5. INR(coords, weights) -> pixel_preds
        6. Total loss = MSE(pixel_preds, pixels)            [reconstruction]
                      + lambda_denoise * MSE(x_0_hat, x_0)  [denoising]

    The reconstruction loss trains both the hypernetwork AND the UNet via
    the end-to-end gradient path through x_0_hat.
    The denoising loss ensures the UNet learns proper image denoising.

    Args:
        model:           DiffusionHyperINR instance
        dataset:         MNISTHyperDataset
        name:            Run name for saving plots/weights
        num_epochs:      Number of training epochs
        batch_size:      Images per batch
        lr:              Learning rate for Adam
        lambda_denoise:  Weight for the denoising loss term (default 1.0)
        device:          'cuda', 'mps', or 'cpu' (auto-detected if None)
        graph_dir:       Directory to save training plots
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on: {device}")
    print(f"Loss = MSE(recon) + {lambda_denoise} * MSE(denoise)")

    os.makedirs(graph_dir, exist_ok=True)
    model = model.to(device)

    # Verify trainable params are only unet + hypernet
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(
        "unet" in n or "hypernet" in n for n in trainable_names
    ), "Found trainable params outside unet/hypernet — check model definition!"
    print(f"Trainable tensors: {len(trainable_names)} ({sum(p.numel() for p in model.parameters()):,} params total)")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    n_batches = len(train_loader)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.unet.parameters(), "lr": lr},
            {"params": model.hypernet.parameters(), "lr": 1e-3},  # hypernet needs faster updates
        ]
    )

    history: dict = {
        "total_loss": [],
        "recon_loss": [],
        "denoise_loss": [],
    }

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_denoise = 0.0

        for image_32, coords, pixels in train_loader:
            # image_32: (B, 1, 32, 32) — padded MNIST for UNet
            # coords:   (B, 784, 2)    — 28x28 INR coord grid
            # pixels:   (B, 784, 1)    — 28x28 ground truth pixels
            image_32 = image_32.to(device)
            coords = coords.to(device)
            pixels = pixels.to(device)
            optimizer.zero_grad()

            # Forward pass: returns (x0_hat, pixel_preds, t)
            x0_hat, pixel_preds, _t = model(image_32, coords)
            # print(f"[SANITY] pixel_preds range : {pixel_preds.min():.4f} - {pixel_preds.max():.4f}")
            # Reconstruction loss: INR pixel predictions vs ground truth
            recon_loss = criterion(pixel_preds, pixels)

            # Denoising loss: UNet output vs original clean image
            denoise_loss = criterion(x0_hat, image_32)

            loss = recon_loss + lambda_denoise * denoise_loss
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_total += loss.item()
            running_recon += recon_loss.item()
            running_denoise += denoise_loss.item()

        epoch_total = running_total / n_batches
        epoch_recon = running_recon / n_batches
        epoch_denoise = running_denoise / n_batches

        history["total_loss"].append(epoch_total)
        history["recon_loss"].append(epoch_recon)
        history["denoise_loss"].append(epoch_denoise)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            {
                "total": f"{epoch_total:.5f}",
                "recon": f"{epoch_recon:.5f}",
                "denoise": f"{epoch_denoise:.5f}",
                "lr": f"{current_lr:.2e}",
            }
        )

        _save_plot(history, name, graph_dir, epoch, num_epochs)

    model.cpu()
    print(f"\nTraining complete. Plot saved to {graph_dir}/{name}.png")
    print("Saving final reconstruction...")
    _save_reconstruction_diffusion(model, dataset, name, graph_dir, device="cpu")
    print("Done.")
    return model
