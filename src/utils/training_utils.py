import logging
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ndm import NDM
from src.utils.plot_utils import plot_ndm_training, plot_training_and_reconstruction, plot_vae_training


def train_inr_siren(
    model: nn.Module,
    dataset,
    name: str,
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    graph_dir: str = "src/results/basic_inr/training_graphs",
):
    """
    Train an INR MLP on a coordinate → pixel value dataset.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(graph_dir, exist_ok=True)
    model = model.to(device)

    # initialize dataloader
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize history dict for storing training metrics (losses, etc) to plot later
    history: dict = {"train_mse": []}

    # Run actual training:
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    for _ in epoch_bar:
        # --- Training ---
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            # x = (B, N, 2) pixel coordinates
            # y = (B, N, 1) pixel values
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward pass: feed pixel coordinates through the INR to get predicted pixel values
            pred = model(x)

            # Compute MSE loss against ground truth pixel values, backpropagate, and step optimizer
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / n_train

        epoch_bar.set_postfix({"train_loss": f"{train_loss:.5f}"})
        history["train_mse"].append(train_loss)

    model.cpu()
    print("Saving final reconstruction...")
    plot_training_and_reconstruction(history, name, graph_dir, num_epochs, num_epochs, model, dataset, device="cpu")
    print("Done.")
    return model


def train_ddpm(
    device: str = "cpu",
    T: int = 1000,  # noqa: N803
    img_size: int = 32,
    channels: int = 32,
    time_dim: int = 256,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs: int = 10,
    experiment_name: str = "ddpm_mnist",
    weights_dir: str = "src/DDPM/weights",
    graphs_dir: str = "src/DDPM/graphs",
    results_dir: str = "src/DDPM/results",
    data_root: str = "./data",
):
    """Train a DDPM UNet on MNIST."""
    # Directories
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # imports:
    from src.models.ddpm import Diffusion
    from src.models.unet import UNet
    from src.utils.ddpm_utils import get_mnist_dataloader, plot_loss

    # Data
    dataloader = get_mnist_dataloader(batch_size, img_size=img_size, data_root=data_root)

    # Model & diffusion
    model = UNet(img_size=img_size, c_in=1, c_out=1, time_dim=time_dim, channels=channels, device=device).to(device)

    diffusion = Diffusion(T=T, beta_start=1e-4, beta_end=0.02, img_size=img_size, img_channels=1, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    all_losses = []

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs}")
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for images, _labels in pbar:
            images = images.to(device)

            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.q_sample(images, t)
            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            all_losses.append(loss.item())
            pbar.set_postfix(MSE=f"{loss.item():.4f}")

        avg = sum(epoch_losses) / len(epoch_losses)
        logging.info(f"Epoch {epoch} avg MSE: {avg:.4f}")

    # Save weights
    weight_path = os.path.join(weights_dir, f"{experiment_name}_epoch{epoch:03d}.pt")
    torch.save(model.state_dict(), weight_path)
    logging.info(f"Weights saved to {weight_path}")

    # Save loss plot
    plot_loss(all_losses, save_path=os.path.join(graphs_dir, f"{experiment_name}_loss.png"))

    logging.info("Training complete.")
    return model, all_losses


def train_inr_mlp_hypernet(
    model: nn.Module,
    dataset,
    name: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str | None = None,
    graph_dir: str = "src/results/hypernet_inr/training_graphs",
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
    # Imports:
    from src.utils.general_utils import _save_plot, _save_reconstruction_hyper

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


def train_ndm(
    model: NDM,
    dataloader: DataLoader,
    name: str,
    num_epochs: int = 100,
    lr: float = 2e-4,
    grad_clip: float = 1.0,
    sample_every: int = 5,
    sample_steps: int | None = None,
    device: str | None = None,
    sample_dir: str = "src/results/ndm/samples",
    graph_dir: str = "src/results/ndm/training_graph",
) -> NDM:
    """
    Train an NDM on a dataloader that yields (image, label) batches.
    Images should be shape (B, 1, 28, 28) in range [-1, 1].

    Returns the trained model (on CPU).
    """

    # ---- Device ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(sample_dir, exist_ok=True)

    model = model.to(device)

    # ---- Optimiser — lower lr for F_phi, it only needs to learn a smooth warp ----
    optimizer = optim.Adam(
        [
            {"params": model.denoiser.parameters(), "lr": lr},
            {"params": model.fphi.parameters(), "lr": lr * 0.5},
        ],
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---- History dict ----
    history: dict = {"loss": [], "ldiff": [], "lprior": [], "lrec": []}

    # ---- Training loop ----
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        running = {"loss": 0.0, "ldiff": 0.0, "lprior": 0.0, "lrec": 0.0}

        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=False)
        for x, _ in batch_bar:
            x = x.to(device)

            losses = model.loss(x)

            optimizer.zero_grad()
            losses["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item() * x.size(0)

        scheduler.step()

        # ---- Epoch averages ----
        n = len(dataloader.dataset)
        avg = {k: v / n for k, v in running.items()}
        for k, v in avg.items():
            history[k].append(v)

        epoch_bar.set_postfix(
            {
                "loss": f"{avg['loss']:.4f}",
                "ldiff": f"{avg['ldiff']:.4f}",
                "lprior": f"{avg['lprior']:.4f}",
                "lrec": f"{avg['lrec']:.4f}",
            }
        )

        # Periodically sample to visualize training progress.
        # Use EMA weights for sampling, but restore original weights immediately after.
        if epoch % sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample(4, device=torch.device(device), steps=sample_steps)

            import matplotlib.pyplot as plt
            import numpy as np

            imgs = ((samples * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
            rows = [np.concatenate([imgs[r * 2 + c, 0] for c in range(2)], axis=1) for r in range(2)]
            grid = np.concatenate(rows, axis=0)

            fig, ax = plt.subplots()
            ax.imshow(grid, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            ax.set_title(f"Epoch {epoch} / {num_epochs}", fontsize=12)
            fig.savefig(os.path.join(sample_dir, f"{name}.png"), bbox_inches="tight", dpi=150)
            plt.close(fig)

    model.cpu()
    print("Saving training plot...")
    plot_ndm_training(history, name, graph_dir)
    print("Done.")
    return model
    return model


def train_vae(
    model,
    optimizer,
    data_loader,
    epochs,
    device,
    name: str = "vae",
    graph_dir: str = "src/results/vae/training_graphs",
    log_every_n_steps: int = 20,
):
    os.makedirs(graph_dir, exist_ok=True)
    model.train()
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    history: dict = {"train_elbo": [], "steps": []}
    global_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(epochs):
        for x in iter(data_loader):
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_count += 1
            global_step += 1

            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch + 1}/{epochs}")
            progress_bar.update()

            if global_step % log_every_n_steps == 0:
                avg_loss = running_loss / running_count
                fractional_epoch = global_step / len(data_loader)
                history["train_elbo"].append(avg_loss)
                history["steps"].append(fractional_epoch)
                running_loss = 0.0
                running_count = 0

    print("Last Loss: ", loss)
    plot_vae_training(history, name, graph_dir)
    return model


def train_inr_vae(
    model,
    mnist_train_loader,
    epochs,
    name: str,
    lr: float = 1e-4,
    device: str | None = None,
    graph_dir: str = "src/results/vae_inr_hypernet/training_graphs",
):
    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = len(mnist_train_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    history: dict = {"train_elbo": [], "steps": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # beta = max(0.0, min(beta_max, beta_max * (epoch - warmup_start) / warmup_epochs))
        model.beta = 0.001

        for image_flat, coords, pixels in mnist_train_loader:
            image_flat = image_flat.to(device)  # (B, 784)
            coords = coords.to(device)  # (B, 784, 2)
            pixels = pixels.to(device)  # (B, 784, 1)

            optimizer.zero_grad()
            loss, recon_loss, kl = model(image_flat, coords, pixels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                epoch=f"{epoch + 1}/{epochs}",
                loss=f"{loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                kl=f"{kl.item():.2f}",
                beta=f"{model.beta:.4f}",
            )
            progress_bar.update()

        # Record average loss once per epoch, after all batches
        avg_loss = total_loss / len(mnist_train_loader)
        history["train_elbo"].append(avg_loss)
        history["steps"].append(epoch)

    print("Last Loss: ", loss)
    plot_vae_training(history, name, graph_dir)
    return model
