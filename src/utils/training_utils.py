import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.plot_utils import plot_training_and_reconstruction, plot_vae_training


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
    model: nn.Module,
    optimizer,
    data_loader,
    epochs,
    device,
    name: str = "ddpm",
    graph_dir: str = "src/results/ddpm/training_graphs",
    log_every_n_steps: int = 20,
):
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    history: dict = {"train_elbo": [], "steps": []}
    global_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, list | tuple):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_count += 1
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch + 1}/{epochs}")
            progress_bar.update()

            if global_step % log_every_n_steps == 0:
                avg_loss = running_loss / running_count
                fractional_epoch = global_step / len(data_loader)
                history["train_elbo"].append(avg_loss)
                history["steps"].append(fractional_epoch)
                running_loss = 0.0
                running_count = 0

    # Save loss plot
    plot_vae_training(history, name, graph_dir)

    return model


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
    model: nn.Module,
    optimizer,
    data_loader,
    epochs: int,
    device: str,
    name: str = "ndm",
    graph_dir: str = "src/results/ndm/training_graphs",
    log_every_n_steps: int = 20,
    warmup_steps: int = 5000,  # paper uses 45k for CIFAR, 20k for ImageNet
    peak_lr: float = 4e-4,  # must match the lr used to construct optimizer
):
    model.train()

    total_steps = len(data_loader) * epochs

    def lr_lambda(current_step: int) -> float:
        """
        Linear warmup then linear decay back to ~0, matching the paper's
        polynomial decay schedule (Appendix C).
        Returns a multiplier on the base lr set in the optimizer.
        """
        if current_step < warmup_steps:
            # Ramp from 1e-8/peak_lr up to 1.0 (i.e. peak_lr)
            return max(1e-8 / peak_lr, current_step / max(warmup_steps, 1))
        else:
            # Linear decay from 1.0 down to 1e-8/peak_lr
            progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            floor = 1e-8 / peak_lr
            return max(floor, 1.0 - progress * (1.0 - floor))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    progress_bar = tqdm(range(total_steps), desc="Training NDM")
    history: dict = {"train_elbo": [], "steps": [], "lr": []}
    global_step = 0
    running_loss = 0.0
    running_count = 0
    """
    alpha_T = model.sqrt_alpha_cumprod[-1].item()
    sigma_T = model.sigma[-1].item()
    alpha_1 = model.sqrt_alpha_cumprod[0].item()
    print(f"\n── Schedule diagnostics ──────────────────────────────")
    print(f"  T={model.T}")
    print(f"  a_T (sqrt_ᾱ_T)  = {alpha_T:.6f}  (should be ~0 for signal to be gone)")
    print(f"  a_T              = {sigma_T:.6f}  (should be ~1)")
    print(f"  a_1 (sqrt_ᾱ_1)  = {alpha_1:.6f}  (should be ~1)")
    print(f"  a_T² contribution to L_prior will be ~{alpha_T**2:.2e}")
    print(f"  Total steps: {total_steps},  Warmup: {warmup_steps} ({100*warmup_steps/total_steps:.1f}%)")
    print(f"─────────────────────────────────────────────────────\n")
    """
    for _ in range(epochs):
        for x in data_loader:
            if isinstance(x, list | tuple):
                x = x[0]
            x = x.to(device)

            optimizer.zero_grad()
            loss, diff, prior = model.loss(x)
            loss.backward()

            # Gradient clipping helps stabilise joint F_phi + network training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # step every iteration, not every epoch

            running_loss += loss.item()
            running_count += 1
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                kl=f"{prior.item():.4f}",
                diff=f"{diff.item():.4f}",
                lr=f"{current_lr:.2e}",
            )
            progress_bar.update()

            if global_step % log_every_n_steps == 0:
                avg_loss = running_loss / running_count
                fractional_epoch = global_step / len(data_loader)
                history["train_elbo"].append(avg_loss)
                history["steps"].append(fractional_epoch)
                history["lr"].append(current_lr)
                running_loss = 0.0
                running_count = 0

    plot_vae_training(history, name, graph_dir)
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
