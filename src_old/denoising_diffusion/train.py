"""
train.py  -  training logic for DDPM on MNIST
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")

from src_old.denoising_diffusion.model import Diffusion, UNet

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_mnist_dataloader(batch_size: int, img_size: int = 32, data_root: str = "./data"):
    """
    Downloads MNIST and returns a DataLoader.
    Images are resized to img_size x img_size and normalised to [-1, 1].
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),  # [0, 255] → [0.0, 1.0]
            transforms.Normalize((0.5,), (0.5,)),  # [0, 1]   → [-1, 1]
        ]
    )
    dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def save_grid(images: torch.Tensor, path: str, nrow: int = 10, title: str = ""):
    """Save a uint8 image tensor as a grid PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(nrow, nrow))
    if title:
        plt.title(title)
    plt.imshow(ndarr.squeeze(), cmap="gray")  # squeeze handles 1-channel
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_loss(losses: list, save_path: str):
    """Save a loss curve to save_path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
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
