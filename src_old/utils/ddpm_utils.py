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
from torch.utils.data import DataLoader

sys.path.append(".")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


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
