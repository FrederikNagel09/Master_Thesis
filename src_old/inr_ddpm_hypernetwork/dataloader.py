import gzip
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset


class MNISTHyperDataset(Dataset):
    """
    Dataset for DiffusionHyperINR training.

    Each sample contains:
        image_32:  (1, 32, 32)  — zero-padded MNIST image, input to the UNet
        image_28:  (1, 28, 28)  — original MNIST image (for target pixel matching)
        coords:    (784, 2)     — 28x28 pixel coords normalized to [-1, 1], input to INR
        pixels:    (784, 1)     — 28x28 pixel values in [0, 1], INR regression target

    The UNet operates on 32x32 images (clean power-of-2 spatial dims).
    The INR reconstructs at 28x28 to match the original MNIST resolution.
    The denoising target is also 32x32.
    """

    # Padding: 28 -> 32 means 2px on each side
    PAD = 2

    def __init__(self, mnist_raw_dir: str = "data/MNIST/raw", split: str = "train"):
        self.mnist_raw_dir = Path(mnist_raw_dir)
        images_28 = self._load_all_images(split)  # (N, 28, 28) float32 in [0,1]

        _, h, w = images_28.shape  # h=w=28

        # Pad 28x28 -> 32x32
        images_tensor = torch.from_numpy(images_28).unsqueeze(1)  # (N, 1, 28, 28)
        self.images_32 = F.pad(images_tensor, [self.PAD] * 4, mode="constant", value=0.0)  # (N, 1, 32, 32)
        self.images_28 = images_tensor  # (N, 1, 28, 28)

        # 28x28 coordinate grid for INR target
        rows = torch.linspace(-1, 1, h)
        cols = torch.linspace(-1, 1, w)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        self.coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)  # (784, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_images(self, split: str) -> np.ndarray:
        if split == "train":
            candidates = [
                self.mnist_raw_dir / "train-images-idx3-ubyte",
                self.mnist_raw_dir / "train-images-idx3-ubyte.gz",
            ]
        else:
            candidates = [
                self.mnist_raw_dir / "t10k-images-idx3-ubyte",
                self.mnist_raw_dir / "t10k-images-idx3-ubyte.gz",
            ]

        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No MNIST image file found in {self.mnist_raw_dir} for split='{split}'.")

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"Not an MNIST image file (magic={magic})"
            buf = f.read(n_images * rows * cols)

        return np.frombuffer(buf, dtype=np.uint8).reshape(n_images, rows, cols).astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images_32)

    def __getitem__(self, idx: int):
        """
        Returns:
            image_32: FloatTensor (1, 32, 32) — padded image for UNet input/target
            coords:   FloatTensor (784, 2)    — 28x28 coords for INR
            pixels:   FloatTensor (784, 1)    — 28x28 pixel values for INR target
        """
        image_32 = self.images_32[idx]  # (1, 32, 32)
        image_28 = self.images_28[idx]  # (1, 28, 28)
        pixels = image_28.flatten().unsqueeze(-1)  # (784, 1)
        return image_32, self.coords, pixels

    @property
    def image_shape(self):
        """Original 28x28 shape."""
        return (28, 28)


# ---------------------------------------------------------------------------
# Legacy single-image dataset — kept for inference / visualisation
# ---------------------------------------------------------------------------


class MNISTCoordDataset(Dataset):
    """
    Single-image dataset for inference and per-image visualisation.
    """

    PAD = 2  # 28 -> 32

    def __init__(self, mnist_raw_dir: str = "data/MNIST/raw", image_index: int = 0):
        self.mnist_raw_dir = Path(mnist_raw_dir)
        self.image_28 = self._load_image(image_index)  # (28, 28) float32

        height, width = self.image_28.shape
        rows = torch.linspace(-1, 1, height)
        cols = torch.linspace(-1, 1, width)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        self.coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)  # (784, 2)
        self.pixels = torch.from_numpy(self.image_28).flatten().unsqueeze(-1)  # (784, 1)

        img_tensor = torch.from_numpy(self.image_28).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
        self.image_32 = F.pad(img_tensor, [self.PAD] * 4).squeeze(0)  # (1, 32, 32)
        self.image_flat_28 = torch.from_numpy(self.image_28).flatten()  # (784,)

    def _load_image(self, index: int) -> np.ndarray:
        candidates = [
            self.mnist_raw_dir / "train-images-idx3-ubyte",
            self.mnist_raw_dir / "train-images-idx3-ubyte.gz",
            self.mnist_raw_dir / "t10k-images-idx3-ubyte",
            self.mnist_raw_dir / "t10k-images-idx3-ubyte.gz",
        ]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No MNIST image file found in {self.mnist_raw_dir}.")

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051
            if index >= n_images:
                raise IndexError(f"image_index={index} out of range ({n_images} images)")
            f.read(index * rows * cols)
            buf = f.read(rows * cols)

        return np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols).astype(np.float32) / 255.0

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        return self.coords[idx], self.pixels[idx]

    @property
    def image_shape(self):
        return self.image_28.shape
