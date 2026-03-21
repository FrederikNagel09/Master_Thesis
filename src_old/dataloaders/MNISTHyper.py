import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTHyperDataset(Dataset):
    """
    Dataset for hypernetwork training.

    Each sample represents ONE full MNIST image and contains:
        image:  (784,)  — flattened pixel values in [0, 1], input to the hypernetwork
        coords: (784, 2) — all pixel coordinates normalized to [-1, 1], input to the INR
        pixels: (784, 1) — all pixel values in [0, 1], regression target

    The hypernetwork sees the full image to produce INR weights.
    The INR then maps coordinates -> pixel values.
    """

    def __init__(self, mnist_raw_dir: str = "data/MNIST/raw", split: str = "train"):
        """
        Args:
            mnist_raw_dir: Path to the MNIST raw directory.
            split:         'train' or 'test'
        """
        self.mnist_raw_dir = Path(mnist_raw_dir)
        self.images = self._load_all_images(split)  # (N, H, W) float32 in [0, 1]

        _, h, w = self.images.shape  # n=60000 for train, 10000 for test; h=w=28

        # Build coordinate grid, normalized to [-1, 1] — same for every image
        rows = torch.linspace(-1, 1, h)
        cols = torch.linspace(-1, 1, w)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")  # (H, W)
        self.coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)  # (H*W, 2)

        # Convert images to tensors
        self.images_tensor = torch.from_numpy(self.images)  # (N, H, W)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_images(self, split: str) -> np.ndarray:
        """Load all images from the MNIST idx file."""
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

        path = None
        for c in candidates:
            if c.exists():
                path = c
                break

        if path is None:
            raise FileNotFoundError(f"No MNIST image file found in {self.mnist_raw_dir} for split='{split}'.")

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"Not an MNIST image file (magic={magic})"
            buf = f.read(n_images * rows * cols)

        images = np.frombuffer(buf, dtype=np.uint8).reshape(n_images, rows, cols).astype(np.float32) / 255.0
        return images

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images_tensor)

    def __getitem__(self, idx: int):
        """
        Returns:
            image:  FloatTensor (784,)   — flattened image, input to hypernetwork
            coords: FloatTensor (784, 2) — pixel coordinates, input to INR
            pixels: FloatTensor (784, 1) — pixel values, reconstruction target
        """
        img = self.images_tensor[idx]  # (H, W)
        image_flat = img.flatten()  # (784,)
        pixels = image_flat.unsqueeze(-1)  # (784, 1)
        return image_flat, self.coords, pixels

    @property
    def image_shape(self):
        return self.images.shape[1:]  # (H, W)
