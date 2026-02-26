import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTCoordDataset(Dataset):
    """
    Dataset that represents a single MNIST image as a collection of (coordinate, pixel_value) pairs.

    Each sample is:
        x: (coord_u, coord_v) — pixel coordinates normalized to [-1, 1]
        y: pixel value normalized to [0, 1]

    Compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, mnist_raw_dir: str = "data/MNIST/raw", image_index: int = 0):
        """
        Args:
            mnist_raw_dir: Path to the MNIST raw directory containing the binary files.
            image_index:   Which image in the dataset to use (0-indexed).
        """
        self.mnist_raw_dir = Path(mnist_raw_dir)
        self.image = self._load_image(image_index)  # (H, W) float32 in [0, 1]

        height, width = self.image.shape

        # Build coordinate grid, normalized to [-1, 1]
        rows = torch.linspace(-1, 1, height)
        cols = torch.linspace(-1, 1, width)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")  # both (H, W)

        # Flatten everything: N = H * W samples
        self.coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)  # (N, 2)
        self.pixels = torch.from_numpy(self.image).flatten().unsqueeze(-1)  # (N, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, index: int) -> np.ndarray:
        """Load a single image from the MNIST idx file (handles plain + gzip)."""
        candidates = [
            self.mnist_raw_dir / "train-images-idx3-ubyte",
            self.mnist_raw_dir / "train-images-idx3-ubyte.gz",
            self.mnist_raw_dir / "t10k-images-idx3-ubyte",
            self.mnist_raw_dir / "t10k-images-idx3-ubyte.gz",
        ]

        path = None
        for c in candidates:
            if c.exists():
                path = c
                break

        if path is None:
            raise FileNotFoundError(
                f"No MNIST image file found in {self.mnist_raw_dir}. Expected train-images-idx3-ubyte[.gz] or t10k-images-idx3-ubyte[.gz]."
            )

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"Not an MNIST image file (magic={magic})"
            if index >= n_images:
                raise IndexError(f"image_index={index} out of range (dataset has {n_images} images)")
            # Skip to the requested image
            f.read(index * rows * cols)
            buf = f.read(rows * cols)

        image = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols).astype(np.float32) / 255.0
        return image

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        """
        Returns:
            x: FloatTensor of shape (2,) — normalized (row, col) coordinate
            y: FloatTensor of shape (1,) — normalized pixel value in [0, 1]
        """
        return self.coords[idx], self.pixels[idx]

    @property
    def image_shape(self):
        return self.image.shape
