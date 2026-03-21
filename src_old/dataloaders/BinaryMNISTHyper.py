import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryMNISTHyperDataset(Dataset):
    """
    Dataset for hypernetwork training using binarized MNIST.
    Each sample represents ONE full MNIST image and contains:
        image:  (784,)   — flattened binary pixel values {0, 1}, input to the hypernetwork
        coords: (784, 2) — all pixel coordinates normalized to [-1, 1], input to the INR
        pixels: (784, 1) — binary pixel values {0, 1}, reconstruction target
    """

    THRESHOLD = 0.5

    def __init__(self, mnist_raw_dir: str = "data/MNIST/raw", split: str = "train"):
        self.mnist_raw_dir = Path(mnist_raw_dir)
        self.images = self._load_all_images(split)  # (N, H, W) float32 in [0, 1]
        _, h, w = self.images.shape

        # Build coordinate grid, normalized to [-1, 1] — same for every image
        rows = torch.linspace(-1, 1, h)
        cols = torch.linspace(-1, 1, w)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        self.coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)  # (784, 2)

        # Binarize: values above threshold become 1.0, rest 0.0
        images_tensor = torch.from_numpy(self.images)  # (N, H, W) float32
        self.images_tensor = (images_tensor > self.THRESHOLD).float()  # (N, H, W) binary

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

    def __len__(self) -> int:
        return len(self.images_tensor)

    def __getitem__(self, idx: int):
        """
        Returns:
            image:  FloatTensor (784,)   — flattened binary image, input to hypernetwork
            coords: FloatTensor (784, 2) — pixel coordinates, input to INR
            pixels: FloatTensor (784, 1) — binary pixel values, reconstruction target
        """
        img = self.images_tensor[idx]  # (H, W)
        image_flat = img.flatten()  # (784,)
        pixels = image_flat.unsqueeze(-1)  # (784, 1)
        return image_flat, self.coords, pixels

    @property
    def image_shape(self):
        return self.images.shape[1:]  # (H, W)
