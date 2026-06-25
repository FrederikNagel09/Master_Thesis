import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from torch.utils.data import Dataset, random_split

DATA_DIR = "data/shapenet_voxels"
NUM_SAMPLES = 4
THRESHOLD = 0.5
VAL_FRAC = 0.05


# ── Dataset ───────────────────────────────────────────────────────────────────


class ShapeNetVoxelDataset(Dataset):
    """Dataset wrapping a directory of .pt voxel grid files.

    Args:
        file_paths: List of paths to .pt files, each a (32, 32, 32) tensor.
    Returns:
        Tuple of (voxel_tensor, dummy_label) where voxel_tensor is (1, 32, 32, 32).
    """

    def __init__(self, file_paths: list[str]) -> None:
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        voxels = torch.load(self.file_paths[idx], map_location="cpu").float()
        return voxels.unsqueeze(0), 0


def build_splits(data_dir: str, val_frac: float = VAL_FRAC) -> tuple:
    """Load all .pt files and split into train/val datasets.

    Args:
        data_dir: Directory containing .pt voxel files.
        val_frac: Fraction of data to use for validation.
    Returns:
        train_dataset: Training split.
        val_dataset:   Validation split.
        full_dataset:  Full unsplit dataset (for statistics).
    """
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])
    full_dataset = ShapeNetVoxelDataset(files)
    n_val = int(len(full_dataset) * val_frac)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, val_dataset, full_dataset


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_statistics(dataset: ShapeNetVoxelDataset) -> dict:
    """Compute voxel statistics over the full dataset.

    Args:
        dataset: Full ShapeNetVoxelDataset (unplit).
    Returns:
        Dict with keys: min, max, mean, std, occupancy_rates (array of per-sample rates).
    """
    mins, maxs, means, stds, occupancy_rates = [], [], [], [], []
    for i in range(len(dataset)):
        voxels, _ = dataset[i]  # (1, 32, 32, 32)
        v = voxels.squeeze()  # (32, 32, 32)
        mins.append(v.min().item())
        maxs.append(v.max().item())
        means.append(v.mean().item())
        stds.append(v.std().item())
        occupancy_rates.append((v > THRESHOLD).float().mean().item())

    return {
        "min": np.min(mins),
        "max": np.max(maxs),
        "mean": np.mean(means),
        "std": np.mean(stds),
        "occupancy_rates": np.array(occupancy_rates),
    }


def print_statistics(stats: dict, n_total: int, n_train: int, n_val: int) -> None:
    """Print dataset split sizes and voxel statistics to stdout.

    Args:
        stats:   Output of compute_statistics.
        n_total: Total number of samples.
        n_train: Number of training samples.
        n_val:   Number of validation samples.
    Returns:
        None
    """
    print("\n── Dataset Statistics ───────────────────────────────")
    print(f"  Total samples : {n_total:,}")
    print(f"  Train         : {n_train:,}")
    print(f"  Val           : {n_val:,}")
    print(f"  Voxel min     : {stats['min']:.4f}")
    print(f"  Voxel max     : {stats['max']:.4f}")
    print(f"  Voxel mean    : {stats['mean']:.4f}")
    print(f"  Voxel std     : {stats['std']:.4f}")
    print(f"  Avg occupancy : {stats['occupancy_rates'].mean():.2%}")
    print("─────────────────────────────────────────────────────\n")


def plot_statistics(stats: dict, n_train: int, n_val: int) -> None:
    """Plot dataset statistics: split sizes and occupancy distribution.

    Args:
        stats:   Output of compute_statistics.
        n_train: Number of training samples.
        n_val:   Number of validation samples.
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Dataset Statistics", fontsize=13)

    # Split size bar chart
    axes[0].bar(["Train", "Val"], [n_train, n_val], color=["steelblue", "salmon"])
    axes[0].set_title("Train / Val Split")
    axes[0].set_ylabel("Samples")
    for ax_bar, val in zip(axes[0].patches, [n_train, n_val], strict=False):
        axes[0].text(ax_bar.get_x() + ax_bar.get_width() / 2, ax_bar.get_height() + 5, str(val), ha="center", va="bottom", fontsize=10)

    # Occupancy histogram
    axes[1].hist(stats["occupancy_rates"], bins=40, color="steelblue", edgecolor="white")
    axes[1].set_title("Occupancy Rate Distribution")
    axes[1].set_xlabel("Fraction of voxels occupied (> threshold)")
    axes[1].set_ylabel("Number of samples")

    plt.tight_layout()


# ── Mesh visualisation ────────────────────────────────────────────────────────


def load_sample(path: str) -> torch.Tensor:
    """Load a single voxel grid from a .pt file.

    Args:
        path: Path to the .pt file.
    Returns:
        Tensor of shape (32, 32, 32).
    """
    return torch.load(path, map_location="cpu")


def voxel_to_mesh(voxels: torch.Tensor, threshold: float = THRESHOLD) -> tuple:
    """Extract mesh vertices and triangles from a voxel grid via marching cubes.

    Args:
        voxels:    Tensor of shape (32, 32, 32).
        threshold: Occupancy threshold for surface extraction.
    Returns:
        vertices:  (N, 3) array of vertex positions.
        triangles: (M, 3) array of triangle indices.
    """
    volume = voxels.numpy().astype(np.float32)
    volume = np.pad(volume, 1, mode="constant", constant_values=0)
    vertices, triangles, _, _ = marching_cubes(volume, level=threshold)
    return vertices, triangles


def plot_mesh(ax: plt.Axes, vertices: np.ndarray, triangles: np.ndarray, title: str = "") -> None:
    """Render a mesh on a 3D matplotlib axis.

    Args:
        ax:        Matplotlib 3D axis to draw on.
        vertices:  (N, 3) vertex positions.
        triangles: (M, 3) triangle indices.
        title:     Optional title for the subplot.
    Returns:
        None
    """
    mesh = Poly3DCollection(vertices[triangles], alpha=0.7, edgecolor=None)
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax.add_collection3d(mesh)
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)


def plot_samples(data_dir: str, num_samples: int = NUM_SAMPLES) -> None:
    """Load and plot a few voxel samples as 3D meshes.

    Args:
        data_dir:    Directory containing .pt voxel files.
        num_samples: Number of samples to visualise.
    Returns:
        None
    """
    files = sorted(os.listdir(data_dir))[:num_samples]
    fig = plt.figure(figsize=(4 * num_samples, 4))
    fig.suptitle("Sample Voxel Grids", fontsize=13)
    for i, fname in enumerate(files):
        voxels = load_sample(os.path.join(data_dir, fname))
        vertices, triangles = voxel_to_mesh(voxels)
        ax = fig.add_subplot(1, num_samples, i + 1, projection="3d")
        plot_mesh(ax, vertices, triangles, title=fname)
    plt.tight_layout()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    train_dataset, val_dataset, full_dataset = build_splits(DATA_DIR)

    print("Computing statistics (this may take a moment)...")
    stats = compute_statistics(full_dataset)
    print_statistics(stats, len(full_dataset), len(train_dataset), len(val_dataset))

    plot_statistics(stats, len(train_dataset), len(val_dataset))
    plot_samples(DATA_DIR)
    plt.show()


if __name__ == "__main__":
    main()
