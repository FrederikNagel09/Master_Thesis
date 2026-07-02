import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.datasets as datasets
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# ── 3D Helper Functions ───────────────────────────────────────────────────────


def load_sample(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def voxel_to_mesh(voxels: torch.Tensor, threshold: float = 0.5) -> tuple:
    # Swap the Y and Z axes to align ShapeNet's "up" with Matplotlib's "up"
    voxels = voxels.transpose(1, 2)

    volume = voxels.numpy().astype(np.float32)
    volume = np.pad(volume, 1, mode="constant", constant_values=0)
    vertices, triangles, _, _ = marching_cubes(volume, level=threshold)
    return vertices, triangles


def plot_mesh(ax: plt.Axes, vertices: np.ndarray, triangles: np.ndarray) -> None:
    mesh = Poly3DCollection(vertices[triangles], alpha=0.7, edgecolor=None)
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax.add_collection3d(mesh)
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_axis_off()


# ── Main Grid Generation ──────────────────────────────────────────────────────


def create_seamless_grid(
    voxel_dir="data/shapenet_voxels", output_filename="mnist_voxels_perfect_grid.png"
):
    print("Loading datasets...")
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True)

    # Grab voxel files
    voxel_files = sorted(
        [os.path.join(voxel_dir, f) for f in os.listdir(voxel_dir) if f.endswith(".pt")]
    )
    if len(voxel_files) < 25:
        raise ValueError(
            f"Need at least 25 voxel samples to fill the grid, found {len(voxel_files)}"
        )

    # 1. Control the exact gap between the two main grids
    gap_width = 0.25
    width_ratios = [1, 1, 1, 1, 1, gap_width, 1, 1, 1, 1, 1]

    # 2. Math to perfectly match the canvas to the grid aspect ratio
    scaling_factor = 1.1
    fig_width = (10 + gap_width) * scaling_factor
    fig_height = (5 * scaling_factor) + 0.4  # Extra room for the bottom labels

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # 3. Create a unified 5x11 grid with ZERO internal padding
    gs = gridspec.GridSpec(5, 11, width_ratios=width_ratios, wspace=0, hspace=0)

    mnist_idx = 0
    voxel_idx = 0

    print("Rendering grid (3D meshes may take a moment)...")
    for row in range(5):
        for col in range(11):
            # Skip drawing anything in the middle divider column
            if col == 5:
                continue

            # --- Left Side: MNIST (2D Axis) ---
            if col < 5:
                ax = fig.add_subplot(gs[row, col])
                img, _ = mnist_dataset[mnist_idx]
                ax.imshow(img, cmap="gray")
                mnist_idx += 1

                # Clean up 2D axes
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if row == 4 and col == 2:
                    ax.text(
                        0.5,
                        -0.2,
                        "a) MNIST",
                        transform=ax.transAxes,
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="top",
                    )

            # --- Right Side: ShapeNet Voxels (3D Axis) ---
            else:
                # CRITICAL: You must specify projection='3d' for this half of the grid
                ax = fig.add_subplot(gs[row, col], projection="3d")
                ax.view_init(elev=20, azim=30)
                voxels = load_sample(voxel_files[voxel_idx]).squeeze()
                vertices, triangles = voxel_to_mesh(voxels)
                plot_mesh(ax, vertices, triangles)
                voxel_idx += 1

                if row == 4 and col == 8:
                    # 3D axes don't handle standard labels well with set_axis_off(), use transform
                    ax.text2D(
                        0.5,
                        -0.2,
                        "b) ShapeNet Chairs",
                        transform=ax.transAxes,
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="top",
                    )

    # Save with tight bounding bounds
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Success! Seamless plot saved as '{output_filename}'")


if __name__ == "__main__":
    create_seamless_grid()
