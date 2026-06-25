import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

THRESHOLD = 0.5


def voxel_to_mesh(voxels: torch.Tensor, threshold: float = THRESHOLD) -> tuple:
    """Extract mesh from voxel grid via marching cubes.

    Args:
        voxels:    Tensor of shape (32, 32, 32) or (1, 32, 32, 32).
        threshold: Occupancy threshold.
    Returns:
        vertices:  (N, 3) array.
        triangles: (M, 3) array.
    """
    if voxels.dim() == 4:
        voxels = voxels.squeeze(0)
    volume = voxels.cpu().numpy().astype(np.float32)
    volume = np.pad(volume, 1, mode="constant", constant_values=0)
    vertices, triangles, _, _ = marching_cubes(volume, level=threshold)
    return vertices, triangles


def render_voxel_to_image(voxels: torch.Tensor, resolution: int = 64) -> np.ndarray:
    """Render a voxel grid as a 3D mesh and return it as an RGB numpy image.

    Args:
        voxels:     Tensor of shape (32, 32, 32) or (1, 32, 32, 32).
        resolution: Pixel resolution of the output image (square).
    Returns:
        img: (resolution, resolution, 3) uint8 numpy array.
    """
    vertices, triangles = voxel_to_mesh(voxels)

    dpi = 100
    size = resolution / dpi
    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(vertices[triangles], alpha=0.7, edgecolor=None)
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax.add_collection3d(mesh)

    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Grab pixel buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    from PIL import Image

    img = np.array(Image.open(buf).convert("RGB").resize((resolution, resolution)))
    return img
