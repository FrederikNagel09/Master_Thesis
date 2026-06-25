"""
plotting.py
Universal training plot for all models.

Expects a history dict with keys:
    "steps"  : list of fractional epoch values
    "total"  : list of total loss values
    "diff"   : list of diffusion loss values   (0.0 if unused)
    "prior"  : list of KL/prior loss values    (0.0 if unused)
    "rec"    : list of reconstruction loss     (0.0 if unused)
    "lr"     : list of learning rate values    (only used if use_scheduler=True)

Panels are shown dynamically — any component whose values are all zero is dropped.
Maximum 4 panels. LR is overlaid as a twin y-axis on the total loss panel when
use_scheduler=True.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# =============================================================================
# Helpers
# =============================================================================
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from torchvision import datasets, transforms
from tqdm import tqdm

from src.configs.train_plot_config import _COLORS, _LABELS

VOXEL_THRESHOLD = 0.5
# Slight per-sample camera rotation so each mesh is seen from a different angle
_AZIM_OFFSETS = [-30, -15, 0, 15, 30, 45]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _voxels_to_mesh(voxels: np.ndarray, threshold: float = VOXEL_THRESHOLD) -> tuple[np.ndarray, np.ndarray] | None:
    """Run marching cubes on a (D, H, W) voxel array.

    Args:
        voxels:    (D, H, W) float numpy array.
        threshold: Surface extraction threshold.
    Returns:
        (vertices, triangles) or None if no surface found.
    """
    volume = np.pad(voxels, 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = marching_cubes(volume, level=threshold)
        return verts, faces
    except ValueError:
        # marching_cubes raises if no surface exists at this threshold
        return None


def _render_mesh_on_ax(
    ax: plt.Axes,
    voxels: np.ndarray,
    title: str = "",
    azim: float = 0.0,
    elev: float = 25.0,
) -> None:
    """Render a voxel grid as a 3D mesh on a matplotlib 3D axis.

    Args:
        ax:     Matplotlib 3D axis.
        voxels: (D, H, W) float numpy array.
        title:  Optional subplot title.
        azim:   Camera azimuth angle in degrees.
        elev:   Camera elevation angle in degrees.
    Returns:
        None
    """
    result = _voxels_to_mesh(voxels)
    if result is not None:
        verts, faces = result
        mesh = Poly3DCollection(verts[faces], alpha=0.75, edgecolor=None)
        mesh.set_facecolor([0.5, 0.7, 1.0])
        ax.add_collection3d(mesh)
        scale = verts.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
    else:
        # Empty grid — draw a wireframe cube as placeholder
        ax.text(0.5, 0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="#aaaaaa")

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7)


def _samples_to_voxel_grids(
    raw_samples: torch.Tensor,
    channels: int,
    grid_size: int,
) -> np.ndarray:
    """Reshape flat sample tensor to per-sample voxel grids.

    Args:
        raw_samples: (B, data_dim) float tensor from model.sample().
        channels:    Number of channels (1 for ShapeNet).
        grid_size:   Spatial size of voxel grid (32).
    Returns:
        (B, D, H, W) numpy array — channel dim squeezed out.
    """
    B = raw_samples.shape[0]  # noqa: N806
    grids = raw_samples.view(B, channels, grid_size, grid_size, grid_size)
    return grids[:, 0].cpu().numpy()  # (B, D, H, W)


def _smooth(values: list[float], n_points: int) -> tuple[list[float], list[float]]:
    """Return (smoothed_values, trimmed_indices) using a uniform moving average."""
    kernel = max(1, len(values) // n_points)
    smoothed = np.convolve(values, np.ones(kernel) / kernel, mode="valid")
    return smoothed, kernel


def _style_ax(ax: plt.Axes) -> None:
    """Apply shared axis styling."""
    spine_color = "#cccccc"
    ax.tick_params(colors="#555555", labelsize=9)
    ax.set_xlabel("Epoch", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _plot_loss_panel(
    ax: plt.Axes,
    steps: list[float],
    values: list[float],
    key: str,
) -> None:
    """Plot raw (faint) + smoothed (bold) loss curve onto ax."""
    color = _COLORS[key]
    ax.set_title(_LABELS[key], fontsize=12, fontweight="medium", pad=8)
    ax.set_ylabel("Loss", fontsize=10)
    _style_ax(ax)

    ax.plot(steps, values, color=color, linewidth=1.2, alpha=0.35)

    if len(steps) >= 10:
        smoothed, kernel = _smooth(values, n_points=20)
        ax.plot(steps[kernel - 1 :], smoothed, color=color, linewidth=2.2, alpha=0.9)


def _add_lr_twin(ax: plt.Axes, steps: list[float], lr_values: list[float]) -> None:
    """Overlay the LR schedule on a twin y-axis of the given axes."""
    ax2 = ax.twinx()
    color = _COLORS["lr"]
    ax2.plot(steps, lr_values, color=color, linewidth=1.2, linestyle="--", alpha=0.6, label="LR")
    ax2.set_ylabel("Learning Rate", fontsize=9, color=color)
    ax2.tick_params(axis="y", colors=color, labelsize=8)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    # Remove twin spines except the right one
    for spine_name, spine in ax2.spines.items():
        if spine_name != "right":
            spine.set_visible(False)
        else:
            spine.set_edgecolor("#ddaaaa")


# =============================================================================
# Main plotting function
# =============================================================================


def plot_training(
    history: dict,
    name: str,
    graph_dir: str,
    use_scheduler: bool = False,
) -> None:
    """
    Save a training plot to <graph_dir>/<name>.png, overwriting each call.

    Parameters
    ----------
    history       : Dict produced by the universal training loop.
    name          : Run name — used in the title and filename.
    graph_dir     : Directory to save the PNG into (created if absent).
    use_scheduler : When True, overlays the LR curve on the total loss panel.
    """
    os.makedirs(graph_dir, exist_ok=True)

    steps = history["steps"]
    if not steps:
        return  # nothing to plot yet

    # ── Determine which loss panels to show (drop all-zero components) ────────
    candidates: list[str] = ["total", "diff", "prior", "rec"]
    active = [k for k in candidates if any(v != 0.0 for v in history.get(k, [0.0]))]

    n_panels = len(active)
    if n_panels == 0:
        return

    # ── Figure layout ─────────────────────────────────────────────────────────
    if n_panels == 4:
        fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes_grid.flatten().tolist()
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]  # ensure iterable

    fig.suptitle(
        f"Training — {name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # ── Draw each active panel ────────────────────────────────────────────────
    for ax, key in zip(axes, active, strict=False):
        _plot_loss_panel(ax, steps, history[key], key)

        # Overlay LR on the total loss panel only
        if key == "total" and use_scheduler and history.get("lr"):
            _add_lr_twin(ax, steps, history["lr"])

    fig.tight_layout()

    save_path = os.path.join(graph_dir, "training_graph.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Sample plotting
# =============================================================================


def _model_to_grid(
    model: object,
    model_type: str,
    n_samples: int,
    device: str,
    data_config: dict,
    collect_snapshots: bool = False,
    debug: bool = True,
) -> tuple[np.ndarray, dict | None]:
    """Draw n_samples from model and return rendered grid + optional snapshots.

    Args:
        model:             Trained model.
        model_type:        Model type string.
        n_samples:         Number of samples to draw.
        device:            Device string.
        data_config:       Dict with 'channels', 'img_size', 'data_dim', optionally 'is_3d'.
        collect_snapshots: If True, collect weight snapshots (NDM transinr only).
        debug:             Passed through to model.sample().
    Returns:
        grid:      2D: (n_samples, H, W) or (n_samples, H, W, C) numpy array in [0,1].
                   3D: (n_samples, D, H, W) numpy array of raw voxel grids.
        snapshots: {t_value: flat np.ndarray} or None if not collected.
    """
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    is_3d = data_config.get("is_3d", False)
    snapshots = None

    model.eval()
    with torch.no_grad():
        if model_type == "ndm":
            samples = model.sample(n_samples)
            samples = (samples * 0.5 + 0.5).clamp(0, 1)
            samples = samples.reshape(n_samples, channels, img_size, img_size)

        elif model_type == "inr_vae":
            dev = torch.device(device)
            lin = torch.linspace(-1, 1, img_size, device=dev)
            grid_r, grid_c = torch.meshgrid(lin, lin, indexing="ij")
            coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)
            coords_batch = coords.unsqueeze(0).expand(n_samples, -1, -1)
            z = model.prior().sample(torch.Size([n_samples])).to(dev)
            flat_weights = model.decode_to_weights(z)
            pixels = model.inr(coords_batch, flat_weights)
            samples = pixels.permute(0, 2, 1).reshape(n_samples, channels, img_size, img_size).clamp(0, 1)

        elif model_type == "ndm_inr" or model_type in ("latent_inr_diffusion", "weight_inr_diffusion", "weight_inr_ndm_diffusion"):
            if collect_snapshots:
                raw_samples, snapshots = model.sample(n_samples, collect_snapshots=True, debug=debug)
            else:
                raw_samples = model.sample(n_samples, debug=debug)

            if is_3d:
                # Return raw voxel grids — rendering happens in the plot functions
                grid = _samples_to_voxel_grids(raw_samples, channels, img_size)
                model.train()
                return grid, snapshots

            samples = (raw_samples * 0.5 + 0.5).clamp(0, 1).reshape(n_samples, channels, img_size, img_size)

        else:
            raise ValueError(f"Unknown model_type '{model_type}' for sampling.")

    samples = samples.cpu().numpy()
    grid = samples[:, 0, :, :] if channels == 1 else samples.transpose(0, 2, 3, 1)

    model.train()
    return grid, snapshots


def plot_final_samples(
    model: object,
    model_type: str,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    n_samples: int = 64,
    n_fid_samples: int = 512,
    val_loader: torch.utils.data.DataLoader = None,
    debug=False,
) -> None:
    """
    Sample an 8x8 grid from the model, compute MNIST + Inception FID scores,
    reconstruction loss, ELBO, and class uniformity. Saves figure and metrics JSON.

    Args:
        model:          Trained model, already on device.
        model_type:     One of "ndm", "inr_vae", "ndm_inr", "ndm_temporal_transinr", etc.
        epoch:          Current epoch number, used in the filename.
        run_dir:        Run results directory (src/train_results/{run_name}).
        device:         Device string.
        data_config:    Dict with "channels", "img_size", "data_dim", "dataset".
        n_samples:      Total grid samples; displayed as sqrt x sqrt grid.
        n_fid_samples:  Number of samples used for FID computation.
        val_loader:     Validation DataLoader; required for ELBO and rec loss computation.
    Returns:
        None
    """
    import json

    import torch

    from src.utility.classifier_utils import (
        _get_inception,
        _inception_features,
        _load_classifier,
        _load_or_compute_real_features,
        _mnist_features,
    )
    from src.utility.metrics_util import _fid

    os.makedirs(run_dir, exist_ok=True)

    dataset = data_config.get("dataset", "mnist").lower()
    channels = data_config["channels"]
    img_size = data_config["img_size"]  # noqa: F841
    is_mnist = dataset == "mnist"

    # ── Grid samples (for display) ────────────────────────────────────────────
    n_side = int(np.sqrt(n_samples))
    grid, _ = _model_to_grid(model, model_type, n_side * n_side, device, data_config, debug=debug)

    # ── FID samples ───────────────────────────────────────────────────────────
    print(f"  Computing FID ({n_fid_samples} samples) …")
    fid_batch_size = 1024
    fid_batches = []

    for start in range(0, n_fid_samples, fid_batch_size):
        print(f"    Sampling FID batch {start} to {min(start + fid_batch_size, n_fid_samples)} …")
        batch_n = min(fid_batch_size, n_fid_samples - start)
        batch, _ = _model_to_grid(model, model_type, batch_n, device, data_config, debug=debug)
        fid_batches.append(batch)

    fid_grid = np.concatenate(fid_batches, axis=0)  # (n_fid_samples, H, W) or (n_fid_samples, H, W, C)

    # Convert numpy grid back to (N, C, H, W) float tensor in [0, 1]
    if channels == 1:
        fid_tensor = torch.from_numpy(fid_grid).unsqueeze(1).float()
    else:
        fid_tensor = torch.from_numpy(fid_grid).permute(0, 3, 1, 2).float()

    inception = _get_inception(device)

    if is_mnist:
        classifier = _load_classifier(device)
        real_mnist_feats, real_inception_feats, _ = _load_or_compute_real_features(classifier, inception, device)
        gen_mnist_feats, gen_preds = _mnist_features(fid_tensor, classifier, device)
        mnist_fid = _fid(real_mnist_feats, gen_mnist_feats)
    else:
        raise NotImplementedError(
            "CIFAR-10 real Inception features cache not yet wired into plot_final_samples. "
            "Add a CIFAR-10 equivalent of _load_or_compute_real_features."
        )

    gen_inception_feats = _inception_features(fid_tensor, inception, device)
    inception_fid = _fid(real_inception_feats, gen_inception_feats)

    # ── Uniformity (normalized entropy of predicted class distribution) ───────
    print("  Computing class uniformity …")
    predicted_classes = torch.from_numpy(gen_preds)
    n_classes = 10
    class_counts = torch.bincount(predicted_classes, minlength=n_classes).float()
    class_probs = class_counts / class_counts.sum()
    entropy = -(class_probs * (class_probs + 1e-8).log()).sum()
    uniformity_score = float(entropy / np.log(n_classes))
    class_breakdown = {str(i): int(class_counts[i].item()) for i in range(n_classes)}

    # ── Reconstruction loss (optional) ────────────────────────────────────────
    rec_loss = None
    if val_loader is not None and hasattr(model, "compute_rec_loss"):
        print("  Computing reconstruction loss over validation set …")
        rec_loss = model.compute_rec_loss(val_loader)

    # ── ELBO (optional) ───────────────────────────────────────────────────────
    elbo_val = None
    if val_loader is not None and hasattr(model, "compute_full_elbo"):
        print("  Computing full ELBO over validation set …")
        elbo_val = model.compute_full_elbo(val_loader)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 45}")
    print(f"  Eval Summary  —  epoch {epoch}")
    print(f"{'=' * 45}")
    print(f"  Rec Loss      : {rec_loss:.4f}" if rec_loss is not None else "  Rec Loss      : None")
    print(f"  ELBO          : {elbo_val:.4f}" if elbo_val is not None else "  ELBO          : None")
    print(f"  MNIST FID     : {mnist_fid:.2f}")
    print(f"  Inception FID : {inception_fid:.2f}")
    print(f"  Uniformity    : {uniformity_score:.4f}  (0=collapsed, 1=uniform)")
    print(f"{'=' * 45}\n")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "epoch": epoch,
        "rec_loss": rec_loss,
        "elbo": elbo_val,
        "mnist_fid": mnist_fid,
        "inception_fid": inception_fid,
        "uniformity_score": uniformity_score,
        "class_breakdown": class_breakdown,
    }
    metrics_path = os.path.join(run_dir, f"eval_metrics_ep{epoch}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    # ── Main grid figure (no text) ────────────────────────────────────────────
    fig, axes = plt.subplots(n_side, n_side, figsize=(n_side * 1.5, n_side * 1.5))
    for i, ax in enumerate(axes.flatten()):
        if channels == 1:
            ax.imshow(grid[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(grid[i], vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    save_path = os.path.join(run_dir, f"final_samples_ep{epoch}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Final samples saved → {save_path}")

    # ── Single-row figures (10, 8, 6 samples) ────────────────────────────────
    # Sample without replacement across all three rows so they're all different
    all_indices = np.random.choice(len(grid), size=10 + 8 + 6, replace=False)
    row_sizes = [10, 8, 6]
    row_names = ["10", "8", "6"]
    offset = 0
    for n_row, name in zip(row_sizes, row_names):  # noqa: B905
        indices = all_indices[offset : offset + n_row]
        offset += n_row

        fig, axes = plt.subplots(1, n_row, figsize=(n_row * 1.5, 1.5))
        for ax, idx in zip(axes, indices):  # noqa: B905
            if channels == 1:
                ax.imshow(grid[idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(grid[idx], vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        row_path = os.path.join(run_dir, f"samples_row{name}_ep{epoch}.png")
        fig.savefig(row_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Row-{name} samples saved → {row_path}")


def plot_sample_progression(
    model: object,
    model_type: str,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "sample_progression",
    collect_snapshots: bool = False,
) -> None:
    """Append a row of samples to the progression figure, saved to <run_dir>/<filename>.png.

    Renders 2D samples with imshow, 3D samples as marching-cubes meshes.
    Always renders 5 rows — empty rows shown as blank until filled.

    Args:
        model:             Trained model, already on device.
        model_type:        Model type string.
        epoch:             Current epoch, used as the row label.
        run_dir:           Run results directory.
        device:            Device string.
        data_config:       Dict with 'channels', 'img_size', 'data_dim', optionally 'is_3d'.
        filename:          Base filename for outputs.
        collect_snapshots: If True, also plot denoising trajectory histograms.
    Returns:
        None
    """
    os.makedirs(run_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    n_cols = 6
    channels = data_config["channels"]
    is_3d = data_config.get("is_3d", False)

    new_row, snapshots = _model_to_grid(model, model_type, n_cols, device, data_config, collect_snapshots)

    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    rows_path = os.path.join(metadata_dir, f"{filename}_rows.npy")

    if os.path.exists(meta_path) and os.path.exists(rows_path):
        with open(meta_path) as f:
            meta = json.load(f)
        existing_rows = np.load(rows_path)
        all_rows = np.concatenate([existing_rows, new_row[None]], axis=0)
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_rows = new_row[None]
        all_epochs = [epoch]

    np.save(rows_path, all_rows)
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs}, f)

    n_existing = len(all_epochs)
    blank_shape = (n_cols, *new_row.shape[1:])
    blank = np.ones(blank_shape)
    padded_rows = list(all_rows) + [blank] * (N_ROWS_TOTAL - n_existing)
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - n_existing)

    label_width = 0.5
    img_inches = 1.8 if is_3d else 1.2  # meshes need a bit more room
    row_gap = 0.15
    title_pad = 0.35

    fig_w = label_width + n_cols * img_inches
    fig_h = title_pad + N_ROWS_TOTAL * img_inches + (N_ROWS_TOTAL - 1) * row_gap
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    for r, (row_samples, ep) in enumerate(zip(padded_rows, padded_epochs, strict=False)):
        for c in range(n_cols):
            left = (label_width + c * img_inches) / fig_w
            bottom = 1.0 - (title_pad / fig_h) - (r + 1) * (img_inches / fig_h) - r * (row_gap / fig_h)
            width = img_inches / fig_w
            height = img_inches / fig_h

            if is_3d:
                ax = fig.add_axes([left, bottom, width, height], projection="3d")
                # row_samples is (n_cols, D, H, W); blank is all-ones → skip mesh
                is_blank = np.all(row_samples[c] == 1.0)
                if not is_blank:
                    _render_mesh_on_ax(ax, row_samples[c], azim=_AZIM_OFFSETS[c])
                else:
                    ax.set_axis_off()
            else:
                ax = fig.add_axes([left, bottom, width, height])
                if channels == 1:
                    ax.imshow(row_samples[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(row_samples[c], vmin=0, vmax=1, interpolation="nearest")
                ax.axis("off")

        fig.text(
            (label_width * 0.5) / fig_w,
            1.0 - (title_pad / fig_h) - (r + 0.5) * (img_inches / fig_h) - r * (row_gap / fig_h),
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("Sample Progression", fontsize=11, fontweight="bold", y=0.99)
    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if collect_snapshots and snapshots is not None:
        denoising_filename = filename.replace("sample_progression", "Reverse_denoising_progression")
        if denoising_filename == filename:
            denoising_filename = f"Reverse_denoising_progression_{filename}"
        plot_denoising_trajectory_progression(
            snapshots=snapshots,
            epoch=epoch,
            run_dir=run_dir,
            filename=denoising_filename,
        )


def plot_denoising_trajectory_progression(
    snapshots: dict[int, np.ndarray],
    epoch: int,
    run_dir: str,
    filename: str = "Reverse_denoising_progression",
) -> None:
    """
    Append a row of 4 weight distribution histograms to the denoising trajectory
    progression figure, saved to <run_dir>/<filename>.png.

    Now safely handles resets, restarts, and dynamic segmentation by grouping
    and sorting snapshots explicitly by epoch.
    """
    import json
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(run_dir, exist_ok=True)
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    t_keys_sorted = sorted(snapshots.keys(), reverse=True)
    n_cols = len(t_keys_sorted)

    # New row data to inject
    new_row_data = [snapshots[t] for t in t_keys_sorted]

    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    data_path = os.path.join(metadata_dir, f"{filename}_data.npy")

    # ── 1. Load, Deduplicate, and Synchronize History ─────────────────────────
    if os.path.exists(meta_path) and os.path.exists(data_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            loaded_rows = list(np.load(data_path, allow_pickle=True))
            loaded_epochs = meta["epochs"]
            all_t_keys = meta["t_keys"]

            # Map epoch -> row data to protect layout budgets from duplicate rows
            history_map = {ep: row for ep, row in zip(loaded_epochs, loaded_rows)}  # noqa: B905, C416
        except Exception:
            history_map = {}
            all_t_keys = t_keys_sorted
    else:
        history_map = {}
        all_t_keys = t_keys_sorted

    # Insert or overwrite current epoch row
    history_map[epoch] = new_row_data

    # Sort everything chronologically by epoch
    sorted_epochs = sorted(history_map.keys())
    all_rows = [history_map[ep] for ep in sorted_epochs]

    # Save tracking history back to disk
    np.save(data_path, np.array(all_rows, dtype=object))
    with open(meta_path, "w") as f:
        json.dump({"epochs": sorted_epochs, "t_keys": all_t_keys}, f)

    # ── 2. Pad to N_ROWS_TOTAL (Strict Layout Isolation) ──────────────────────
    padded_rows = all_rows[:N_ROWS_TOTAL] + [None] * (N_ROWS_TOTAL - len(all_rows))
    padded_epochs = sorted_epochs[:N_ROWS_TOTAL] + [""] * (N_ROWS_TOTAL - len(sorted_epochs))

    # ── 3. Build Figure Layout ────────────────────────────────────────────────
    col_width, row_height = 2.2, 1.6
    label_width = 0.75
    col_gap, row_gap = 0.35, 0.25
    title_pad, header_pad = 0.5, 0.35

    fig_w = label_width + n_cols * col_width + (n_cols - 1) * col_gap
    fig_h = title_pad + header_pad + N_ROWS_TOTAL * row_height + (N_ROWS_TOTAL - 1) * row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Draw Column Headers
    for c, t_val in enumerate(all_t_keys):
        cx = (label_width + c * (col_width + col_gap) + col_width * 0.5) / fig_w
        cy = 1.0 - (title_pad / fig_h) - (header_pad * 0.6 / fig_h)
        fig.text(cx, cy, f"t = {t_val}", ha="center", va="center", fontsize=8, color="#555555", fontweight="bold")

    # Draw Rows safely bound within N_ROWS_TOTAL
    for r in range(N_ROWS_TOTAL):
        row_data = padded_rows[r]
        ep = padded_epochs[r]
        row_bottom = 1.0 - (title_pad / fig_h) - (header_pad / fig_h) - (r + 1) * (row_height / fig_h) - r * (row_gap / fig_h)

        for c in range(n_cols):
            left = (label_width + c * (col_width + col_gap)) / fig_w
            ax = fig.add_axes([left, row_bottom, col_width / fig_w, row_height / fig_h])

            if row_data is not None:
                w = row_data[c]
                mu_val, std_val = np.mean(w), np.std(w)
                ax.hist(w, bins=80, color="#4A90E2", alpha=0.75, density=True)

                # Overlay ideal N(0,1) reference Gaussian
                xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 300)
                gaussian = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs**2)
                ax.plot(xs, gaussian, color="#333333", linewidth=1.0, linestyle="--")

                ax.text(
                    0.97,
                    0.93,
                    f"μ:{mu_val:.2f}\nx:{std_val:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "ec": "none"},
                )

            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)

            if r == N_ROWS_TOTAL - 1:
                ax.set_xlabel("weight value", fontsize=7)

        # Draw left-hand epoch index label
        fig.text(
            (label_width * 0.5) / fig_w,
            row_bottom + (row_height * 0.5) / fig_h,
            f"ep {ep}" if ep != "" else "",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("Denoising Trajectory — Weight Distributions", fontsize=11, fontweight="bold", y=0.99)

    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fphi_progression(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "fphi_progression",
    model_name: str = "",
) -> None:
    """
    Append a row showing one image passed through F_phi at 6 evenly spaced
    timesteps (t=0 to t=T) to the progression figure.

    Always renders 5 rows — empty rows shown as white until filled.
    First row has t-labels along the top. Each row is labelled with its
    epoch on the left.

    Parameters
    ----------
    model       : Trained NDM model with F_phi, already on device.
    batch       : Current training batch (N, data_dim), used to pick one image.
    epoch       : Current epoch, used as the row label.
    run_dir     : Run results directory (src/train_results/{run_name}).
    device      : Device string.
    data_config : Dict with "channels", "img_size", "data_dim".
    filename    : Base name for the saved png and metadata files.
    """

    os.makedirs(run_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    n_cols = 6
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    T = model.T  # noqa: N806

    # ── Timesteps: 6 evenly spaced from 0 to T ───────────────────────────────
    timesteps = [round(T * i / (n_cols - 1)) for i in range(n_cols)]
    timesteps[-1] = T - 1  # clamp to valid index

    # ── Pick one image from the batch ────────────────────────────────────────
    x = batch[0][0:1].to(device)  # batch[0] gets the images tensor, [0:1] gets first image

    # ── Run F_phi at each timestep ────────────────────────────────────────────
    model.eval()

    if model_name == "latent_ndm_inr_diffusion":
        row_images = []
        with torch.no_grad():
            for t in timesteps:
                t_norm = torch.full((1, 1), t / max(T - 1, 1), device=device)
                if x.dim() == 2:
                    channels = x.shape[1] // (model.img_size * model.img_size)
                    x = x.view(x.shape[0], channels, model.img_size, model.img_size)
                mu, logvar = model.latent_encoder(x)
                z = model.latent_encoder.reparameterize(mu, logvar)
                z_trans = model.latent_transformer(z, t_norm)
                x_recon = model._decode_latent(z_trans)
                img = (x_recon * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
                img = img.reshape(channels, img_size, img_size).cpu().numpy()
                if channels == 1:  # noqa: SIM108
                    img = img[0]  # (H, W)
                else:
                    img = img.transpose(1, 2, 0)  # (H, W, C)
                row_images.append(img)
    else:
        row_images = []
        with torch.no_grad():
            for t in timesteps:
                t_norm = torch.full((1, 1), t / max(T - 1, 1), device=device)
                z_t = model.F_phi(x, t_norm)  # (1, data_dim)
                img = (z_t * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
                img = img.reshape(channels, img_size, img_size).cpu().numpy()
                if channels == 1:  # noqa: SIM108
                    img = img[0]  # (H, W)
                else:
                    img = img.transpose(1, 2, 0)  # (H, W, C)
                row_images.append(img)

    model.train()

    new_row = np.stack(row_images, axis=0)  # (6, H, W) or (6, H, W, C)

    # ── Load existing rows from disk if available ─────────────────────────────
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    rows_path = os.path.join(metadata_dir, f"{filename}_rows.npy")

    if os.path.exists(meta_path) and os.path.exists(rows_path):
        with open(meta_path) as f:
            meta = json.load(f)
        existing_rows = np.load(rows_path)
        all_rows = np.concatenate([existing_rows, new_row[None]], axis=0)
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_rows = new_row[None]
        all_epochs = [epoch]

    # ── Persist updated rows ──────────────────────────────────────────────────
    np.save(rows_path, all_rows)
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs, "timesteps": timesteps}, f)

    # ── Pad to always have N_ROWS_TOTAL rows ──────────────────────────────────
    n_existing = len(all_epochs)
    blank_shape = (n_cols, *new_row.shape[1:])
    blank = np.ones(blank_shape)
    padded_rows = list(all_rows) + [blank] * (N_ROWS_TOTAL - n_existing)
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - n_existing)

    # ── Build figure ──────────────────────────────────────────────────────────
    label_width = 0.5
    img_inches = 1.2
    row_gap = 0.15
    title_pad = 0.35
    t_label_pad = 0.25  # extra space for t labels on first row

    fig_w = label_width + n_cols * img_inches
    fig_h = title_pad + t_label_pad + N_ROWS_TOTAL * img_inches + (N_ROWS_TOTAL - 1) * row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # t= labels along the top (above first row only)
    for c, t in enumerate(timesteps):
        label_x = (label_width + (c + 0.5) * img_inches) / fig_w
        label_y = 1.0 - (title_pad / fig_h) - (t_label_pad * 0.5 / fig_h)
        fig.text(label_x, label_y, f"t={t}", ha="center", va="center", fontsize=7, color="#555555")

    for r, (row_samples, ep) in enumerate(zip(padded_rows, padded_epochs, strict=False)):
        for c in range(n_cols):
            left = (label_width + c * img_inches) / fig_w
            bottom = 1.0 - (title_pad / fig_h) - (t_label_pad / fig_h) - (r + 1) * (img_inches / fig_h) - r * (row_gap / fig_h)
            width = img_inches / fig_w
            height = img_inches / fig_h

            ax = fig.add_axes([left, bottom, width, height])
            if channels == 1:
                ax.imshow(row_samples[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(row_samples[c], vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

        # Epoch label on the left
        fig.text(
            (label_width * 0.5) / fig_w,
            (1.0 - (title_pad / fig_h) - (t_label_pad / fig_h) - (r + 0.5) * (img_inches / fig_h) - r * (row_gap / fig_h)),
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("F_phi Corruption Progression", fontsize=11, fontweight="bold", y=0.99)

    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fphi_weight_histograms(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,  # noqa: ARG001
    filename: str = "fphi_weight_histogram",
    model_name: str = "",  # noqa: ARG001
) -> None:
    """
    Append a row showing the weight distribution (histogram) of theta_prime
    passed through F_phi at 6 evenly spaced timesteps to a progression figure.

    Column 0: theta_prime (normalized if model.normalize, else raw encoder output).
    Columns 1-5: F_phi(theta_prime, t) at increasing timesteps.

    X-axis is shared per row (computed from column 0 range) so distribution
    shifts are visually comparable. Always renders N_ROWS_TOTAL rows; empty
    rows shown as blank axes until filled.

    Parameters
    ----------
    model       : WeightNDMDiffusion model with F_phi, scaler, weight_encoder.
    batch       : Current training batch — tuple (images, labels) or raw tensor.
    epoch       : Current epoch, used as the row label.
    run_dir     : Run results directory.
    device      : Device string.
    data_config : Dict with at least "data_dim".
    filename    : Base name for saved png and metadata files.
    model_name  : Optional model name string (unused but kept for API parity).
    """

    os.makedirs(run_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    N_BINS = 60  # histogram bins  # noqa: N806
    n_cols = 6

    T = model.T  # noqa: N806

    # ── Timesteps: t=0 is column 0 (baseline), cols 1-5 are evenly spaced ───
    timesteps = [round(T * i / (n_cols - 1)) for i in range(n_cols)]
    timesteps[-1] = T - 1  # clamp last to valid index
    # Column 0 is the baseline (theta_prime itself), so F_phi timesteps start at col 1
    fphi_timesteps = timesteps[1:]  # 5 timesteps for F_phi columns

    # ── Extract image tensor from batch (handles (imgs, labels) tuples) ──────
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    x = x.to(device)
    # print("DEBUG: [F_phi]", x.shape, "device", x.device, "dtype", x.dtype, "mean", x.mean().item(), "std", x.std().item())

    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    # print("DEBUG: [F_phi]", x.shape, "device", x.device, "dtype", x.dtype, "mean", x.mean().item(), "std", x.std().item())

    # ── Encode full batch to theta_prime ─────────────────────────────────────
    model.eval()
    with torch.no_grad():
        # print("DEBUG: [F_phi]", model.probablistic, "normalize", model.normalize)
        if model.probablistic:
            mean, logvar = model.weight_encoder(x)
            theta_prime_raw = model.weight_encoder._reparameterize(mean, logvar)
        else:
            theta_prime_raw = model.weight_encoder(x)

        # Normalize if the model uses normalization (mirrors training flow)
        theta_prime = model.scaler(theta_prime_raw, reverse=False) if model.normalize else theta_prime_raw
        # print(f"DEBUG: F_phi | Mean: {theta_prime.mean():.4f} | Std: {theta_prime.std():.4f}")

        # ── Build row: col 0 = baseline, cols 1-5 = F_phi at each timestep ──
        row_values = []

        # Column 0: baseline distribution
        row_values.append(theta_prime.detach().cpu().numpy().flatten())

        # Columns 1-5: F_phi(theta_prime, t)
        for t in fphi_timesteps:
            t_norm = torch.full((x.shape[0], 1), t / max(T - 1, 1), device=device)
            z_t = model.F_phi(theta_prime, t_norm)  # (B, modulation_dim)
            row_values.append(z_t.detach().cpu().numpy().flatten())

    model.train()

    # ── Compute shared x-axis range from baseline ─────────────────────────────
    baseline = row_values[0]
    x_min = float(np.percentile(baseline, 0.5))
    x_max = float(np.percentile(baseline, 99.5))
    margin = (x_max - x_min) * 0.1
    x_range = (x_min - margin, x_max + margin)

    # ── Precompute histogram arrays for persistence ───────────────────────────
    # Store as (n_cols, N_BINS) bin counts + one shared edges array per row
    row_counts = np.zeros((n_cols, N_BINS), dtype=np.float32)
    row_edges = np.zeros((n_cols, N_BINS + 1), dtype=np.float32)
    for c, vals in enumerate(row_values):
        counts, edges = np.histogram(vals, bins=N_BINS, range=x_range)
        row_counts[c] = counts.astype(np.float32)
        row_edges[c] = edges.astype(np.float32)

    # ── Load existing rows from disk if available ─────────────────────────────
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    counts_path = os.path.join(metadata_dir, f"{filename}_counts.npy")
    edges_path = os.path.join(metadata_dir, f"{filename}_edges.npy")

    if os.path.exists(meta_path) and os.path.exists(counts_path) and os.path.exists(edges_path):
        with open(meta_path) as f:
            meta = json.load(f)
        all_counts = np.concatenate([np.load(counts_path), row_counts[None]], axis=0)
        all_edges = np.concatenate([np.load(edges_path), row_edges[None]], axis=0)
        all_epochs = meta["epochs"] + [epoch]
        all_xranges = meta["xranges"] + [list(x_range)]
    else:
        all_counts = row_counts[None]
        all_edges = row_edges[None]
        all_epochs = [epoch]
        all_xranges = [list(x_range)]

    # ── Persist updated rows ──────────────────────────────────────────────────
    np.save(counts_path, all_counts)
    np.save(edges_path, all_edges)
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs, "timesteps": timesteps, "xranges": all_xranges}, f)

    # ── Pad to always render N_ROWS_TOTAL rows ────────────────────────────────
    n_existing = len(all_epochs)
    padded_counts = list(all_counts) + [None] * (N_ROWS_TOTAL - n_existing)
    padded_edges = list(all_edges) + [None] * (N_ROWS_TOTAL - n_existing)
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - n_existing)
    padded_xranges = list(all_xranges) + [None] * (N_ROWS_TOTAL - n_existing)

    # ── Build figure ──────────────────────────────────────────────────────────
    label_width = 0.75
    hist_w_inches = 1.4
    hist_h_inches = 0.9
    row_gap = 0.25
    title_pad = 0.35
    header_pad = 0.35

    fig_w = label_width + n_cols * hist_w_inches + (n_cols - 1) * 0.1  # adjust spacing
    fig_h = title_pad + header_pad + N_ROWS_TOTAL * hist_h_inches + (N_ROWS_TOTAL - 1) * row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Column headers
    col_labels = ["θ' (base)"] + [f"t={t}" for t in fphi_timesteps]
    for c, label in enumerate(col_labels):
        label_x = (label_width + c * hist_w_inches + hist_w_inches * 0.5) / fig_w
        label_y = 1.0 - (title_pad / fig_h) - (header_pad * 0.3 / fig_h)
        fig.text(label_x, label_y, label, ha="center", va="center", fontsize=8, color="#555555", fontweight="bold")

    for r, (counts_row, edges_row, ep, xrange_row) in enumerate(
        zip(padded_counts, padded_edges, padded_epochs, padded_xranges, strict=False)
    ):
        row_bottom = 1.0 - (title_pad / fig_h) - (header_pad / fig_h) - (r + 1) * (hist_h_inches / fig_h) - r * (row_gap / fig_h)

        for c in range(n_cols):
            left = (label_width + c * hist_w_inches) / fig_w
            ax = fig.add_axes([left, row_bottom, hist_w_inches / fig_w, hist_h_inches / fig_h])

            if counts_row is None:
                ax.set_axis_off()
                continue

            # Density calculation
            bin_width = edges_row[c][1] - edges_row[c][0]
            density = counts_row[c] / (counts_row[c].sum() * bin_width + 1e-9)

            # Colors: col 0 blue, others orange
            color = "#5b7fa6" if c == 0 else "#E2844A"
            ax.bar(edges_row[c][:-1], density, width=bin_width, align="edge", color=color, alpha=0.75, linewidth=0)

            # Gaussian Overlay for F_phi (cols > 0)
            if c > 0:
                xs = np.linspace(xrange_row[0], xrange_row[1], 300)
                gaussian = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs**2)
                ax.plot(xs, gaussian, color="#333333", linewidth=1.0, linestyle="--")

            # Statistics Box
            bin_centers = (edges_row[c][:-1] + edges_row[c][1:]) / 2
            total = counts_row[c].sum()
            mu_val = float(np.sum(bin_centers * counts_row[c]) / (total + 1e-9))
            std_val = float(np.sqrt(np.sum(counts_row[c] * (bin_centers - mu_val) ** 2) / (total + 1e-9)))
            ax.text(
                0.97,
                0.93,
                f"μ:{mu_val:.2f}\nx:{std_val:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "ec": "none"},
            )

            # Style adjustments to match trajectory plot
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)
            ax.set_xlim(xrange_row)

            # Y-axis labels only on column 0
            if c > 0:
                ax.tick_params(left=False, labelleft=False)

            if r == N_ROWS_TOTAL - 1:
                ax.set_xlabel("weight value", fontsize=7)

        # Epoch label
        fig.text(
            (label_width * 0.4) / fig_w,
            row_bottom + (hist_h_inches * 0.5) / fig_h,
            f"ep {ep}",
            ha="right",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("F_phi Weight Distribution Progression", fontsize=11, fontweight="bold", y=0.99)

    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ztrans_histogram(
    model: object,
    batch: tuple,
    epoch: int,
    run_dir: str,
    device: str,
    filename: str = "ztrans_histogram",
) -> None:
    """
    Compute z_trans for the entire batch at 6 evenly spaced t values and append
    a row of histograms to a persistent figure. One row per epoch call, always
    rendering all epochs together in one saved figure.

    Args:
        model    : Trained latent_ndm_inr_diffusion model, already on device.
        batch    : Current training batch tuple; batch[0] is the images tensor (N, C, H, W).
        epoch    : Current epoch number, used as the row label.
        run_dir  : Run results directory (src/train_results/{run_name}).
        device   : Device string.
        filename : Base name for the saved png and metadata files.

    Returns:
        None
    """
    os.makedirs(run_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    N_COLS = 6  # noqa: N806
    T = model.T  # noqa: N806

    timesteps = [round(T * i / (N_COLS - 1)) for i in range(N_COLS)]
    timesteps[-1] = T - 1

    x = batch[0].to(device)
    if x.dim() == 2:
        channels = x.shape[1] // (model.img_size * model.img_size)
        x = x.view(x.shape[0], channels, model.img_size, model.img_size)

    # ── Compute z_trans for full batch at each t ──────────────────────────────
    model.eval()
    row_hists = []  # list of (N*latent_dim,) arrays, one per t
    with torch.no_grad():
        mu, logvar = model.latent_encoder(x)
        z = model.latent_encoder.reparameterize(mu, logvar)  # (N, latent_dim)
        for t in timesteps:
            t_norm = torch.full((x.shape[0], 1), t / max(T - 1, 1), device=device)
            z_trans = model.latent_transformer(z, t_norm)  # (N, latent_dim)
            row_hists.append(z_trans.cpu().numpy().flatten())
    model.train()

    # ── Persist histogram data ────────────────────────────────────────────────
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    rows_path = os.path.join(metadata_dir, f"{filename}_rows.npy")

    new_row = np.array(row_hists, dtype=object)  # (N_COLS,) of flat arrays

    if os.path.exists(meta_path) and os.path.exists(rows_path):
        with open(meta_path) as f:
            meta = json.load(f)
        existing_rows = np.load(rows_path, allow_pickle=True)
        all_rows = np.concatenate([existing_rows, new_row[None]], axis=0)
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_rows = new_row[None]
        all_epochs = [epoch]

    np.save(rows_path, all_rows)
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs, "timesteps": timesteps}, f)

    # ── Pad to N_ROWS_TOTAL ───────────────────────────────────────────────────
    n_existing = len(all_epochs)
    padded_rows = list(all_rows) + [None] * (N_ROWS_TOTAL - n_existing)
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - n_existing)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        N_ROWS_TOTAL,
        N_COLS,
        figsize=(N_COLS * 2.5, N_ROWS_TOTAL * 2.0),
        sharey=False,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle("z_trans Distribution Progression", fontsize=11, fontweight="bold")

    # t= labels along the top
    for c, t in enumerate(timesteps):
        axes[0, c].set_title(f"t={t}", fontsize=8, color="#555555", pad=4)

    for r, (row_hists_r, ep) in enumerate(zip(padded_rows, padded_epochs, strict=False)):
        for c in range(N_COLS):
            ax = axes[r, c]
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)

            if row_hists_r is not None:
                ax.hist(row_hists_r[c], bins=60, color="#4C72B0", edgecolor="none", alpha=0.85)
            else:
                # Empty placeholder row
                ax.set_facecolor("#f5f5f5")
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Epoch label on the left of each row
        axes[r, 0].set_ylabel(f"ep {ep}", fontsize=8, color="#333333", labelpad=6)

    fig.tight_layout()

    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_reconstruction_progression(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "reconstruction_progression",
    model_name: str = "",
) -> None:
    """Append a row of originals + reconstructions to the progression figure.

    Renders 2D data with imshow, 3D data as marching-cubes meshes side-by-side.
    Always renders 5 rows — empty rows shown as blank until filled.

    Args:
        model:       Model already on device.
        batch:       Current training batch — list/tuple where batch[0] is the data tensor.
        epoch:       Current epoch, used as the row label.
        run_dir:     Run results directory.
        device:      Device string.
        data_config: Dict with 'channels', 'img_size', 'data_dim', optionally 'is_3d'.
        filename:    Base name for the saved png and metadata files.
        model_name:  Model variant name string.
    Returns:
        None
    """
    os.makedirs(run_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    n_cols = 6  # 3 originals + 3 reconstructions
    n_pairs = n_cols // 2
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    is_3d = data_config.get("is_3d", False)

    x = batch[0][:n_pairs].to(device)

    model.eval()
    if model_name in ("latent_inr_diffusion", "weight_inr_diffusion", "weight_inr_ndm_diffusion"):
        with torch.no_grad():
            if is_3d:
                # x arrives as (B, 1, D, H, W) from the dataloader
                if x.dim() == 4:  # (B, D, H, W) — add channel
                    x = x.unsqueeze(1)
            else:
                if x.dim() == 2:
                    x = x.view(x.shape[0], channels, img_size, img_size)
            x_recon = model.get_reconstructions(x)  # same shape as x
    else:
        with torch.no_grad():
            if hasattr(model, "F_phi"):
                t0_norm = torch.zeros(x.shape[0], 1, device=device)
                weights = model.F_phi(x, t0_norm)
            elif hasattr(model, "W") and hasattr(model.W, "inflate"):
                x_spatial = x.view(x.shape[0], channels, img_size, img_size)
                weights = model.weight_encoder(x_spatial)
            else:
                t0_norm = torch.zeros(x.shape[0], device=device)
                weights = model.weight_encoder(x)
            x_recon = model._inr_decode(weights)
    model.train()

    # ── Convert to plottable format ───────────────────────────────────────────
    if is_3d:
        # Both x and x_recon are (B, 1, D, H, W) — squeeze channel for marching cubes
        orig_grids = x.squeeze(1).detach().cpu().numpy()  # (n_pairs, D, H, W)
        recon_grids = x_recon.squeeze(1).detach().cpu().numpy()  # (n_pairs, D, H, W)
        new_row_data = {  # noqa: F841
            "orig": orig_grids,
            "recon": recon_grids,
        }
    else:

        def _to_img(t: torch.Tensor) -> np.ndarray:
            """Flat/spatial tensor → numpy HxW or HxWxC in [0,1]."""
            img = t.cpu().numpy().reshape(channels, img_size, img_size)
            return img[0] if channels == 1 else img.transpose(1, 2, 0)

        originals = [(x[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]
        recons = [(x_recon[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]
        new_row = np.stack([_to_img(t) for t in originals + recons], axis=0)  # (6, H, W[,C])

    # ── Persist rows ──────────────────────────────────────────────────────────
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")

    if is_3d:
        # Store 3D grids in a single .npz per epoch; track filenames in JSON
        npz_path = os.path.join(metadata_dir, f"{filename}_ep{epoch}.npz")
        np.savez_compressed(npz_path, orig=orig_grids, recon=recon_grids)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["epochs"].append(epoch)
            meta["npz_paths"].append(npz_path)
        else:
            meta = {"epochs": [epoch], "npz_paths": [npz_path]}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        all_epochs = meta["epochs"]
        all_npz = meta["npz_paths"]
    else:
        rows_path = os.path.join(metadata_dir, f"{filename}_rows.npy")
        if os.path.exists(meta_path) and os.path.exists(rows_path):
            with open(meta_path) as f:
                meta = json.load(f)
            existing_rows = np.load(rows_path)
            all_rows = np.concatenate([existing_rows, new_row[None]], axis=0)
            all_epochs = meta["epochs"] + [epoch]
        else:
            all_rows = new_row[None]
            all_epochs = [epoch]
        np.save(rows_path, all_rows)
        with open(meta_path, "w") as f:
            json.dump({"epochs": all_epochs}, f)

    # ── Build figure ──────────────────────────────────────────────────────────
    label_width = 0.5
    img_inches = 1.8 if is_3d else 1.2
    row_gap = 0.15
    title_pad = 0.35
    divider_gap = 0.08

    fig_w = label_width + n_cols * img_inches + divider_gap
    fig_h = title_pad + N_ROWS_TOTAL * img_inches + (N_ROWS_TOTAL - 1) * row_gap
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Column headers
    for c, header in enumerate(["", "Originals", "", "", "Reconstructions", ""]):
        extra = divider_gap if c >= n_pairs else 0.0
        cx = (label_width + (c + 0.5) * img_inches + extra) / fig_w
        fig.text(cx, 1.0 - (title_pad * 0.7 / fig_h), header, ha="center", va="center", fontsize=7, color="#555555")

    # Pad epochs to N_ROWS_TOTAL
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - len(all_epochs))

    for r, ep in enumerate(padded_epochs):
        # Load this row's data
        if is_3d:
            if r < len(all_npz):
                npz = np.load(all_npz[r])
                row_orig = npz["orig"]  # (n_pairs, D, H, W)
                row_recon = npz["recon"]
                row_is_blank = False
            else:
                row_is_blank = True
        else:
            row_is_blank = r >= len(all_epochs)

        for c in range(n_cols):
            extra = divider_gap if c >= n_pairs else 0.0
            left = (label_width + c * img_inches + extra) / fig_w
            bottom = 1.0 - (title_pad / fig_h) - (r + 1) * (img_inches / fig_h) - r * (row_gap / fig_h)
            width = img_inches / fig_w
            height = img_inches / fig_h

            if is_3d:
                ax = fig.add_axes([left, bottom, width, height], projection="3d")
                if not row_is_blank:
                    pair_idx = c % n_pairs
                    voxels = row_orig[pair_idx] if c < n_pairs else row_recon[pair_idx]
                    _render_mesh_on_ax(ax, voxels, azim=_AZIM_OFFSETS[c])
                else:
                    ax.set_axis_off()
            else:
                ax = fig.add_axes([left, bottom, width, height])
                if not row_is_blank:
                    row_samples = all_rows[r]
                    if channels == 1:
                        ax.imshow(row_samples[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    else:
                        ax.imshow(row_samples[c], vmin=0, vmax=1, interpolation="nearest")
                ax.axis("off")

        fig.text(
            (label_width * 0.5) / fig_w,
            1.0 - (title_pad / fig_h) - (r + 0.5) * (img_inches / fig_h) - r * (row_gap / fig_h),
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    # Vertical divider between originals and reconstructions
    divider_x = (label_width + n_pairs * img_inches + divider_gap * 0.5) / fig_w
    fig.add_artist(
        plt.Line2D(
            [divider_x, divider_x],
            [0.01, 1.0 - title_pad / fig_h],
            transform=fig.transFigure,
            color="#cccccc",
            linewidth=0.8,
            linestyle="--",
        )
    )
    fig.suptitle("Reconstruction Progression", fontsize=11, fontweight="bold", y=1.02)

    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_reconstruction_norm_progression(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "reconstruction_norm_progression",
) -> None:
    """
    Like plot_reconstruction_progression but passes weights through
    normalize -> denormalize before INR decode, to sanity-check the scaler.
    Pipeline: w_raw = F_phi(x) -> w_norm = scaler(w_raw) -> w_denorm = scaler(w_norm, reverse=True) -> INR(w_denorm)
    Parameters
    ----------
    model       : NeuralDiffusionModelINR with model.scaler (WeightScaler), already on device.
    batch       : Current training batch — list/tuple where batch[0] is images.
    epoch       : Current epoch, used as the row label.
    run_dir     : Run results directory.
    device      : Device string.
    data_config : Dict with "channels", "img_size", "data_dim".
    filename    : Base name for the saved png and metadata files.
    """
    import json

    os.makedirs(run_dir, exist_ok=True)
    N_ROWS_TOTAL = 5  # noqa: N806
    n_cols = 6
    n_pairs = n_cols // 2
    channels = data_config["channels"]
    img_size = data_config["img_size"]

    x = batch[0][:n_pairs].to(device)

    model.eval()
    with torch.no_grad():
        # ── Encode to raw weights (mirrors existing plot) ─────────────────────
        if hasattr(model, "F_phi"):
            t0_norm = torch.zeros(x.shape[0], 1, device=device)
            weights_raw = model.F_phi(x, t0_norm)
        elif hasattr(model, "W") and hasattr(model.W, "inflate"):
            x_spatial = x.view(x.shape[0], channels, img_size, img_size)
            weights_raw = model.weight_encoder(x_spatial)
        else:
            t0_norm = torch.zeros(x.shape[0], device=device)
            weights_raw = model.weight_encoder(x)

        # ── Normalize then denormalize (the sanity check) ─────────────────────
        weights_norm = model.scaler(weights_raw, reverse=False, training=False)
        weights_denorm = model.scaler(weights_norm, reverse=True)

        x_recon = model._inr_decode(weights_denorm)
    model.train()

    def _to_img(tensor_1d):
        """Flat tensor → numpy HxW or HxWxC in [0,1]."""
        img = tensor_1d.cpu().numpy().reshape(channels, img_size, img_size)
        if channels == 1:
            return img[0]
        return img.transpose(1, 2, 0)

    originals = [(x[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]
    recons = [(x_recon[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]
    new_row = np.stack([_to_img(t) for t in originals + recons], axis=0)

    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    rows_path = os.path.join(metadata_dir, f"{filename}_rows.npy")

    if os.path.exists(meta_path) and os.path.exists(rows_path):
        with open(meta_path) as f:
            meta = json.load(f)
        existing_rows = np.load(rows_path)
        all_rows = np.concatenate([existing_rows, new_row[None]], axis=0)
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_rows = new_row[None]
        all_epochs = [epoch]

    np.save(rows_path, all_rows)
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs}, f)

    n_existing = len(all_epochs)
    blank_shape = (n_cols, *new_row.shape[1:])
    blank = np.ones(blank_shape)
    padded_rows = list(all_rows) + [blank] * (N_ROWS_TOTAL - n_existing)
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - n_existing)

    label_width = 0.5
    img_inches = 1.2
    row_gap = 0.15
    title_pad = 0.35
    divider_gap = 0.08
    fig_w = label_width + n_cols * img_inches + divider_gap
    fig_h = title_pad + N_ROWS_TOTAL * img_inches + (N_ROWS_TOTAL - 1) * row_gap
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    for c, header in enumerate(["", "Originals", "", "", "Reconstructions", ""]):
        extra = divider_gap if c >= n_pairs else 0.0
        cx = (label_width + (c + 0.5) * img_inches + extra) / fig_w
        fig.text(cx, 1.0 - (title_pad * 0.7 / fig_h), header, ha="center", va="center", fontsize=7, color="#555555")

    for r, (row_samples, ep) in enumerate(zip(padded_rows, padded_epochs, strict=False)):
        for c in range(n_cols):
            extra = divider_gap if c >= n_pairs else 0.0
            left = (label_width + c * img_inches + extra) / fig_w
            bottom = 1.0 - (title_pad / fig_h) - (r + 1) * (img_inches / fig_h) - r * (row_gap / fig_h)
            width = img_inches / fig_w
            height = img_inches / fig_h
            ax = fig.add_axes([left, bottom, width, height])
            if channels == 1:
                ax.imshow(row_samples[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(row_samples[c], vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
        fig.text(
            (label_width * 0.5) / fig_w,
            1.0 - (title_pad / fig_h) - (r + 0.5) * (img_inches / fig_h) - r * (row_gap / fig_h),
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    divider_x = (label_width + n_pairs * img_inches + divider_gap * 0.5) / fig_w
    fig.add_artist(
        plt.Line2D(
            [divider_x, divider_x],
            [0.01, 1.0 - title_pad / fig_h],
            transform=fig.transFigure,
            color="#cccccc",
            linewidth=0.8,
            linestyle="--",
        )
    )
    fig.suptitle("Reconstruction Norm Progression", fontsize=11, fontweight="bold", y=1.02)
    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_training_summary(
    name: str,
    history: dict,
    global_step: int,
    completed_steps: int,
    start_epoch: int,
    epochs: int,
    lr: float,
) -> None:
    if not history.get("total"):
        return

    losses = history["total"]
    steps = history["steps"]
    n = len(losses)

    first_loss = losses[0]
    final_loss = losses[-1]
    best_loss = min(losses)
    best_step = steps[losses.index(best_loss)]
    checkpoints = {pct: losses[int((pct / 100) * (n - 1))] for pct in (25, 50, 75, 100)}
    still_improving = losses[-1] < losses[max(0, n - max(1, n // 10))]
    total_steps_run = global_step - completed_steps
    final_lr = history["lr"][-1] if history["lr"] else lr

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  TRAINING COMPLETE — {name}")
    print(sep)
    print(f"  Steps trained   : {total_steps_run:,}  (epochs {start_epoch + 1}-{start_epoch + epochs})")
    print(f"  First loss      : {first_loss:.4f}")
    print(f"  Final loss      : {final_loss:.4f}  ({'+' if final_loss > first_loss else ''}{final_loss - first_loss:.4f})")
    print(f"  Best loss       : {best_loss:.4f}  @ step {best_step:.1f}")
    print(f"  Final LR        : {final_lr:.2e}")
    print(f"  Still improving : {'YES — loss still dropping at end' if still_improving else 'NO  — loss had plateaued'}")
    print(sep)
    print("  Loss at training milestones:")
    for pct, val in checkpoints.items():
        bar = "█" * int((val / (first_loss + 1e-8)) * 20)
        print(f"    {pct:3d}%  {val:.4f}  {bar}")
    print(sep)

    if any(history["diff"]):
        print("  Final loss components (last logged):")
        print(f"    diff  : {history['diff'][-1]:.4f}")
        print(f"    prior : {history['prior'][-1]:.4f}")
        print(f"    rec   : {history['rec'][-1]:.4f}")
        print(sep)
    print()


def plot_reconstruction_diffusion_progression(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "reconstruction_diffusion_progression",
) -> None:
    """
    Sanity-check plot: for each T in [100, 450, 700], encodes images to weights,
    noises to t=T, denoises back to t=0, then decodes to pixel space.
    Pipeline: x -> weight_encoder -> scaler(normalize) -> noise to t=T
              -> reverse diffusion t=T..0 -> scaler(denormalize) -> INR -> x_recon
    Parameters
    ----------
    model       : WeightDiffusion, already on device.
    batch       : Current training batch — list/tuple where batch[0] is images.
    epoch       : Current epoch, used as the row label.
    run_dir     : Run results directory.
    device      : Device string.
    data_config : Dict with "channels", "img_size", "data_dim".
    filename    : Base name for the saved png and metadata files.
    """
    import json

    T_VALUES = [model.T // 4, model.T // 2, 3 * model.T // 4, model.T - 1]  # noise levels to evaluate  # noqa: N806

    os.makedirs(run_dir, exist_ok=True)
    N_ROWS_TOTAL = 5  # noqa: N806
    n_pairs = 2
    n_orig = n_pairs
    n_recon_cols = n_pairs * len(T_VALUES)
    n_cols = n_orig + n_recon_cols  # 2 originals + 2*4 reconstructions = 10
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    x = batch[0][5 : 5 + n_pairs].to(device)

    def _run_diffusion(t_noise: int) -> torch.Tensor:
        """
        Encodes x to weights, noises to t=t_noise, denoises back to t=0, decodes.
        Args:
            t_noise : Timestep index to noise to before reversing.
        Returns:
            x_recon : Reconstructed images, same shape as x.
        """
        weights_raw = model.weight_encoder(x)
        if model.normalize:
            weights_raw = model.scaler(weights_raw, reverse=False, training=False)
        t_idx = torch.full((x.shape[0],), t_noise, dtype=torch.long, device=device)
        curr_theta, _ = model._construct_theta_t(weights_raw, t_idx)

        for t in tqdm(
            range(t_noise, -1, -1),
            desc=f"Denoising T={t_noise}",
            total=t_noise + 1,
            file=sys.stderr,
        ):
            t_norm = torch.full((x.shape[0], 1), t / (model.T - 1), device=device)
            theta0_hat = model.denoiser(curr_theta, t_norm)  # direct x0 prediction

            if t > 0:
                alpha_bar = model.alpha_cumprod[t]
                alpha_bar_prev = model.alpha_cumprod[t - 1]
                alpha_t = model.alpha[t]
                beta_t = model.beta[t]

                coeff_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar)
                coeff_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                mean = coeff_x0 * theta0_hat + coeff_xt * curr_theta
                sigma = torch.sqrt(beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                z = torch.randn_like(curr_theta) if t > 1 else torch.zeros_like(curr_theta)
                curr_theta = mean + sigma * z
            else:
                curr_theta = theta0_hat

        if model.normalize:
            curr_theta = model.scaler(curr_theta, reverse=True)
        return model._inr_decode(curr_theta)

    model.eval()
    with torch.no_grad():
        recons_per_t = [_run_diffusion(t) for t in T_VALUES]
    model.train()

    def _to_img(tensor_1d: torch.Tensor) -> np.ndarray:
        """Flat tensor → numpy HxW or HxWxC in [0,1]."""
        img = tensor_1d.cpu().numpy().reshape(channels, img_size, img_size)
        if channels == 1:
            return img[0]
        return img.transpose(1, 2, 0)

    originals = [(x[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]
    # Shape: (n_cols,) — originals first, then recons grouped by T value
    all_imgs = originals + [(recons_per_t[t_idx][i] * 0.5 + 0.5).clamp(0, 1) for t_idx in range(len(T_VALUES)) for i in range(n_pairs)]
    new_row = np.stack([_to_img(t) for t in all_imgs], axis=0)

    # ── Persist rows across epochs ────────────────────────────────────────────
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    rows_path = os.path.join(metadata_dir, f"{filename}_rows.npy")
    if os.path.exists(meta_path) and os.path.exists(rows_path):
        with open(meta_path) as f:
            meta = json.load(f)
        existing_rows = np.load(rows_path)
        all_rows = np.concatenate([existing_rows, new_row[None]], axis=0)
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_rows = new_row[None]
        all_epochs = [epoch]
    np.save(rows_path, all_rows)
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs}, f)

    # ── Layout constants ──────────────────────────────────────────────────────
    n_existing = len(all_epochs)
    blank_shape = (n_cols, *new_row.shape[1:])
    blank = np.ones(blank_shape)
    padded_rows = list(all_rows) + [blank] * (N_ROWS_TOTAL - n_existing)
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - n_existing)

    label_width = 0.5
    img_inches = 1.2
    row_gap = 0.15
    title_pad = 0.55  # extra height for two-line header
    divider_gap = 0.08
    n_dividers = len(T_VALUES) + 1  # one before originals group, one before each T group

    fig_w = label_width + n_cols * img_inches + n_dividers * divider_gap
    fig_h = title_pad + N_ROWS_TOTAL * img_inches + (N_ROWS_TOTAL - 1) * row_gap
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Column index → (group_label, divider_offset_from_left)
    # Groups: orig(0-1), T=100(2-3), T=450(4-5), T=700(6-7)
    group_starts = [0, n_orig] + [n_orig + t * n_pairs for t in range(1, len(T_VALUES))]
    group_labels = ["Originals"] + [f"T={t}" for t in T_VALUES]

    def _col_x(col: int) -> float:
        """Compute figure-space x-center for a given column index."""
        # Count how many dividers precede this column
        n_div = sum(1 for gs in group_starts[1:] if col >= gs)
        return (label_width + (col + 0.5) * img_inches + n_div * divider_gap) / fig_w

    def _col_left(col: int) -> float:
        """Compute figure-space left edge for a given column index."""
        n_div = sum(1 for gs in group_starts[1:] if col >= gs)
        return (label_width + col * img_inches + n_div * divider_gap) / fig_w

    # ── Group header labels ───────────────────────────────────────────────────
    header_y = 1.0 - (title_pad * 0.4 / fig_h)
    for g_idx, (g_start, g_label) in enumerate(zip(group_starts, group_labels, strict=False)):
        g_end = g_start + (n_orig if g_idx == 0 else n_pairs)
        cx = (_col_x(g_start) + _col_x(g_end - 1)) / 2
        fig.text(cx, header_y, g_label, ha="center", va="center", fontsize=7, color="#555555")

    # ── Image axes ───────────────────────────────────────────────────────────
    for r, (row_samples, ep) in enumerate(zip(padded_rows, padded_epochs, strict=False)):
        for c in range(n_cols):
            left = _col_left(c)
            bottom = 1.0 - (title_pad / fig_h) - (r + 1) * (img_inches / fig_h) - r * (row_gap / fig_h)
            width = img_inches / fig_w
            height = img_inches / fig_h
            ax = fig.add_axes([left, bottom, width, height])
            if channels == 1:
                ax.imshow(row_samples[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(row_samples[c], vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

        # Row epoch label
        fig.text(
            (label_width * 0.5) / fig_w,
            1.0 - (title_pad / fig_h) - (r + 0.5) * (img_inches / fig_h) - r * (row_gap / fig_h),
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    # ── Dividers between groups ───────────────────────────────────────────────
    divider_ys = [0.01, 1.0 - title_pad / fig_h]
    for g_start in group_starts[1:]:
        div_x = _col_left(g_start) - divider_gap * 0.5 / fig_w
        fig.add_artist(
            plt.Line2D(
                [div_x, div_x],
                divider_ys,
                transform=fig.transFigure,
                color="#cccccc",
                linewidth=0.8,
                linestyle="--",
            )
        )

    fig.suptitle("Reconstruction Diffusion Progression", fontsize=11, fontweight="bold", y=1.02)
    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_weight_profile_progression(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "weight_profile_progression",
) -> None:
    """
    Plots the raw sequence of weights for a SINGLE sample to see the 'signature'.
    """
    import json
    import os

    os.makedirs(run_dir, exist_ok=True)
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    channels, img_size = data_config["channels"], data_config["img_size"]

    # ── 1. Get One Sample Weight ──────────────────────────────────────────────
    x = batch[0][0:1].to(device)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "F_phi"):
            weights = model.F_phi(x, torch.zeros(1, 1, device=device))
        elif hasattr(model, "W") and hasattr(model.W, "inflate"):
            weights = model.weight_encoder(x.view(1, channels, img_size, img_size))  # noqa: F841
        else:
            mean, logvar = model.weight_encoder(x)
            weights_raw_np = model.weight_encoder._reparameterize(mean, logvar)
        weights_raw_np = weights_raw_np
        if model.normalize:
            weights_norm = model.scaler(weights_raw_np, reverse=False, training=False)
            weights_np = weights_norm.detach().cpu().numpy().flatten()
        else:
            weights_np = weights_raw_np.detach().cpu().numpy().flatten()
    model.train()

    # ── 2. Persist ────────────────────────────────────────────────────────────
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    rows_path = os.path.join(metadata_dir, f"{filename}_weights.npy")

    if os.path.exists(meta_path) and os.path.exists(rows_path):
        with open(meta_path) as f:
            meta = json.load(f)
        all_weights = list(np.load(rows_path, allow_pickle=True)) + [weights_np]  # noqa: RUF005
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_weights = [weights_np]
        all_epochs = [epoch]

    np.save(rows_path, np.array(all_weights, dtype=object))
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs}, f)

    if model.normalize:
        raw_rows_path = os.path.join(metadata_dir, f"{filename}_raw_weights.npy")
        all_raw = list(np.load(raw_rows_path, allow_pickle=True)) + [weights_raw_np] if os.path.exists(raw_rows_path) else [weights_raw_np]  # noqa: RUF005
        np.save(raw_rows_path, np.array(all_raw, dtype=object))

    # ── 3. Plotting ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(N_ROWS_TOTAL, 1, figsize=(10, 10), sharex=True)
    fig.patch.set_facecolor("white")

    # Determine consistent Y-limits based on seen data
    all_vals = np.concatenate(all_weights)
    y_min, y_max = np.percentile(all_vals, [0.5, 99.5])

    for i in range(N_ROWS_TOTAL):
        ax = axes[i]
        if i < len(all_weights):
            ax.plot(all_weights[i], color="#E67E22", linewidth=0.6, label="normalized")
            if model.normalize and i < len(all_raw):
                ax2 = ax.twinx()
                ax2.plot(all_raw[i], color="#4A90E2", linewidth=0.6, alpha=0.6, label="raw")
                ax2.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(y_min * 1.2, y_max * 1.2)
            ax.set_ylabel(f"ep {all_epochs[i]}", fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    plt.xlabel("Weight Index (0 to N)")
    fig.suptitle("Weight Vector Profile Progression (Single Sample)", fontsize=12, fontweight="bold", y=0.96)
    plt.savefig(os.path.join(run_dir, f"{filename}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weight_distribution_progression(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "weight_dist_progression",
) -> None:
    """
    Computes weights for the WHOLE BATCH and plots three distribution progression plots:
    1. Raw encoder weight vectors across epochs.
    2. Normalized weight vectors (after scaler) — what the diffusion actually trains on.
    3. Noised weight vectors at t=T across epochs (should approach N(0,1)).

    Args:
        model:       Model with weight_encoder, scaler, alpha_cumprod, sigma buffers.
        batch:       (images, labels) tuple.
        epoch:       Current epoch number.
        run_dir:     Directory to save plots and metadata.
        device:      Device string.
        data_config: Dict with 'channels' and 'img_size' keys.
        filename:    Base filename for outputs.
    Returns: None
    """
    import json
    import os

    os.makedirs(run_dir, exist_ok=True)
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    channels, img_size = data_config["channels"], data_config["img_size"]

    # ── 1. Get Batch Weights ──────────────────────────────────────────────────
    x = batch[0].to(device)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "F_phi"):
            t = torch.zeros(x.shape[0], 1, device=device)
            weights = model.F_phi(x, t)
        elif hasattr(model, "W") and hasattr(model.W, "inflate"):
            x_spatial = x.view(x.shape[0], channels, img_size, img_size)
            weights = model.weight_encoder(x_spatial)
        else:
            weights = model.weight_encoder(x)

        # ── 2. Normalized weights (what diffusion trains on) ──────────────────
        if model.normalize:
            weights = model.scaler(weights, reverse=False, training=False)

        weights_batch_np = weights.detach().cpu().numpy().flatten()

        # ── 3. Noise weights at t=T ───────────────────────────────────────────
        T_idx = model.T - 1  # noqa: N806
        alpha_T = model.sqrt_alpha_cumprod[T_idx]  # noqa: N806
        sigma_T = model.sigma[T_idx]  # noqa: N806
        epsilon = torch.randn_like(weights)
        theta_T = alpha_T * weights + sigma_T * epsilon  # noqa: N806
        theta_T_np = theta_T.detach().cpu().numpy().flatten()  # noqa: N806

    model.train()

    # ── 4. Persist raw, normalized, and noised weights ────────────────────────
    def _load_or_init(meta_path: str, data_path: str, new_data: np.ndarray, new_epoch: int):
        """Load existing history and append new data, or start fresh."""
        if os.path.exists(meta_path) and os.path.exists(data_path):
            with open(meta_path) as f:
                meta = json.load(f)
            all_data = list(np.load(data_path, allow_pickle=True)) + [new_data]  # noqa: RUF005
            all_epochs = meta["epochs"] + [new_epoch]
        else:
            all_data = [new_data]
            all_epochs = [new_epoch]
        np.save(data_path, np.array(all_data, dtype=object))
        with open(meta_path, "w") as f:
            json.dump({"epochs": all_epochs}, f)
        return all_data, all_epochs

    raw_meta = os.path.join(metadata_dir, f"{filename}_meta.json")
    raw_data = os.path.join(metadata_dir, f"{filename}_weights.npy")
    # norm_meta = os.path.join(metadata_dir, f"{filename}_normalized_meta.json")
    # norm_data = os.path.join(metadata_dir, f"{filename}_normalized_weights.npy")
    noised_meta = os.path.join(metadata_dir, f"{filename}_noised_meta.json")
    noised_data = os.path.join(metadata_dir, f"{filename}_noised_weights.npy")

    all_weights, all_epochs = _load_or_init(raw_meta, raw_data, weights_batch_np, epoch)
    # all_normalized, all_epochs_norm = _load_or_init(norm_meta, norm_data, weights_normalized_np, epoch)
    all_noised, all_epochs_noised = _load_or_init(noised_meta, noised_data, theta_T_np, epoch)

    # ── 5. Plotting helper ────────────────────────────────────────────────────
    def _plot_progression(
        all_data: list,
        epochs: list,
        title: str,
        xlabel: str,
        save_path: str,
        reference_gaussian: bool = False,
    ) -> None:
        """Plot a weight distribution histogram progression across epochs."""
        fig, axes = plt.subplots(N_ROWS_TOTAL, 1, figsize=(7, 10), sharex=True)
        fig.patch.set_facecolor("white")

        all_vals = np.concatenate(all_data)
        x_min, x_max = np.percentile(all_vals, [0.5, 99.5])

        for i in range(N_ROWS_TOTAL):
            ax = axes[i]
            if i < len(all_data):
                w = all_data[i]
                mu_val, std_val = np.mean(w), np.std(w)
                ax.hist(w, bins=100, color="#4A90E2", alpha=0.7, range=(x_min, x_max), density=True)

                if reference_gaussian:
                    xs = np.linspace(x_min, x_max, 300)
                    gaussian = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs**2)
                    ax.plot(
                        xs,
                        gaussian,
                        color="#E25050",
                        linewidth=1.2,
                        linestyle="--",
                        label="N(0,1)",
                    )

                ax.set_ylabel(f"ep {epochs[i]}", fontsize=9, fontweight="bold")
                ax.text(
                    0.98,
                    0.85,
                    f"$\mu$:{mu_val:.3f}\n$\sigma$:{std_val:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "ec": "none"},
                )
            ax.spines[["top", "right"]].set_visible(False)

        plt.xlabel(xlabel)
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.96)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 6. Save all three plots ───────────────────────────────────────────────
    # 3) Distribution of raw weight vector from the weight encoder
    _plot_progression(
        all_weights,
        all_epochs,
        title="Batch Weight Distribution Progression (Raw)",
        xlabel="Weight Value Magnitude (Batch Distribution)",
        save_path=os.path.join(run_dir, f"{filename}.png"),
    )

    # 2) Normalized so the weight vector distribution of the normalized weight vector
    """
    _plot_progression(
        all_normalized,
        all_epochs_norm,
        title="Normalized Weight Distribution Progression (Diffusion Input)",
        xlabel="Weight Value Magnitude (Normalized)",
        save_path=os.path.join(run_dir, f"{filename}_normalized.png"),
        reference_gaussian=True,  # diffusion input should be ~N(0,1)
    )
    """
    # 1) Noised_T so the weight vector distribution of the raw weight vector at time T
    _plot_progression(
        all_noised,
        all_epochs_noised,
        title="Noised Weight Distribution at t=T",
        xlabel="Weight Value (Noised at t=T)",
        save_path=os.path.join(run_dir, f"{filename}_noised_T.png"),
        reference_gaussian=True,
    )


def plot_forward_trajectory_progression(
    model: object,
    batch: torch.Tensor,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    filename: str = "forward_trajectory_progression",
    model_name: str = "",
    normalize: bool = False,  # noqa: ARG001
) -> None:
    """
    Appends a row of 5 weight distribution histograms (one per t-value) to the
    forward noising trajectory progression figure, saved to <run_dir>/<filename>.png.
    Always renders 5 rows x 5 columns — empty rows shown as blank until filled.
    T-values match the reverse process: {T-1, 3T//4, T//2, T//4, 0}.
    Histograms are pre-computed and stored as bin counts/edges in JSON,
    avoiding large .npy files.
    Args:
        model:       Model with weight_encoder, sqrt_alpha_cumprod, sigma buffers.
        batch:       (images, labels) tuple.
        epoch:       Current epoch number.
        run_dir:     Directory to save plots and metadata.
        device:      Device string.
        data_config: Dict with 'channels' and 'img_size' keys.
        filename:    Base filename for outputs.
        model_name:  Model variant name string.
        normalize:   Whether to apply scaler normalization.
    Returns: None
    """
    import json

    os.makedirs(run_dir, exist_ok=True)
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    N_BINS = 80  # noqa: N806
    N_ROWS_TOTAL = 5  # noqa: N806
    channels, img_size = data_config["channels"], data_config["img_size"]

    T_values_sorted = sorted(  # noqa: N806
        [model.T - 1, 3 * model.T // 4, model.T // 2, model.T // 4, 0],
        reverse=True,
    )

    # ── 1. Encode batch to weights ────────────────────────────────────────────
    x = batch[0].to(device)
    # print("DEBUG: [Trajectory]", x.shape, "device", x.device, "dtype", x.dtype, "mean", x.mean().item(), "std", x.std().item())
    model.eval()
    # print("DEBUG: [Trajectory]", model.probablistic, "normalize", model.normalize)

    if model_name in ("latent_inr_diffusion", "weight_inr_diffusion", "weight_inr_ndm_diffusion"):
        if x.dim() == 2:
            channels = x.shape[1] // (model.img_size * model.img_size)
            x = x.view(x.shape[0], channels, model.img_size, model.img_size)

        z, _, _ = model.encode(x)

        # if model has a self.normalize apply normalization to z
        if hasattr(model, "normalize") and model.normalize:
            z = model.scaler(z, reverse=False) if model.normalize else z

        raw_arrays = []
        for t in T_values_sorted:
            theta_t = z if t == 0 else model._forward_process(z, torch.tensor([t], device=z.device))[0]
            raw_arrays.append(theta_t.detach().cpu().numpy().flatten())

    else:
        with torch.no_grad():
            if hasattr(model, "F_phi"):
                t_zero = torch.zeros(x.shape[0], 1, device=device)
                weights = model.F_phi(x, t_zero)
            elif hasattr(model, "W") and hasattr(model.W, "inflate"):
                x_spatial = x.view(x.shape[0], channels, img_size, img_size)
                weights = model.weight_encoder(x_spatial)
            else:
                weights = model.weight_encoder(x)
            if model.normalize:
                print("\n####################################")
                print("##########Applying Scaler Normalization##########")
                print("####################################")
                weights = model.scaler(weights, reverse=False, training=False)
                print("weights stats after scaler:", weights.mean(), weights.std())
            raw_arrays = []
            for t in T_values_sorted:
                theta_t = weights if t == 0 else model._forward_process(weights, t)
                raw_arrays.append(theta_t.detach().cpu().numpy().flatten())

    model.train()

    # ── 2. Pre-compute histograms — store counts/edges instead of raw arrays ──
    new_row_data = []
    for flat_arr in raw_arrays:
        counts, edges = np.histogram(flat_arr, bins=N_BINS, density=True)
        new_row_data.append(
            {
                "counts": counts.tolist(),
                "edges": edges.tolist(),
                "mu": float(np.mean(flat_arr)),
                "std": float(np.std(flat_arr)),
            }
        )

    # ── 3. Persist data (single JSON, no .npy) ───────────────────────────────
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["rows"].append(new_row_data)
        meta["epochs"].append(epoch)
    else:
        meta = {
            "epochs": [epoch],
            "t_keys": T_values_sorted,
            "rows": [new_row_data],
        }

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    all_rows = meta["rows"]
    all_epochs = meta["epochs"]

    # ── 4. Pad to N_ROWS_TOTAL ────────────────────────────────────────────────
    padded_rows = all_rows + [None] * (N_ROWS_TOTAL - len(all_rows))
    padded_epochs = all_epochs + [""] * (N_ROWS_TOTAL - len(all_epochs))

    # ── 5. Build figure ───────────────────────────────────────────────────────
    n_cols = len(T_values_sorted)
    col_width = 1.9
    row_height = 1.6
    label_width = 0.75
    col_gap = 0.35
    row_gap = 0.25
    title_pad = 0.5
    header_pad = 0.35

    fig_w = label_width + n_cols * col_width + (n_cols - 1) * col_gap
    fig_h = title_pad + header_pad + N_ROWS_TOTAL * row_height + (N_ROWS_TOTAL - 1) * row_gap
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Column headers
    for c, t_val in enumerate(T_values_sorted):
        cx = (label_width + c * (col_width + col_gap) + col_width * 0.5) / fig_w
        cy = 1.0 - (title_pad / fig_h) - (header_pad * 0.6 / fig_h)
        label = f"t = {t_val}" if t_val > 0 else "t = 0 (raw)"
        fig.text(cx, cy, label, ha="center", va="center", fontsize=8, color="#555555", fontweight="bold")

    for r, (row_data, ep) in enumerate(zip(padded_rows, padded_epochs, strict=False)):
        row_bottom = 1.0 - (title_pad / fig_h) - (header_pad / fig_h) - (r + 1) * (row_height / fig_h) - r * (row_gap / fig_h)
        for c in range(n_cols):
            left = (label_width + c * (col_width + col_gap)) / fig_w
            ax = fig.add_axes([left, row_bottom, col_width / fig_w, row_height / fig_h])

            if row_data is not None:
                hist = row_data[c]
                counts = np.array(hist["counts"])
                edges = np.array(hist["edges"])
                mu_val, std_val = hist["mu"], hist["std"]

                # Reconstruct bar plot from pre-computed histogram
                ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#E2844A", alpha=0.75)

                if T_values_sorted[c] > 0:
                    xs = np.linspace(edges[0], edges[-1], 300)
                    gaussian = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs**2)
                    ax.plot(xs, gaussian, color="#333333", linewidth=1.0, linestyle="--", label="N(0,1)")

                ax.text(
                    0.97,
                    0.93,
                    f"μ:{mu_val:.2f}\nx:{std_val:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "ec": "none"},
                )

            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)
            if r == N_ROWS_TOTAL - 1:
                ax.set_xlabel("weight value", fontsize=7)

        fig.text(
            (label_width * 0.5) / fig_w,
            row_bottom + (row_height * 0.5) / fig_h,
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("Forward Noising Trajectory — Weight Distributions", fontsize=11, fontweight="bold", y=0.99)
    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Plotting for FID table (per-model sample quality metrics)
# =============================================================================
def plot_val_elbo_progression(
    model: torch.nn.Module,
    data_loader_val: torch.utils.data.DataLoader,
    epoch: int,
    run_dir: str,
    filename: str = "val_elbo_progression",
) -> None:
    """
    Invokes compute_full_elbo, appends the result to a persistent JSON file,
    and updates a single continuous progression graph across restarts/resumes.
    """
    os.makedirs(run_dir, exist_ok=True)
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")

    # ── 1. Extract Validation ELBO ───────────────────────────────────────────
    avg_elbo = model.compute_full_elbo(data_loader_val)

    # ── 2. Load and Append History ───────────────────────────────────────────
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:  # noqa: UP015
                meta = json.load(f)
            all_epochs = meta.get("epochs", []) + [epoch]  # noqa: RUF005
            all_elbos = meta.get("elbo_values", []) + [avg_elbo]  # noqa: RUF005
        except (json.JSONDecodeError, KeyError):
            all_epochs = [epoch]
            all_elbos = [avg_elbo]
    else:
        all_epochs = [epoch]
        all_elbos = [avg_elbo]

    # Remove duplicates and sort by epoch to keep the line moving clean linearly
    # (Crucial for handling crashes/resumes where an epoch might get re-logged)
    history_dict = {}
    for ep, val in zip(all_epochs, all_elbos):  # noqa: B905
        history_dict[ep] = val  # overwrites old duplicate epoch keys with fresh values

    sorted_epochs = sorted(history_dict.keys())
    sorted_elbos = [history_dict[ep] for ep in sorted_epochs]

    with open(meta_path, "w") as f:
        json.dump({"epochs": sorted_epochs, "elbo_values": sorted_elbos}, f, indent=4)

    # ── 3. Build & Render Progression Graph ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfc")

    # Plot historical trajectory tracking lines
    ax.plot(sorted_epochs, sorted_elbos, marker="o", color="#2b5c8f", linestyle="-", linewidth=2, markersize=5, label="Validation ELBO")

    # Text annotation for the most recent tracking point
    ax.annotate(
        f"{avg_elbo:.4f}", xy=(epoch, avg_elbo), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold", color="#1a365d"
    )

    # Layout and styling cleanups
    ax.set_title("Validation Full ELBO Progression", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10, color="#333333")
    ax.set_ylabel("ELBO (higher is better)", fontsize=10, color="#333333")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

    if len(sorted_epochs) > 1:
        ax.set_xlim(min(sorted_epochs) - 0.5, max(sorted_epochs) + 0.5)

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    fig.tight_layout()

    save_path = os.path.join(run_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  [Eval Checkpoint] Epoch {epoch} Validation ELBO: {avg_elbo:.6f}")


def _build_figure(
    metrics: dict,
    sample_images: dict[str, np.ndarray],
    real_dist: np.ndarray,
    out_path: str,
) -> None:
    """
    Builds and saves the comparison figure with three sections:
    table (top, 3 models only), 4x4 image grids (middle, real + 3 models),
    bar charts (bottom, real + 3 models).

    Args:
        metrics:       dict keyed by slot ("model_1/2/3") →
                       {mnist_fid, inception_fid, uniformity, dist_gen, label, color}
        sample_images: dict keyed by slot → np.ndarray (16, C, H, W) in [0,1]
        real_dist:     (10,) ground-truth class distribution for real MNIST
        out_path:      path to save the figure
    Returns:
        None
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model_keys = list(metrics.keys())
    n_models = len(model_keys)
    n_grids = 1 + n_models
    digits = np.arange(10)

    REAL_COLOR = "#444444"  # noqa: N806
    REAL_LABEL = "Real MNIST"  # noqa: N806

    # Load 16 real MNIST images for the grid
    mnist = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
    indices = np.random.choice(len(mnist), 16, replace=False)
    real_grid = np.stack([mnist[i][0].numpy() for i in indices])  # (16, 1, 28, 28)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5 * n_grids, 13))
    fig.patch.set_facecolor("white")

    ax_table = fig.add_axes([0.05, 0.76, 0.90, 0.21])
    ax_table.axis("off")

    grid_gap = 0.02
    grid_w = (0.88 - (n_grids - 1) * grid_gap) / n_grids
    grid_bottom = 0.495
    grid_height = 0.245
    grid_axes = []
    grid_labels = [REAL_LABEL] + [metrics[k]["label"] for k in model_keys]
    grid_colors = [REAL_COLOR] + [metrics[k]["color"] for k in model_keys]

    for g in range(n_grids):
        axes_row = []
        for row in range(4):
            for col in range(4):
                left = 0.06 + g * (grid_w + grid_gap) + col * (grid_w / 4)
                bottom = grid_bottom + (3 - row) * (grid_height / 4)
                ax = fig.add_axes([left, bottom, grid_w / 4, grid_height / 4])
                ax.axis("off")
                axes_row.append(ax)
        grid_axes.append(axes_row)

    bar_w = 0.82 / n_grids
    bar_gap = 0.02
    bar_axes = []
    for i in range(n_grids):
        ax = fig.add_axes([0.06 + i * (bar_w + bar_gap), 0.06, bar_w, 0.40])
        bar_axes.append(ax)

    # ── Table (3 models only) ─────────────────────────────────────────────────
    col_labels = ["Model", "MNIST FID ↓", "Inception FID ↓", "Uniformity ↓"]
    table_data = []
    for key in model_keys:
        m = metrics[key]
        table_data.append(
            [
                m["label"],
                f"{m['mnist_fid']:.2f}",
                f"{m['inception_fid']:.2f}",
                f"{m['uniformity']:.2f}",
            ]
        )
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    best_mnist = min(range(n_models), key=lambda i: metrics[model_keys[i]]["mnist_fid"])
    best_inception = min(range(n_models), key=lambda i: metrics[model_keys[i]]["inception_fid"])
    best_uniformity = min(range(n_models), key=lambda i: metrics[model_keys[i]]["uniformity"])
    best_cols = {1: best_mnist, 2: best_inception, 3: best_uniformity}

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
        cell.set_text_props(color="#111111")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold", color="#111111")
        if row > 0 and col == 0:
            cell.set_text_props(color=metrics[model_keys[row - 1]]["color"], fontweight="bold")
        if row > 0 and col in best_cols:  # noqa: SIM102
            if best_cols[col] == row - 1:
                cell.set_text_props(color="#2a9d3a", fontweight="bold")

    ax_table.set_title(
        "Model Comparison — MNIST Generation",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#111111",
    )

    # ── Image grids (real + 3 models) ─────────────────────────────────────────
    all_grids = [real_grid] + [sample_images[k] for k in model_keys]

    for g, (images, label, color) in enumerate(zip(all_grids, grid_labels, grid_colors, strict=False)):
        for idx in range(16):
            ax = grid_axes[g][idx]
            img = images[idx]
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray", vmin=0, vmax=1, aspect="auto")
            else:
                ax.imshow(np.transpose(img, (1, 2, 0)).clip(0, 1), aspect="auto")
        grid_centre_x = 0.06 + g * (grid_w + grid_gap) + grid_w / 2
        fig.text(
            grid_centre_x,
            grid_bottom + grid_height + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    # ── Bar plots (real + 3 models) ───────────────────────────────────────────
    all_dists = [real_dist] + [metrics[k]["dist_gen"] for k in model_keys]
    all_colors = [REAL_COLOR] + [metrics[k]["color"] for k in model_keys]
    all_labels = [REAL_LABEL] + [metrics[k]["label"] for k in model_keys]
    y_max = max(d.max() for d in all_dists) * 100 * 1.25

    for i, (ax, dist, color, label) in enumerate(zip(bar_axes, all_dists, all_colors, all_labels, strict=False)):
        ax.bar(digits, dist * 100, color=color, alpha=0.85, width=0.65)
        ax.axhline(10, color="#999999", linewidth=1.0, linestyle="--", label="Uniform (10%)")
        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits], fontsize=10)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Digit", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold", color=color, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_edgecolor("#cccccc")
        ax.spines["bottom"].set_edgecolor("#cccccc")
        ax.tick_params(colors="#555555")
        ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("% of samples", fontsize=10)
            ax.legend(fontsize=9, framealpha=0.8, loc="upper right")
        else:
            ax.set_yticklabels([])

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved → {out_path}")
