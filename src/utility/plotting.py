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
from tqdm import tqdm

from src.configs.results_config import MODEL_COLORS, MODEL_LABELS
from src.configs.train_plot_config import _COLORS, _LABELS

# =============================================================================
# Helpers
# =============================================================================


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
) -> tuple[np.ndarray, dict[int, np.ndarray] | None]:
    """
    Draw n_samples from model and return rendered grid + optional denoising snapshots.
    Args:
        model:              Trained model.
        model_type:         Model type string.
        n_samples:          Number of samples to draw.
        device:             Device string.
        data_config:        Dict with 'channels', 'img_size', 'data_dim'.
        collect_snapshots:  If True, collect weight snapshots at T-values (NDM transinr only).
    Returns:
        grid:      (n_samples, H, W) or (n_samples, H, W, C) numpy array in [0, 1].
        snapshots: {t_value: flat np.ndarray} or None if not collected.
    """
    import torch

    channels = data_config["channels"]
    img_size = data_config["img_size"]
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

        elif model_type == "ndm_inr" or model_type == "latent_inr_diffusion":
            samples = model.sample(n_samples)
            samples = (samples * 0.5 + 0.5).clamp(0, 1).reshape(n_samples, channels, img_size, img_size)

        elif model_type == "ndm_transinr" or model_type in ("ndm_static_transinr", "ndm_temporal_transinr", "ndm_static_mlpinr"):
            if collect_snapshots:
                raw_samples, snapshots = model.sample_weight(n_samples=128, collect_snapshots=True)
            else:
                raw_samples = model.sample_weight(n_samples)
            # Use only first n_samples for the image grid
            samples = model._inr_decode(raw_samples[:n_samples])
            samples = (samples * 0.5 + 0.5).clamp(0, 1).reshape(n_samples, channels, img_size, img_size)

        else:
            raise ValueError(f"Unknown model_type '{model_type}' for sampling.")

    samples = samples.cpu().numpy()
    grid = samples[:, 0, :, :] if channels == 1 else samples.transpose(0, 2, 3, 1)

    return grid, snapshots


def plot_final_samples(
    model: object,
    model_type: str,
    epoch: int,
    run_dir: str,
    device: str,
    data_config: dict,
    n_samples: int = 64,
) -> None:
    """
    Sample an 8x8 grid from the model and save to
    <run_dir>/final_samples_ep{epoch}.png.

    Parameters
    ----------
    model       : Trained model, already on device.
    model_type  : One of "ndm", "inr_vae", "ndm_inr", "ndm_temporal_transinr".
    epoch       : Current epoch number, used in the filename.
    run_dir     : Run results directory (src/train_results/{run_name}).
    device      : Device string.
    data_config : Dict with "channels", "img_size", "data_dim".
    n_samples   : Total samples; displayed as sqrt x sqrt grid.
    """
    os.makedirs(run_dir, exist_ok=True)

    n_side = int(np.sqrt(n_samples))
    channels = data_config["channels"]
    samples = _model_to_grid(model, model_type, n_side * n_side, device, data_config)

    fig, axes = plt.subplots(n_side, n_side, figsize=(n_side * 1.5, n_side * 1.5))
    fig.suptitle(f"Final samples — epoch {epoch}", fontsize=11)

    for i, ax in enumerate(axes.flatten()):
        if channels == 1:
            ax.imshow(samples[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(samples[i], vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")

    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    save_path = os.path.join(run_dir, f"final_samples_ep{epoch}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Final samples saved → {save_path}")


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
    """
    Append a row of 6 samples to the training progression figure and save to
    <run_dir>/sample_progression.png, overwriting each call.
    Always renders 5 rows — empty rows shown as blank until filled.
    Args:
        model:              Trained model, already on device.
        model_type:         One of "ndm", "inr_vae", "ndm_inr", "ndm_temporal_transinr".
        epoch:              Current epoch, used as the row label.
        run_dir:            Run results directory.
        device:             Device string.
        data_config:        Dict with "channels", "img_size", "data_dim".
        filename:           Base filename for outputs.
        collect_snapshots:  If True, also plot denoising trajectory histograms.
    Returns: None
    """
    import json

    os.makedirs(run_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    n_cols = 6
    channels = data_config["channels"]

    # ── Draw new row of samples ───────────────────────────────────────────────
    new_row, snapshots = _model_to_grid(model, model_type, n_cols, device, data_config, collect_snapshots)

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
        json.dump({"epochs": all_epochs}, f)

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

    # ── Plot denoising trajectory if snapshots were collected ─────────────────
    print(f"\n\nSample progression saved {collect_snapshots and snapshots is not None}\n\n")
    if collect_snapshots and snapshots is not None:
        print("################### Plotting denoising trajectory progression... ###################")
        plot_denoising_trajectory_progression(
            snapshots=snapshots,
            epoch=epoch,
            run_dir=run_dir,
        )


def plot_denoising_trajectory_progression(
    snapshots: dict[int, np.ndarray],
    epoch: int,
    run_dir: str,
    filename: str = "Reverse_denoising_progression",
) -> None:
    """
    Append a row of 4 weight distribution histograms to the denoising trajectory
    progression figure and save to <run_dir>/<filename>.png, overwriting each call.
    Always renders 5 rows x 4 columns — empty rows shown as blank until filled.
    Each row is one sampling run (epoch), each column one T-value snapshot.
    Args:
        snapshots: {t_value: flat np.ndarray} from sample_weight with collect_snapshots=True.
        epoch:     Current epoch, used as row label.
        run_dir:   Run results directory.
        filename:  Base filename for outputs.
    Returns: None
    """
    import json

    os.makedirs(run_dir, exist_ok=True)
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    # Sorted descending so columns go T-1 → T//4 (high noise → clean)
    t_keys_sorted = sorted(snapshots.keys(), reverse=True)
    n_cols = len(t_keys_sorted)  # always 4

    # New row: list of flat arrays in column order
    new_row_data = [snapshots[t] for t in t_keys_sorted]

    # ── Load or init persisted data ───────────────────────────────────────────
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    data_path = os.path.join(metadata_dir, f"{filename}_data.npy")

    if os.path.exists(meta_path) and os.path.exists(data_path):
        with open(meta_path) as f:
            meta = json.load(f)
        all_rows = list(np.load(data_path, allow_pickle=True))
        all_rows.append(new_row_data)
        all_epochs = meta["epochs"] + [epoch]
        all_t_keys = meta["t_keys"]  # reuse column order from first call
    else:
        all_rows = [new_row_data]
        all_epochs = [epoch]
        all_t_keys = t_keys_sorted

    np.save(data_path, np.array(all_rows, dtype=object))
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs, "t_keys": all_t_keys}, f)

    # ── Pad to N_ROWS_TOTAL ───────────────────────────────────────────────────
    padded_rows = all_rows + [None] * (N_ROWS_TOTAL - len(all_rows))
    padded_epochs = all_epochs + [""] * (N_ROWS_TOTAL - len(all_epochs))

    # ── Build figure: 5 rows x 4 cols of histograms ───────────────────────────
    col_width = 2.2  # inches per histogram
    row_height = 1.6  # inches per histogram
    label_width = 0.75
    col_gap = 0.35
    row_gap = 0.25
    title_pad = 0.5
    header_pad = 0.35

    fig_w = label_width + n_cols * col_width + (n_cols - 1) * col_gap
    fig_h = title_pad + header_pad + N_ROWS_TOTAL * row_height + (N_ROWS_TOTAL - 1) * row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Column headers (T-value labels)
    for c, t_val in enumerate(all_t_keys):
        cx = (label_width + c * (col_width + col_gap) + col_width * 0.5) / fig_w
        cy = 1.0 - (title_pad / fig_h) - (header_pad * 0.6 / fig_h)
        fig.text(cx, cy, f"t = {t_val}", ha="center", va="center", fontsize=8, color="#555555", fontweight="bold")

    for r, (row_data, ep) in enumerate(zip(padded_rows, padded_epochs, strict=False)):
        row_bottom = 1.0 - (title_pad / fig_h) - (header_pad / fig_h) - (r + 1) * (row_height / fig_h) - r * (row_gap / fig_h)

        for c in range(n_cols):
            left = (label_width + c * (col_width + col_gap)) / fig_w
            width = col_width / fig_w
            height = row_height / fig_h
            ax = fig.add_axes([left, row_bottom, width, height])

            if row_data is not None:
                w = row_data[c]
                mu_val, std_val = np.mean(w), np.std(w)
                ax.hist(w, bins=80, color="#4A90E2", alpha=0.75, density=True)

                # Reference N(0,1)
                xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 300)
                gaussian = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs**2)
                ax.plot(xs, gaussian, color="#333333", linewidth=1.0, linestyle="--")

                ax.text(
                    0.97,
                    0.93,
                    f"μ:{mu_val:.2f}\n σ:{std_val:.2f}",  # noqa: RUF001
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "ec": "none"},
                )

            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)

            # X-axis label only on bottom row
            if r == N_ROWS_TOTAL - 1:
                ax.set_xlabel("weight value", fontsize=7)

        # Epoch label on the left, vertically centred on the row
        fig.text(
            (label_width * 0.5) / fig_w,
            row_bottom + (row_height * 0.5) / fig_h,
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("Denoising Trajectory — Weight Distributions", fontsize=11, fontweight="bold", y=0.99)

    print(f"\n\nDenoising trajectory progression saved → {os.path.join(run_dir, f'{filename}.png')}\n\n")
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
    """
    Append a row of 6 reconstructions to the progression figure and save to
    <run_dir>/<filename>.png, overwriting each call.

    Always renders 5 rows — empty rows are shown as blank until filled.
    Each row is labelled with its epoch on the left. Left half shows originals,
    right half shows reconstructions.

    Reconstruction pipeline (mirrors _l_rec):
        w = F_phi(x, t=0)
        x_recon = INR(coords, w)

    Parameters
    ----------
    model       : NeuralDiffusionModelINR, already on device.
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
    n_cols = 6  # 3 originals + 3 reconstructions
    n_pairs = n_cols // 2
    channels = data_config["channels"]
    img_size = data_config["img_size"]

    # ── Get images from batch ─────────────────────────────────────────────────
    x = batch[0][:n_pairs].to(device)  # (3, data_dim)

    # ── Reconstruct via encoder(t=0) → INR decode ────────────────────────────
    model.eval()

    if model_name == "latent_inr_diffusion":
        with torch.no_grad():
            if x.dim() == 2:
                channels = x.shape[1] // (model.img_size * model.img_size)
                x = x.view(x.shape[0], channels, model.img_size, model.img_size)
            z = model.latent_encoder(x)
            x_recon = model._decode_latent(z)
    else:
        with torch.no_grad():
            if hasattr(model, "F_phi"):
                t0_norm = torch.zeros(x.shape[0], 1, device=device)
                weights = model.F_phi(x, t0_norm)  # temporal encoder
            elif hasattr(model, "W") and hasattr(model.W, "inflate"):
                # TransInrEncoder expects spatial (B, C, H, W)
                x_spatial = x.view(x.shape[0], channels, img_size, img_size)
                weights = model.weight_encoder(x_spatial)  # ← spatial reshape
            else:
                t0_norm = torch.zeros(x.shape[0], device=device)
                weights = model.weight_encoder(x)  # static MLP/CNN encoder

            x_recon = model._inr_decode(weights)  # (3, data_dim)

    model.train()

    def _to_img(tensor_1d):
        """Flat tensor → numpy HxW or HxWxC in [0,1]."""
        img = tensor_1d.cpu().numpy().reshape(channels, img_size, img_size)
        if channels == 1:
            return img[0]
        return img.transpose(1, 2, 0)

    # ── Build new row: [orig_0, orig_1, orig_2, recon_0, recon_1, recon_2] ────
    originals = [(x[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]  # [-1,1] → [0,1]
    recons = [(x_recon[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]  # already [0,1]
    new_row = np.stack([_to_img(t) for t in originals + recons], axis=0)  # (6, H, W[,C])

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
        json.dump({"epochs": all_epochs}, f)

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
    divider_gap = 0.08  # extra horizontal gap between originals and recons

    fig_w = label_width + n_cols * img_inches + divider_gap
    fig_h = title_pad + N_ROWS_TOTAL * img_inches + (N_ROWS_TOTAL - 1) * row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Column header labels (only drawn once, above the axes area)
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

        # Epoch label on the left
        fig.text(
            (label_width * 0.5) / fig_w,
            1.0 - (title_pad / fig_h) - (r + 0.5) * (img_inches / fig_h) - r * (row_gap / fig_h),
            f"ep {ep}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    # Vertical divider line between originals and reconstructions
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
    model       : NDMStaticTransInr, already on device.
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
            weights = model.weight_encoder(x.view(1, channels, img_size, img_size))
        else:
            weights = model.weight_encoder(x)
        weights_raw_np = weights.detach().cpu().numpy().flatten()
        if model.normalize:
            weights_norm = model.scaler(weights, reverse=False, training=False)
            weights_np = weights_norm.detach().cpu().numpy().flatten()
        else:
            weights_np = weights_raw_np
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
) -> None:
    """
    Appends a row of 5 weight distribution histograms (one per t-value) to the
    forward noising trajectory progression figure, saved to <run_dir>/<filename>.png.
    Always renders 5 rows x 5 columns — empty rows shown as blank until filled.
    T-values match the reverse process: {T-1, 3T//4, T//2, T//4, 0}.
    Args:
        model:       Model with weight_encoder, sqrt_alpha_cumprod, sigma buffers.
        batch:       (images, labels) tuple.
        epoch:       Current epoch number.
        run_dir:     Directory to save plots and metadata.
        device:      Device string.
        data_config: Dict with 'channels' and 'img_size' keys.
        filename:    Base filename for outputs.
    Returns: None
    """
    import json

    os.makedirs(run_dir, exist_ok=True)
    metadata_dir = os.path.join(run_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    channels, img_size = data_config["channels"], data_config["img_size"]

    # T-values match the reverse process snapshot points
    T_values_sorted = sorted(  # noqa: N806
        [model.T - 1, 3 * model.T // 4, model.T // 2, model.T // 4, 0],
        reverse=True,
    )

    # ── 1. Encode batch to weights ────────────────────────────────────────────
    x = batch[0].to(device)
    model.eval()
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

        # ── 2. Apply forward noising at each t-value ──────────────────────────
        new_row_data = []
        for t in T_values_sorted:
            if t == 0:
                # t=0 is the raw weight vector, no noise added
                theta_t = weights
            else:
                alpha_t = model.sqrt_alpha_cumprod[t]
                sigma_t = model.sigma[t]
                epsilon = torch.randn_like(weights)
                theta_t = alpha_t * weights + sigma_t * epsilon
            new_row_data.append(theta_t.detach().cpu().numpy().flatten())

    model.train()

    # ── 3. Persist data ───────────────────────────────────────────────────────
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    data_path = os.path.join(metadata_dir, f"{filename}_data.npy")

    if os.path.exists(meta_path) and os.path.exists(data_path):
        with open(meta_path) as f:
            meta = json.load(f)
        all_rows = list(np.load(data_path, allow_pickle=True))
        all_rows.append(new_row_data)
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_rows = [new_row_data]
        all_epochs = [epoch]

    np.save(data_path, np.array(all_rows, dtype=object))
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs, "t_keys": T_values_sorted}, f)

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
                w = row_data[c]
                mu_val, std_val = np.mean(w), np.std(w)
                ax.hist(w, bins=80, color="#E2844A", alpha=0.75, density=True)

                # Reference N(0,1) on all columns except t=0 (raw weights)
                if T_values_sorted[c] > 0:
                    xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 300)
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


def _build_figure(
    metrics: dict,
    out_path: str,
) -> None:
    """
    metrics: dict keyed by model_key, each with:
        mnist_fid, inception_fid, uniformity, dist_gen (np array len 10)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model_keys = list(metrics.keys())
    n_models = len(model_keys)
    digits = np.arange(10)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5 * n_models, 9))
    fig.patch.set_facecolor("white")

    # Table takes top 30%, bar plots take bottom 60%, small gap in between
    ax_table = fig.add_axes([0.05, 0.68, 0.90, 0.28])
    ax_table.axis("off")

    bar_axes = []
    bar_w = 0.82 / n_models
    for i in range(n_models):
        ax = fig.add_axes([0.08 + i * (bar_w + 0.02), 0.08, bar_w, 0.52])
        bar_axes.append(ax)

    # ── Table ─────────────────────────────────────────────────────────────────
    col_labels = ["Model", "MNIST FID ↓", "Inception FID ↓", "Uniformity ↓"]
    table_data = []
    for key in model_keys:
        m = metrics[key]
        table_data.append(
            [
                MODEL_LABELS[key],
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

    # Find best (lowest) value per metric column
    best_mnist = min(range(n_models), key=lambda i: metrics[model_keys[i]]["mnist_fid"])
    best_inception = min(range(n_models), key=lambda i: metrics[model_keys[i]]["inception_fid"])
    best_uniformity = min(range(n_models), key=lambda i: metrics[model_keys[i]]["uniformity"])
    best_cols = {1: best_mnist, 2: best_inception, 3: best_uniformity}

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
        cell.set_text_props(color="#111111")

        if row == 0:  # header
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold", color="#111111")

        if row > 0 and col == 0:  # model name — colour coded
            key = model_keys[row - 1]
            cell.set_text_props(color=MODEL_COLORS[key], fontweight="bold")

        if row > 0 and col in best_cols:  # best value — bold green  # noqa: SIM102
            if best_cols[col] == row - 1:
                cell.set_text_props(color="#2a9d3a", fontweight="bold")

    ax_table.set_title(
        "Model Comparison — MNIST Generation",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#111111",
    )

    # ── Bar plots ─────────────────────────────────────────────────────────────
    y_max = max(metrics[k]["dist_gen"].max() for k in model_keys) * 100 * 1.25

    for i, (ax, key) in enumerate(zip(bar_axes, model_keys, strict=False)):
        dist = metrics[key]["dist_gen"]
        color = MODEL_COLORS[key]

        ax.bar(digits, dist * 100, color=color, alpha=0.85, width=0.65)
        ax.axhline(10, color="#999999", linewidth=1.0, linestyle="--", label="Uniform (10%)")

        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits], fontsize=10)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Digit", fontsize=10)
        ax.set_title(MODEL_LABELS[key], fontsize=11, fontweight="bold", color=color, pad=6)

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
