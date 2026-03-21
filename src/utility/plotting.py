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

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# =============================================================================
# Colour palette
# =============================================================================

_COLORS = {
    "total": "#111111",
    "diff": "#2a6fdb",
    "prior": "#2ca05a",
    "rec": "#e07b39",
    "lr": "#cc3333",
}

_LABELS = {
    "total": "Total Loss",
    "diff": "Diffusion Loss",
    "prior": "KL / Prior Loss",
    "rec": "Reconstruction Loss",
}


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
) -> np.ndarray:
    """
    Draw n_samples from model and return a (n_samples, H, W) or
    (n_samples, H, W, C) numpy array in [0, 1].
    """
    import torch

    channels = data_config["channels"]
    img_size = data_config["img_size"]

    model.eval()
    with torch.no_grad():
        if model_type == "ndm":
            samples = model.sample(n_samples)  # (N, data_dim) in [-1, 1]
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
            pixels = model.inr(coords_batch, flat_weights)  # (N, H*W, C)
            samples = pixels.permute(0, 2, 1).reshape(n_samples, channels, img_size, img_size).clamp(0, 1)

            print(f"  [PLOT DEBUG] pixels shape: {pixels.shape}")
            print(f"  [PLOT DEBUG] pixels range: [{pixels.min():.3f}, {pixels.max():.3f}]")
            print(f"  [PLOT DEBUG] samples shape: {samples.shape}")
            print(f"  [PLOT DEBUG] samples range: [{samples.min():.3f}, {samples.max():.3f}]")
            print(f"  [PLOT DEBUG] z range: [{z.min():.3f}, {z.max():.3f}]")
            print(f"  [PLOT DEBUG] flat_weights range: [{flat_weights.min():.3f}, {flat_weights.max():.3f}]")
            print(f"  [PLOT DEBUG] channels: {channels}")

        elif model_type == "ndm_inr":
            samples = model.sample(n_samples)  # (N, H*W) in [0, 1]
            samples = samples.clamp(0, 1).reshape(n_samples, channels, img_size, img_size)

        else:
            raise ValueError(f"Unknown model_type '{model_type}' for sampling.")

    samples = samples.cpu().numpy()
    if channels == 1:
        return samples[:, 0, :, :]  # (N, H, W)
    return samples.transpose(0, 2, 3, 1)  # (N, H, W, C)


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
    model_type  : One of "ndm", "inr_vae", "ndm_inr".
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
) -> None:
    """
    Append a row of 6 samples to the training progression figure and save to
    <run_dir>/sample_progression.png, overwriting each call.

    Always renders 5 rows — empty rows are shown as blank until filled.
    Each row is labelled with its epoch on the left. Rows have a small gap
    between them; images within a row have no spacing.

    Parameters
    ----------
    model       : Trained model, already on device.
    model_type  : One of "ndm", "inr_vae", "ndm_inr".
    epoch       : Current epoch, used as the row label.
    run_dir     : Run results directory (src/train_results/{run_name}).
    device      : Device string.
    data_config : Dict with "channels", "img_size", "data_dim".
    """
    import json

    os.makedirs(run_dir, exist_ok=True)

    N_ROWS_TOTAL = 5  # noqa: N806
    n_cols = 6
    channels = data_config["channels"]

    # ── Draw new row of samples ───────────────────────────────────────────────
    new_row = _model_to_grid(model, model_type, n_cols, device, data_config)  # (6, H, W[, C])

    # ── Load existing rows from disk if available ─────────────────────────────
    metadata_dir = os.path.join(run_dir, "metadata")
    meta_path = os.path.join(metadata_dir, f"{filename}_meta.json")
    rows_path = os.path.join(metadata_dir, f"{filename}_rows.npy")

    if os.path.exists(meta_path) and os.path.exists(rows_path):
        with open(meta_path) as f:
            meta = json.load(f)
        existing_rows = np.load(rows_path)
        all_rows = np.concatenate([existing_rows, new_row[None]], axis=0)
        all_epochs = meta["epochs"] + [epoch]
    else:
        all_rows = new_row[None]  # (1, 6, H, W[, C])
        all_epochs = [epoch]

    # ── Persist updated rows ──────────────────────────────────────────────────
    np.save(rows_path, all_rows)
    with open(meta_path, "w") as f:
        json.dump({"epochs": all_epochs}, f)

    # ── Pad to always have N_ROWS_TOTAL rows ──────────────────────────────────
    n_existing = len(all_epochs)
    blank_shape = (n_cols,) + new_row.shape[1:]  # (6, H, W) or (6, H, W, C)
    blank = np.ones(blank_shape)
    padded_rows = list(all_rows) + [blank] * (N_ROWS_TOTAL - n_existing)
    padded_epochs = list(all_epochs) + [""] * (N_ROWS_TOTAL - n_existing)

    # ── Build figure ──────────────────────────────────────────────────────────
    label_width = 0.5  # inches for epoch label column
    img_inches = 1.2  # inches per image
    row_gap = 0.15  # inches of vertical gap between rows
    title_pad = 0.35  # inches reserved for title above first row

    fig_w = label_width + n_cols * img_inches
    fig_h = title_pad + N_ROWS_TOTAL * img_inches + (N_ROWS_TOTAL - 1) * row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    for r, (row_samples, ep) in enumerate(zip(padded_rows, padded_epochs)):  # noqa: B905
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

        # Epoch label centred to the left of each row
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
