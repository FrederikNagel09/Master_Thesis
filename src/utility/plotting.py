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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

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
    blank_shape = (n_cols, *new_row.shape[1:])
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

    # ── Reconstruct via F_phi(t=0) → INR decode ───────────────────────────────
    model.eval()
    with torch.no_grad():
        t0_norm = torch.zeros(x.shape[0], 1, device=device)
        weights = model.F_phi(x, t0_norm)  # (3, weight_dim)
        x_recon = model._inr_decode(weights)  # (3, data_dim)  in [0, 1]
    model.train()

    def _to_img(tensor_1d):
        """Flat tensor → numpy HxW or HxWxC in [0,1]."""
        img = tensor_1d.cpu().numpy().reshape(channels, img_size, img_size)
        if channels == 1:
            return img[0]
        return img.transpose(1, 2, 0)

    # ── Build new row: [orig_0, orig_1, orig_2, recon_0, recon_1, recon_2] ────
    originals = [(x[i] * 0.5 + 0.5).clamp(0, 1) for i in range(n_pairs)]  # [-1,1] → [0,1]
    recons = [x_recon[i].clamp(0, 1) for i in range(n_pairs)]  # already [0,1]
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
