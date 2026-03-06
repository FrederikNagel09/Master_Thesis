"""
Generates two comparison plots for VAE and VAE-INR models.

    Plot 1 — Sampling comparison (all 4 models, 8 samples each)
    Plot 2 — Upscaling showcase (INR models only, 4 samples x 5 resolutions)

Output directory: src/results/general/
"""

import sys

sys.path.append(".")

import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ============================================================
#  HARDCODE CONFIG PATHS HERE
# ============================================================
VAE_GAUSS_CONFIG = "src/results/vae/experiments/vae_gauss_06-03-13:34.json"
VAE_MOG_CONFIG = "src/results/vae/experiments/vae_MoG_06-03-14:02.json"
INR_GAUSS_CONFIG = "src/results/vae_inr_hypernet/experiments/inr_vae_gauss_06-03-13:58.json"
INR_MOG_CONFIG = "src/results/vae_inr_hypernet/experiments/inr_vae_mog_06-03-13:59.json"
# ============================================================

OUT_DIR = "src/results/general"
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_vae(config: dict, device: str):
    from src.models.prior import GaussianPrior, MoGPrior, VAMPPrior
    from src.models.vae import VAE
    from src.models.vae_coders import BernoulliFullDecoder, GaussianFullEncoder

    latent_dim = config["latent_dim"]
    hidden_dims = config["hidden_dims"]
    prior_type = config["prior"]

    encoder = GaussianFullEncoder(input_dim=784, latent_dim=latent_dim, hidden_dims=hidden_dims)
    decoder = BernoulliFullDecoder(latent_dim=latent_dim, output_shape=(28, 28), hidden_dims=hidden_dims)

    if prior_type == "gaussian":
        prior = GaussianPrior(latent_dim=latent_dim)
    elif prior_type == "mog":
        prior = MoGPrior(latent_dim=latent_dim)
    elif prior_type == "vampp":
        prior = VAMPPrior()
    else:
        raise ValueError(f"Unsupported prior: {prior_type}")

    model = VAE(encoder=encoder, decoder=decoder, prior=prior)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval().to(device)
    return model


def build_inr_vae(config: dict, device: str):
    from src.models.inr_vae_hypernet import INR, VAEINR
    from src.models.prior import GaussianPrior, MoGPrior
    from src.models.vae_coders import GaussianEncoder

    latent_dim = config["latent_dim"]

    inr = INR(
        coord_dim=2,
        hidden_dim=config["inr_hidden_dim"],
        n_hidden=config["inr_layers"],
        out_dim=config["inr_out_dim"],
    )

    encoder_net = nn.Sequential(
        nn.Linear(784, config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], config["vae_enc_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_enc_dim"], latent_dim * 2),
    )
    encoder = GaussianEncoder(encoder_net)

    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], config["vae_dec_dim"]),
        nn.ReLU(),
        nn.Linear(config["vae_dec_dim"], inr.num_weights),
    )

    if config["prior"] == "gaussian":
        prior = GaussianPrior(latent_dim=latent_dim)
    elif config["prior"] == "mog":
        prior = MoGPrior(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported prior: {config['prior']}")

    model = VAEINR(prior, encoder, decoder_net, inr, beta=1.0, prior_type=config["prior"]).to(device)
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.eval()
    return model


def make_coord_grid(size: int, device: str) -> torch.Tensor:
    """(size*size, 2) coordinate grid normalised to [-1, 1]."""
    lin = torch.linspace(-1, 1, size)
    gr, gc = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([gr.flatten(), gc.flatten()], dim=-1).to(device)


# ------------------------------------------------------------
# Sampling helpers
# ------------------------------------------------------------


def sample_vae(model, n: int, device: str) -> np.ndarray:  # noqa: ARG001
    """Returns (n, 28, 28) numpy array in [0, 1]."""
    with torch.no_grad():
        imgs = model.sample(n_samples=n).cpu().numpy()  # (n, 28, 28)
    return imgs


def sample_inr_vae(model, n: int, res: int, device: str) -> np.ndarray:
    """Returns (n, res, res) numpy array in [0, 1]."""
    with torch.no_grad():
        z = model.prior().sample(torch.Size([n])).to(device)
        flat_w = model.decode_to_weights(z)
        coords = make_coord_grid(res, device)  # (res^2, 2)
        coords_b = coords.unsqueeze(0).expand(n, -1, -1)  # (n, res^2, 2)
        pixels = model.inr(coords_b, flat_w)  # (n, res^2, 1)
        imgs = pixels.squeeze(-1).view(n, res, res).cpu().numpy()  # (n, res, res)
    # normalise to [0, 1]
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
    return imgs


def sample_inr_vae_fixed_z(model, n: int, resolutions: list, device: str) -> dict:
    """Sample n images at multiple resolutions using the *same* z vectors.
    Returns {res: (n, res, res) numpy array}."""
    with torch.no_grad():
        z = model.prior().sample(torch.Size([n])).to(device)
        flat_w = model.decode_to_weights(z)
        results = {}
        for res in resolutions:
            coords = make_coord_grid(res, device)
            coords_b = coords.unsqueeze(0).expand(n, -1, -1)
            pixels = model.inr(coords_b, flat_w)
            imgs = pixels.squeeze(-1).view(n, res, res).cpu().numpy()
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
            results[res] = imgs
    return results


# ============================================================
# PLOT 1 — Sampling comparison
# ============================================================


def plot_sampling(device: str):
    N_SAMPLES = 8  # noqa: N806
    RESOLUTIONS_INR = 28  # resolution used for INR rows  # noqa: N806

    configs = [
        (VAE_GAUSS_CONFIG, "vae", "VAE  —  Gaussian prior"),
        (VAE_MOG_CONFIG, "vae", "VAE  —  MoG prior"),
        (INR_GAUSS_CONFIG, "vae_inr", "VAE-INR  —  Gaussian prior"),
        (INR_MOG_CONFIG, "vae_inr", "VAE-INR  —  MoG prior"),
    ]

    rows_images = []
    row_labels = []

    for cfg_path, model_type, label_base in configs:
        cfg = load_config(cfg_path)
        latent_dim = cfg["latent_dim"]
        full_label = f"{label_base}   |   z = {latent_dim}"
        if model_type == "vae":
            model = build_vae(cfg, device)
            imgs = sample_vae(model, N_SAMPLES, device)
        else:
            model = build_inr_vae(cfg, device)
            imgs = sample_inr_vae(model, N_SAMPLES, RESOLUTIONS_INR, device)
        rows_images.append(imgs)
        row_labels.append(full_label)
        del model

    n_rows = len(rows_images)

    # Each row: images + title space above
    # Use GridSpec so we can control title rows explicitly
    title_h = 0.3  # relative height for title rows
    img_h = 1.0  # relative height for image rows
    row_heights = []
    for _ in range(n_rows):
        row_heights += [title_h, img_h]

    fig = plt.figure(figsize=(N_SAMPLES * 1.6, n_rows * 2.2))
    gs = gridspec.GridSpec(
        n_rows * 2,
        N_SAMPLES,
        figure=fig,
        height_ratios=row_heights,
        hspace=0.08,
        wspace=0.05,
    )

    for r, (imgs, label) in enumerate(zip(rows_images, row_labels)):  # noqa: B905
        title_row = r * 2
        img_row = r * 2 + 1

        # --- Title: one invisible axes spanning all columns, text centred ---
        ax_title = fig.add_subplot(gs[title_row, :])
        ax_title.set_axis_off()
        ax_title.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            transform=ax_title.transAxes,
        )

        # --- Image row ---
        for c in range(N_SAMPLES):
            ax = fig.add_subplot(gs[img_row, c])
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

    out_path = os.path.join(OUT_DIR, "sampling_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 1] Saved → {out_path}")


# ============================================================
# PLOT 2 — Upscaling showcase (INR only)
# ============================================================


def plot_upscaling(device: str):
    N_SAMPLES = 4  # noqa: N806
    RESOLUTIONS = [64, 128, 256, 512, 1024]  # noqa: N806

    inr_configs = [
        (INR_GAUSS_CONFIG, "VAE-INR  —  Gaussian prior"),
        (INR_MOG_CONFIG, "VAE-INR  —  MoG prior"),
    ]

    n_models = len(inr_configs)
    n_res = len(RESOLUTIONS)

    # Layout per model-row:
    #   title_row  (thin)
    #   res_label_row (thin) — one label centred over each 2x2 block
    #   img_row_0  \  2 image rows making up the 2x2 grid
    #   img_row_1  /
    #
    # Between each 2x2 block we add a gap column.
    # Columns: [img, img, GAP, img, img, GAP, ...] x n_res  (last gap omitted)

    IMG_COLS = 2  # images per block horizontally  # noqa: N806
    GAP_COLS = 1  # spacer columns between blocks  # noqa: N806
    n_img_cols = n_res * IMG_COLS + (n_res - 1) * GAP_COLS

    # Row structure per model: title, res-label, img, gap, img  → 5 sub-rows
    ROWS_PER_MODEL = 5  # noqa: N806
    title_h = 0.25
    res_lbl_h = 0.18
    img_h = 1.0
    img_gap_h = 0.06  # vertical gap between the two image rows in each 2x2 block
    row_h = [title_h, res_lbl_h, img_h, img_gap_h, img_h]

    all_row_heights = []
    model_gap_h = 0.4  # extra gap between the two model blocks (as a dummy row)
    for m in range(n_models):
        all_row_heights += row_h
        if m < n_models - 1:
            all_row_heights.append(model_gap_h)

    total_rows = len(all_row_heights)

    # Column widths: image cols = 1.0, gap cols = 0.7
    col_widths = []
    for i in range(n_res):
        col_widths += [1.0, 1.0]
        if i < n_res - 1:
            col_widths.append(0.7)

    fig = plt.figure(figsize=(n_res * 4.2, n_models * 4.0 + 0.4))

    gs = gridspec.GridSpec(
        total_rows,
        n_img_cols,
        figure=fig,
        height_ratios=all_row_heights,
        width_ratios=col_widths,
        hspace=0.0,
        wspace=0.0,
    )

    for row_idx, (cfg_path, model_label) in enumerate(inr_configs):
        cfg = load_config(cfg_path)
        latent_dim = cfg["latent_dim"]
        full_label = f"{model_label}   |   z = {latent_dim}"

        model = build_inr_vae(cfg, device)
        res_imgs = {res: sample_inr_vae(model, N_SAMPLES, res, device) for res in RESOLUTIONS}
        del model

        # Row offsets for this model block
        base_row = row_idx * (ROWS_PER_MODEL + 1)  # +1 for the inter-model gap row
        if row_idx == n_models - 1:
            base_row = row_idx * ROWS_PER_MODEL + row_idx  # same formula works

        r_title = base_row
        r_res_lbl = base_row + 1
        r_img0 = base_row + 2
        # base_row + 3 is the gap row (no axes added)
        r_img1 = base_row + 4  # was base_row + 3

        # --- Model title spanning all columns ---
        ax_title = fig.add_subplot(gs[r_title, :])
        ax_title.set_axis_off()
        ax_title.text(
            0.5,
            0.1,
            full_label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            transform=ax_title.transAxes,
        )

        for col_idx, res in enumerate(RESOLUTIONS):
            imgs = res_imgs[res]  # (4, res, res)

            # Column index of the left image in this block
            c_left = col_idx * (IMG_COLS + GAP_COLS)

            # --- Resolution label: invisible axes spanning the 2 image cols,
            #     text anchored at centre ---
            ax_res = fig.add_subplot(gs[r_res_lbl, c_left : c_left + IMG_COLS])
            ax_res.set_axis_off()
            ax_res.text(
                0.5,
                0.5,
                f"{res}x{res}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                transform=ax_res.transAxes,
            )

            # --- 2x2 image grid ---
            for sub_r, gs_row in enumerate([r_img0, r_img1]):
                for sub_c in range(IMG_COLS):
                    img_idx = sub_r * IMG_COLS + sub_c
                    ax = fig.add_subplot(gs[gs_row, c_left + sub_c])
                    ax.imshow(imgs[img_idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    ax.axis("off")

    out_path = os.path.join(OUT_DIR, "upscaling_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 2] Saved → {out_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n--- Generating Plot 1: Sampling comparison ---")
    plot_sampling(device)

    print("\n--- Generating Plot 2: Upscaling showcase ---")
    plot_upscaling(device)

    print("\nDone. Both plots saved to:", OUT_DIR)
