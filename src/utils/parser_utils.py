import argparse
import json


def parse_args_sample():
    parser = argparse.ArgumentParser(description="INR inference — reconstruct an image at any resolution.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the experiment directory.")
    parser.add_argument("--height", type=int, help="Output image height in pixels.")
    parser.add_argument("--width", type=int, help="Output image width in pixels.")
    parser.add_argument("--grid_size", type=int, default=4, help="Grid is grid_size x grid_size images")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps (None = full T)")
    args = parser.parse_args()

    return args


def parse_config_vars(config_path: str) -> tuple[int, int, int, int]:
    """
    Extract image index and layer sizes from a JSON metadata file.
    The JSON file should be in the same directory as the weights file,
    with the same name but a .json extension.
    Returns: (img_index, h1, h2, h3)
    """
    print(f"Parsing config variables from: {config_path}")
    with open(config_path) as f:
        meta = json.load(f)
    try:
        return meta
    except KeyError as e:
        raise ValueError(f"Missing expected field {e} in metadata file: {config_path}")  # noqa: B904


# Define which args to save per model type
MODEL_SAVE_KEYS = {
    "basic_inr": ["model", "epochs", "batch_size", "lr", "name", "h1", "h2", "h3", "omega_0"],
    "hypernet_inr": ["model", "epochs", "batch_size", "lr", "name", "subset_frac", "hyper_h", "h1", "h2", "h3", "omega_0"],
    "vae_inr_hypernet": [
        "model",
        "epochs",
        "batch_size",
        "lr",
        "latent_dim",
        "prior",
        "device",
        "name",
        "subset_frac",
        "inr_hidden_dim",
        "inr_layers",
        "inr_out_dim",
        "vae_enc_dim",
        "vae_dec_dim",
    ],
    "vae": ["model", "epochs", "batch_size", "lr", "latent_dim", "prior", "device", "name", "subset_frac", "hidden_dims"],
    "ddpm": ["model", "epochs", "batch_size", "lr", "T", "name", "subset_frac", "device"],
    "ndm": [
        "model",
        "epochs",
        "batch_size",
        "lr",
        "T",
        "name",
        "subset_frac",
        "device",
        "f_phi_type",
        "f_phi_hidden",
        "f_phi_t_embed",
        "sigma_tilde",
        "dataset",
    ],
    # add more model types here as needed
}


def save_config(args, save_dir, weights_path=None):
    full_config = {**vars(args), "weights_path": weights_path}

    # Filter to only the keys defined for this model, otherwise save everything
    allowed_keys = MODEL_SAVE_KEYS.get(args.model)
    if allowed_keys is not None:
        config = {k: full_config[k] for k in allowed_keys if k in full_config}
        config["weights_path"] = weights_path  # always include this
        config["model_type"] = args.model  # always include this
    else:
        config = full_config

    with open(save_dir, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Experiment config saved to: {save_dir}")


def parse_args_training():
    parser = argparse.ArgumentParser(description="Train an INR MLP on a single MNIST image.")
    parser.add_argument(
        "--model",
        type=str,
        default="basic_inr",
        help="Model to train (default: 'basic_inr').",
        choices=["basic_inr", "ddpm", "hypernet_inr", "ndm", "vae", "vae_inr_hypernet"],
    )

    parser.add_argument("--index", type=int, default=0, help="Index of the MNIST image to fit (default: 0).")
    parser.add_argument(
        "--name", type=str, default="img_", help="Base name for the run. The image index is appended automatically (default: 'trial_')."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--h1", type=int, default=20, help="Size of first hidden layer (default: 20).")
    parser.add_argument("--h2", type=int, default=20, help="Size of second hidden layer (default: 20).")
    parser.add_argument("--h3", type=int, default=20, help="Size of third hidden layer (default: 20).")
    parser.add_argument("--omega_0", type=float, default=20.0, help="Omega_0 parameter for SIREN layers (default: 20.0).")
    parser.add_argument("--hyper_h", type=int, default=256, help="Size of hidden layer in hypernetwork (default: 256).")
    parser.add_argument("--subset_frac", type=float, default=1.0, help="Fraction of the dataset to use (default: 1.0, i.e. 100%).")

    # ---- NDM architecture ----
    parser.add_argument("--T", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="lr warm up")

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth file to resume from")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch number the checkpoint was saved at")

    parser.add_argument(
        "--f_phi_type",
        type=str,
        default="mlp",
        choices=["mlp", "unet"],
        help="Architecture for the learnable data-transformation F_phi. 'mlp' is faster; 'unet' matches the noise-predictor architecture.",
    )
    parser.add_argument(
        "--f_phi_hidden",
        type=int,
        nargs="+",
        default=[512, 512, 512],
        help="Hidden layer sizes for the MLP transformation F_phi (ignored when --f_phi_type unet). Example: --f_phi_hidden 512 512 512",
    )
    parser.add_argument(
        "--f_phi_t_embed",
        type=int,
        default=32,
        help="Dimension of the time embedding in the MLP transformation (ignored when --f_phi_type unet).",
    )
    parser.add_argument(
        "--sigma_tilde",
        type=float,
        default=1.0,
        help="Stochasticity factor for the NDM reverse process. "
        "0.0 → deterministic DDIM-style sampling. "
        "1.0 → full DDPM-style stochasticity (default).",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=20,
        help="How often (in optimizer steps) to log the running average loss.",
    )
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])

    # ---- NDM training ----
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--sample_every", type=int, default=5, help="Save sample grid every N epochs")
    parser.add_argument("--sample_steps", type=int, default=None, help="Steps for sampling (None = full T)")

    # ---- VAE architecture ----
    parser.add_argument("--latent_dim", type=int, default=20, help="Dimension of the VAE latent space (default: 20).")
    parser.add_argument(
        "--prior", type=str, default="gaussian", help="Type of prior for VAE (default: 'gaussian'). Choices: 'gaussian', 'mog', 'vamp'."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (default: 'cpu'). Use 'cuda' for GPU training.")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512, 512])

    parser.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE style). Default 1.0.")

    parser.add_argument("--inr_hidden_dim", type=int, default=32)
    parser.add_argument("--inr_layers", type=int, default=3)
    parser.add_argument("--inr_out_dim", type=int, default=1)
    parser.add_argument("--vae_enc_dim", type=int, default=512)
    parser.add_argument("--vae_dec_dim", type=int, default=512)

    args = parser.parse_args()
    return args


def parse_args_training_2() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NDM Training")

    # ---- Shared ----
    p.add_argument("--model", type=str, required=True, help="Model type: ndm")
    p.add_argument("--name", type=str, required=True, help="Run name (used in file names)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dataset", type=str, default="mnist")

    # ---- NDM architecture ----
    p.add_argument("--T", type=int, default=1000, help="Diffusion timesteps")
    p.add_argument("--fphi_ch", type=int, default=32, help="F_phi base channels")
    p.add_argument("--denoiser_ch", type=int, default=64, help="Denoiser base channels")
    p.add_argument("--time_emb_dim", type=int, default=256)

    # ---- NDM training ----
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_every", type=int, default=5, help="Save sample grid every N epochs")
    p.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--sample_steps", type=int, default=None, help="Steps for sampling (None = full T)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()
