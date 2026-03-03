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


def save_config(args, save_dir, weights_path=None):
    config = {**vars(args), "weights_path": weights_path}

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
        choices=["basic_inr", "ddpm", "hypernet_inr", "ndm"],
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
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use (default: 'mnist').")
    parser.add_argument("--omega_0", type=float, default=20.0, help="Omega_0 parameter for SIREN layers (default: 20.0).")
    parser.add_argument("--hyper_h", type=int, default=256, help="Size of hidden layer in hypernetwork (default: 256).")
    parser.add_argument("--subset_frac", type=float, default=1.0, help="Fraction of the dataset to use (default: 1.0, i.e. 100%).")

    # ---- NDM architecture ----
    parser.add_argument("--T", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--fphi_ch", type=int, default=32, help="F_phi base channels")
    parser.add_argument("--denoiser_ch", type=int, default=64, help="Denoiser base channels")
    parser.add_argument("--time_emb_dim", type=int, default=256)

    # ---- NDM training ----
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--sample_every", type=int, default=5, help="Save sample grid every N epochs")
    parser.add_argument("--sample_steps", type=int, default=None, help="Steps for sampling (None = full T)")
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
