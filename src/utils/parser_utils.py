import argparse
import json


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
    config = {
        "model": args.model,
        "index": args.index,
        "name": args.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "h1": args.h1,
        "h2": args.h2,
        "h3": args.h3,
        "dataset": args.dataset,
        "omega_0": args.omega_0,
        "weights_path": weights_path,
    }

    with open(save_dir, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Experiment config saved to: {save_dir}")


def parse_args_basic_inr():
    parser = argparse.ArgumentParser(description="Train an INR MLP on a single MNIST image.")
    parser.add_argument(
        "--model",
        type=str,
        default="basic_inr",
        help="Model to train (default: 'basic_inr').",
        choices=["basic_inr", "ddpm", "inr_mlp_hypernet"],
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
    args = parser.parse_args()

    return args
