import os
import sys

import torch
from torch.utils.data import Subset

sys.path.append(".")

from src.utils.general_utils import get_current_datetime
from src.utils.parser_utils import save_config

RES_DIR = "src/results/"


def run_training_siren_inr(args):
    """
    Runs training of an INR MLP on a single MNIST image, with layer sizes chosen to match the number of pixels.
    then saves weights and training plot
    """

    # Imports
    from src.models.inr_siren import SirenINR
    from src.utils.training_utils import train_inr_siren

    if args.dataset == "mnist":
        from src.dataloaders.MNISTCoord import MNISTCoordDataset

        # Loads the MNIST image at the given index number
        dataset = MNISTCoordDataset(mnist_raw_dir="data/MNIST/raw", image_index=args.index)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Initialize the SirenINR model with the computed layer sizes
    model = SirenINR(h1=args.h1, h2=args.h2, h3=args.h3, omega_0=args.omega_0)

    # Create name for training run
    run_name = f"{args.name}_{get_current_datetime()}"

    # Run training:
    model = train_inr_siren(
        model=model,
        dataset=dataset,
        name=run_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save weights
    weights_path = os.path.join(RES_DIR, f"{args.model}/weights", f"{run_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")

    save_dir = os.path.join(f"src/results/{args.model}/experiments", f"{run_name}.json")
    save_config(args, save_dir, weights_path)


def run_training_ddpm(args):
    # Imports
    from src.utils.training_utils import train_ddpm

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Training on device: {device}")

    train_ddpm(
        device=device,
        T=args.T,
        img_size=args.img_size,
        channels=args.channels,
        time_dim=args.time_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        experiment_name=args.experiment_name,
        weights_dir=args.weights_dir,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        data_root=args.data_root,
    )


def run_training_inr_mlp_hypernet(args):
    """
    Train a HyperINR model on MNIST.

    The hypernetwork learns to map a full MNIST image to a set of INR weights.
    The INR then maps pixel coordinates to pixel values using those weights.
    """
    # Imports
    from src.dataloaders.MNISTHyper import MNISTHyperDataset
    from src.models.inr_mlp_hypernet import HyperINR
    from src.utils.training_utils import train_inr_mlp_hypernet

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    dataset = MNISTHyperDataset(mnist_raw_dir="data/MNIST/raw")
    # Subset: use first N images (or random sample)
    n = int(len(dataset) * args.subset_frac)
    dataset = Subset(dataset, range(n))

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = HyperINR(
        h1=args.h1,
        h2=args.h2,
        h3=args.h3,
        omega_0=args.omega_0,
        hyper_h=args.hyper_h,
    )

    run_name = f"{args.name}_{get_current_datetime()}"

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    model = train_inr_mlp_hypernet(
        model=model,
        dataset=dataset,
        name=run_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save weights
    weights_path = os.path.join(RES_DIR, f"{args.model}/weights", f"{run_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")

    save_dir = os.path.join(f"src/results/{args.model}/experiments", f"{run_name}.json")
    save_config(args, save_dir, weights_path)
