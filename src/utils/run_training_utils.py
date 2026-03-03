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
    # -----------------------------------------------------------------
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


def run_training_ndm(args):
    """
    Sets up data, model, and calls train_ndm().
    Saves final checkpoint and config after training.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from src.models.ndm import NDM
    from src.utils.training_utils import train_ndm

    # ---- Dataloader ----
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
        ]
    )
    dataset = datasets.MNIST("data/MNIST/raw", train=True, download=True, transform=transform)
    if args.subset_frac < 1.0:
        n = int(len(dataset) * args.subset_frac)
        dataset = torch.utils.data.Subset(dataset, range(n))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Model ----
    model = NDM(
        in_channels=1,
        T=args.T,
        fphi_base_ch=args.fphi_ch,
        denoiser_base_ch=args.denoiser_ch,
        time_emb_dim=args.time_emb_dim,
    )

    n_fphi = sum(p.numel() for p in model.fphi.parameters() if p.requires_grad)
    n_denoiser = sum(p.numel() for p in model.denoiser.parameters() if p.requires_grad)
    print("Trainable parameters:")
    print(f"  F_phi    : {n_fphi:,}")
    print(f"  Denoiser : {n_denoiser:,}")
    print(f"  Total    : {n_fphi + n_denoiser:,}")

    # ---- Run name ----
    run_name = f"{args.name}_{get_current_datetime()}"

    # ---- Train ----
    model = train_ndm(
        model=model,
        dataloader=dataloader,
        name=run_name,
        num_epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        sample_every=args.sample_every,
        sample_steps=args.sample_steps,
    )

    # ---- Save final weights ----
    weights_dir = os.path.join(RES_DIR, "ndm/weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{run_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")

    save_dir = os.path.join(f"src/results/{args.model}/experiments", f"{run_name}.json")
    save_config(args, save_dir, weights_path)
