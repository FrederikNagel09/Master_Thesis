import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset

sys.path.append(".")

from src.models.vae_coders import GaussianEncoder
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
    from torchvision import datasets, transforms

    from src.models.ddpm import DDPM, Unet
    from src.utils.training_utils import train_ddpm

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )

    train_data = datasets.MNIST("data/", train=True, download=True, transform=transform)
    # Subset: use first N images (or random sample)
    n = int(len(train_data) * args.subset_frac)
    train_data = Subset(train_data, range(n))
    print(f"Training on {len(train_data)} samples ({args.subset_frac * 100:.1f}% of full dataset)")
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    run_name = f"{args.name}_{get_current_datetime()}"

    network = Unet()

    # Define model
    model = DDPM(network, t=args.T).to(args.device)
    print(f"# Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    print("\nStarting DDPM training...")
    model = train_ddpm(model, optimizer, train_loader, args.epochs, args.device, name=run_name)
    print("DDPM training completed.")
    # Save weights
    weights_path = os.path.join(RES_DIR, f"{args.model}/weights", f"{run_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")

    save_dir = os.path.join(f"src/results/{args.model}/experiments", f"{run_name}.json")
    save_config(args, save_dir, weights_path)


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
    from torchvision import datasets, transforms

    from src.models.ndm import (
        MLPTransformation,  # <-- your ndm.py
        NeuralDiffusionModel,
        UnetNDM,
        UNetTransformation,
    )
    from src.utils.training_utils import train_ndm

    # ---- Data ----
    if args.dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
                transforms.Lambda(lambda x: (x - 0.5) * 2.0),
                transforms.Lambda(lambda x: x.flatten()),
            ]
        )
        train_data = datasets.CIFAR10("data/", train=True, download=True, transform=transform)
        data_dim = 32 * 32 * 3  # 3072
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img, _ = train_data[i]
            img = (img * 0.5 + 0.5).clamp(0, 1).numpy().reshape(3, 32, 32)  # back to (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # to (H, W, C) for imshow
            ax.imshow(img)
            ax.axis("off")

        fig.tight_layout()
        os.makedirs("src/results", exist_ok=True)
        fig.savefig("src/results/samples_CIFAR.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved to src/results/samples_CIFAR.png")
        print("##### DATASET: CIFAR-10 #####")
    else:  # mnist
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
                transforms.Lambda(lambda x: (x - 0.5) * 2.0),
                transforms.Lambda(lambda x: x.flatten()),
            ]
        )
        train_data = datasets.MNIST("data/", train=True, download=True, transform=transform)
        data_dim = 28 * 28  # 784
        print("##### DATASET: MNIST #####")

    single_class = args.single_class

    if single_class:
        print("##### Using only class 1 for training #####.")
        indices = [i for i, (_, label) in enumerate(train_data) if label == 1]
        train_data = Subset(train_data, indices)

    n = int(len(train_data) * args.subset_frac)
    train_data = Subset(train_data, range(n))
    print(f"Training on {len(train_data)} samples ({args.subset_frac * 100:.1f}% of full dataset)")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    run_name = f"{args.name}_{get_current_datetime()}"

    # ---- Transformation network F_phi ----
    if args.f_phi_type == "mlp":
        F_phi = MLPTransformation(  # noqa: N806
            data_dim=data_dim,
            hidden_dims=args.f_phi_hidden,
            t_embed_dim=args.f_phi_t_embed,
        )
        print(f"F_phi: MLP  hidden={args.f_phi_hidden}  t_embed={args.f_phi_t_embed}")
    elif args.f_phi_type == "unet":
        F_phi = UNetTransformation(data_dim=data_dim, base_channels=args.base_channels)  # noqa: N806
        print("F_phi: UNet")
    else:
        raise ValueError(f"Unknown f_phi_type: {args.f_phi_type!r}. Choose 'mlp' or 'unet'.")

    # ---- Noise-prediction network (same Unet as DDPM) ----
    network = UnetNDM(data_dim=data_dim, base_channels=args.base_channels)

    # ---- Model ----
    model = NeuralDiffusionModel(
        network=network,
        F_phi=F_phi,
        T=args.T,
        sigma_tilde_factor=args.sigma_tilde,
        data_dim=data_dim,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    fphi_params = sum(p.numel() for p in F_phi.parameters())
    net_params = sum(p.numel() for p in network.parameters())
    print(f"# Total parameters : {total_params:,}")
    print(f"  ε_theta (Unet)   : {net_params:,}")
    print(f"  F_phi            : {fphi_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    # ---- Resume from checkpoint (optional) ----
    start_epoch = 0
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)

        # Handle both old-style (bare state_dict) and new-style (full checkpoint)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", args.resume_epoch)
        else:
            model.load_state_dict(checkpoint)
            start_epoch = args.resume_epoch

    print(f"Resumed from epoch {start_epoch}. Training for {args.epochs} more epochs.")

    # ---- Train ----
    print("\nStarting NDM training...")
    model = train_ndm(
        model,
        optimizer,
        train_loader,
        epochs=args.epochs,
        device=args.device,
        name=run_name,
        log_every_n_steps=args.log_every_n_steps,
        warmup_steps=args.warmup_steps,
        peak_lr=args.lr,
        dataset=args.dataset,
        start_epoch=start_epoch,
    )
    print("NDM training completed.")

    # ---- Save weights ----
    weights_dir = os.path.join(RES_DIR, f"{args.model}/weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{run_name}.pth")

    # Save full checkpoint instead of just state_dict
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "run_name": run_name,
    }
    torch.save(checkpoint, weights_path)
    print(f"Checkpoint saved to: {weights_path}")
    save_dir = os.path.join(RES_DIR, f"{args.model}/experiments", f"{run_name}.json")
    save_config(args, save_dir, weights_path)


def run_vae_training(args):
    from torchvision import datasets, transforms

    from src.models.prior import GaussianPrior, MoGPrior, VAMPPrior
    from src.models.vae import VAE
    from src.models.vae_coders import BernoulliFullDecoder, GaussianFullEncoder

    # Imports for encoder, decoder, and prior
    from src.utils.training_utils import train_vae

    # Load MNIST as binarized at 'thresshold' and create data loaders
    threshold = 0.5
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])

    train_dataset = datasets.MNIST("data/", train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST("data/", train=False, download=False, transform=transform)

    if args.subset_frac < 1.0:
        n_train = int(len(train_dataset) * args.subset_frac)
        n_test = int(len(test_dataset) * args.subset_frac)
        train_dataset = torch.utils.data.Subset(train_dataset, range(n_train))
        test_dataset = torch.utils.data.Subset(test_dataset, range(n_test))

    mnist_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # mnist_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = args.device
    latent_dim = args.latent_dim
    # Define encoder, decoder, and prior based on args

    encoder = GaussianFullEncoder(input_dim=28 * 28, latent_dim=latent_dim, hidden_dims=args.hidden_dims)
    decoder = BernoulliFullDecoder(latent_dim=latent_dim, output_shape=(28, 28), hidden_dims=args.hidden_dims)

    if args.prior == "gaussian":
        prior = GaussianPrior(latent_dim=latent_dim)
    elif args.prior == "mog":
        prior = MoGPrior(latent_dim=latent_dim)
    elif args.prior == "vampp":
        prior = VAMPPrior()
    else:
        raise ValueError(f"Unsupported prior: {args.prior}")

    model = VAE(encoder=encoder, decoder=decoder, prior=prior, type=args.prior).to(device)

    run_name = f"{args.name}_{get_current_datetime()}"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = train_vae(
        model=model,
        optimizer=optimizer,
        data_loader=mnist_train_loader,
        epochs=args.epochs,
        device=device,
        name=run_name,
    )

    # Save weights
    weights_path = os.path.join(RES_DIR, f"{args.model}/weights", f"{run_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")

    save_dir = os.path.join(f"src/results/{args.model}/experiments", f"{run_name}.json")
    save_config(args, save_dir, weights_path)


def run_inr_vae_training(args):
    """
    Train an INRVAE model on MNIST.

    The encoder maps images -> q(z|x) in a small latent space.
    A linear projection lifts z -> flat INR weights.
    The INR reconstructs pixels from coordinates using those weights.
    """
    import torch.nn as nn
    from torch.utils.data import Subset

    from src.dataloaders.BinaryMNISTHyper import BinaryMNISTHyperDataset
    from src.models.inr_vae_hypernet import INR, VAEINR
    from src.models.prior import GaussianPrior, MoGPrior
    from src.utils.training_utils import train_inr_vae

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    # Load data using your INR-aware dataset instead of raw torchvision MNIST
    train_dataset = BinaryMNISTHyperDataset(mnist_raw_dir="data/MNIST/raw", split="train")
    n = int(len(train_dataset) * args.subset_frac)
    train_dataset = Subset(train_dataset, range(n))

    mnist_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- INR ---
    # Small MLP: (x,y) coords -> pixel value. Has NO nn.Parameters — weights come from decoder.
    inr = INR(coord_dim=2, hidden_dim=args.inr_hidden_dim, n_hidden=args.inr_layers, out_dim=args.inr_out_dim)

    # --- Encoder ---
    # Takes flattened binary image (784,) -> outputs mean+logvar for q(z|x)
    encoder_net = nn.Sequential(
        nn.Linear(784, args.vae_enc_dim),
        nn.ReLU(),
        nn.Linear(args.vae_enc_dim, args.vae_enc_dim),
        nn.ReLU(),
        nn.Linear(args.vae_enc_dim, args.vae_enc_dim),
        nn.ReLU(),
        nn.Linear(args.vae_enc_dim, args.latent_dim * 2),  # outputs [mean | log-var], split inside GaussianEncoder
    )
    encoder = GaussianEncoder(encoder_net)

    # --- Decoder (hypernetwork) ---
    # Takes z (M,) -> flat INR weight vector
    decoder_net = nn.Sequential(
        nn.Linear(args.latent_dim, args.vae_dec_dim),
        nn.ReLU(),
        nn.Linear(args.vae_dec_dim, args.vae_dec_dim),
        nn.ReLU(),
        nn.Linear(args.vae_dec_dim, args.vae_dec_dim),
        nn.ReLU(),
        nn.Linear(args.vae_dec_dim, inr.num_weights),  # output dim must match INR parameter count exactly
    )
    # After defining decoder_net:
    nn.init.zeros_(decoder_net[-1].bias)
    nn.init.normal_(decoder_net[-1].weight, std=0.01)  # tiny weights → INR starts near 0.5

    # ------------------------------------------------------------------
    # 2. Prior
    # ------------------------------------------------------------------
    if args.prior == "gaussian":
        prior = GaussianPrior(latent_dim=args.latent_dim)
    elif args.prior == "mog":
        prior = MoGPrior(latent_dim=args.latent_dim)
    else:
        raise ValueError(f"Unsupported prior for INRVAE: {args.prior}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    # --- Full model ---
    model = VAEINR(prior, encoder, decoder_net, inr, beta=1.0, prior_type=args.prior).to(args.device)

    print(f"\n# INR has {inr.num_weights} weights  (decoder output dim = {inr.num_weights})")
    print(f"# Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    run_name = f"{args.name}_{get_current_datetime()}"

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    model = train_inr_vae(
        model=model,
        mnist_train_loader=mnist_train_loader,
        name=run_name,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    weights_path = os.path.join(RES_DIR, f"{args.model}/weights", f"{run_name}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")

    save_dir = os.path.join(f"src/results/{args.model}/experiments", f"{run_name}.json")
    save_config(args, save_dir, weights_path)
