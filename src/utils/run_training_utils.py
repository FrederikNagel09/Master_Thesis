import os
import sys

import torch
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
