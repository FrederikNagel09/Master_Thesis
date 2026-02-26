"""
train.py  -  training logic for Latent NDM on MNIST
=====================================================

Training has two phases (or can be run jointly):

Phase 1  -  VAE warm-up
    Train only the VAE (encoder + decoder) with a standard ELBO:
        L_vae = L_recon + β_kl · L_kl
    This gives the diffusion model a stable latent space to work in.

Phase 2  -  Joint NDM training
    Train the UNet, NDM transform F_φ, and (optionally) the VAE jointly.
    Loss:
        L_total = L_ndm + λ_vae · (L_recon + β_kl · L_kl)

    The NDM loss follows eq. (9) of Bartosh et al. (2024):
        L_ndm = || F_φ(z, t) - UNet(z_t, t) ||²
    where z_t is the noised transformed latent and F_φ(z,t) is the target.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")

from src.neural_latent_diffusion.model import VAE, LatentNDMDiffusion, NDMTransform, UNet

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_mnist_dataloader(batch_size: int, img_size: int = 32, data_root: str = "./data"):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # → [-1, 1]
        ]
    )
    dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def save_grid(images: torch.Tensor, path: str, nrow: int = 10, title: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(nrow, nrow))
    if title:
        plt.title(title)
    plt.imshow(ndarr.squeeze(), cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_loss(losses: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 4))
    for name, vals in losses.items():
        plt.plot(vals, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    device: str = "cpu",
    # Diffusion
    T: int = 500,  # noqa: N803
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    # VAE architecture
    latent_channels: int = 4,
    latent_size: int = 4,  # spatial size of latent (4x4 for img_size=32)
    # NDM Transform
    ndm_hidden_dim: int = 256,
    ndm_num_layers: int = 3,
    ndm_time_dim: int = 128,
    # UNet architecture
    img_size: int = 32,  # pixel-space image size (for MNIST → 32)
    unet_channels: int = 64,  # UNet base channels
    time_dim: int = 256,  # UNet time embedding dim
    # Training
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs_vae: int = 5,  # VAE warm-up epochs
    num_epochs_ndm: int = 10,  # joint NDM training epochs
    beta_kl: float = 1e-3,  # KL weight in VAE loss
    lambda_vae: float = 0.1,  # weight of VAE loss during joint training
    # Paths
    experiment_name: str = "latent_ndm_mnist",
    weights_dir: str = "src/neural_latent_diffusion/weights",
    graphs_dir: str = "src/neural_latent_diffusion/graphs",
    results_dir: str = "src/neural_latent_diffusion/results",
    data_root: str = "./data",
):
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    dataloader = get_mnist_dataloader(batch_size, img_size=img_size, data_root=data_root)

    # -----------------------------------------------------------------------
    # Build models
    # -----------------------------------------------------------------------
    vae = VAE(img_channels=1, latent_channels=latent_channels, latent_size=latent_size).to(device)

    latent_dim = latent_channels * latent_size * latent_size
    transform = NDMTransform(
        latent_dim=latent_dim,
        time_dim=ndm_time_dim,
        hidden_dim=ndm_hidden_dim,
        num_layers=ndm_num_layers,
    ).to(device)

    unet = UNet(
        img_size=latent_size,
        c_in=latent_channels,
        c_out=latent_channels,
        time_dim=time_dim,
        device=device,
        channels=unet_channels,
    ).to(device)

    latent_shape = (latent_channels, latent_size, latent_size)
    diffusion = LatentNDMDiffusion(
        T=T,
        beta_start=beta_start,
        beta_end=beta_end,
        latent_shape=latent_shape,
        device=device,
    )

    mse = torch.nn.MSELoss()

    # -----------------------------------------------------------------------
    # Phase 1: VAE warm-up
    # -----------------------------------------------------------------------
    logging.info(f"=== Phase 1: VAE warm-up ({num_epochs_vae} epochs) ===")
    vae_optimizer = optim.AdamW(vae.parameters(), lr=lr)
    vae_losses = []

    for epoch in range(1, num_epochs_vae + 1):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch}/{num_epochs_vae}")
        for images, _ in pbar:
            images = images.to(device)
            recon, mu, logvar = vae(images)

            l_recon = mse(recon, images)
            l_kl = VAE.kl_loss(mu, logvar)
            loss = l_recon + beta_kl * l_kl

            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()

            epoch_losses.append(loss.item())
            vae_losses.append(loss.item())
            pbar.set_postfix(recon=f"{l_recon.item():.4f}", kl=f"{l_kl.item():.4f}")

        avg = sum(epoch_losses) / len(epoch_losses)
        logging.info(f"VAE Epoch {epoch} avg loss: {avg:.4f}")

    # Save VAE weights
    vae_path = os.path.join(weights_dir, f"{experiment_name}_vae.pt")
    torch.save(vae.state_dict(), vae_path)
    logging.info(f"VAE weights saved to {vae_path}")

    # -----------------------------------------------------------------------
    # Phase 2: Joint NDM training
    # -----------------------------------------------------------------------
    logging.info(f"=== Phase 2: Joint NDM training ({num_epochs_ndm} epochs) ===")
    ndm_optimizer = optim.AdamW(
        list(unet.parameters()) + list(transform.parameters()) + list(vae.parameters()),
        lr=lr,
    )

    ndm_losses, joint_vae_losses, total_losses = [], [], []

    for epoch in range(1, num_epochs_ndm + 1):
        logging.info(f"NDM Epoch {epoch}/{num_epochs_ndm}")
        ep_ndm, ep_vae, ep_total = [], [], []

        pbar = tqdm(dataloader, desc=f"NDM Epoch {epoch}")
        for images, _ in pbar:
            images = images.to(device)

            # Encode to latent (with gradient — joint training)
            mu, logvar = vae.encode(images)
            z = vae.reparameterize(mu, logvar)

            # Sample timestep and apply NDM forward process
            t = diffusion.sample_timesteps(images.shape[0])
            z_t, Fz_target, _noise = diffusion.q_sample(z, t, transform)  # noqa: N806

            # UNet predicts F_φ(ẑ, t)  — target is F_φ(z, t)
            pred_Fz = unet(z_t, t)  # noqa: N806
            l_ndm = mse(Fz_target.detach(), pred_Fz)

            # VAE regularisation loss (keeps the encoder/decoder well-behaved)
            recon = vae.decode(z)
            l_recon = mse(recon, images)
            l_kl = VAE.kl_loss(mu, logvar)
            l_vae = l_recon + beta_kl * l_kl

            loss = l_ndm + lambda_vae * l_vae

            ndm_optimizer.zero_grad()
            loss.backward()
            ndm_optimizer.step()

            ep_ndm.append(l_ndm.item())
            ep_vae.append(l_vae.item())
            ep_total.append(loss.item())
            ndm_losses.append(l_ndm.item())
            joint_vae_losses.append(l_vae.item())
            total_losses.append(loss.item())

            pbar.set_postfix(
                ndm=f"{l_ndm.item():.4f}",
                vae=f"{l_vae.item():.4f}",
                total=f"{loss.item():.4f}",
            )

        logging.info(
            f"Epoch {epoch} - NDM: {sum(ep_ndm)/len(ep_ndm):.4f} | "
            f"VAE: {sum(ep_vae)/len(ep_vae):.4f} | "
            f"Total: {sum(ep_total)/len(ep_total):.4f}"
        )

    # Save all weights
    unet_path = os.path.join(weights_dir, f"{experiment_name}_unet_epoch{epoch:03d}.pt")
    transform_path = os.path.join(weights_dir, f"{experiment_name}_transform_epoch{epoch:03d}.pt")
    vae_final_path = os.path.join(weights_dir, f"{experiment_name}_vae_final.pt")
    torch.save(unet.state_dict(), unet_path)
    torch.save(transform.state_dict(), transform_path)
    torch.save(vae.state_dict(), vae_final_path)
    logging.info(f"UNet → {unet_path}")
    logging.info(f"Transform → {transform_path}")
    logging.info(f"VAE final → {vae_final_path}")

    # Loss plots
    plot_loss(
        {
            "NDM Loss": ndm_losses,
            "VAE Loss (joint)": joint_vae_losses,
            "Total Loss": total_losses,
        },
        save_path=os.path.join(graphs_dir, f"{experiment_name}_loss.png"),
    )
    plot_loss(
        {"VAE Warm-up Loss": vae_losses},
        save_path=os.path.join(graphs_dir, f"{experiment_name}_vae_warmup_loss.png"),
    )

    logging.info("Training complete.")
    return unet, transform, vae, {"ndm": ndm_losses, "vae": joint_vae_losses, "total": total_losses}
