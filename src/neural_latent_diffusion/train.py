import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, Subset
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


def get_mnist_dataloader(
    batch_size: int, img_size: int = 32, data_root: str = "./data", subset_pct: float = 1.0, seed: int = 42
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # → [-1, 1]
        ]
    )
    dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    if 0 < subset_pct < 1.0:
        rng = random.Random(seed)
        n = int(len(dataset) * subset_pct)
        indices = rng.sample(range(len(dataset)), n)
        dataset = Subset(dataset, indices)
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
    T: int = 1000,  # noqa: N803
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    # VAE architecture
    latent_channels: int = 8,
    latent_size: int = 8,
    # NDM Transform
    ndm_hidden_dim: int = 256,
    ndm_num_layers: int = 3,
    ndm_time_dim: int = 128,
    # UNet architecture
    img_size: int = 32,
    unet_channels: int = 64,
    time_dim: int = 256,
    # Training
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs_vae: int = 5,
    num_epochs_unet_warmup: int = 1,  # FIX: warm up UNet before joint training
    num_epochs_ndm: int = 10,
    beta_kl: float = 1e-3,
    lambda_vae: float = 0.1,
    grad_clip_norm: float = 1.0,  # FIX: gradient clipping max norm
    # Paths
    experiment_name: str = "latent_ndm_mnist",
    weights_dir: str = "src/neural_latent_diffusion/weights",
    graphs_dir: str = "src/neural_latent_diffusion/graphs",
    results_dir: str = "src/neural_latent_diffusion/results",
    data_root: str = "./data",
    subset_pct: float = 1.0,  # Use a fraction of the dataset for faster testing
):
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    dataloader = get_mnist_dataloader(batch_size, img_size=img_size, data_root=data_root, subset_pct=subset_pct)

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
    vae_path = os.path.join(weights_dir, f"{experiment_name}_vae.pt")
    vae_losses = []

    if os.path.exists(vae_path):
        logging.info(f"=== Phase 1: found existing VAE weights at {vae_path} - skipping warm-up ===")
        vae.load_state_dict(torch.load(vae_path, map_location=device))
    else:
        logging.info(f"=== Phase 1: VAE warm-up ({num_epochs_vae} epochs) ===")
        vae_optimizer = optim.AdamW(vae.parameters(), lr=lr)

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

        torch.save(vae.state_dict(), vae_path)
        logging.info(f"VAE weights saved to {vae_path}")

    # -----------------------------------------------------------------------
    # Phase 2: UNet warm-up  (FIX: train UNet with frozen transform before joint)
    #
    # With the transform initialised near-identity (zero output_proj), F_φ(z,t) ≈ z.
    # This phase lets the UNet learn a sensible baseline before the transform
    # starts moving the targets.  The VAE is also frozen here.
    # -----------------------------------------------------------------------
    unet_warmup_path = os.path.join(weights_dir, f"{experiment_name}_unet_warmup.pt")
    unet_warmup_losses = []

    if os.path.exists(unet_warmup_path):
        logging.info(f"=== Phase 2: found existing UNet warm-up weights at {unet_warmup_path} - skipping ===")
        unet.load_state_dict(torch.load(unet_warmup_path, map_location=device))
    else:
        logging.info(f"=== Phase 2: UNet warm-up with frozen VAE + frozen transform ({num_epochs_unet_warmup} epochs) ===")
        unet_warmup_optimizer = optim.AdamW(unet.parameters(), lr=lr)

        # Freeze VAE and transform during UNet warm-up
        for p in vae.parameters():
            p.requires_grad_(False)
        for p in transform.parameters():
            p.requires_grad_(False)

        for epoch in range(1, num_epochs_unet_warmup + 1):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"UNet Warmup Epoch {epoch}/{num_epochs_unet_warmup}")
            for images, _ in pbar:
                images = images.to(device)

                with torch.no_grad():
                    mu, logvar = vae.encode(images)
                    z = vae.reparameterize(mu, logvar)
                    # FIX: use fixed per-dataset latent stats instead of per-batch norm.
                    # During warmup we just use z as-is (transform ≈ identity anyway).

                t = diffusion.sample_timesteps(images.shape[0])

                # q_sample through frozen transform (≈ identity at init)
                z_t, Fz_target, _ = diffusion.q_sample(z, t, transform)  # noqa: N806

                pred_Fz = unet(z_t, t)  # noqa: N806
                loss = mse(Fz_target, pred_Fz)

                unet_warmup_optimizer.zero_grad()
                loss.backward()
                # FIX: gradient clipping
                torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip_norm)
                unet_warmup_optimizer.step()

                epoch_losses.append(loss.item())
                unet_warmup_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg = sum(epoch_losses) / len(epoch_losses)
            logging.info(f"UNet Warmup Epoch {epoch} avg loss: {avg:.4f}")

        # Unfreeze for joint training
        for p in vae.parameters():
            p.requires_grad_(True)
        for p in transform.parameters():
            p.requires_grad_(True)

        torch.save(unet.state_dict(), unet_warmup_path)
        logging.info(f"UNet warm-up weights saved to {unet_warmup_path}")

    # -----------------------------------------------------------------------
    # Phase 3: Joint NDM training
    #
    # FIX summary vs original:
    #   1. Removed per-batch z normalization (z /= z.std()) — it destabilises
    #      the transform by shifting target scales every batch.
    #   2. Fz_target is NOT detached — gradients flow through the transform
    #      so it can learn jointly with the UNet.  (Detaching Fz_target would
    #      stop the transform from ever improving.)
    #   3. Gradient clipping on all parameters prevents runaway updates.
    # -----------------------------------------------------------------------
    logging.info(f"=== Phase 3: Joint NDM training ({num_epochs_ndm} epochs) ===")
    all_params = list(unet.parameters()) + list(transform.parameters()) + list(vae.parameters())
    ndm_optimizer = optim.AdamW(all_params, lr=lr)

    ndm_losses, joint_vae_losses, total_losses = [], [], []

    for epoch in range(1, num_epochs_ndm + 1):
        logging.info(f"NDM Epoch {epoch}/{num_epochs_ndm}")
        ep_ndm, ep_vae, ep_total = [], [], []

        pbar = tqdm(dataloader, desc=f"NDM Epoch {epoch}")
        for images, _ in pbar:
            images = images.to(device)

            # Encode to latent (with gradient for joint training)
            mu, logvar = vae.encode(images)
            z = vae.reparameterize(mu, logvar)

            # FIX: removed per-batch z /= z.std() normalization.
            # Per-batch rescaling shifts the scale of Fz_target every step,
            # making the UNet chase a moving target and causing loss explosion.

            # Sample timestep and apply NDM forward process
            t = diffusion.sample_timesteps(images.shape[0])
            z_t, Fz_target, _ = diffusion.q_sample(z, t, transform)  # noqa: N806

            # UNet predicts F_φ(z, t) — target is F_φ(z, t)
            pred_Fz = unet(z_t, t)  # noqa: N806

            # FIX: do NOT detach Fz_target.
            # Detaching would block gradients to the transform, preventing it
            # from learning. The transform and UNet must co-train.
            l_ndm = mse(Fz_target, pred_Fz)

            # VAE regularisation loss
            recon = vae.decode(z)
            l_recon = mse(recon, images)
            l_kl = VAE.kl_loss(mu, logvar)
            l_vae = l_recon + beta_kl * l_kl

            loss = l_ndm + lambda_vae * l_vae

            ndm_optimizer.zero_grad()
            loss.backward()
            # FIX: gradient clipping prevents runaway transform/UNet updates
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)
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
            f"Epoch {epoch} - NDM: {sum(ep_ndm) / len(ep_ndm):.4f} | "
            f"VAE: {sum(ep_vae) / len(ep_vae):.4f} | "
            f"Total: {sum(ep_total) / len(ep_total):.4f}"
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
        {
            "VAE Warm-up Loss": vae_losses,
            "UNet Warm-up Loss": unet_warmup_losses,
        },
        save_path=os.path.join(graphs_dir, f"{experiment_name}_warmup_loss.png"),
    )

    logging.info("Training complete.")
    return {"ndm": ndm_losses, "vae": joint_vae_losses, "total": total_losses}
