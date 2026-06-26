import os

import json
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T  # noqa: N812
from torch.utils.data import DataLoader

from src.models.vae.vae_wrapper import VAEWrapper
from src.models.two_stage_models.latent_two_stage import TwoStageLDM
from src.utility.metrics_util import _fid
from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
    _mnist_features,
)
from src.utility.voxel_metrics import compute_mmd_cov


@torch.no_grad()
def run_evaluation(model, config, output_path, device, data_root="./data"):
    """Performs reconstruction and sampling using a model instance."""
    model.eval()  # Ensure model is in eval mode

    # 1. Prepare Data for Original & Reconstructions
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    val_set = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    val_loader = DataLoader(val_set, batch_size=36, shuffle=True)

    # Get a batch of 36 images
    originals, _ = next(iter(val_loader))
    originals = originals.to(device)

    # 2. Process Reconstructions
    reconstructions, _, _ = model(originals)

    # 3. Process Samples
    z_shape = (36, config["latent_chan"], config["latent_res"], config["latent_res"])
    z_prior = torch.randn(z_shape).to(device)
    samples = model.trans_inr(z_prior)

    # 4. Plotting (Stitching images for a perfect 6x6 grid)
    _, axes = plt.subplots(1, 3, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.3)

    titles = ["Original MNIST", "Reconstructions", "Samples"]
    data_sources = [originals, reconstructions, samples]

    for ax, title, data in zip(axes, titles, data_sources, strict=False):
        grid_img = torchvision.utils.make_grid(data, nrow=6, padding=0)
        grid_img = (grid_img + 1.0) / 2.0
        grid_img = grid_img.clamp(0, 1)
        grid_np = grid_img.permute(1, 2, 0).cpu().numpy()

        ax.imshow(grid_np)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()  # Important to close plot to free up memory
    print(f"[done] Evaluation saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# FID EVALUATION
# ──────────────────────────────────────────────────────────────────────────────


def compute_fid(
    ldm: TwoStageLDM,
    data_config: dict,
    n_samples: int,
    fid_batch_size: int,
    device: torch.device,
) -> float:
    """
    Generate samples and compute Inception FID. Skipped for 3D voxel data.

    Args:
        ldm           (TwoStageLDM): model in eval mode
        data_config   (dict):        dataset config
        n_samples     (int):         number of samples to generate
        fid_batch_size(int):         generation batch size
        device        (torch.device):target device
    Returns:
        float: Inception FID score, or nan if 3D
    """
    if data_config.get("is_3d", False):
        print("  FID skipped — Inception FID is not defined for 3D voxel data.")
        return float("nan")

    dataset_name = data_config.get("dataset", "mnist").lower()
    is_mnist = dataset_name == "mnist"

    print(f"  Generating {n_samples} samples for FID …")
    all_samples = []
    remaining = n_samples
    ldm.eval()
    with torch.no_grad():
        while remaining > 0:
            n = min(fid_batch_size, remaining)
            imgs = (ldm.p_sample_loop(n) * 0.5 + 0.5).clamp(0, 1)
            all_samples.append(imgs.cpu())
            remaining -= n
    fid_tensor = torch.cat(all_samples, dim=0)

    inception = _get_inception(device)

    if is_mnist:
        classifier = _load_classifier(device)
        _, real_inception_feats, _ = _load_or_compute_real_features(
            classifier, inception, device
        )
    else:
        _, real_inception_feats, _ = _load_or_compute_real_features(
            None, inception, device
        )

    print("  Computing Inception FID …")
    gen_feats = _inception_features(fid_tensor, inception, device)
    return float(_fid(real_inception_feats, gen_feats))


# ──────────────────────────────────────────────────────────────────────────────
# FINAL EVAL
# ──────────────────────────────────────────────────────────────────────────────


def compute_final_eval(
    vae: VAEWrapper | None,
    ldm: TwoStageLDM,
    hparams: dict,
    val_loader: DataLoader,
    data_config: dict,
    args: argparse.Namespace,
    results_dir: str,
    device: torch.device,
    vae_epochs: int,
    ddpm_epochs: int,
    skip_vae_eval: bool = False,
) -> None:
    """
    Compute and save final eval metrics for VAE and/or LDM.
    2D: FID + sample grids. 3D: recon MSE + MMD/COV on generated vs val set.

    Args:
        vae           (VAEWrapper | None):  trained VAE, or None if skipped
        ldm           (TwoStageLDM):        trained LDM
        hparams       (dict):               LDM arch hparams
        val_loader    (DataLoader):         validation data loader
        data_config   (dict):               dataset config
        args          (argparse.Namespace): CLI args
        results_dir   (str):                output directory
        device        (torch.device):       target device
        vae_epochs    (int):                VAE epochs trained (0 if skipped)
        ddpm_epochs   (int):                DDPM epochs trained
        skip_vae_eval (bool):               if True, skip VAE eval entirely
    Returns:
        None
    """
    import torchvision.utils as vutils

    channels = data_config["channels"]  # noqa: F841
    img_size = data_config["img_size"]  # noqa: F841
    is_3d = data_config.get("is_3d", False)
    dataset_name = data_config.get("dataset", "mnist").lower()
    is_mnist = dataset_name == "mnist"

    # FID infrastructure only needed for 2D
    if not is_3d:
        inception = _get_inception(device)
        if is_mnist:
            classifier = _load_classifier(device)
            real_mnist_feats, real_inception_feats, _ = _load_or_compute_real_features(
                classifier, inception, device
            )
        else:
            _, real_inception_feats, _ = _load_or_compute_real_features(
                None, inception, device
            )
            classifier = None
    else:
        inception = classifier = real_inception_feats = real_mnist_feats = None

    # ── VAE eval ──────────────────────────────────────────────────────────────
    vae_recon_mse = None
    vae_mnist_fid = None
    vae_inception_fid = None

    if not skip_vae_eval:
        print("\n--- VAE Final Eval ---")
        vae.eval()
        latent_dim = hparams["latent_dim"]
        latent_size = hparams["latent_size"]

        # Reconstruction MSE on val set
        total_mse, n_seen = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_recon, _, _ = vae(x)
                x_flat = x.reshape(x.shape[0], -1)
                x_hat_flat = x_recon.reshape(x.shape[0], -1)
                if not is_3d:
                    x_flat = x_flat.clamp(-1, 1)
                total_mse += ((x_flat - x_hat_flat) ** 2).sum(dim=-1).sum().item()
                n_seen += x.shape[0]
        vae_recon_mse = total_mse / n_seen

        # FID + sample grid — 2D only
        if not is_3d:
            print(f"  Generating {args.n_fid_samples} VAE samples …")
            all_vae_samples = []
            remaining = args.n_fid_samples
            with torch.no_grad():
                while remaining > 0:
                    n = min(args.fid_batch_size, remaining)
                    z = torch.randn(
                        n, latent_dim, latent_size, latent_size, device=device
                    )
                    imgs = (vae._decode_latent(z) * 0.5 + 0.5).clamp(0, 1)
                    all_vae_samples.append(imgs.cpu())
                    remaining -= n
            vae_tensor = torch.cat(all_vae_samples, dim=0)

            if is_mnist:
                gen_mnist_feats, _ = _mnist_features(vae_tensor, classifier, device)
                vae_mnist_fid = float(_fid(real_mnist_feats, gen_mnist_feats))

            gen_vae_inception = _inception_features(vae_tensor, inception, device)
            vae_inception_fid = float(_fid(real_inception_feats, gen_vae_inception))

            vutils.save_image(
                vae_tensor[:64],
                os.path.join(results_dir, f"{args.run_name}_vae_samples_8x8.png"),
                nrow=8,
                padding=2,
            )
        else:
            print("  VAE sample grid + FID skipped for 3D data.")
    else:
        print("\n--- VAE Final Eval SKIPPED (pre-trained VAE reused) ---")

    # ── LDM eval ──────────────────────────────────────────────────────────────
    print("\n--- LDM Final Eval ---")

    ldm_fid = None
    ldm_mmd = None
    ldm_cov = None

    if is_3d:
        # Generate samples
        print(f"  Generating {args.n_fid_samples} LDM samples for MMD/COV …")
        all_samples = []
        remaining = args.n_fid_samples
        ldm.eval()
        with torch.no_grad():
            while remaining > 0:
                n = min(args.fid_batch_size, remaining)
                all_samples.append(ldm.p_sample_loop(n).cpu())
                remaining -= n
        generated = torch.cat(all_samples, dim=0)

        # Collect full reference val set
        ref_batches = [batch[0] for batch in val_loader]
        reference = torch.cat(ref_batches, dim=0)

        print(
            f"  Computing MMD/COV ({generated.shape[0]} generated vs {reference.shape[0]} reference) …"
        )
        ldm_mmd, ldm_cov = compute_mmd_cov(generated, reference)
        print(f"  MMD: {ldm_mmd:.4f} | COV: {ldm_cov:.4f}")
    else:
        ldm_fid = compute_fid(
            ldm, data_config, args.n_fid_samples, args.fid_batch_size, device
        )

        ldm.eval()
        with torch.no_grad():
            ldm_samples = (ldm.p_sample_loop(64) * 0.5 + 0.5).clamp(0, 1)
        vutils.save_image(
            ldm_samples,
            os.path.join(results_dir, f"{args.run_name}_ldm_samples_8x8.png"),
            nrow=8,
            padding=2,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Final Eval Summary — {args.run_name}")
    print(f"{'=' * 50}")
    if skip_vae_eval:
        print("  VAE eval             : skipped (pre-trained)")
    else:
        print(f"  VAE epochs trained   : {vae_epochs}")
        print(f"  VAE recon MSE        : {vae_recon_mse:.6f}")
        if vae_mnist_fid is not None:
            print(f"  VAE MNIST FID        : {vae_mnist_fid:.2f}")
        if vae_inception_fid is not None:
            print(f"  VAE Inception FID    : {vae_inception_fid:.2f}")
    print(f"  DDPM epochs trained  : {ddpm_epochs}")
    if is_3d:
        print(f"  LDM MMD              : {ldm_mmd:.4f}")
        print(f"  LDM COV              : {ldm_cov:.4f}")
    else:
        print(f"  LDM Inception FID    : {ldm_fid:.2f}")
    print(f"{'=' * 50}\n")

    metrics = {
        "run_name": args.run_name,
        "mode": args.mode,
        "vae_epochs": vae_epochs,
        "ddpm_epochs": ddpm_epochs,
        "vae_recon_mse": vae_recon_mse,
        "vae_mnist_fid": vae_mnist_fid,
        "vae_inception_fid": vae_inception_fid,
        "ldm_inception_fid": ldm_fid,
        "ldm_mmd": ldm_mmd,
        "ldm_cov": ldm_cov,
    }
    metrics_path = os.path.join(results_dir, f"{args.run_name}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Eval metrics saved → {metrics_path}")


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _build_val_noise_cache(
    val_loader: DataLoader,
    n_timesteps: int,
    device: torch.device,  # noqa: ARG001
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-sample fixed (x, t) tuples for consistent DDPM validation.

    Args:
        val_loader  (DataLoader):   validation data loader
        n_timesteps (int):          diffusion timestep count T
        device      (torch.device): target device
        seed        (int):          RNG seed for reproducibility
    Returns:
        list of (x_batch, t_batch) tuples on CPU;
              x_batch : (B, C, ...) — 2D or 3D
              t_batch : (B,) int64 in [0, T-1]
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    cache = []
    for batch in val_loader:
        x = batch[0]
        B = x.shape[0]  # noqa: N806
        t = torch.randint(0, n_timesteps, (B,), generator=rng)
        cache.append((x.cpu(), t.cpu()))
    return cache


@torch.no_grad()
def compute_ddpm_val_loss(
    ldm: TwoStageLDM,
    val_cache: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    noise_seed: int = 42,
) -> float:
    """
    Compute MSE between predicted and actual noise on the validation set.

    Args:
        ldm        (TwoStageLDM):                  model in eval mode
        val_cache  (list of (x_cpu, t_cpu)):        pre-built fixed val pairs
        device     (torch.device):                  target device
        noise_seed (int):                           seed for noise generation
    Returns:
        float: mean MSE over validation set
    """
    ldm.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(noise_seed)

    total_loss = 0.0
    n_seen = 0

    for x_cpu, t_cpu in val_cache:
        x = x_cpu.to(device)
        t = t_cpu.to(device)

        mu, logvar = ldm.latent_encoder(x)
        z0 = ldm.latent_encoder.reparameterize(mu, logvar)

        noise = torch.randn(z0.shape, device=device, generator=rng)
        z_t = ldm.q_sample(z0, t, noise)

        t_norm = (t.float() / (ldm.T - 1)).unsqueeze(1)
        eps_pred = ldm.noise_predictor(z_t, t_norm)

        loss = ((eps_pred - noise) ** 2).mean()
        total_loss += loss.item() * x.shape[0]
        n_seen += x.shape[0]

    return total_loss / n_seen
