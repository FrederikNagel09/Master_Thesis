import argparse
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")

from src.models.LatentEncoder import ResNetLatentEncoder
from src.models.trans_inr import TransInr, make_coord_grid
from src.models.LatentNoisePredictor import LatentTransformerNoisePredictor
from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
    _mnist_features,
)
from src.utility.dataset_builders import build_dataset
from src.utility.metrics_util import _fid

warnings.filterwarnings("ignore", message="The operator 'aten::im2col'")

"""
Fixed-budget mode (version 1):
python src/scripts/two-stage-training.py \
    --run_name two_stage_fixed \
    --mode fixed \
    --ldm_config src/train_results/latent-diffusion-1/metadata/config.json \
    --total_epochs 15 \
    --vae_epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 32 \
    --fid_batch_size 32

Convergence mode (version 2):
python src/scripts/two-stage-latent-training.py \
    --run_name two_stage_convergence \
    --mode convergence \
    --ldm_config src/train_results/latent-diffusion-4/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 5 \
    --vae_patience 10 \
    --vae_delta 1e-4 \
    --ddpm_check_every 5 \
    --ddpm_patience 30 \
    --ddpm_delta 7e-5 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 1024 \
    --fid_batch_size 64

Skip-VAE mode (re-run DDPM only, VAE files left untouched):
python src/scripts/two-stage-latent-training.py \
    --run_name two_stage_convergence \
    --mode convergence \
    --skip_vae \
    --vae_weights src/results/two_stage_convergence/two_stage_convergence_vae_weights.pt \
    --ldm_config src/train_results/latent-diffusion-4/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --ddpm_check_every 5 \
    --ddpm_patience 30 \
    --ddpm_delta 5e-5 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 128 \
    --fid_batch_size 64
"""

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for two-stage LDM training.

    Returns:
        argparse.Namespace: parsed arguments
    """
    p = argparse.ArgumentParser(description="Train a two-stage VAE + LDM model")

    # Run
    p.add_argument("--run_name",    type=str,   required=True)
    p.add_argument("--ldm_config",  type=str,   required=True, help="Path to LDM config .json")
    p.add_argument("--results_dir", type=str,   default="src/train_results")
    p.add_argument(
        "--mode", type=str, required=True, choices=["fixed", "convergence"],
        help="'fixed': explicit epoch counts; 'convergence': early-stop both stages",
    )

    # Skip-VAE: load pre-trained VAE weights and go straight to DDPM
    p.add_argument("--skip_vae",    action="store_true", default=False,
                   help="Skip VAE training and load pre-trained weights instead")
    p.add_argument("--vae_weights", type=str, default=None,
                   help="Path to _vae_weights.pt (required when --skip_vae is set)")

    # Shared training
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--subset_frac",   type=float, default=1.0)

    # KL annealing
    p.add_argument("--lambda_kl_max",   type=float, default=1.0)
    p.add_argument("--kl_warmup_frac",  type=float, default=0.4)

    # Fixed-mode epochs
    p.add_argument("--total_epochs", type=int, default=400,
                   help="[fixed mode] Total epochs across both stages")
    p.add_argument("--vae_epochs",   type=int, default=100,
                   help="[fixed mode] How many of total_epochs go to VAE training")

    # Convergence-mode VAE stopping
    p.add_argument("--vae_check_every", type=int,   default=5,
                   help="[convergence mode] Validate VAE every N epochs")
    p.add_argument("--vae_patience",    type=int,   default=10,
                   help="[convergence mode] Stop VAE after this many checks without improvement")
    p.add_argument("--vae_delta",       type=float, default=1e-4,
                   help="[convergence mode] Minimum ELBO improvement to count as progress")
    p.add_argument("--vae_max_epochs",  type=int,   default=1000,
                   help="[convergence mode] Hard cap on VAE epochs")

    # Convergence-mode DDPM stopping
    p.add_argument("--ddpm_check_every", type=int,   default=5,
                   help="[convergence mode] Validate DDPM every N epochs")
    p.add_argument("--ddpm_patience",    type=int,   default=20,
                   help="[convergence mode] Window size (in checks) for smoothed plateau detection")
    p.add_argument("--ddpm_delta",       type=float, default=1e-4,
                   help="[convergence mode] Minimum avg-loss improvement (first vs second half of window)")
    p.add_argument("--ddpm_max_epochs",  type=int,   default=5000,
                   help="[convergence mode] Hard cap on DDPM epochs")

    # Noise schedule
    p.add_argument("--T",      type=int,   default=1000)
    p.add_argument("--beta_1", type=float, default=1e-4)
    p.add_argument("--beta_T", type=float, default=0.02)

    # FID evaluation
    p.add_argument("--n_fid_samples",  type=int, default=1024)
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument(
        "--fid_fractions", type=float, nargs="+",
        default=[0.4, 0.55, 0.7, 0.8, 0.9, 1.0],
        help="Fractional checkpoints (of ddpm epochs) at which to log FID during DDPM training",
    )

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_ldm_config(path: str) -> dict:
    """
    Load hparams from a trained LDM config JSON.

    Args:
        path (str): path to config .json
    Returns:
        dict: hparams block from config
    """
    with open(path, "r") as f:
        config = json.load(f)
    required_keys = [
        "latent_dim", "latent_size", "latent_patch_size",
        "latent_enc_hidden_dim", "dec_trans_dim", "dec_trans_n_head",
        "dec_trans_head_dim", "dec_trans_ff_dim", "dec_trans_enc_depth",
        "dec_trans_dec_depth", "dec_trans_n_groups", "dec_trans_update_strategy",
        "inr_hidden_dim", "inr_layers", "dataset",
        "pred_d_model", "pred_n_heads", "pred_n_layers", "pred_d_ff",
        "pred_t_embed_dim", "noise_predictor_dropout",
    ]
    hparams = config["hparams"]
    missing = [k for k in required_keys if k not in hparams]
    if missing:
        raise ValueError(f"LDM config missing required keys: {missing}")
    return hparams


# ──────────────────────────────────────────────────────────────────────────────
# MODEL CLASSES
# ──────────────────────────────────────────────────────────────────────────────

class VAEWrapper(nn.Module):
    """
    Wraps encoder + TransInr decoder into a VAE.
    Designed so encoder and decoder can be extracted individually after training.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        img_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.latent_encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.device = device
        coord_grid = make_coord_grid((img_size, img_size), (-1, 1))
        self.register_buffer("coord_grid", coord_grid)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent tensor through TransInr decoder.

        Args:
            z (torch.Tensor): latent sample (B, latent_dim, H, W)
        Returns:
            torch.Tensor: reconstructed image (B, C, img_size, img_size)
        """
        B = z.shape[0]
        coords = self.coord_grid.unsqueeze(0).repeat(B, 1, 1, 1).to(self.device)
        return self.decoder(z, coords)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass: encode → reparameterise → decode.

        Args:
            x (torch.Tensor): input image (B, C, H, W)
        Returns:
            tuple: (x_recon, mu, logvar) each (B, ...)
        """
        mu, logvar = self.latent_encoder(x)
        z = self.latent_encoder.reparameterize(mu, logvar)
        x_recon = self._decode_latent(z)
        return x_recon, mu, logvar


class TwoStageLDM(nn.Module):
    """
    Two-stage LDM: frozen VAE encoder/decoder + trainable noise predictor.
    The VAE weights are loaded from a pre-trained VAEWrapper and frozen.
    The noise schedule buffers live here so the model is self-contained on load.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        noise_predictor: nn.Module,
        img_size: int,
        latent_dim: int,
        latent_size: tuple[int, int],
        T: int,
        beta_1: float,
        beta_T: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.latent_encoder = encoder
        self.decoder = decoder
        self.noise_predictor = noise_predictor
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.T = T
        self.device = device

        # Freeze encoder and decoder — they must not be updated
        for p in self.latent_encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        # Noise schedule buffers (same as one-stage model)
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)
        self.register_buffer("beta",              beta)
        self.register_buffer("alpha",             alpha)
        self.register_buffer("alpha_cumprod",     alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod",alpha_cumprod.sqrt())
        self.register_buffer("sigma_sq",          1.0 - alpha_cumprod)
        self.register_buffer("sigma",             (1.0 - alpha_cumprod).sqrt())

        coord_grid = make_coord_grid((img_size, img_size), (-1, 1))
        self.register_buffer("coord_grid", coord_grid, persistent=False)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent through the frozen decoder.

        Args:
            z (torch.Tensor): latent (B, latent_dim, H, W)
        Returns:
            torch.Tensor: image (B, C, img_size, img_size)
        """
        B = z.shape[0]
        coords = self.coord_grid.unsqueeze(0).repeat(B, 1, 1, 1).to(self.device)
        return self.decoder(z, coords)

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean latent at timestep t.

        Args:
            z0    (torch.Tensor): clean latent (B, latent_dim, H, W)
            t     (torch.Tensor): timestep indices (B,) in [0, T-1]
            noise (torch.Tensor): noise tensor same shape as z0
        Returns:
            torch.Tensor: noisy latent z_t, same shape as z0
        """
        sqrt_ac = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sigma   = self.sigma[t].view(-1, 1, 1, 1)
        return sqrt_ac * z0 + sigma * noise

    @torch.no_grad()
    def p_sample_loop(self, n_samples: int) -> torch.Tensor:
        """
        Full reverse diffusion chain to generate images.

        Args:
            n_samples (int): number of images to generate
        Returns:
            torch.Tensor: generated images in [-1, 1], (n_samples, C, img_size, img_size)
        """
        H, W = self.latent_size
        z = torch.randn(n_samples, self.latent_dim, H, W, device=self.device)
        for t_idx in reversed(range(self.T)):
            t_tensor = torch.full((n_samples, 1), t_idx / (self.T - 1),
                                  device=self.device, dtype=torch.float32)
            t_int = torch.full((n_samples,), t_idx, device=self.device, dtype=torch.long)
            eps_pred = self.noise_predictor(z, t_tensor)
            alpha_t  = self.alpha[t_int].view(-1, 1, 1, 1)
            beta_t   = self.beta[t_int].view(-1, 1, 1, 1)
            sigma_t  = self.sigma[t_int].view(-1, 1, 1, 1)
            # DDPM reverse step
            z = (1.0 / alpha_t.sqrt()) * (z - (beta_t / sigma_t) * eps_pred)
            if t_idx > 0:
                z = z + beta_t.sqrt() * torch.randn_like(z)
        return self._decode_latent(z)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def build_encoder_decoder(
    hparams: dict,
    channels: int,
    img_size: int,
) -> tuple[nn.Module, nn.Module]:
    """
    Instantiate encoder and decoder from hparams (no device placement yet).

    Args:
        hparams  (dict): LDM hparams
        channels (int):  image channels
        img_size (int):  spatial image size
    Returns:
        tuple: (ResNetLatentEncoder, TransInr)
    """
    latent_dim  = hparams["latent_dim"]
    latent_size = hparams["latent_size"]

    encoder = ResNetLatentEncoder(
        in_channels=channels,
        latent_dim=latent_dim,
        latent_size=(latent_size, latent_size),
        hidden_dim=hparams["latent_enc_hidden_dim"],
    )
    decoder = TransInr(
        tokenizer={
            "target": "src.models.trans_inr_helpers.LatentTokenizer",
            "params": {
                "latent_dim":  latent_dim,
                "latent_size": latent_size,
                "patch_size":  hparams["latent_patch_size"],
                "dim":         hparams["dec_trans_dim"],
                "n_head":      hparams["dec_trans_n_head"],
                "head_dim":    hparams["dec_trans_head_dim"],
            },
        },
        inr={
            "target": "src.models.trans_inr_helpers.SIREN",
            "params": {
                "depth":      hparams["inr_layers"],
                "in_dim":     2,
                "out_dim":    channels,
                "hidden_dim": hparams["inr_hidden_dim"],
                "out_bias":   0.5,
            },
        },
        data_shape=(img_size, img_size),
        n_groups=hparams["dec_trans_n_groups"],
        transformer={
            "target": "src.models.trans_inr_helpers.Transformer",
            "params": {
                "dim":           hparams["dec_trans_dim"],
                "encoder_depth": hparams["dec_trans_enc_depth"],
                "decoder_depth": hparams["dec_trans_dec_depth"],
                "n_head":        hparams["dec_trans_n_head"],
                "head_dim":      hparams["dec_trans_head_dim"],
                "ff_dim":        hparams["dec_trans_ff_dim"],
            },
        },
        update_strategy=hparams["dec_trans_update_strategy"],
    )
    return encoder, decoder


def build_vae(
    hparams: dict,
    channels: int,
    img_size: int,
    device: torch.device,
) -> VAEWrapper:
    """
    Build a VAEWrapper from hparams and place on device.

    Args:
        hparams  (dict):          LDM hparams
        channels (int):           image channels
        img_size (int):           spatial image size
        device   (torch.device):  target device
    Returns:
        VAEWrapper: assembled model on device
    """
    encoder, decoder = build_encoder_decoder(hparams, channels, img_size)
    return VAEWrapper(encoder, decoder, img_size, device).to(device)


def load_pretrained_vae(
    weights_path: str,
    hparams: dict,
    channels: int,
    img_size: int,
    device: torch.device,
) -> VAEWrapper:
    """
    Build a VAEWrapper and load pre-trained encoder/decoder weights from disk.

    Args:
        weights_path (str):          path to _vae_weights.pt
        hparams      (dict):         LDM arch hparams (for model construction)
        channels     (int):          image channels
        img_size     (int):          spatial image size
        device       (torch.device): target device
    Returns:
        VAEWrapper: model with loaded weights on device, in eval mode
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"VAE weights not found at: {weights_path}")

    vae = build_vae(hparams, channels, img_size, device)
    ckpt = torch.load(weights_path, map_location=device)

    # Prefer individual encoder/decoder state dicts; fall back to full VAE state dict
    if "encoder_state_dict" in ckpt and "decoder_state_dict" in ckpt:
        vae.latent_encoder.load_state_dict(ckpt["encoder_state_dict"])
        vae.decoder.load_state_dict(ckpt["decoder_state_dict"])
    else:
        vae.load_state_dict(ckpt["vae_state_dict"])

    vae.eval()
    print(f"  Loaded pre-trained VAE weights from: {weights_path}")
    return vae


def build_ldm(
    hparams: dict,
    args: argparse.Namespace,
    channels: int,
    img_size: int,
    device: torch.device,
) -> TwoStageLDM:
    """
    Build TwoStageLDM with a fresh noise predictor (VAE weights loaded separately).

    Args:
        hparams  (dict):              LDM hparams
        args     (argparse.Namespace): CLI args (noise predictor dims, schedule)
        channels (int):               image channels
        img_size (int):               spatial image size
        device   (torch.device):      target device
    Returns:
        TwoStageLDM: assembled model on device (VAE weights NOT yet loaded)
    """
    latent_dim  = hparams["latent_dim"]
    latent_size = hparams["latent_size"]

    encoder, decoder = build_encoder_decoder(hparams, channels, img_size)
    noise_predictor = LatentTransformerNoisePredictor(
        latent_dim=latent_dim,
        latent_size=(latent_size, latent_size),
        d_model=hparams["pred_d_model"],
        n_heads=hparams["pred_n_heads"],
        n_layers=hparams["pred_n_layers"],
        d_ff=hparams["pred_d_ff"],
        dropout=hparams["noise_predictor_dropout"],
        t_embed_dim=hparams["pred_t_embed_dim"],
    )
    return TwoStageLDM(
        encoder=encoder,
        decoder=decoder,
        noise_predictor=noise_predictor,
        img_size=img_size,
        latent_dim=latent_dim,
        latent_size=(latent_size, latent_size),
        T=args.T,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        device=device,
    ).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# KL ANNEALING
# ──────────────────────────────────────────────────────────────────────────────

def _get_beta(
    global_step: int,
    beta_final: float,
    warmup_steps: int,
    burnin_steps: int = 0,
) -> float:
    """
    Linear KL warmup with optional burn-in period.

    Args:
        global_step  (int):   current training step
        beta_final   (float): target beta
        warmup_steps (int):   steps to ramp from 0 → beta_final after burnin
        burnin_steps (int):   steps to hold at 0 before ramping
    Returns:
        float: current beta
    """
    if global_step < burnin_steps:
        return 0.0
    return beta_final * min(1.0, (global_step - burnin_steps) / warmup_steps)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT & MODEL SAVING
# ──────────────────────────────────────────────────────────────────────────────

def save_vae_checkpoint(
    model: VAEWrapper,
    optimizer: optim.Optimizer,
    epoch: int,
    history: dict,
    results_dir: str,
    run_name: str,
) -> None:
    """
    Save full VAE training checkpoint (resumable).

    Args:
        model       (VAEWrapper):       model to save
        optimizer   (optim.Optimizer):  optimizer state
        epoch       (int):              last completed epoch
        history     (dict):             loss history
        results_dir (str):              output directory
        run_name    (str):              run identifier
    Returns:
        None
    """
    path = os.path.join(results_dir, f"{run_name}_vae_checkpoint.pt")
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        "history":              history,
    }, path)


def save_vae_weights(
    model: VAEWrapper,
    hparams: dict,
    results_dir: str,
    run_name: str,
) -> None:
    """
    Save standalone VAE weights + config for independent later use.

    Args:
        model       (VAEWrapper): trained model
        hparams     (dict):       arch hparams
        results_dir (str):        output directory
        run_name    (str):        run identifier
    Returns:
        None
    """
    weights_path = os.path.join(results_dir, f"{run_name}_vae_weights.pt")
    config_path  = os.path.join(results_dir, f"{run_name}_vae_config.json")

    # Save only encoder + decoder state dicts so this file is VAE-only
    torch.save({
        "encoder_state_dict": model.latent_encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        # Full VAEWrapper state dict also included for convenience
        "vae_state_dict":     model.state_dict(),
    }, weights_path)

    arch_keys = [
        "dataset", "latent_dim", "latent_size", "latent_patch_size",
        "latent_enc_hidden_dim", "dec_trans_dim", "dec_trans_n_head",
        "dec_trans_head_dim", "dec_trans_ff_dim", "dec_trans_enc_depth",
        "dec_trans_dec_depth", "dec_trans_n_groups", "dec_trans_update_strategy",
        "inr_hidden_dim", "inr_layers",
    ]
    config = {k: hparams[k] for k in arch_keys}
    config["run_name"] = run_name
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"VAE weights → {weights_path}")
    print(f"VAE config  → {config_path}")


def save_ldm_checkpoint(
    model: TwoStageLDM,
    optimizer: optim.Optimizer,
    epoch: int,
    history: dict,
    results_dir: str,
    run_name: str,
) -> None:
    """
    Save full LDM training checkpoint (resumable).

    Args:
        model       (TwoStageLDM):      model to save
        optimizer   (optim.Optimizer):  optimizer state (noise predictor only)
        epoch       (int):              last completed epoch
        history     (dict):             loss history
        results_dir (str):              output directory
        run_name    (str):              run identifier
    Returns:
        None
    """
    path = os.path.join(results_dir, f"{run_name}_ldm_checkpoint.pt")
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        "history":              history,
    }, path)


def save_ldm_weights(
    model: TwoStageLDM,
    hparams: dict,
    args: argparse.Namespace,
    results_dir: str,
    run_name: str,
) -> None:
    """
    Save standalone LDM weights + config for independent later use.
    Includes noise predictor only, VAE, and the full model state dict.

    Args:
        model       (TwoStageLDM):       trained model
        hparams     (dict):              arch hparams
        args        (argparse.Namespace): CLI args (noise predictor dims)
        results_dir (str):               output directory
        run_name    (str):               run identifier
    Returns:
        None
    """
    weights_path = os.path.join(results_dir, f"{run_name}_ldm_weights.pt")
    config_path  = os.path.join(results_dir, f"{run_name}_ldm_config.json")

    torch.save({
        # Full model for loading TwoStageLDM directly
        "ldm_state_dict":             model.state_dict(),
        # Noise predictor alone for independent use
        "noise_predictor_state_dict": model.noise_predictor.state_dict(),
        # VAE components for independent use
        "encoder_state_dict":         model.latent_encoder.state_dict(),
        "decoder_state_dict":         model.decoder.state_dict(),
    }, weights_path)

    arch_keys = [
        "dataset", "latent_dim", "latent_size", "latent_patch_size",
        "latent_enc_hidden_dim", "dec_trans_dim", "dec_trans_n_head",
        "dec_trans_head_dim", "dec_trans_ff_dim", "dec_trans_enc_depth",
        "dec_trans_dec_depth", "dec_trans_n_groups", "dec_trans_update_strategy",
        "inr_hidden_dim", "inr_layers", "pred_d_model", "pred_n_heads", "pred_n_layers", "pred_d_ff",
        "pred_t_embed_dim", "noise_predictor_dropout",
    ]
    config = {k: hparams[k] for k in arch_keys}
    config["run_name"] = run_name
    config["T"]        = args.T
    config["beta_1"]   = args.beta_1
    config["beta_T"]   = args.beta_T

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"LDM weights → {weights_path}")
    print(f"LDM config  → {config_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def save_vae_training_graph(
    history: dict,
    steps_per_epoch: int,
    total_epochs: int,
    save_path: str,
    plot_every_n: int = 100,
) -> None:
    """
    Save 3-panel VAE training curve (ELBO, recon, KL).

    Args:
        history        (dict): keys "elbo", "recon", "kl" — per-step values
        steps_per_epoch(int):  optimizer steps per epoch
        total_epochs   (int):  total epochs completed
        save_path      (str):  output .png path
        plot_every_n   (int):  downsample factor for plotting
    Returns:
        None
    """
    max_ticks = 10
    step = max(1, total_epochs // max_ticks)
    tick_pos    = [i * steps_per_epoch // plot_every_n for i in range(0, total_epochs + 1, step)]
    tick_labels = [str(i) for i in range(0, total_epochs + 1, step)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("elbo",  "Total ELBO",           "tab:blue"),
        ("recon", "Reconstruction Loss",  "tab:orange"),
        ("kl",    "KL Loss",              "tab:green"),
    ]
    for ax, (key, title, color) in zip(axes, panels):
        downsampled = history[key][::plot_every_n]
        ax.plot(range(len(downsampled)), downsampled, color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("VAE Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_ddpm_training_graph(
    history: dict,
    steps_per_epoch: int,
    total_epochs: int,
    save_path: str,
    plot_every_n: int = 100,
) -> None:
    """
    Save DDPM training curve (train loss, val loss, periodic FID).

    Args:
        history        (dict): keys "train_loss", "val_loss" (per-step and per-check),
                               "fid_epochs", "fid_scores"
        steps_per_epoch(int):  optimizer steps per epoch
        total_epochs   (int):  total DDPM epochs completed
        save_path      (str):  output .png path
        plot_every_n   (int):  downsample factor for train loss
    Returns:
        None
    """
    max_ticks = 10
    step = max(1, total_epochs // max_ticks)

    has_fid = len(history.get("fid_epochs", [])) > 0

    n_panels = 3 if has_fid else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))

    # Train loss (per step, downsampled)
    tick_pos    = [i * steps_per_epoch // plot_every_n for i in range(0, total_epochs + 1, step)]
    tick_labels = [str(i) for i in range(0, total_epochs + 1, step)]
    downsampled = history["train_loss"][::plot_every_n]
    axes[0].plot(range(len(downsampled)), downsampled, color="tab:blue", linewidth=0.8, alpha=0.85)
    axes[0].set_title("DDPM Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_xticks(tick_pos)
    axes[0].set_xticklabels(tick_labels)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Val loss (per check epoch)
    val_epochs = history.get("val_epochs", [])
    val_losses = history.get("val_loss",   [])
    axes[1].plot(val_epochs, val_losses, color="tab:orange", linewidth=1.2, marker="o", markersize=3)
    axes[1].set_title("DDPM Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # FID log (optional)
    if has_fid:
        axes[2].plot(history["fid_epochs"], history["fid_scores"],
                     color="tab:red", linewidth=1.2, marker="o", markersize=4)
        axes[2].set_title("Inception FID (periodic)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("FID")
        axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("DDPM Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_val_noise_cache(
    val_loader: DataLoader,
    n_timesteps: int,
    device: torch.device,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Pre-sample fixed (x, t, noise) tuples for consistent DDPM validation.
    Seeded so every validation check uses identical noise/timesteps.

    Args:
        val_loader  (DataLoader):   validation data loader
        n_timesteps (int):          diffusion timestep count T
        device      (torch.device): target device
        seed        (int):          RNG seed for reproducibility
    Returns:
        list of (x_batch, t_batch) tuples, all on CPU
              x_batch : (B, C, H, W)
              t_batch : (B,)  int64 in [0, T-1]
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    cache = []
    for batch in val_loader:
        x = batch[0]
        B = x.shape[0]
        t = torch.randint(0, n_timesteps, (B,), generator=rng)
        # Noise shape is determined lazily in compute_ddpm_val_loss
        # (we don't have the latent yet here), so store t only
        cache.append((x.cpu(), t.cpu()))
    return cache


@torch.no_grad()
def compute_ddpm_val_loss(
    ldm: TwoStageLDM,
    val_cache: list[tuple[torch.Tensor, torch.Tensor]],
    channels: int,
    img_size: int,
    device: torch.device,
    noise_seed: int = 42,
) -> float:
    """
    Compute MSE between predicted and actual noise on the validation set.
    Uses fixed (x, t) pairs from val_cache and a seeded noise tensor
    so every call is directly comparable.

    Args:
        ldm        (TwoStageLDM):                  model in eval mode
        val_cache  (list of (x_cpu, t_cpu)):        pre-built fixed val pairs
        channels   (int):                           image channels
        img_size   (int):                           spatial image size
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

        if x.dim() == 2:
            B = x.shape[0]
            x = x.view(B, channels, img_size, img_size)

        # Encode to latent space using frozen encoder
        mu, logvar = ldm.latent_encoder(x)
        z0 = ldm.latent_encoder.reparameterize(mu, logvar)

        noise = torch.randn(z0.shape, device=device, generator=rng)
        z_t   = ldm.q_sample(z0, t, noise)

        # Normalise t to [0, 1] as expected by noise predictor
        t_norm = (t.float() / (ldm.T - 1)).unsqueeze(1)
        eps_pred = ldm.noise_predictor(z_t, t_norm)

        loss = ((eps_pred - noise) ** 2).mean()
        total_loss += loss.item() * x.shape[0]
        n_seen += x.shape[0]

    return total_loss / n_seen


# ──────────────────────────────────────────────────────────────────────────────
# CONVERGENCE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

class SmoothedPlateauDetector:
    """
    Stops training when a smoothed validation signal stops improving.

    Compares the mean of the first half vs second half of a rolling window
    of recent check values. Stops when the improvement is below delta.
    """

    def __init__(self, patience: int, delta: float) -> None:
        """
        Args:
            patience (int):   number of checks to keep in the rolling window
            delta    (float): minimum improvement (first→second half avg) to count as progress
        """
        self.patience = patience
        self.delta    = delta
        self._window: list[float] = []

    def step(self, value: float) -> bool:
        """
        Record a new check value and return True if training should stop.

        Args:
            value (float): latest validation metric (lower is better)
        Returns:
            bool: True if plateau detected (stop training)
        """
        self._window.append(value)
        if len(self._window) < self.patience:
            return False  # not enough history yet

        # Keep only the most recent `patience` checks
        window = self._window[-self.patience:]
        mid    = len(window) // 2
        first_half_avg  = sum(window[:mid]) / mid
        second_half_avg = sum(window[mid:]) / (len(window) - mid)

        # Improvement = how much better the second half is vs the first
        improvement = first_half_avg - second_half_avg
        return improvement < self.delta


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
    Generate samples via full reverse diffusion chain and compute Inception FID.

    Args:
        ldm           (TwoStageLDM): model in eval mode
        data_config   (dict):        dataset config with "dataset", "channels"
        n_samples     (int):         number of samples to generate
        fid_batch_size(int):         generation batch size
        device        (torch.device):target device
    Returns:
        float: Inception FID score
    """
    dataset_name = data_config.get("dataset", "mnist").lower()
    is_mnist     = dataset_name == "mnist"

    print(f"  Generating {n_samples} samples for FID …")
    all_samples = []
    remaining   = n_samples
    ldm.eval()
    with torch.no_grad():
        while remaining > 0:
            n    = min(fid_batch_size, remaining)
            imgs = (ldm.p_sample_loop(n) * 0.5 + 0.5).clamp(0, 1)
            all_samples.append(imgs.cpu())
            remaining -= n
    fid_tensor = torch.cat(all_samples, dim=0)

    inception = _get_inception(device)

    if is_mnist:
        classifier = _load_classifier(device)
        _, real_inception_feats, _ = _load_or_compute_real_features(classifier, inception, device)
    else:
        _, real_inception_feats, _ = _load_or_compute_real_features(None, inception, device)

    print("  Computing Inception FID …")
    gen_feats = _inception_features(fid_tensor, inception, device)
    return float(_fid(real_inception_feats, gen_feats))


# ──────────────────────────────────────────────────────────────────────────────
# VAE TRAINING STAGE
# ──────────────────────────────────────────────────────────────────────────────

def train_vae(
    args: argparse.Namespace,
    hparams: dict,
    dataloader: DataLoader,
    val_loader: DataLoader,
    channels: int,
    img_size: int,
    results_dir: str,
    device: torch.device,
) -> VAEWrapper:
    """
    Train the VAE stage and save weights. Returns the trained VAEWrapper.

    In fixed mode  : trains for exactly args.vae_epochs epochs.
    In convergence mode: trains until ELBO plateaus (patience/delta on val set).

    Args:
        args        (argparse.Namespace): CLI args
        hparams     (dict):               LDM arch hparams
        dataloader  (DataLoader):         training data loader
        val_loader  (DataLoader):         validation data loader
        channels    (int):                image channels
        img_size    (int):                spatial image size
        results_dir (str):                output directory
        device      (torch.device):       target device
    Returns:
        VAEWrapper: trained model (on device, frozen weights saved to disk)
    """
    print("\n" + "=" * 60)
    print("  STAGE 1 — VAE TRAINING")
    print("=" * 60)

    model     = build_vae(hparams, channels, img_size, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"elbo": [], "recon": [], "kl": []}
    graph_path = os.path.join(results_dir, f"{args.run_name}_vae_training_curves.png")

    # Determine training length
    if args.mode == "fixed":
        max_epochs = args.vae_epochs
        detector   = None
    else:
        max_epochs = args.vae_max_epochs
        detector   = SmoothedPlateauDetector(patience=args.vae_patience, delta=args.vae_delta)

    steps_per_epoch = len(dataloader)
    # KL warmup is relative to the VAE training budget
    kl_warmup_steps = max(1, int(args.kl_warmup_frac * max_epochs)) * steps_per_epoch
    global_step     = 0

    progress = tqdm(total=max_epochs * steps_per_epoch, desc="VAE Training", unit="step")

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_mse = 0.0
        running_kl  = 0.0

        for batch in dataloader:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)

            optimizer.zero_grad()
            beta_kl = _get_beta(global_step, args.lambda_kl_max, kl_warmup_steps)

            x_recon, mu, logvar = model(x)
            x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
            x_flat     = x.reshape(x.shape[0], -1).clamp(-1, 1)

            loss_recon = 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1).mean()
            loss_kl    = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
            )
            total_loss = loss_recon + beta_kl * loss_kl
            total_loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            history["elbo"].append(total_loss.item())
            history["recon"].append(loss_recon.item())
            history["kl"].append(loss_kl.item())
            running_mse += loss_recon.item()
            running_kl  += loss_kl.item()

            progress.set_postfix({
                "epoch": f"{epoch}/{max_epochs}",
                "MSE":   f"{loss_recon.item():.4f}",
                "KL":    f"{loss_kl.item():.2f}",
                "β":     f"{beta_kl:.3f}",
            })
            progress.update(1)
            global_step += 1

        epoch_mse = running_mse / steps_per_epoch
        epoch_kl  = running_kl  / steps_per_epoch
        print(f"  [VAE epoch {epoch}] MSE: {epoch_mse:.5f} | KL: {epoch_kl:.3f} | β: {beta_kl:.4f}")

        save_vae_checkpoint(model, optimizer, epoch, history, results_dir, args.run_name)
        save_vae_training_graph(history, steps_per_epoch, epoch, graph_path)

        # Convergence check
        if detector is not None and epoch % args.vae_check_every == 0:
            model.eval()
            val_elbo = 0.0
            n_seen   = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    if x.dim() == 2:
                        x = x.view(x.shape[0], channels, img_size, img_size)
                    x_recon, mu, logvar = model(x)
                    x_hat_flat = x_recon.reshape(x_recon.shape[0], -1)
                    x_flat     = x.reshape(x.shape[0], -1).clamp(-1, 1)
                    recon = 0.5 * ((x_flat - x_hat_flat) ** 2).sum(dim=-1).mean()
                    kl    = -0.5 * torch.mean(
                        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
                    )
                    val_elbo += (recon + args.lambda_kl_max * kl).item() * x.shape[0]
                    n_seen   += x.shape[0]
            val_elbo /= n_seen
            print(f"  [VAE convergence check @ epoch {epoch}] Val ELBO: {val_elbo:.5f}")

            if detector.step(val_elbo):
                print(f"  VAE converged at epoch {epoch} — switching to DDPM stage.")
                break

    progress.close()

    # Save standalone VAE weights for independent later use
    save_vae_weights(model, hparams, results_dir, args.run_name)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# DDPM TRAINING STAGE
# ──────────────────────────────────────────────────────────────────────────────

def train_ddpm(
    args: argparse.Namespace,
    hparams: dict,
    vae: VAEWrapper,
    dataloader: DataLoader,
    val_loader: DataLoader,
    channels: int,
    img_size: int,
    data_config: dict,
    results_dir: str,
    device: torch.device,
    vae_epochs_done: int,
) -> TwoStageLDM:
    """
    Train the DDPM stage on top of the frozen pre-trained VAE.
    Saves full LDM weights at the end for independent later use.

    In fixed mode:      trains for (total_epochs - vae_epochs) epochs.
    In convergence mode: trains until val loss plateaus.

    Args:
        args           (argparse.Namespace): CLI args
        hparams        (dict):               LDM arch hparams
        vae            (VAEWrapper):         trained VAE (encoder/decoder reused)
        dataloader     (DataLoader):         training data loader
        val_loader     (DataLoader):         validation data loader
        channels       (int):                image channels
        img_size       (int):                spatial image size
        data_config    (dict):               dataset config
        results_dir    (str):                output directory
        device         (torch.device):       target device
        vae_epochs_done(int):                how many epochs VAE was trained for (for logging)
    Returns:
        TwoStageLDM: trained model on device
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 — DDPM TRAINING")
    print("=" * 60)

    ldm = build_ldm(hparams, args, channels, img_size, device)

    # Load trained VAE weights into the LDM's frozen encoder/decoder
    ldm.latent_encoder.load_state_dict(vae.latent_encoder.state_dict())
    ldm.decoder.load_state_dict(vae.decoder.state_dict())
    print("  Loaded pre-trained VAE weights into LDM (encoder + decoder frozen).")

    # Only optimise the noise predictor
    optimizer = optim.AdamW(
        ldm.noise_predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_epochs": [],
        "fid_scores": [],
        "fid_epochs": [],
    }
    graph_path = os.path.join(results_dir, f"{args.run_name}_ddpm_training_curves.png")

    # Determine training length
    if args.mode == "fixed":
        max_epochs = args.total_epochs - vae_epochs_done
        detector   = None
        print(f"  Fixed mode: {max_epochs} DDPM epochs "
              f"({args.total_epochs} total − {vae_epochs_done} VAE).")
    else:
        max_epochs = args.ddpm_max_epochs
        detector   = SmoothedPlateauDetector(patience=args.ddpm_patience, delta=args.ddpm_delta)

    steps_per_epoch = len(dataloader)

    # Pre-build fixed val cache for reproducible DDPM validation
    val_cache = _build_val_noise_cache(val_loader, args.T, device)

    # Pre-compute FID checkpoint epochs from fractional schedule
    fid_check_epochs = set(
        max(1, int(f * max_epochs)) for f in args.fid_fractions
    )
    # Skip the first 25% — too early to be meaningful
    warmup_cutoff    = int(0.25 * max_epochs)
    fid_check_epochs = {e for e in fid_check_epochs if e > warmup_cutoff}

    progress = tqdm(total=max_epochs * steps_per_epoch, desc="DDPM Training", unit="step")

    for epoch in range(1, max_epochs + 1):
        ldm.train()
        running_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.view(x.shape[0], channels, img_size, img_size)

            B = x.shape[0]
            optimizer.zero_grad()

            # Encode with frozen VAE encoder
            with torch.no_grad():
                mu, logvar = ldm.latent_encoder(x)
                z0 = ldm.latent_encoder.reparameterize(mu, logvar)

            # Sample random timesteps and add noise
            t     = torch.randint(0, args.T, (B,), device=device)
            noise = torch.randn_like(z0)
            z_t   = ldm.q_sample(z0, t, noise)

            # Normalise t to [0, 1] for noise predictor
            t_norm   = (t.float() / (args.T - 1)).unsqueeze(1)
            eps_pred = ldm.noise_predictor(z_t, t_norm)
            loss     = ((eps_pred - noise) ** 2).mean()

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(ldm.noise_predictor.parameters(), args.grad_clip)
            optimizer.step()

            history["train_loss"].append(loss.item())
            running_loss += loss.item()

            progress.set_postfix({
                "epoch":     f"{epoch}/{max_epochs}",
                "train_MSE": f"{loss.item():.5f}",
            })
            progress.update(1)

        epoch_loss = running_loss / steps_per_epoch
        print(f"  [DDPM epoch {epoch}] Train MSE: {epoch_loss:.5f}")

        # Periodic validation loss check
        if epoch % args.ddpm_check_every == 0:
            val_loss = compute_ddpm_val_loss(ldm, val_cache, channels, img_size, device)
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            print(f"  [DDPM val @ epoch {epoch}] Val MSE: {val_loss:.5f}")

            if detector is not None and detector.step(val_loss):
                print(f"  DDPM converged at epoch {epoch} — stopping DDPM training.")
                save_ldm_checkpoint(ldm, optimizer, epoch, history, results_dir, args.run_name)
                save_ddpm_training_graph(history, steps_per_epoch, epoch, graph_path)
                break

        # Periodic FID logging (evidence curve for thesis, not used for stopping)
        if epoch in fid_check_epochs:
            fid_score = compute_fid(ldm, data_config, args.n_fid_samples, args.fid_batch_size, device)
            history["fid_scores"].append(fid_score)
            history["fid_epochs"].append(epoch)
            print(f"  [FID @ epoch {epoch}] Inception FID: {fid_score:.2f}")

        save_ldm_checkpoint(ldm, optimizer, epoch, history, results_dir, args.run_name)
        save_ddpm_training_graph(history, steps_per_epoch, epoch, graph_path)

    progress.close()

    # Save standalone LDM weights for independent later use
    save_ldm_weights(ldm, hparams, args, results_dir, args.run_name)

    return ldm


# ──────────────────────────────────────────────────────────────────────────────
# FINAL EVAL
# ──────────────────────────────────────────────────────────────────────────────

def compute_final_eval(
    vae: "VAEWrapper | None",
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
    When skip_vae_eval is True, only the LDM eval runs; VAE results
    are assumed to already exist in the results folder from a prior run.

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

    channels     = data_config["channels"]
    img_size     = data_config["img_size"]
    dataset_name = data_config.get("dataset", "mnist").lower()
    is_mnist     = dataset_name == "mnist"

    inception = _get_inception(device)
    if is_mnist:
        classifier = _load_classifier(device)
        real_mnist_feats, real_inception_feats, _ = _load_or_compute_real_features(
            classifier, inception, device
        )
    else:
        _, real_inception_feats, _ = _load_or_compute_real_features(None, inception, device)
        classifier = None

    # ── VAE eval (skipped when reusing a pre-trained VAE) ────────────────────
    vae_recon_mse     = None
    vae_mnist_fid     = None
    vae_inception_fid = None

    if not skip_vae_eval:
        print("\n--- VAE Final Eval ---")
        vae.eval()
        latent_dim  = hparams["latent_dim"]
        latent_size = hparams["latent_size"]

        print(f"  Generating {args.n_fid_samples} VAE samples …")
        all_vae_samples = []
        remaining = args.n_fid_samples
        with torch.no_grad():
            while remaining > 0:
                n = min(args.fid_batch_size, remaining)
                z = torch.randn(n, latent_dim, latent_size, latent_size, device=device)
                imgs = (vae._decode_latent(z) * 0.5 + 0.5).clamp(0, 1)
                all_vae_samples.append(imgs.cpu())
                remaining -= n
        vae_tensor = torch.cat(all_vae_samples, dim=0)

        if is_mnist:
            gen_mnist_feats, _ = _mnist_features(vae_tensor, classifier, device)
            vae_mnist_fid = float(_fid(real_mnist_feats, gen_mnist_feats))

        gen_vae_inception = _inception_features(vae_tensor, inception, device)
        vae_inception_fid = float(_fid(real_inception_feats, gen_vae_inception))

        # Reconstruction MSE on val set
        total_mse, n_seen = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                if x.dim() == 2:
                    x = x.view(x.shape[0], channels, img_size, img_size)
                x_recon, _, _ = vae(x)
                x_flat     = x.reshape(x.shape[0], -1).clamp(-1, 1)
                x_hat_flat = x_recon.reshape(x.shape[0], -1)
                total_mse += ((x_flat - x_hat_flat) ** 2).sum(dim=-1).sum().item()
                n_seen    += x.shape[0]
        vae_recon_mse = total_mse / n_seen

        vutils.save_image(
            vae_tensor[:64],
            os.path.join(results_dir, f"{args.run_name}_vae_samples_8x8.png"),
            nrow=8, padding=2,
        )
    else:
        print("\n--- VAE Final Eval SKIPPED (pre-trained VAE reused) ---")

    # ── LDM eval ─────────────────────────────────────────────────────────────
    print("\n--- LDM Final Eval ---")
    ldm_fid = compute_fid(ldm, data_config, args.n_fid_samples, args.fid_batch_size, device)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Final Eval Summary — {args.run_name}")
    print(f"{'=' * 50}")
    if skip_vae_eval:
        print(f"  VAE eval             : skipped (pre-trained)")
    else:
        print(f"  VAE epochs trained   : {vae_epochs}")
        print(f"  VAE recon MSE        : {vae_recon_mse:.6f}")
        if vae_mnist_fid is not None:
            print(f"  VAE MNIST FID        : {vae_mnist_fid:.2f}")
        print(f"  VAE Inception FID    : {vae_inception_fid:.2f}")
    print(f"  DDPM epochs trained  : {ddpm_epochs}")
    print(f"  LDM Inception FID    : {ldm_fid:.2f}")
    print(f"{'=' * 50}\n")

    metrics = {
        "run_name":          args.run_name,
        "mode":              args.mode,
        "vae_epochs":        vae_epochs,
        "ddpm_epochs":       ddpm_epochs,
        "vae_recon_mse":     vae_recon_mse,
        "vae_mnist_fid":     vae_mnist_fid,
        "vae_inception_fid": vae_inception_fid,
        "ldm_inception_fid": ldm_fid,
    }
    metrics_path = os.path.join(results_dir, f"{args.run_name}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Eval metrics saved → {metrics_path}")

    # LDM sample grid
    ldm.eval()
    with torch.no_grad():
        ldm_samples = (ldm.p_sample_loop(64) * 0.5 + 0.5).clamp(0, 1)
    vutils.save_image(
        ldm_samples,
        os.path.join(results_dir, f"{args.run_name}_ldm_samples_8x8.png"),
        nrow=8, padding=2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS FOLDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# Files produced exclusively by Stage 2 — safe to delete on a DDPM-only re-run
_DDPM_ONLY_FILES = [
    "_ldm_checkpoint.pt",
    "_ldm_weights.pt",
    "_ldm_config.json",
    "_ddpm_training_curves.png",
    "_eval_metrics.json",
    "_ldm_samples_8x8.png",
]


def _clear_ddpm_files(results_dir: str, run_name: str) -> None:
    """
    Delete only Stage-2 output files, leaving VAE files intact.

    Args:
        results_dir (str): results directory
        run_name    (str): run identifier prefix
    Returns:
        None
    """
    for suffix in _DDPM_ONLY_FILES:
        path = os.path.join(results_dir, f"{run_name}{suffix}")
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale DDPM file: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> None:
    """
    Orchestrate two-stage training: VAE → DDPM, then final eval.
    When --skip_vae is set, loads a pre-trained VAE and runs only Stage 2.

    Args:
        args (argparse.Namespace): parsed CLI arguments
    Returns:
        None
    """
    # Validate skip-VAE args early so we fail fast
    if args.skip_vae and not args.vae_weights:
        raise ValueError("--vae_weights must be provided when --skip_vae is set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"--- Two-Stage LDM Training: {args.run_name} | mode={args.mode} ---")

    hparams = load_ldm_config(args.ldm_config)
    print(f"Dataset: {hparams['dataset']}")

    dataset, val_dataset, data_config = build_dataset(
        dataset_name=hparams["dataset"],
        data_root="data/",
        subset_frac=args.subset_frac,
        single_class=False,
    )
    dataloader = DataLoader(dataset,     batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    channels = data_config["channels"]
    img_size = data_config["img_size"]

    results_dir = os.path.join(args.results_dir, args.run_name)

    if args.skip_vae:
        # Folder must already exist from the prior VAE run; only wipe DDPM files
        if not os.path.exists(results_dir):
            raise FileNotFoundError(
                f"Results directory '{results_dir}' not found. "
                "Run full training first before using --skip_vae."
            )
        print(f"  --skip_vae: clearing stale DDPM outputs from {results_dir}")
        _clear_ddpm_files(results_dir, args.run_name)
    else:
        # Full run: wipe and recreate the entire results folder
        if os.path.exists(results_dir):
            import shutil
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

    # ── Stage 1: VAE ──────────────────────────────────────────────────────────
    if args.skip_vae:
        print("\n" + "=" * 60)
        print("  STAGE 1 — VAE TRAINING SKIPPED (loading pre-trained weights)")
        print("=" * 60)
        vae = load_pretrained_vae(args.vae_weights, hparams, channels, img_size, device)
        vae_epochs_done = 0
    else:
        vae = train_vae(
            args, hparams, dataloader, val_loader,
            channels, img_size, results_dir, device,
        )
        vae_ckpt_path   = os.path.join(results_dir, f"{args.run_name}_vae_checkpoint.pt")
        vae_ckpt        = torch.load(vae_ckpt_path, map_location=device)
        vae_epochs_done = vae_ckpt["epoch"]

    # ── Stage 2: DDPM ─────────────────────────────────────────────────────────
    ldm = train_ddpm(
        args, hparams, vae, dataloader, val_loader,
        channels, img_size, data_config, results_dir, device,
        vae_epochs_done=vae_epochs_done,
    )

    ldm_ckpt_path    = os.path.join(results_dir, f"{args.run_name}_ldm_checkpoint.pt")
    ldm_ckpt         = torch.load(ldm_ckpt_path, map_location=device)
    ddpm_epochs_done = ldm_ckpt["epoch"]

    # ── Final eval ────────────────────────────────────────────────────────────
    compute_final_eval(
        vae=vae if not args.skip_vae else None,
        ldm=ldm,
        hparams=hparams,
        val_loader=val_loader,
        data_config=data_config,
        args=args,
        results_dir=results_dir,
        device=device,
        vae_epochs=vae_epochs_done,
        ddpm_epochs=ddpm_epochs_done,
        skip_vae_eval=args.skip_vae,
    )


def main() -> None:
    args = parse_args()
    os.makedirs("src/logs", exist_ok=True)
    log_path = f"src/logs/{args.run_name}.log"
    log_file = open(log_path, "w")
    sys.stdout = log_file
    try:
        run_training(args)
    finally:
        log_file.close()
        sys.stdout = sys.__stdout__
        print(f"Training complete. Log saved to {log_path}")


if __name__ == "__main__":
    main()