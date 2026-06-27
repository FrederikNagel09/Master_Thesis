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

from src.models.weight_diffusion.modules.WeightnoisePredictor import (
    TransInrNoisePredictor,
)
from src.models.weight_diffusion.modules.WeightEncoder import TransInrEncoder
from src.models.latent_diffusion.modules.trans_inr import make_coord_grid
from src.models.weight_diffusion.WeightDiffusion import WeightDiffusion

from src.utility.classifier_utils import (
    _get_inception,
    _inception_features,
    _load_classifier,
    _load_or_compute_real_features,
)
from src.utility.dataset_builders import build_dataset
from src.utility.metrics_util import _fid

warnings.filterwarnings("ignore", message="The operator 'aten::im2col'")

"""
Fixed-budget mode:
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_fixed \
    --mode fixed \
    --wd_config src/train_results/Weight-Diffusion-Probabilistic-test/metadata/config.json \
    --total_epochs 300 \
    --vae_epochs 150 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 256 \
    --fid_batch_size 64

Convergence mode:
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_convergence \
    --mode convergence \
    --wd_config src/train_results/Weight-Diffusion-Probabilistic-test/metadata/config.json \
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
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 4096 \
    --fid_batch_size 1024

Skip-encoder mode (re-run diffusion only, encoder files left untouched):
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_convergence \
    --mode convergence \
    --skip_vae \
    --encoder_weights src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_encoder_weights.pt \
    --wd_config src/train_results/weight-Diffusion/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --ddpm_check_every 5 \
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 256 \
    --fid_batch_size 64

3D ShapeNet voxels mode (convergence):
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_shapenet \
    --mode convergence \
    --wd_config src/train_results/weight-diffusion-shapenet/metadata/config.json \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 5 \
    --vae_patience 10 \
    --vae_delta 1e-4 \
    --ddpm_check_every 5 \
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 128 \
    --fid_batch_size 16
"""


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for two-stage WeightDiffusion training.

    Returns:
        argparse.Namespace: parsed arguments
    """
    p = argparse.ArgumentParser(
        description="Two-stage WeightEncoder + Diffusion training"
    )

    # Run
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument(
        "--wd_config",
        type=str,
        required=True,
        help="Path to WeightDiffusion config .json",
    )
    p.add_argument("--results_dir", type=str, default="src/train_results")
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fixed", "convergence"],
        help="'fixed': explicit epoch counts; 'convergence': early-stop both stages",
    )

    # Skip-encoder: load pre-trained encoder weights and go straight to diffusion
    p.add_argument(
        "--skip_vae",
        action="store_true",
        default=False,
        help="Skip encoder training and load pre-trained weights instead",
    )
    p.add_argument(
        "--encoder_weights",
        type=str,
        default=None,
        help="Path to _encoder_weights.pt (required when --skip_vae is set)",
    )

    # Shared training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--subset_frac", type=float, default=1.0)

    # KL annealing (stage 1 only)
    p.add_argument(
        "--lambda_kl_max",
        type=float,
        default=0.01,
        help="Max KL weight — keep small since KL is in weight space",
    )
    p.add_argument("--kl_warmup_frac", type=float, default=0.4)

    # Fixed-mode epochs
    p.add_argument(
        "--total_epochs",
        type=int,
        default=300,
        help="[fixed mode] Total epochs across both stages",
    )
    p.add_argument(
        "--vae_epochs",
        type=int,
        default=100,
        help="[fixed mode] How many of total_epochs go to encoder training",
    )

    # Convergence-mode encoder stopping
    p.add_argument("--vae_check_every", type=int, default=5)
    p.add_argument("--vae_patience", type=int, default=10)
    p.add_argument("--vae_delta", type=float, default=1e-4)
    p.add_argument("--vae_max_epochs", type=int, default=1000)

    # Convergence-mode diffusion stopping
    p.add_argument("--ddpm_check_every", type=int, default=5)
    p.add_argument("--ddpm_patience", type=int, default=20)
    p.add_argument("--ddpm_delta", type=float, default=1e-4)
    p.add_argument("--ddpm_max_epochs", type=int, default=5000)

    # Noise schedule
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_1", type=float, default=1e-4)
    p.add_argument("--beta_T", type=float, default=0.02)

    # FID / MMD+COV evaluation
    p.add_argument("--n_fid_samples", type=int, default=1024)
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument(
        "--fid_fractions",
        type=float,
        nargs="+",
        default=[0.4, 0.55, 0.7, 0.8, 0.9, 1.0],
        help="Fractional checkpoints (of diffusion epochs) at which to log FID",
    )

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────


def load_wd_config(path: str) -> dict:
    """
    Load hparams from a trained WeightDiffusion config JSON.

    Args:
        path (str): path to config .json
    Returns:
        dict: hparams block from config
    """
    with open(path) as f:
        config = json.load(f)

    required_keys = [
        "dataset",
        "encoder_trans_dim",
        "encoder_trans_n_head",
        "encoder_trans_head_dim",
        "encoder_trans_ff_dim",
        "encoder_trans_enc_depth",
        "encoder_trans_dec_depth",
        "encoder_trans_patch_size",
        "encoder_trans_n_groups",
        "encoder_trans_update_strategy",
        "inr_hidden_dim",
        "inr_layers",
        "noise_predictor_type",
        "noise_predictor_dim",
        "noise_predictor_n_head",
        "noise_predictor_head_dim",
        "noise_predictor_ff_dim",
        "noise_predictor_t_embed_dim",
        "noise_predictor_depth",
        "noise_predictor_dropout",
        "noise_predictor_chunk_size",
    ]
    hparams = config["hparams"]
    missing = [k for k in required_keys if k not in hparams]
    if missing:
        raise ValueError(f"WeightDiffusion config missing required keys: {missing}")
    return hparams


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ──────────────────────────────────────────────────────────────────────────────


def build_weight_encoder(
    hparams: dict,
    channels: int,
    img_size: int,
    is_3d: bool = False,
) -> TransInrEncoder:
    """
    Instantiate TransInrEncoder using VolumeTokenizer for 3D or ImageTokenizer for 2D.

    Args:
        hparams  (dict): WeightDiffusion hparams
        channels (int):  image/volume channels
        img_size (int):  spatial size per dimension
        is_3d    (bool): whether input is volumetric
    Returns:
        TransInrEncoder
    """
    if is_3d:
        vol_size = (img_size, img_size, img_size)
        tokenizer_cfg = {
            "target": "src.models.tokenizers.volume_tokenizer.VolumeTokenizer",
            "params": {
                "in_channels": channels,
                "vol_size": vol_size,
                "patch_size": hparams["encoder_trans_patch_size"],
                "dim": hparams["encoder_trans_dim"],
                "n_head": hparams["encoder_trans_n_head"],
                "head_dim": hparams["encoder_trans_head_dim"],
            },
        }
        # 3D INR: 3D coords, sigmoid output (voxels are binary [0,1])
        inr_cfg = {
            "target": "src.models.inr.siren.SIREN",
            "params": {
                "depth": hparams["inr_layers"],
                "in_dim": 3,
                "out_dim": channels,
                "hidden_dim": hparams["inr_hidden_dim"],
                "out_bias": 0.5,
                "out_activation": "sigmoid",
            },
        }
    else:
        tokenizer_cfg = {
            "target": "src.models.tokenizers.image_tokenizer.ImageTokenizer",
            "params": {
                "in_channels": channels,
                "image_size": img_size,
                "patch_size": hparams["encoder_trans_patch_size"],
                "n_head": hparams["encoder_trans_n_head"],
                "head_dim": hparams["encoder_trans_head_dim"],
            },
        }
        inr_cfg = {
            "target": "src.models.inr.siren.SIREN",
            "params": {
                "depth": hparams["inr_layers"],
                "in_dim": 2,
                "out_dim": channels,
                "hidden_dim": hparams["inr_hidden_dim"],
                "out_bias": 0.5,
                "out_activation": "tanh",
            },
        }

    transformer_cfg = {
        "target": "src.models.utils.transformer.Transformer",
        "params": {
            "dim": hparams["encoder_trans_dim"],
            "encoder_depth": hparams["encoder_trans_enc_depth"],
            "decoder_depth": hparams["encoder_trans_dec_depth"],
            "n_head": hparams["encoder_trans_n_head"],
            "head_dim": hparams["encoder_trans_head_dim"],
            "ff_dim": hparams["encoder_trans_ff_dim"],
        },
    }

    return TransInrEncoder(
        tokenizer=tokenizer_cfg,
        inr=inr_cfg,
        n_groups=hparams["encoder_trans_n_groups"],
        transformer=transformer_cfg,
        update_strategy=hparams["encoder_trans_update_strategy"],
        in_channels=channels,
        img_size=img_size,
    )


def build_noise_predictor(hparams: dict, weight_dim: int) -> nn.Module:
    """
    Instantiate the noise predictor from hparams.

    Args:
        hparams    (dict): WeightDiffusion hparams
        weight_dim (int):  dimensionality of the weight/modulation space
    Returns:
        nn.Module: noise predictor
    """
    predictor_type = hparams.get("noise_predictor_type", "transinr").lower()

    if predictor_type == "transinr":
        return TransInrNoisePredictor(
            weight_dim=weight_dim,
            dim=hparams["noise_predictor_dim"],
            depth=hparams["noise_predictor_depth"],
            n_head=hparams["noise_predictor_n_head"],
            head_dim=hparams["noise_predictor_head_dim"],
            ff_dim=hparams["noise_predictor_ff_dim"],
            chunk_size=hparams["noise_predictor_chunk_size"],
            t_embed_dim=hparams["noise_predictor_t_embed_dim"],
            dropout=hparams["noise_predictor_dropout"],
        )
    elif predictor_type == "paramdit":
        raise ValueError(
            "paramdit noise predictor requires modulation_shapes from the encoder. "
            "Build it manually via build_full_wd_model() instead."
        )
    else:
        raise ValueError(
            f"Unknown noise_predictor_type '{predictor_type}'. Expected: transinr, paramdit."
        )


def build_full_wd_model(
    hparams: dict,
    args: argparse.Namespace,
    channels: int,
    img_size: int,
    data_dim: int,
    device: torch.device,
    is_3d: bool = False,
) -> WeightDiffusion:
    """
    Build a full WeightDiffusion model from hparams and CLI args.

    Args:
        hparams   (dict):              WeightDiffusion hparams
        args      (argparse.Namespace): CLI args (noise schedule, etc.)
        channels  (int):               image/volume channels
        img_size  (int):               spatial size per dimension
        data_dim  (int):               flattened data dimension
        device    (torch.device):      target device
        is_3d     (bool):              whether input is volumetric
    Returns:
        WeightDiffusion: assembled model on device
    """
    encoder = build_weight_encoder(hparams, channels, img_size, is_3d=is_3d)
    weight_dim = encoder.modulation_dim

    predictor_type = hparams.get("noise_predictor_type", "transinr").lower()
    if predictor_type == "transinr":
        noise_predictor = build_noise_predictor(hparams, weight_dim)
    elif predictor_type == "paramdit":
        from models.latent_diffusion.modules.param_dit import ParamDiT

        param_shapes = {
            name: (shape[1], shape[0])
            for name, shape in encoder.modulation_shapes.items()
        }
        noise_predictor = ParamDiT(
            param_shapes=param_shapes,
            hidden_dim=hparams["noise_predictor_dim"],
            depth=hparams["noise_predictor_depth"],
            num_heads=hparams["noise_predictor_n_head"],
            mlp_ratio=hparams.get("paramdit_mlp_ratio", 4.0),
            dropout=hparams["noise_predictor_dropout"],
            time_dim=hparams["noise_predictor_t_embed_dim"],
            tokenizer=hparams.get("paramdit_tokenizer", "column"),
            tokens_per_tensor=hparams.get("paramdit_tokens_per_tensor", 1),
            chunk_size=hparams.get("paramdit_chunk_size"),
        )
    else:
        raise ValueError(f"Unknown noise_predictor_type '{predictor_type}'.")

    # Coord grid shape depends on dimensionality
    data_shape = (img_size, img_size, img_size) if is_3d else (img_size, img_size)
    coord_grid = make_coord_grid(data_shape, (-1, 1))

    return WeightDiffusion(
        NoisePredictor=noise_predictor,
        WeightEncoder=encoder,
        coord_grid=coord_grid,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        sigma_tilde_factor=hparams.get("sigma_tilde", 1.0),
        data_dim=data_dim,
        img_size=img_size,
        stop_gradient_flow=False,
    ).to(device)


def load_pretrained_encoder(
    weights_path: str,
    hparams: dict,
    args: argparse.Namespace,
    channels: int,
    img_size: int,
    data_dim: int,
    device: torch.device,
    is_3d: bool = False,
) -> WeightDiffusion:
    """
    Build a full WeightDiffusion model and load pre-trained encoder weights.

    Args:
        weights_path (str):           path to _encoder_weights.pt
        hparams      (dict):          arch hparams
        args         (argparse.Namespace): CLI args
        channels     (int):           image/volume channels
        img_size     (int):           spatial size per dimension
        data_dim     (int):           flattened data dimension
        device       (torch.device):  target device
        is_3d        (bool):          whether input is volumetric
    Returns:
        WeightDiffusion: model with pre-trained encoder loaded, in eval mode
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Encoder weights not found at: {weights_path}")

    model = build_full_wd_model(
        hparams, args, channels, img_size, data_dim, device, is_3d=is_3d
    )
    ckpt = torch.load(weights_path, map_location=device)
    model.weight_encoder.load_state_dict(ckpt["weight_encoder_state_dict"])
    model.eval()
    print(f"  Loaded pre-trained encoder weights from: {weights_path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# INPUT PREPARATION HELPER
# ──────────────────────────────────────────────────────────────────────────────


def _prepare_input(x: torch.Tensor, is_3d: bool) -> torch.Tensor:
    """
    Prepare batch input for the weight encoder.
    3D: keep as (B, C, D, H, W). 2D: flatten to (B, data_dim).

    Args:
        x     (torch.Tensor): raw batch tensor
        is_3d (bool):         whether input is volumetric
    Returns:
        torch.Tensor: prepared input
    """
    if is_3d:
        return x  # VolumeTokenizer expects (B, C, D, H, W)
    B = x.shape[0]
    return x.reshape(B, -1) if x.dim() > 2 else x


# ──────────────────────────────────────────────────────────────────────────────
# KL ANNEALING
# ──────────────────────────────────────────────────────────────────────────────


def _get_beta_kl(
    global_step: int,
    beta_final: float,
    warmup_steps: int,
    burnin_steps: int = 0,
) -> float:
    """
    Linear KL warmup with optional burn-in period.

    Args:
        global_step  (int):   current training step
        beta_final   (float): target KL weight
        warmup_steps (int):   steps to ramp 0 → beta_final after burnin
        burnin_steps (int):   steps to hold at 0 before ramping
    Returns:
        float: current KL weight
    """
    if global_step < burnin_steps:
        return 0.0
    return beta_final * min(1.0, (global_step - burnin_steps) / warmup_steps)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT & WEIGHT SAVING
# ──────────────────────────────────────────────────────────────────────────────


def save_encoder_checkpoint(
    model: WeightDiffusion,
    optimizer: optim.Optimizer,
    epoch: int,
    history: dict,
    results_dir: str,
    run_name: str,
) -> None:
    """
    Save full encoder training checkpoint (resumable).

    Args:
        model       (WeightDiffusion):  model containing the encoder
        optimizer   (optim.Optimizer):  optimizer state
        epoch       (int):              last completed epoch
        history     (dict):             loss history
        results_dir (str):              output directory
        run_name    (str):              run identifier
    Returns:
        None
    """
    path = os.path.join(results_dir, f"{run_name}_encoder_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def save_encoder_weights(
    model: WeightDiffusion,
    hparams: dict,
    results_dir: str,
    run_name: str,
    is_3d: bool = False,
) -> None:
    """
    Save standalone encoder weights + config for independent later use.

    Args:
        model       (WeightDiffusion): trained model
        hparams     (dict):            arch hparams
        results_dir (str):             output directory
        run_name    (str):             run identifier
        is_3d       (bool):            whether input is volumetric
    Returns:
        None
    """
    weights_path = os.path.join(results_dir, f"{run_name}_encoder_weights.pt")
    config_path = os.path.join(results_dir, f"{run_name}_encoder_config.json")

    torch.save(
        {"weight_encoder_state_dict": model.weight_encoder.state_dict()}, weights_path
    )

    arch_keys = [
        "dataset",
        "encoder_trans_dim",
        "encoder_trans_n_head",
        "encoder_trans_head_dim",
        "encoder_trans_ff_dim",
        "encoder_trans_enc_depth",
        "encoder_trans_dec_depth",
        "encoder_trans_patch_size",
        "encoder_trans_n_groups",
        "encoder_trans_update_strategy",
        "inr_hidden_dim",
        "inr_layers",
    ]
    config = {k: hparams[k] for k in arch_keys if k in hparams}
    config["run_name"] = run_name
    config["is_3d"] = is_3d
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Encoder weights → {weights_path}")
    print(f"Encoder config  → {config_path}")


def save_diffusion_checkpoint(
    model: WeightDiffusion,
    optimizer: optim.Optimizer,
    epoch: int,
    history: dict,
    results_dir: str,
    run_name: str,
) -> None:
    """
    Save full diffusion stage training checkpoint (resumable).

    Args:
        model       (WeightDiffusion):  model (frozen encoder + active denoiser)
        optimizer   (optim.Optimizer):  optimizer state (denoiser only)
        epoch       (int):              last completed epoch
        history     (dict):             loss history
        results_dir (str):              output directory
        run_name    (str):              run identifier
    Returns:
        None
    """
    path = os.path.join(results_dir, f"{run_name}_diffusion_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def save_diffusion_weights(
    model: WeightDiffusion,
    hparams: dict,
    args: argparse.Namespace,
    results_dir: str,
    run_name: str,
    is_3d: bool = False,
) -> None:
    """
    Save standalone full WeightDiffusion weights + config for independent later use.

    Args:
        model       (WeightDiffusion):   trained model
        hparams     (dict):              arch hparams
        args        (argparse.Namespace): CLI args (noise schedule)
        results_dir (str):               output directory
        run_name    (str):               run identifier
        is_3d       (bool):              whether input is volumetric
    Returns:
        None
    """
    weights_path = os.path.join(results_dir, f"{run_name}_wd_weights.pt")
    config_path = os.path.join(results_dir, f"{run_name}_wd_config.json")

    torch.save(
        {
            "full_model_state_dict": model.state_dict(),
            "weight_encoder_state_dict": model.weight_encoder.state_dict(),
            "noise_predictor_state_dict": model.denoiser.state_dict(),
        },
        weights_path,
    )

    arch_keys = [
        "dataset",
        "encoder_trans_dim",
        "encoder_trans_n_head",
        "encoder_trans_head_dim",
        "encoder_trans_ff_dim",
        "encoder_trans_enc_depth",
        "encoder_trans_dec_depth",
        "encoder_trans_patch_size",
        "encoder_trans_n_groups",
        "encoder_trans_update_strategy",
        "inr_hidden_dim",
        "inr_layers",
        "noise_predictor_type",
        "noise_predictor_dim",
        "noise_predictor_n_head",
        "noise_predictor_head_dim",
        "noise_predictor_ff_dim",
        "noise_predictor_t_embed_dim",
        "noise_predictor_depth",
        "noise_predictor_dropout",
        "noise_predictor_chunk_size",
    ]
    config = {k: hparams[k] for k in arch_keys if k in hparams}
    config["run_name"] = run_name
    config["T"] = args.T
    config["beta_1"] = args.beta_1
    config["beta_T"] = args.beta_T
    config["is_3d"] = is_3d

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"WeightDiffusion weights → {weights_path}")
    print(f"WeightDiffusion config  → {config_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────


def save_encoder_training_graph(
    history: dict,
    steps_per_epoch: int,
    total_epochs: int,
    save_path: str,
    plot_every_n: int = 100,
) -> None:
    """
    Save 3-panel encoder training curve (ELBO, recon, KL).

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
    tick_pos = [
        i * steps_per_epoch // plot_every_n for i in range(0, total_epochs + 1, step)
    ]
    tick_labels = [str(i) for i in range(0, total_epochs + 1, step)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("elbo", "Total ELBO (Recon + KL)", "tab:blue"),
        ("recon", "Reconstruction Loss", "tab:orange"),
        ("kl", "KL Loss", "tab:green"),
    ]
    for ax, (key, title, color) in zip(axes, panels, strict=False):
        downsampled = history[key][::plot_every_n]
        ax.plot(
            range(len(downsampled)), downsampled, color=color, linewidth=0.8, alpha=0.85
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("WeightEncoder Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_diffusion_training_graph(
    history: dict,
    steps_per_epoch: int,
    total_epochs: int,
    save_path: str,
    plot_every_n: int = 100,
) -> None:
    """
    Save diffusion stage training curve (train loss, val loss, periodic FID/MMD).

    Args:
        history        (dict): keys "train_loss", "val_loss", "val_epochs",
                               "fid_epochs", "fid_scores"
        steps_per_epoch(int):  optimizer steps per epoch
        total_epochs   (int):  total diffusion epochs completed
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

    tick_pos = [
        i * steps_per_epoch // plot_every_n for i in range(0, total_epochs + 1, step)
    ]
    tick_labels = [str(i) for i in range(0, total_epochs + 1, step)]
    downsampled = history["train_loss"][::plot_every_n]
    axes[0].plot(
        range(len(downsampled)),
        downsampled,
        color="tab:blue",
        linewidth=0.8,
        alpha=0.85,
    )
    axes[0].set_title("Diffusion Train Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(tick_pos)
    axes[0].set_xticklabels(tick_labels)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    val_epochs = history.get("val_epochs", [])
    val_losses = history.get("val_loss", [])
    axes[1].plot(
        val_epochs,
        val_losses,
        color="tab:orange",
        linewidth=1.2,
        marker="o",
        markersize=3,
    )
    axes[1].set_title("Diffusion Validation Loss (MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    if has_fid:
        axes[2].plot(
            history["fid_epochs"],
            history["fid_scores"],
            color="tab:red",
            linewidth=1.2,
            marker="o",
            markersize=4,
        )
        axes[2].set_title("Inception FID (periodic)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("FID")
        axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Diffusion Stage Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _build_val_noise_cache(
    val_loader: DataLoader,
    n_timesteps: int,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-sample fixed (x, t) pairs for consistent diffusion validation.

    Args:
        val_loader  (DataLoader): validation data loader
        n_timesteps (int):        diffusion timestep count T
        seed        (int):        RNG seed for reproducibility
    Returns:
        list of (x_batch_cpu, t_batch_cpu) tuples
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    cache = []
    for batch in val_loader:
        x = batch[0]
        B = x.shape[0]
        t = torch.randint(0, n_timesteps, (B,), generator=rng)
        cache.append((x.cpu(), t.cpu()))
    return cache


@torch.no_grad()
def compute_diffusion_val_loss(
    model: WeightDiffusion,
    val_cache: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    is_3d: bool = False,
    noise_seed: int = 42,
) -> float:
    """
    Compute MSE between predicted and actual v-target on the validation set.

    Args:
        model      (WeightDiffusion):               model in eval mode
        val_cache  (list of (x_cpu, t_cpu)):         pre-built fixed val pairs
        device     (torch.device):                   target device
        is_3d      (bool):                           whether input is volumetric
        noise_seed (int):                            seed for reproducibility
    Returns:
        float: mean MSE over validation set
    """
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(noise_seed)

    total_loss = 0.0
    n_seen = 0

    for x_cpu, t_cpu in val_cache:
        x = x_cpu.to(device)
        t = t_cpu.to(device)
        B = x.shape[0]

        x_in = _prepare_input(x, is_3d)

        mean, logvar = model.weight_encoder(x_in)
        theta_prime = model.weight_encoder._reparameterize(mean, logvar)

        theta_t, epsilon = model._forward_process(theta_prime, t)

        t_norm = (t.float() / (model.T - 1)).unsqueeze(1)
        sqrt_ab = model.sqrt_alpha_cumprod[t].unsqueeze(1)
        sqrt_1mab = model.sigma[t].unsqueeze(1)
        v_target = sqrt_ab * epsilon - sqrt_1mab * theta_prime
        v_hat = model.denoiser(theta_t, t_norm)

        loss = ((v_hat - v_target) ** 2).mean()
        total_loss += loss.item() * B
        n_seen += B

    return total_loss / n_seen


# ──────────────────────────────────────────────────────────────────────────────
# CONVERGENCE DETECTION
# ──────────────────────────────────────────────────────────────────────────────


class SmoothedPlateauDetector:
    """
    Stops training when a smoothed validation signal stops improving.
    Compares mean of first vs second half of a rolling window of check values.
    """

    def __init__(self, patience: int, delta: float) -> None:
        """
        Args:
            patience (int):   rolling window size (number of checks)
            delta    (float): minimum improvement between window halves to count as progress
        """
        self.patience = patience
        self.delta = delta
        self._window: list[float] = []

    def step(self, value: float) -> bool:
        """
        Record a new check value and return True if training should stop.

        Args:
            value (float): latest validation metric (lower is better)
        Returns:
            bool: True if plateau detected
        """
        self._window.append(value)
        if len(self._window) < self.patience:
            return False
        window = self._window[-self.patience :]
        mid = len(window) // 2
        first_half_avg = sum(window[:mid]) / mid
        second_half_avg = sum(window[mid:]) / (len(window) - mid)
        return (first_half_avg - second_half_avg) < self.delta


# ──────────────────────────────────────────────────────────────────────────────
# FID / MMD+COV EVALUATION
# ──────────────────────────────────────────────────────────────────────────────


def compute_fid(
    model: WeightDiffusion,
    data_config: dict,
    n_samples: int,
    fid_batch_size: int,
    device: torch.device,
) -> float:
    """
    Generate samples and compute Inception FID (2D) or skip and return nan (3D).

    Args:
        model          (WeightDiffusion): model in eval mode
        data_config    (dict):            dataset config
        n_samples      (int):             number of samples to generate
        fid_batch_size (int):             generation batch size
        device         (torch.device):    target device
    Returns:
        float: Inception FID score, or nan if 3D
    """
    if data_config.get("is_3d", False):
        print("  FID skipped — not defined for 3D voxel data.")
        return float("nan")

    dataset_name = data_config.get("dataset", "mnist").lower()
    is_mnist = dataset_name == "mnist"
    img_size = data_config["img_size"]
    channels = data_config["channels"]

    print(f"  Generating {n_samples} samples for FID …")
    all_samples = []
    remaining = n_samples
    model.eval()
    with torch.no_grad():
        while remaining > 0:
            n = min(fid_batch_size, remaining)
            imgs = model.sample(n_samples=n)
            imgs = imgs.reshape(n, channels, img_size, img_size)
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
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
# RESULTS FOLDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# Files produced exclusively by Stage 2 — safe to delete on a diffusion-only re-run
_DIFFUSION_ONLY_FILES = [
    "_diffusion_checkpoint.pt",
    "_wd_weights.pt",
    "_wd_config.json",
    "_diffusion_training_curves.png",
    "_eval_metrics.json",
    "_diffusion_samples_8x8.png",
    "_encoder_samples_8x8.png",
]


def _clear_diffusion_files(results_dir: str, run_name: str) -> None:
    """
    Delete only Stage-2 output files, leaving encoder files intact.

    Args:
        results_dir (str): results directory
        run_name    (str): run identifier prefix
    Returns:
        None
    """
    for suffix in _DIFFUSION_ONLY_FILES:
        path = os.path.join(results_dir, f"{run_name}{suffix}")
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale diffusion file: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — WEIGHT ENCODER TRAINING
# ──────────────────────────────────────────────────────────────────────────────


def train_weight_encoder(
    args: argparse.Namespace,
    hparams: dict,
    dataloader: DataLoader,
    val_loader: DataLoader,
    channels: int,
    img_size: int,
    data_dim: int,
    data_config: dict,
    results_dir: str,
    device: torch.device,
    is_3d: bool = False,
) -> WeightDiffusion:
    """
    Train Stage 1: WeightEncoder end-to-end with pixel recon + KL loss.

    In fixed mode     : trains for exactly args.vae_epochs epochs.
    In convergence mode: trains until ELBO plateaus.

    Args:
        args        (argparse.Namespace): CLI args
        hparams     (dict):               arch hparams
        dataloader  (DataLoader):         training data loader
        val_loader  (DataLoader):         validation data loader
        channels    (int):                image/volume channels
        img_size    (int):                spatial size per dimension
        data_dim    (int):                flattened data size
        data_config (dict):               dataset config dict
        results_dir (str):                output directory
        device      (torch.device):       target device
        is_3d       (bool):               whether input is volumetric
    Returns:
        WeightDiffusion: model with trained WeightEncoder (on device)
    """
    print("\n" + "=" * 60)
    print("  STAGE 1 — WEIGHT ENCODER TRAINING")
    print("=" * 60)

    model = build_full_wd_model(
        hparams, args, channels, img_size, data_dim, device, is_3d=is_3d
    )
    optimizer = optim.AdamW(
        model.weight_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {"elbo": [], "recon": [], "kl": []}
    graph_path = os.path.join(
        results_dir, f"{args.run_name}_encoder_training_curves.png"
    )

    if args.mode == "fixed":
        max_epochs = args.vae_epochs
        detector = None
    else:
        max_epochs = args.vae_max_epochs
        detector = SmoothedPlateauDetector(
            patience=args.vae_patience, delta=args.vae_delta
        )

    steps_per_epoch = len(dataloader)
    kl_warmup_steps = max(1, int(args.kl_warmup_frac * max_epochs)) * steps_per_epoch
    global_step = 0

    progress = tqdm(
        total=max_epochs * steps_per_epoch, desc="Encoder Training", unit="step"
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_recon = 0.0
        running_kl = 0.0

        for batch in dataloader:
            x = batch[0].to(device)
            x_in = _prepare_input(x, is_3d)
            B = x.shape[0]

            optimizer.zero_grad()
            beta_kl = _get_beta_kl(global_step, args.lambda_kl_max, kl_warmup_steps)

            mu, logvar = model.weight_encoder(x_in)
            theta_prime = model.weight_encoder._reparameterize(mu, logvar)
            theta = model.weight_encoder.decode_modulations(theta_prime)
            x_recon = model._inr_decode(theta)

            # Flatten target for loss; clamp only for 2D (images are [-1,1])
            x_target = x.reshape(B, -1)
            if not is_3d:
                x_target = x_target.clamp(-1, 1)
            if x_recon.shape != x_target.shape:
                x_recon = x_recon.view_as(x_target)

            loss_recon = 0.5 * ((x_target - x_recon) ** 2).sum(dim=-1).mean()
            loss_kl = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            )
            total_loss = loss_recon + beta_kl * loss_kl
            total_loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.weight_encoder.parameters(), args.grad_clip
                )
            optimizer.step()

            history["elbo"].append(total_loss.item())
            history["recon"].append(loss_recon.item())
            history["kl"].append(loss_kl.item())
            running_recon += loss_recon.item()
            running_kl += loss_kl.item()

            progress.set_postfix(
                {
                    "epoch": f"{epoch}/{max_epochs}",
                    "recon": f"{loss_recon.item():.4f}",
                    "KL": f"{loss_kl.item():.3f}",
                    "β_kl": f"{beta_kl:.4f}",
                }
            )
            progress.update(1)
            global_step += 1

        epoch_recon = running_recon / steps_per_epoch
        epoch_kl = running_kl / steps_per_epoch
        print(
            f"  [Encoder epoch {epoch}] Recon: {epoch_recon:.5f} | KL: {epoch_kl:.3f} | β_kl: {beta_kl:.4f}"
        )

        save_encoder_checkpoint(
            model, optimizer, epoch, history, results_dir, args.run_name
        )
        save_encoder_training_graph(history, steps_per_epoch, epoch, graph_path)

        if detector is not None and epoch % args.vae_check_every == 0:
            model.eval()
            val_elbo = 0.0
            n_seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    x_in = _prepare_input(x, is_3d)
                    B = x.shape[0]

                    mu, logvar = model.weight_encoder(x_in)
                    theta_prime = model.weight_encoder._reparameterize(mu, logvar)
                    theta = model.weight_encoder.decode_modulations(theta_prime)
                    x_recon = model._inr_decode(theta)

                    x_target = x.reshape(B, -1)
                    if not is_3d:
                        x_target = x_target.clamp(-1, 1)
                    if x_recon.shape != x_target.shape:
                        x_recon = x_recon.view_as(x_target)

                    recon = 0.5 * ((x_target - x_recon) ** 2).sum(dim=-1).mean()
                    kl = -0.5 * torch.mean(
                        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                    )
                    val_elbo += (recon + args.lambda_kl_max * kl).item() * B
                    n_seen += B

            val_elbo /= n_seen
            print(
                f"  [Encoder convergence check @ epoch {epoch}] Val ELBO: {val_elbo:.5f}"
            )

            if detector.step(val_elbo):
                print(
                    f"  WeightEncoder converged at epoch {epoch} — switching to diffusion stage."
                )
                break

    progress.close()
    save_encoder_weights(model, hparams, results_dir, args.run_name, is_3d=is_3d)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DIFFUSION TRAINING
# ──────────────────────────────────────────────────────────────────────────────


def train_diffusion(
    args: argparse.Namespace,
    hparams: dict,
    stage1_model: WeightDiffusion,
    dataloader: DataLoader,
    val_loader: DataLoader,
    channels: int,
    img_size: int,
    data_dim: int,
    data_config: dict,
    results_dir: str,
    device: torch.device,
    encoder_epochs_done: int,
    is_3d: bool = False,
) -> WeightDiffusion:
    """
    Train Stage 2: noise predictor / denoiser on frozen WeightEncoder.
    Uses v-prediction objective, matching WeightDiffusion._l_diff exactly.

    In fixed mode     : trains for (total_epochs - encoder_epochs_done) epochs.
    In convergence mode: trains until val loss plateaus.

    Args:
        args                (argparse.Namespace): CLI args
        hparams             (dict):               arch hparams
        stage1_model        (WeightDiffusion):    trained stage-1 model
        dataloader          (DataLoader):         training data loader
        val_loader          (DataLoader):         validation data loader
        channels            (int):                image/volume channels
        img_size            (int):                spatial size per dimension
        data_dim            (int):                flattened data size
        data_config         (dict):               dataset config dict
        results_dir         (str):                output directory
        device              (torch.device):       target device
        encoder_epochs_done (int):                encoder epochs actually completed
        is_3d               (bool):               whether input is volumetric
    Returns:
        WeightDiffusion: fully trained model (on device)
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 — DIFFUSION TRAINING")
    print("=" * 60)

    model = stage1_model
    model.stop_gradient_flow = True

    for p in model.weight_encoder.parameters():
        p.requires_grad_(False)
    print("  WeightEncoder (including decode_modulations) frozen.")

    optimizer = optim.AdamW(
        model.denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_epochs": [],
        "fid_scores": [],
        "fid_epochs": [],
    }
    graph_path = os.path.join(
        results_dir, f"{args.run_name}_diffusion_training_curves.png"
    )

    if args.mode == "fixed":
        max_epochs = args.total_epochs - encoder_epochs_done
        detector = None
        print(
            f"  Fixed mode: {max_epochs} diffusion epochs "
            f"({args.total_epochs} total − {encoder_epochs_done} encoder)."
        )
    else:
        max_epochs = args.ddpm_max_epochs
        detector = SmoothedPlateauDetector(
            patience=args.ddpm_patience, delta=args.ddpm_delta
        )

    steps_per_epoch = len(dataloader)
    val_cache = _build_val_noise_cache(val_loader, args.T)

    fid_check_epochs = {max(1, int(f * max_epochs)) for f in args.fid_fractions}
    warmup_cutoff = int(0.25 * max_epochs)
    fid_check_epochs = {e for e in fid_check_epochs if e > warmup_cutoff}

    progress = tqdm(
        total=max_epochs * steps_per_epoch, desc="Diffusion Training", unit="step"
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.weight_encoder.eval()  # keep encoder bn/dropout in eval mode
        running_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)
            x_in = _prepare_input(x, is_3d)
            B = x.shape[0]

            optimizer.zero_grad()

            with torch.no_grad():
                mu, logvar = model.weight_encoder(x_in)
                theta_prime = model.weight_encoder._reparameterize(mu, logvar)

            t = torch.randint(0, args.T, (B,), device=device)
            theta_t, epsilon = model._forward_process(theta_prime, t)

            t_norm = (t.float() / (args.T - 1)).unsqueeze(1)
            sqrt_ab = model.sqrt_alpha_cumprod[t].unsqueeze(1)
            sqrt_1mab = model.sigma[t].unsqueeze(1)
            v_target = sqrt_ab * epsilon - sqrt_1mab * theta_prime
            v_hat = model.denoiser(theta_t, t_norm)
            loss = ((v_hat - v_target) ** 2).mean()

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.denoiser.parameters(), args.grad_clip)
            optimizer.step()

            history["train_loss"].append(loss.item())
            running_loss += loss.item()

            progress.set_postfix(
                {
                    "epoch": f"{epoch}/{max_epochs}",
                    "train_MSE": f"{loss.item():.5f}",
                }
            )
            progress.update(1)

        epoch_loss = running_loss / steps_per_epoch
        print(f"  [Diffusion epoch {epoch}] Train MSE: {epoch_loss:.5f}")

        if epoch % args.ddpm_check_every == 0:
            val_loss = compute_diffusion_val_loss(model, val_cache, device, is_3d=is_3d)
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            print(f"  [Diffusion val @ epoch {epoch}] Val MSE: {val_loss:.5f}")

            if detector is not None and detector.step(val_loss):
                print(f"  Diffusion converged at epoch {epoch} — stopping.")
                save_diffusion_checkpoint(
                    model, optimizer, epoch, history, results_dir, args.run_name
                )
                save_diffusion_training_graph(
                    history, steps_per_epoch, epoch, graph_path
                )
                break

        if epoch in fid_check_epochs:
            fid_score = compute_fid(
                model, data_config, args.n_fid_samples, args.fid_batch_size, device
            )
            history["fid_scores"].append(fid_score)
            history["fid_epochs"].append(epoch)
            print(f"  [FID @ epoch {epoch}] Inception FID: {fid_score:.2f}")

        save_diffusion_checkpoint(
            model, optimizer, epoch, history, results_dir, args.run_name
        )
        save_diffusion_training_graph(history, steps_per_epoch, epoch, graph_path)

    progress.close()
    save_diffusion_weights(
        model, hparams, args, results_dir, args.run_name, is_3d=is_3d
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# FINAL EVALUATION
# ──────────────────────────────────────────────────────────────────────────────


def compute_final_eval(
    model: WeightDiffusion,
    hparams: dict,
    val_loader: DataLoader,
    data_config: dict,
    args: argparse.Namespace,
    results_dir: str,
    device: torch.device,
    encoder_epochs: int,
    diffusion_epochs: int,
    is_3d: bool = False,
    skip_encoder_eval: bool = False,
) -> None:
    """
    Compute and save final eval metrics.
    2D: recon MSE + Inception FID for encoder prior samples + diffusion samples.
    3D: recon MSE + MMD/COV for diffusion samples vs val set.

    Args:
        model             (WeightDiffusion):   trained model
        hparams           (dict):              arch hparams
        val_loader        (DataLoader):        validation data loader
        data_config       (dict):              dataset config
        args              (argparse.Namespace): CLI args
        results_dir       (str):               output directory
        device            (torch.device):      target device
        encoder_epochs    (int):               encoder epochs trained (0 if skipped)
        diffusion_epochs  (int):               diffusion epochs trained
        is_3d             (bool):              whether input is volumetric
        skip_encoder_eval (bool):              if True, skip encoder-only eval
    Returns:
        None
    """
    import torchvision.utils as vutils

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    dataset_name = data_config.get("dataset", "mnist").lower()
    is_mnist = dataset_name == "mnist"

    # FID infrastructure only needed for 2D
    if not is_3d:
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
            classifier = None
    else:
        inception = classifier = real_inception_feats = None

    model.eval()

    # ── Encoder eval ──────────────────────────────────────────────────────────
    enc_recon_mse = None
    enc_inception_fid = None

    if not skip_encoder_eval:
        print("\n--- Encoder Final Eval ---")

        # Reconstruction MSE on val set
        total_mse, n_seen = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_in = _prepare_input(x, is_3d)
                B = x.shape[0]

                mu, logvar = model.weight_encoder(x_in)
                theta_prime = model.weight_encoder._reparameterize(mu, logvar)
                theta = model.weight_encoder.decode_modulations(theta_prime)
                x_recon = model._inr_decode(theta)

                x_target = x.reshape(B, -1)
                if not is_3d:
                    x_target = x_target.clamp(-1, 1)
                if x_recon.shape != x_target.shape:
                    x_recon = x_recon.view_as(x_target)

                total_mse += ((x_target - x_recon) ** 2).sum(dim=-1).sum().item()
                n_seen += B

        enc_recon_mse = total_mse / n_seen

        # Encoder prior samples — 2D only
        if not is_3d:
            weight_dim = model.weight_encoder.modulation_dim
            all_enc_samples = []
            remaining = args.n_fid_samples
            with torch.no_grad():
                while remaining > 0:
                    n = min(args.fid_batch_size, remaining)
                    z = torch.randn(n, weight_dim, device=device)
                    theta = model.weight_encoder.decode_modulations(z)
                    imgs = model._inr_decode(theta)
                    imgs = imgs.reshape(n, channels, img_size, img_size)
                    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                    all_enc_samples.append(imgs.cpu())
                    remaining -= n

            enc_tensor = torch.cat(all_enc_samples, dim=0)
            gen_enc_inception = _inception_features(enc_tensor, inception, device)
            enc_inception_fid = float(_fid(real_inception_feats, gen_enc_inception))

            vutils.save_image(
                enc_tensor[:64],
                os.path.join(results_dir, f"{args.run_name}_encoder_samples_8x8.png"),
                nrow=8,
                padding=2,
            )
        else:
            print("  Encoder prior samples + FID skipped for 3D data.")
    else:
        print("\n--- Encoder Final Eval SKIPPED (pre-trained encoder reused) ---")

    # ── Diffusion eval ────────────────────────────────────────────────────────
    print("\n--- Diffusion Final Eval ---")

    diff_fid = None
    diff_mmd = None
    diff_cov = None

    if is_3d:
        from src.utility.voxel_metrics import compute_mmd_cov

        print(f"  Generating {args.n_fid_samples} 3D samples for MMD/COV …")
        all_samples = []
        remaining = args.n_fid_samples
        with torch.no_grad():
            while remaining > 0:
                n = min(args.fid_batch_size, remaining)
                imgs = model.sample(n_samples=n)
                imgs = imgs.reshape(n, channels, img_size, img_size, img_size)
                all_samples.append(imgs.cpu())
                remaining -= n
        generated = torch.cat(all_samples, dim=0)

        ref_batches = [batch[0] for batch in val_loader]
        reference = torch.cat(ref_batches, dim=0)

        print(
            f"  Computing MMD/COV ({generated.shape[0]} gen vs {reference.shape[0]} ref) …"
        )
        diff_mmd, diff_cov = compute_mmd_cov(generated, reference)
        print(f"  MMD: {diff_mmd:.4f} | COV: {diff_cov:.4f}")
    else:
        diff_fid = compute_fid(
            model, data_config, args.n_fid_samples, args.fid_batch_size, device
        )

        with torch.no_grad():
            diff_imgs = model.sample(n_samples=64)
            diff_imgs = diff_imgs.reshape(64, channels, img_size, img_size)
            diff_imgs = (diff_imgs * 0.5 + 0.5).clamp(0, 1)
        vutils.save_image(
            diff_imgs.cpu(),
            os.path.join(results_dir, f"{args.run_name}_diffusion_samples_8x8.png"),
            nrow=8,
            padding=2,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Final Eval Summary — {args.run_name}")
    print(f"{'=' * 50}")
    if skip_encoder_eval:
        print("  Encoder eval             : skipped (pre-trained)")
    else:
        print(f"  Encoder epochs trained   : {encoder_epochs}")
        print(f"  Encoder recon MSE        : {enc_recon_mse:.6f}")
        if enc_inception_fid is not None:
            print(f"  Encoder (prior) FID      : {enc_inception_fid:.2f}")
    print(f"  Diffusion epochs trained : {diffusion_epochs}")
    if is_3d:
        print(f"  Diffusion MMD            : {diff_mmd:.4f}")
        print(f"  Diffusion COV            : {diff_cov:.4f}")
    else:
        print(f"  Diffusion FID            : {diff_fid:.2f}")
    print(f"{'=' * 50}\n")

    metrics = {
        "run_name": args.run_name,
        "mode": args.mode,
        "encoder_epochs": encoder_epochs,
        "diffusion_epochs": diffusion_epochs,
        "enc_recon_mse": enc_recon_mse,
        "enc_inception_fid": enc_inception_fid,
        "diff_inception_fid": diff_fid,
        "diff_mmd": diff_mmd,
        "diff_cov": diff_cov,
    }
    metrics_path = os.path.join(results_dir, f"{args.run_name}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Eval metrics saved → {metrics_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────────────


def run_training(args: argparse.Namespace) -> None:
    """
    Orchestrate two-stage training: WeightEncoder → Diffusion, then final eval.
    When --skip_vae is set, loads pre-trained encoder and runs only Stage 2.

    Args:
        args (argparse.Namespace): parsed CLI arguments
    Returns:
        None
    """
    if args.skip_vae and not args.encoder_weights:
        raise ValueError("--encoder_weights must be provided when --skip_vae is set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(
        f"--- Two-Stage WeightDiffusion Training: {args.run_name} | mode={args.mode} ---"
    )

    hparams = load_wd_config(args.wd_config)
    print(f"Dataset: {hparams['dataset']}")

    dataset, val_dataset, data_config = build_dataset(
        dataset_name=hparams["dataset"],
        data_root="data/",
        subset_frac=args.subset_frac,
        single_class=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]
    is_3d = data_config.get("is_3d", False)

    results_dir = os.path.join(args.results_dir, args.run_name)

    if args.skip_vae:
        if not os.path.exists(results_dir):
            raise FileNotFoundError(
                f"Results directory '{results_dir}' not found. "
                "Run full training first before using --skip_vae."
            )
        print(f"  --skip_vae: clearing stale diffusion outputs from {results_dir}")
        _clear_diffusion_files(results_dir, args.run_name)
    else:
        if os.path.exists(results_dir):
            import shutil

            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

    # ── Stage 1: WeightEncoder ────────────────────────────────────────────────
    if args.skip_vae:
        print("\n" + "=" * 60)
        print("  STAGE 1 — ENCODER TRAINING SKIPPED (loading pre-trained weights)")
        print("=" * 60)
        stage1_model = load_pretrained_encoder(
            args.encoder_weights,
            hparams,
            args,
            channels,
            img_size,
            data_dim,
            device,
            is_3d=is_3d,
        )
        encoder_epochs_done = 0
    else:
        stage1_model = train_weight_encoder(
            args,
            hparams,
            dataloader,
            val_loader,
            channels,
            img_size,
            data_dim,
            data_config,
            results_dir,
            device,
            is_3d=is_3d,
        )
        enc_ckpt = torch.load(
            os.path.join(results_dir, f"{args.run_name}_encoder_checkpoint.pt"),
            map_location=device,
        )
        encoder_epochs_done = enc_ckpt["epoch"]

    # ── Stage 2: Diffusion ────────────────────────────────────────────────────
    final_model = train_diffusion(
        args,
        hparams,
        stage1_model,
        dataloader,
        val_loader,
        channels,
        img_size,
        data_dim,
        data_config,
        results_dir,
        device,
        encoder_epochs_done=encoder_epochs_done,
        is_3d=is_3d,
    )

    diff_ckpt = torch.load(
        os.path.join(results_dir, f"{args.run_name}_diffusion_checkpoint.pt"),
        map_location=device,
    )
    diffusion_epochs_done = diff_ckpt["epoch"]

    # ── Final eval ────────────────────────────────────────────────────────────
    compute_final_eval(
        final_model,
        hparams,
        val_loader,
        data_config,
        args,
        results_dir,
        device,
        encoder_epochs=encoder_epochs_done,
        diffusion_epochs=diffusion_epochs_done,
        is_3d=is_3d,
        skip_encoder_eval=args.skip_vae,
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
