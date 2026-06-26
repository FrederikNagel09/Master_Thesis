# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT & MODEL SAVING
# ──────────────────────────────────────────────────────────────────────────────


import argparse
import json
import os

import torch
from torch import optim

from src.models.two_stage_models.latent_two_stage import TwoStageLDM
from src.models.vae.vae_wrapper import VAEWrapper


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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


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
    config_path = os.path.join(results_dir, f"{run_name}_vae_config.json")

    torch.save(
        {
            "encoder_state_dict": model.latent_encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "vae_state_dict": model.state_dict(),
        },
        weights_path,
    )

    arch_keys = [
        "dataset",
        "latent_dim",
        "latent_size",
        "latent_patch_size",
        "latent_enc_hidden_dim",
        "dec_trans_dim",
        "dec_trans_n_head",
        "dec_trans_head_dim",
        "dec_trans_ff_dim",
        "dec_trans_enc_depth",
        "dec_trans_dec_depth",
        "dec_trans_n_groups",
        "dec_trans_update_strategy",
        "inr_hidden_dim",
        "inr_layers",
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def save_ldm_weights(
    model: TwoStageLDM,
    hparams: dict,
    args: argparse.Namespace,
    results_dir: str,
    run_name: str,
    is_3d: bool = False,
) -> None:
    """
    Save standalone LDM weights + config for independent later use.

    Args:
        model       (TwoStageLDM):       trained model
        hparams     (dict):              arch hparams
        args        (argparse.Namespace): CLI args
        results_dir (str):               output directory
        run_name    (str):               run identifier
    Returns:
        None
    """
    weights_path = os.path.join(results_dir, f"{run_name}_ldm_weights.pt")
    config_path = os.path.join(results_dir, f"{run_name}_ldm_config.json")

    torch.save(
        {
            "ldm_state_dict": model.state_dict(),
            "noise_predictor_state_dict": model.noise_predictor.state_dict(),
            "encoder_state_dict": model.latent_encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
        },
        weights_path,
    )

    arch_keys = [
        "dataset",
        "latent_dim",
        "latent_size",
        "latent_patch_size",
        "latent_enc_hidden_dim",
        "dec_trans_dim",
        "dec_trans_n_head",
        "dec_trans_head_dim",
        "dec_trans_ff_dim",
        "dec_trans_enc_depth",
        "dec_trans_dec_depth",
        "dec_trans_n_groups",
        "dec_trans_update_strategy",
        "inr_hidden_dim",
        "inr_layers",
        "pred_d_model",
        "pred_n_heads",
        "pred_n_layers",
        "pred_d_ff",
        "pred_t_embed_dim",
        "noise_predictor_dropout",
    ]
    config = {k: hparams[k] for k in arch_keys}
    config["run_name"] = run_name
    config["T"] = args.T
    config["beta_1"] = args.beta_1
    config["beta_T"] = args.beta_T
    config["is_3d"] = is_3d

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"LDM weights → {weights_path}")
    print(f"LDM config  → {config_path}")
