import argparse

import torch

from src.models.latent_diffusion.modules.LatentNoisePredictor import (
    LatentTransformerNoisePredictor,
)
from src.models.two_stage_models.latent_two_stage import TwoStageLDM
from src.utility.model_builders.util.vae_builder import build_encoder_decoder


def build_ldm(
    hparams: dict,
    args: argparse.Namespace,
    channels: int,
    img_size: int,
    device: torch.device,
    is_3d: bool = False,
) -> TwoStageLDM:
    """
    Build TwoStageLDM with a fresh noise predictor (VAE weights loaded separately).

    Args:
        hparams  (dict):              LDM hparams
        args     (argparse.Namespace): CLI args (noise predictor dims, schedule)
        channels (int):               image channels
        img_size (int):               spatial size per dimension
        device   (torch.device):      target device
        is_3d    (bool):              whether input is volumetric
    Returns:
        TwoStageLDM: assembled model on device (VAE weights NOT yet loaded)
    """
    latent_dim = hparams["latent_dim"]
    latent_size = hparams["latent_size"]
    latent_size_tuple = (
        (latent_size, latent_size) if isinstance(latent_size, int) else latent_size
    )

    encoder, decoder = build_encoder_decoder(hparams, channels, img_size, is_3d)
    noise_predictor = LatentTransformerNoisePredictor(
        latent_dim=latent_dim,
        latent_size=latent_size_tuple,
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
        latent_size=latent_size_tuple,
        T=args.T,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        device=device,
        is_3d=is_3d,
    ).to(device)
