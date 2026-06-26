import os

from torch import nn
import torch

from src.models.latent_diffusion.modules.LatentEncoder import ResNetLatentEncoder
from src.models.latent_diffusion.modules.trans_inr import TransInr
from src.models.vae.vae_wrapper import VAEWrapper


def build_encoder_decoder(
    hparams: dict,
    channels: int,
    img_size: int,
    is_3d: bool = False,
) -> tuple[nn.Module, nn.Module]:
    """
    Instantiate encoder and decoder from hparams. Branches on is_3d:
    - 3D: Conv3DEncoder + TransInr with 3D coord/sigmoid INR
    - 2D: ResNetLatentEncoder + TransInr with 2D coord/tanh INR

    Args:
        hparams  (dict): LDM hparams
        channels (int):  image channels
        img_size (int):  spatial size (per dimension)
        is_3d    (bool): whether input is volumetric
    Returns:
        tuple: (encoder, TransInr decoder)
    """
    latent_dim = hparams["latent_dim"]
    latent_size = hparams["latent_size"]
    latent_size_tuple = (
        (latent_size, latent_size) if isinstance(latent_size, int) else latent_size
    )

    if is_3d:
        from src.models.latent_diffusion.modules.LatentEncoder3D import Conv3DEncoder

        encoder = Conv3DEncoder(
            in_channels=channels,
            dim_z=latent_dim,
            base_channels=hparams["latent_enc_hidden_dim"],
        )
        inr_in_dim = 3
        inr_out_act = "sigmoid"
        data_shape = (img_size, img_size, img_size)
    else:
        encoder = ResNetLatentEncoder(
            in_channels=channels,
            latent_dim=latent_dim,
            latent_size=latent_size_tuple,
            hidden_dim=hparams["latent_enc_hidden_dim"],
        )
        inr_in_dim = 2
        inr_out_act = "tanh"
        data_shape = (img_size, img_size)

    decoder = TransInr(
        tokenizer={
            "target": "src.models.tokenizers.latent_tokenizer.LatentTokenizer",
            "params": {
                "latent_dim": latent_dim,
                "latent_size": latent_size,
                "patch_size": hparams["latent_patch_size"],
                "dim": hparams["dec_trans_dim"],
                "n_head": hparams["dec_trans_n_head"],
                "head_dim": hparams["dec_trans_head_dim"],
            },
        },
        inr={
            "target": "src.models.inr.siren.SIREN",
            "params": {
                "depth": hparams["inr_layers"],
                "in_dim": inr_in_dim,
                "out_dim": channels,
                "hidden_dim": hparams["inr_hidden_dim"],
                "out_bias": 0.5,
                "out_activation": inr_out_act,
            },
        },
        data_shape=data_shape,
        n_groups=hparams["dec_trans_n_groups"],
        transformer={
            "target": "src.models.utils.transformer.Transformer",
            "params": {
                "dim": hparams["dec_trans_dim"],
                "encoder_depth": hparams["dec_trans_enc_depth"],
                "decoder_depth": hparams["dec_trans_dec_depth"],
                "n_head": hparams["dec_trans_n_head"],
                "head_dim": hparams["dec_trans_head_dim"],
                "ff_dim": hparams["dec_trans_ff_dim"],
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
    is_3d: bool = False,
) -> VAEWrapper:
    """
    Build a VAEWrapper from hparams and place on device.

    Args:
        hparams  (dict):          LDM hparams
        channels (int):           image channels
        img_size (int):           spatial size per dimension
        device   (torch.device):  target device
        is_3d    (bool):          whether input is volumetric
    Returns:
        VAEWrapper: assembled model on device
    """
    encoder, decoder = build_encoder_decoder(hparams, channels, img_size, is_3d)
    return VAEWrapper(encoder, decoder, img_size, device, is_3d=is_3d).to(device)


def load_pretrained_vae(
    weights_path: str,
    hparams: dict,
    channels: int,
    img_size: int,
    device: torch.device,
    is_3d: bool = False,
) -> VAEWrapper:
    """
    Build a VAEWrapper and load pre-trained encoder/decoder weights from disk.

    Args:
        weights_path (str):          path to _vae_weights.pt
        hparams      (dict):         LDM arch hparams
        channels     (int):          image channels
        img_size     (int):          spatial size per dimension
        device       (torch.device): target device
        is_3d        (bool):         whether input is volumetric
    Returns:
        VAEWrapper: model with loaded weights on device, in eval mode
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"VAE weights not found at: {weights_path}")

    vae = build_vae(hparams, channels, img_size, device, is_3d=is_3d)
    ckpt = torch.load(weights_path, map_location=device)

    if "encoder_state_dict" in ckpt and "decoder_state_dict" in ckpt:
        vae.latent_encoder.load_state_dict(ckpt["encoder_state_dict"])
        vae.decoder.load_state_dict(ckpt["decoder_state_dict"])
    else:
        vae.load_state_dict(ckpt["vae_state_dict"])

    vae.eval()
    print(f"  Loaded pre-trained VAE weights from: {weights_path}")
    return vae
