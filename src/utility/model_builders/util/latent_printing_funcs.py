from torch import nn

from src.models.latent_diffusion.modules.LatentEncoder3D import Conv3DEncoder
from src.models.latent_diffusion.modules.LatentNoisePredictor import (
    LatentTransformerNoisePredictor,
)
from src.models.latent_diffusion.modules.trans_inr import TransInr


def _print_decoder_info(decoder: TransInr) -> int:
    """
    Prints a parameter count summary for a TransInr decoder.
    Args:
        decoder : instantiated TransInr decoder
    Returns:
        total : total parameter count (int)
    """
    # ── Tokenizer breakdown ───────────────────────────────────────────────
    prefc_params = sum(p.numel() for p in decoder.tokenizer.prefc.parameters())
    posemb_params = decoder.tokenizer.posemb.numel()
    local_params = sum(p.numel() for p in decoder.tokenizer.local_attn.parameters())
    global_params = sum(p.numel() for p in decoder.tokenizer.global_attn.parameters())
    tok_params = sum(p.numel() for p in decoder.tokenizer.parameters())
    n_patches = decoder.tokenizer.posemb.shape[1]
    tok_dim = decoder.tokenizer.posemb.shape[2]

    # ── Transformer breakdown ─────────────────────────────────────────────
    cls_name = decoder.transformer.__class__.__name__
    trans_params = sum(p.numel() for p in decoder.transformer.parameters())
    if cls_name == "Transformer":
        enc_params = sum(p.numel() for p in decoder.transformer.encoder.parameters())
        dec_params = sum(p.numel() for p in decoder.transformer.decoder.parameters())
    else:
        enc_params = trans_params
        dec_params = None

    # ── Wtoken / INR breakdown ────────────────────────────────────────────
    n_wtokens = decoder.wtokens.shape[0]
    wtoken_dim = decoder.wtokens.shape[1]
    wtoken_params = decoder.wtokens.numel()
    postfc_params = sum(p.numel() for p in decoder.wtoken_postfc.parameters())
    base_params = sum(p.numel() for p in decoder.base_params.values())
    inr_params = sum(p.numel() for p in decoder.inr.parameters())
    total = sum(p.numel() for p in decoder.parameters())

    # ── Print: architecture stats ─────────────────────────────────────────
    print("############## Latent Decoder Summary: ##############")
    print("---- Architecture Stats ------------------------------")
    print(f"  Data tokens               : {n_patches:>6}   (dim={tok_dim})")
    print(f"  Weight tokens             : {n_wtokens:>6}   (dim={wtoken_dim})")

    # ── Print: parameter counts ───────────────────────────────────────────
    print("---- Parameters --------------------------------------")
    print(f"Tokenizer                   : {tok_params:>12,}")
    print(f"  Pre-FC                    : {prefc_params:>12,}")
    print(f"  Positional embedding      : {posemb_params:>12,}")
    print(f"  Local attention           : {local_params:>12,}")
    print(f"  Global attention          : {global_params:>12,}")
    print(f"Transformer                 : {trans_params:>12,}")
    if dec_params is not None:
        print(f"  Encoder                   : {enc_params:>12,}")
        print(f"  Decoder                   : {dec_params:>12,}")
    print(f"Weight tokens               : {wtoken_params:>12,}")
    print(f"Wtoken post-FC              : {postfc_params:>12,}")
    print(f"Base INR params             : {base_params:>12,}")
    print(f"SIREN (INR module)          : {inr_params:>12,}")
    print("--------------------------------------------------------------")
    print(f"Total                       : {total:>12,}")
    print("--------------------------------------------------------------")

    return total


def _print_noise_predictor_info(predictor: LatentTransformerNoisePredictor) -> int:
    """
    Prints a parameter count summary for a LatentTransformerNoisePredictor.
    Args:
        predictor : instantiated LatentTransformerNoisePredictor
    Returns:
        total : total parameter count (int)
    """
    # ── Derive architecture stats ─────────────────────────────────────────
    n_layers = len(predictor.blocks)
    d_model = predictor.token_embed.out_features
    n_heads = predictor.blocks[0].attn.num_heads
    d_ff = predictor.blocks[0].mlp[0].out_features

    # ── Parameter counts ──────────────────────────────────────────────────
    time_embed_params = sum(p.numel() for p in predictor.time_embed.parameters())
    time_proj_params = sum(p.numel() for p in predictor.time_proj.parameters())
    embed_params = sum(p.numel() for p in predictor.token_embed.parameters())

    # Per-block component breakdown (assumed uniform across blocks)
    block0 = predictor.blocks[0]
    attn_per_block = sum(p.numel() for p in block0.attn.parameters())
    mlp_per_block = sum(p.numel() for p in block0.mlp.parameters())
    adaLN_per_block = sum(p.numel() for p in block0.adaLN_modulation.parameters())  # noqa: N806
    per_block = attn_per_block + mlp_per_block + adaLN_per_block
    blocks_params = per_block * n_layers

    final_mod_params = sum(p.numel() for p in predictor.final_modulation.parameters())
    readout_params = sum(p.numel() for p in predictor.token_readout.parameters())
    norm_params = sum(p.numel() for p in predictor.final_norm.parameters())
    total = sum(p.numel() for p in predictor.parameters())

    # ── Print: architecture stats ─────────────────────────────────────────
    print("############# Noise Predictor Summary: #############")
    print("---- Architecture Stats ----------------------------")
    print(
        f"  Latent tokens  : {predictor.n_patches:>6}   (latent_dim={predictor.latent_dim})"
    )
    print(f"  d_model        : {d_model:>6}   (n_heads={n_heads}, d_ff={d_ff})")
    print(f"  DiT blocks     : {n_layers:>6}")

    # ── Print: parameter counts ───────────────────────────────────────────
    print("---- Parameters ------------------------------------")
    print(f"Time embedding             : {time_embed_params:>12,}")
    print(f"Time projection            : {time_proj_params:>12,}")
    print(f"Token input projection     : {embed_params:>12,}")
    print(
        f"DiT blocks ({n_layers} layers)      : {blocks_params:>12,}  (~{per_block:,} / block)"
    )
    print(
        f"  Attention                : {attn_per_block * n_layers:>12,}  (~{attn_per_block:,} / block)"
    )
    print(
        f"  MLP                      : {mlp_per_block * n_layers:>12,}  (~{mlp_per_block:,} / block)"
    )
    print(
        f"  AdaLN modulation         : {adaLN_per_block * n_layers:>12,}  (~{adaLN_per_block:,} / block)"
    )
    print(f"Final modulation           : {final_mod_params:>12,}")
    print(f"Final norm                 : {norm_params:>12,}")
    print(f"Token readout              : {readout_params:>12,}")
    print("----------------------------------------------------")
    print(f"Total                      : {total:>12,}")
    print("----------------------------------------------------")

    return total


def _print_latent_encoder_info(encoder: nn.Module) -> int:
    """Prints a parameter count summary for a latent encoder (2D or 3D).

    Args:
        encoder: ResNetLatentEncoder or Conv3DEncoder instance.
    Returns:
        total: Total parameter count.
    """
    total = sum(p.numel() for p in encoder.parameters())
    print("############## Latent Encoder Summary: #############")

    if isinstance(encoder, Conv3DEncoder):
        encoder_params = sum(p.numel() for p in encoder.encoder.parameters())
        head_params = sum(p.numel() for p in encoder.output_head.parameters())
        print("Type                   : Conv3DEncoder (3D)")
        print(f"Conv encoder stack     : {encoder_params:>12,}")
        print(f"Output head (μ/logσ²)  : {head_params:>12,}")  # noqa: RUF001

    else:  # ResNetLatentEncoder
        stem_params = sum(p.numel() for p in encoder.stem.parameters())
        layer1_params = sum(p.numel() for p in encoder.layer1.parameters())
        layer2_params = sum(p.numel() for p in encoder.layer2.parameters())
        layer3_params = sum(p.numel() for p in encoder.layer3.parameters())
        layer4_params = sum(p.numel() for p in encoder.layer4.parameters())
        backbone_params = layer1_params + layer2_params + layer3_params + layer4_params
        upsample_mu = sum(p.numel() for p in encoder.upsample_mu.parameters())
        upsample_logvar = sum(p.numel() for p in encoder.upsample_logvar.parameters())
        upsample_params = upsample_mu + upsample_logvar
        print("Type                   : ResNetLatentEncoder (2D)")
        print(f"Stem                   : {stem_params:>12,}")
        print(f"ResNet backbone        : {backbone_params:>12,}")
        print(f"  layer1               : {layer1_params:>12,}")
        print(f"  layer2               : {layer2_params:>12,}")
        print(f"  layer3               : {layer3_params:>12,}")
        print(f"  layer4               : {layer4_params:>12,}")
        print(f"Learnable upsample     : {upsample_params:>12,}")

    print("-------------------------------------------------------------")
    print(f"Total                  : {total:>12,}")
    print("-------------------------------------------------------------")
    return total
