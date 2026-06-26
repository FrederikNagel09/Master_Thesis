from src.models.latent_diffusion.modules.LatentEncoder import ResNetLatentEncoder
from src.utility.model_builders.util.latent_printing_funcs import (
    _print_decoder_info,
    _print_latent_encoder_info,
    _print_noise_predictor_info,
)


def _build_latent_diffusion(args, data_config: dict):
    """
    Build LatentDiffusion with LatentEncoder + TransInr decoder + noise predictor.

    Encoder variant  : controlled by args.latent_encoder_type  ("mlp" | "transformer")
    Predictor variant: controlled by args.latent_predictor_type ("mlp" | "transformer")
    """
    from src.models.latent_diffusion.LatentInrDiffusion import LatentDiffusion
    from src.models.latent_diffusion.modules.LatentNoisePredictor import (
        LatentTransformerNoisePredictor,
    )
    from src.models.latent_diffusion.modules.trans_inr import TransInr, make_coord_grid

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]
    is_3d = data_config.get("is_3d", False)  # <-- new flag

    latent_dim = getattr(args, "latent_dim", 32)
    latent_size = getattr(args, "latent_size", 14)
    patch_size = getattr(args, "latent_patch_size", 2)
    latent_size_tuple = (
        (latent_size, latent_size) if isinstance(latent_size, int) else latent_size
    )

    # ── Encoder: 3D or 2D ────────────────────────────────────────────────────
    if is_3d:
        from src.models.latent_diffusion.modules.LatentEncoder3D import Conv3DEncoder

        latent_encoder = Conv3DEncoder(
            in_channels=channels,
            dim_z=latent_dim,
            base_channels=getattr(args, "latent_enc_hidden_dim", 64),
            dropout=getattr(args, "dropout", 0.0),
        )
    else:
        latent_encoder = ResNetLatentEncoder(
            in_channels=channels,
            latent_dim=latent_dim,
            latent_size=latent_size_tuple,
            hidden_dim=getattr(args, "latent_enc_hidden_dim", 512),
        )

    # ── INR: 3D uses sigmoid output and 3D coords ────────────────────────────
    inr_in_dim = 3 if is_3d else 2
    inr_out_act = "sigmoid" if is_3d else "tanh"
    data_shape = (img_size, img_size, img_size) if is_3d else (img_size, img_size)

    # ── TransInr decoder ──────────────────────────────────────────────────────
    dec_dim = getattr(args, "dec_trans_dim", 256)
    dec_n_head = getattr(args, "dec_trans_n_head", 8)
    dec_head_dim = getattr(args, "dec_trans_head_dim", 32)
    dec_ff_dim = getattr(args, "dec_trans_ff_dim", 512)
    dec_enc_depth = getattr(args, "dec_trans_enc_depth", 4)
    dec_dec_depth = getattr(args, "dec_trans_dec_depth", 4)
    dec_n_groups = getattr(args, "dec_trans_n_groups", 8)
    dec_update = getattr(args, "dec_trans_update_strategy", "scale")
    inr_hidden = getattr(args, "inr_hidden_dim", 256)
    inr_layers = getattr(args, "inr_layers", 5)

    # LatentTokenizer config — latent_dim and latent_size must match encoder output
    tokenizer_cfg = {
        "target": "src.models.tokenizers.latent_tokenizer.LatentTokenizer",
        "params": {
            "latent_dim": latent_dim,
            "latent_size": latent_size,
            "patch_size": patch_size,
            "dim": dec_dim,
            "n_head": dec_n_head,
            "head_dim": dec_head_dim,
        },
    }
    inr_cfg = {
        "target": "src.models.inr.siren.SIREN",
        "params": {
            "depth": inr_layers,
            "in_dim": inr_in_dim,  # 2 or 3
            "out_dim": channels,
            "hidden_dim": inr_hidden,
            "out_bias": 0.5,
            "out_activation": inr_out_act,  # "tanh" or "sigmoid"
        },
    }
    transformer_cfg = {
        "target": "src.models.utils.transformer.Transformer",
        "params": {
            "dim": dec_dim,
            "encoder_depth": dec_enc_depth,
            "decoder_depth": dec_dec_depth,
            "n_head": dec_n_head,
            "head_dim": dec_head_dim,
            "ff_dim": dec_ff_dim,
        },
    }

    decoder = TransInr(
        tokenizer=tokenizer_cfg,
        inr=inr_cfg,
        data_shape=data_shape,  # (32,32) or (32,32,32)
        n_groups=dec_n_groups,
        transformer=transformer_cfg,
        update_strategy=dec_update,
    )

    # ── Noise predictor ───────────────────────────────────────────────────────

    noise_predictor = LatentTransformerNoisePredictor(
        latent_dim=latent_dim,
        latent_size=latent_size_tuple,
        d_model=getattr(args, "pred_d_model", 256),
        n_heads=getattr(args, "pred_n_heads", 8),
        n_layers=getattr(args, "pred_n_layers", 4),
        d_ff=getattr(args, "pred_d_ff", 1024),
        dropout=getattr(args, "dropout", 0.0),
        t_embed_dim=getattr(args, "pred_t_embed_dim", 128),
    )
    # ── Coordinate grid ───────────────────────────────────────────────────────
    coord_grid = make_coord_grid(data_shape, (-1, 1))

    # ── Assemble ──────────────────────────────────────────────────────────────
    model = LatentDiffusion(
        noise_predictor=noise_predictor,
        latent_encoder=latent_encoder,
        decoder=decoder,
        coord_grid=coord_grid,
        latent_size=latent_size_tuple,
        latent_dim=latent_dim,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        data_dim=data_dim,
        img_size=img_size,
    )

    # Final print of total model params (encoder + decoder + predictor)
    # Layer-by-layer INR table
    print("\n########## Decoder INR Parameter Breakdown: ##############")
    print(f"  {'Layer':<10} | {'Shape':>16}   {'Total':>8}")
    print(f"  {'─'*10}-+-{'─'*16}---{'─'*8}")
    inr_total = 0
    for name, shape in decoder.inr.param_shapes.items():
        total_els = shape[0] * shape[1]
        shape_str = f"{shape[0]}x{shape[1]}"
        print(f"  {name:<10} | {shape_str:>16}   {total_els:>8,}")
        inr_total += total_els
    print(f"  {'─'*10}-+-{'─'*16}---{'─'*8}")
    print(f"  {'TOTAL':<10} | {'':>16}   {inr_total:>8,}")
    print("############## Latent Space & INR Summary: #############")
    print(
        f"Latent variable (diffusion) : ({latent_dim}, {latent_size_tuple[0]}, {latent_size_tuple[1]})"
    )
    print("________________________________________________________")
    print(f"latent dim: {latent_dim * latent_size_tuple[0] * latent_size_tuple[1]}")
    print(f"INR dim.  : {inr_total}")
    print("########################################################")
    encoder_params = _print_latent_encoder_info(latent_encoder)
    decoder_params = _print_decoder_info(decoder)
    noise_params = _print_noise_predictor_info(noise_predictor)

    total_params = sum(p.numel() for p in model.parameters())
    print("\n########## Total Parameter Summary: ##############")
    print("Latent Encoder  : ", f"{encoder_params:,}")
    print("Noise Predictor : ", f"{noise_params:,}")
    print("Latent Decoder  : ", f"{decoder_params:,}")
    print("----------------------------------------------")
    print(f"TOTAL PARAMETERS  : {total_params:,}")
    print("####################################################\n")

    return model
