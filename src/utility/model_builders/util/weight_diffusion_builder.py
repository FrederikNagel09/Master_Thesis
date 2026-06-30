from src.models.latent_diffusion.modules.trans_inr import make_coord_grid
from src.models.weight_diffusion.modules.WeightEncoder import TransInrEncoder
from src.models.weight_diffusion.modules.WeightnoisePredictor import (
    TransInrNoisePredictor,
)
from src.models.weight_diffusion.WeightDiffusion import WeightDiffusion
from src.utility.model_builders.util.weight_printing_funcs import (
    print_encoder_stats,
    print_noise_predictor_stats,
)


def _build_weight_diffusion(args, data_config: dict):
    """
    Build WeightDiffusion:
        TransInrEncoder as W(x) + NDM diffusion in weight space.
        Updated to support 3D volumetric data.
    """
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]
    is_3d = data_config.get("is_3d", False)

    # Handle 3D vs 2D dimensionality
    data_shape = (img_size, img_size, img_size) if is_3d else (img_size, img_size)
    inr_in_dim = 3 if is_3d else 2
    inr_out_act = "sigmoid" if is_3d else "tanh"

    # ── TransInrEncoder config ────────────────────────────────────────────────
    encoder_dim = getattr(args, "encoder_trans_dim", 256)
    encoder_n_head = getattr(args, "encoder_trans_n_head", 8)
    encoder_head_dim = getattr(args, "encoder_trans_head_dim", 32)
    encoder_ff_dim = getattr(args, "encoder_trans_ff_dim", 512)
    encoder_enc_depth = getattr(args, "encoder_trans_enc_depth", 4)
    encoder_dec_depth = getattr(args, "encoder_trans_dec_depth", 4)
    encoder_patch_size = getattr(
        args, "encoder_trans_patch_size", 4
    )  # Should be (pd, ph, pw) if 3D
    encoder_n_groups = getattr(args, "encoder_trans_n_groups", 8)
    encoder_update_strat = getattr(args, "encoder_trans_update_strategy", "scale")
    inr_hidden = getattr(args, "inr_hidden_dim", 256)
    inr_layers = getattr(args, "inr_layers", 5)

    # Use VolumeTokenizer if 3D, else keep ImageTokenizer
    tokenizer_target = (
        "src.models.tokenizers.volume_tokenizer.VolumeTokenizer"
        if is_3d
        else "src.models.tokenizers.image_tokenizer.ImageTokenizer"
    )

    # Ensure patch_size is a tuple (pd, ph, pw)
    if isinstance(encoder_patch_size, int):
        # Assuming isotropic patches if a single int is provided
        patch_size_tuple = (encoder_patch_size, encoder_patch_size, encoder_patch_size)
    else:
        patch_size_tuple = encoder_patch_size

    tokenizer_params = {
        "in_channels": channels,
        "patch_size": patch_size_tuple
        if is_3d
        else patch_size_tuple[0],  # Use the tuple here
        "n_head": encoder_n_head,
        "head_dim": encoder_head_dim,
    }
    # Add size parameter specific to tokenizer type
    if is_3d:
        tokenizer_params["vol_size"] = data_shape
    else:
        tokenizer_params["image_size"] = img_size

    tokenizer_cfg = {
        "target": tokenizer_target,
        "params": tokenizer_params,
    }

    inr_cfg = {
        "target": "src.models.inr.siren.SIREN",
        "params": {
            "depth": inr_layers,
            "in_dim": inr_in_dim,
            "out_dim": channels,
            "hidden_dim": inr_hidden,
            "out_bias": 0.5,
            "out_activation": inr_out_act,
        },
    }

    transformer_cfg = {
        "target": "src.models.utils.transformer.Transformer",
        "params": {
            "dim": encoder_dim,
            "encoder_depth": encoder_enc_depth,
            "decoder_depth": encoder_dec_depth,
            "n_head": encoder_n_head,
            "head_dim": encoder_head_dim,
            "ff_dim": encoder_ff_dim,
        },
    }

    encoder = TransInrEncoder(
        tokenizer=tokenizer_cfg,
        inr=inr_cfg,
        n_groups=encoder_n_groups,
        transformer=transformer_cfg,
        update_strategy=encoder_update_strat,
        in_channels=channels,
        img_size=data_shape if is_3d else img_size,  # Pass full shape tuple
    )
    weight_dim = encoder.modulation_dim
    encoder_params = print_encoder_stats(encoder)

    # ── Noise Predictor ───────────────────────────────────────────────────────
    noise_predictor_type = getattr(args, "noise_predictor_type", "transinr").lower()
    noise_predictor_depth = getattr(args, "noise_predictor_depth", 4)
    noise_predictor_dim = getattr(args, "noise_predictor_dim", 256)
    noise_predictor_n_head = getattr(args, "noise_predictor_n_head", 8)
    dropout = getattr(args, "dropout", 0.0)

    if noise_predictor_type == "transinr":
        network = TransInrNoisePredictor(
            weight_dim=weight_dim,
            dim=noise_predictor_dim,
            depth=noise_predictor_depth,
            n_head=noise_predictor_n_head,
            head_dim=getattr(args, "noise_predictor_head_dim", 32),
            ff_dim=getattr(args, "noise_predictor_ff_dim", 1024),
            chunk_size=getattr(args, "noise_predictor_chunk_size", 64),
            t_embed_dim=getattr(args, "noise_predictor_t_embed", 128),
            dropout=dropout,
        )
    elif noise_predictor_type == "paramdit":
        # encoder._param_shapes stores (in_dim+1, out_dim); ParamDiT expects (out_dim, in_dim+1)
        from src.models.latent_diffusion.modules.param_dit import ParamDiT

        param_shapes = {
            name: (shape[1], shape[0])
            for name, shape in encoder.modulation_shapes.items()
        }
        network = ParamDiT(
            param_shapes=param_shapes,
            hidden_dim=noise_predictor_dim,
            depth=noise_predictor_depth,
            num_heads=noise_predictor_n_head,
            mlp_ratio=getattr(args, "noise_predictor_mlp_ratio", 4.0),
            dropout=dropout,
            time_dim=getattr(args, "noise_predictor_t_embed", 128),
            tokenizer=getattr(args, "paramdit_tokenizer", "column"),
            tokens_per_tensor=getattr(args, "paramdit_tokens_per_tensor", 1),
            chunk_size=getattr(args, "paramdit_chunk_size", None),
        )
    else:
        raise ValueError(
            f"Unknown noise_predictor_type '{noise_predictor_type}'. "
            "Expected one of: transinr, paramdit."
        )

    noise_predictor_params = print_noise_predictor_stats(network)

    # ── Coordinate grid ───────────────────────────────────────────────────────
    coord_grid = make_coord_grid(data_shape, (-1, 1))

    # ── Assemble ──────────────────────────────────────────────────────────────
    model = WeightDiffusion(
        NoisePredictor=network,
        WeightEncoder=encoder,
        coord_grid=coord_grid,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        sigma_tilde_factor=args.sigma_tilde,
        data_dim=data_dim,
        img_size=data_shape if is_3d else img_size,  # Updated to tuple
        stop_gradient_flow=args.stop_gradient_flow,
        normalize=args.normalize,
        is_3d=is_3d,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("\n########## Total Parameter Summary: ##############")
    print(f"Noise Predictor Type : {noise_predictor_type.upper()}")
    print("Weight Encoder  : ", f"{encoder_params:,}")
    print("Noise Predictor : ", f"{noise_predictor_params:,}")
    print("----------------------------------------------")
    print(f"TOTAL PARAMETERS  : {total_params:,}")
    print("####################################################\n")
    return model
