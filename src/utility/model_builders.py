"""
model_builder.py
Builds and returns a model given a model name, args, and data_config.

data_config is the dict returned by dataset_builder.build_dataset():
    {"channels": int, "img_size": int, "data_dim": int}

Supported model names (args.model):
    "ndm"       - NeuralDiffusionModel with MLP or UNet F_phi
    "inr_vae"   - VAE with INR decoder (hypernetwork)
    "ndm_inr"   - NeuralDiffusionModel with INR reconstruction
"""

import torch.nn as nn  # noqa: I001

from src.models.NDM_INR import (
    INR,
    CNNStaticWeightEncoder,
    CNNTemporalWeightEncoder,
    MLPStaticWeightEncoder,
    MLPTemporalWeightEncoder,
    NDMStaticINR,
    NDMTemporalINR,
    NoisePredictor,
    SirenINR,
    TransformerNoisePredictor,
    TransformerStaticWeightEncoder,
    TransformerTemporalWeightEncoder,
)
from src.models.trans_inr_encoder import TransInrNoisePredictor, TransInrTemporalEncoder
from src.models.trans_inr import make_coord_grid

# =============================================================================
# Public API
# =============================================================================


def build_model(args, data_config: dict) -> nn.Module:
    """
    Instantiate and return the model specified by args.model.

    Parameters
    ----------
    args        : argparse.Namespace with all hyperparameters.
    data_config : Dict from build_dataset() with channels/img_size/data_dim.

    Returns
    -------
    model : Untrainable nn.Module (not yet moved to device).
    """
    name = args.model.lower()

    if name == "ndm":
        model = _build_ndm(args, data_config)
    elif name == "inr_vae":
        model = _build_inr_vae(args, data_config)
    elif name == "ndm_inr":
        model = _build_ndm_inr(args, data_config)
    elif name == "ndm_transinr":
        model = _build_ndm_transinr(args, data_config)
    elif name == "ndm_temporal_transinr":
        model = _build_ndm_temporal_transinr(args, data_config)
    elif name == "ndm_static_transinr":
        model = _build_ndm_static_transinr(args, data_config)
    else:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: 'ndm', 'inr_vae', 'ndm_inr'.")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Model   : {args.model.upper()}  | parameters={total:,}")
    return model


# =============================================================================
# Printing functions:
# =============================================================================
def print_encoder_stats(model, mode="Static"):
    def count(params):
        return sum(p.numel() for p in params)

    # Component counts
    tokenizer_p = count(model.tokenizer.parameters())
    transformer_p = count(model.transformer.parameters())
    base_p = count(model.base_params.values())
    wtoken_p = model.wtokens.numel()
    postfc_p = count(model.wtoken_postfc.parameters())

    if mode == "Temporal":
        time_embedding = count(model.time_mlp.parameters())
        total_learnable = tokenizer_p + transformer_p + base_p + wtoken_p + postfc_p + time_embedding
    else:
        total_learnable = tokenizer_p + transformer_p + base_p + wtoken_p + postfc_p
    inr_total = sum(s[0] * s[1] for s in model.inr.param_shapes.values())

    print("\n" + "=" * 60)
    print(f"{'TransInrEncoder WeightEncoder Statistics':^60}")
    print("=" * 60)
    print(f"Architecture: {model.transformer.__class__.__name__}")
    print(f"Weight Dim (Total INR params): {model.weight_dim:,}")
    print(f"Encoder Mode: {mode}")
    print("-" * 60)

    print("Learnable Parameters:")
    print(f"  Vision Tokenizer:     {tokenizer_p:>12,} params")
    print(f"  Main Transformer:     {transformer_p:>12,} params")
    print(f"  Weight Tokens (N_w):  {wtoken_p:>12,} params")
    print(f"  Base INR Weights:     {base_p:>12,} params")
    print(f"  Wtoken Post-FC:       {postfc_p:>12,} params")
    if mode == "Temporal":
        print(f"  Time MLP:            {time_embedding:>12,} params")
    print(f"  {'─'*44}")
    print(f"  Total Learnable:      {total_learnable:>12,} params")

    print("\nGenerated INR Runtime Structure (Non-Learnable Output):")
    for name, shape in model.inr.param_shapes.items():
        l, r = model.wtoken_rng[name]  # noqa: E741
        n_groups = r - l
        print(f"  {name:<12} {shape[0]}x{shape[1]:<6} | Groups: {n_groups:<3} | Total: {shape[0]*shape[1]:>10,} weights")
    print(f"Total INR Weights: {inr_total:,}")
    print("=" * 60 + "\n")


def print_noise_predictor_stats(model):
    def count(params):
        return sum(p.numel() for p in params)

    total = count(model.parameters())

    print("\n" + "=" * 60)
    print(f"{'TransInrNoise Predictor ε_θ Statistics':^60}")
    print(f"{'(Encoder-Only DiT Architecture)':^60}")
    print("=" * 60)

    if isinstance(model, TransInrNoisePredictor):
        # Updated to match new attribute names
        time_p = count(model.time_embed.parameters()) + count(model.time_mlp.parameters())
        token_p = count(model.token_embed.parameters())
        pos_p = model.pos_embed.numel()
        transformer_p = count(model.transformer.parameters())
        head_p = count(model.noise_head.parameters())

        print(f"  Weight Dim:   {model.weight_dim:<10} | Chunk Size: {model.chunk_size}")
        print(f"  Num Tokens:   {model.n_tokens:<10} | Padded Dim: {model.padded_dim}")
        print("-" * 60)
        print("Learnable Parameters:")
        print(f"  Time Conditioning (MLP): {time_p:>12,} params")
        print(f"  Token Embedding:         {token_p:>12,} params")
        print(f"  Positional Embedding:    {pos_p:>12,} params")
        print(f"  Transformer Blocks:      {transformer_p:>12,} params")
        print(f"  Noise Prediction Head:   {head_p:>12,} params")
        print(f"  {'─'*44}")
        print(f"  Total Predictor:         {total:>12,} params")
    else:
        print("  Generic or Legacy Noise Predictor detected.")
        print(f"  Total parameters: {total:,}")

    print("=" * 60 + "\n")


# =============================================================================
# Per-model builders
# =============================================================================


def _build_ndm(args, data_config: dict) -> nn.Module:
    from src.models.NDM import MLPTransformation, NeuralDiffusionModel, UnetNDM, UNetTransformation, WrappedUNetModel
    from src.models.ndm_unet_module import UNetModel  # adjust import path as needed

    data_dim = data_config["data_dim"]
    use_attention_unet = getattr(args, "use_attention_unet", False)
    shape_map = {
        "mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
    }
    dataset_name = args.dataset.lower()
    C, H, W = shape_map.get(dataset_name, (1, 28, 28))  # noqa: N806
    print(f"  Dataset : {dataset_name.upper()}  | shape=({C},{H},{W})  | data_dim={data_dim}")

    def _make_attention_unet(identity_constraint: bool) -> nn.Module:
        unet = UNetModel(
            image_size=H,
            in_channels=C,
            model_channels=getattr(args, "base_channels", 32),  # 256
            out_channels=C,
            num_res_blocks=getattr(args, "num_res_blocks", 2),  # 3
            attention_resolutions=getattr(args, "attention_resolutions", (3,)),  # (16,8)
            channel_mult=getattr(args, "channel_mult", (1, 2, 2)),  # (1,2,2,2)
            num_heads=getattr(args, "num_heads", 3),  # 4
            num_head_channels=getattr(args, "num_heads_channels", 16),  # 64
            dims=2,
        )
        return WrappedUNetModel(unet, C=C, H=H, W=W, identity_constraint=identity_constraint)

    # ── F_phi ────────────────────────────────────────────────────────────────
    if use_attention_unet:
        f_phi = _make_attention_unet(identity_constraint=True)
        print("    F_phi : Attention UNet (wrapped)")
    elif args.f_phi_type == "mlp":
        f_phi = MLPTransformation(
            data_dim=data_dim,
            hidden_dims=args.f_phi_hidden,
            t_embed_dim=args.f_phi_t_embed,
        )
        print(f"    F_phi : MLP  hidden={args.f_phi_hidden}  t_embed={args.f_phi_t_embed}")
    elif args.f_phi_type == "unet":
        f_phi = UNetTransformation(
            data_dim=data_dim,
            base_channels=getattr(args, "base_channels", 32),
        )
        print("    F_phi : UNet")
    else:
        raise ValueError(f"Unknown f_phi_type '{args.f_phi_type}'. Choose 'mlp' or 'unet'.")

    # ── Noise predictor ───────────────────────────────────────────────────────
    if use_attention_unet:
        network = _make_attention_unet(identity_constraint=False)
        print("    ε_θ   : Attention UNet (wrapped)")
    else:
        network = UnetNDM(data_dim=data_dim, base_channels=args.base_channels)

    model = NeuralDiffusionModel(
        network=network,
        F_phi=f_phi,
        T=args.T,
        sigma_tilde_factor=args.sigma_tilde,
        data_dim=data_dim,
        prior_scaling=getattr(args, "prior_scaling", 1.0),
    )

    fphi_params = sum(p.numel() for p in f_phi.parameters())
    net_params = sum(p.numel() for p in network.parameters())
    print(f"    ε_θ   : {net_params:,}  |  F_phi : {fphi_params:,}")
    return model


def _build_inr_vae(args, data_config: dict) -> nn.Module:
    from src.models.VAE_INR import VAEINR
    from src.models.vae_modules import GaussianEncoder, GaussianPrior, MoGPrior

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    flat_dim = channels * img_size * img_size

    # ── INR ───────────────────────────────────────────────────────────────────
    inr = INR(
        coord_dim=2,
        hidden_dim=args.inr_hidden_dim,
        n_hidden=args.inr_layers,
        out_dim=channels,  # 1 for greyscale, 3 for RGB
        output_activation="sigmoid",
    )
    print(f"    INR   : hidden={args.inr_hidden_dim}  layers={args.inr_layers}  out_dim={channels}  weights={inr.num_weights}")

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder_net = nn.Sequential(
        nn.Linear(flat_dim, args.vae_enc_dim),
        nn.ReLU(),
        nn.Linear(args.vae_enc_dim, args.vae_enc_dim),
        nn.ReLU(),
        nn.Linear(args.vae_enc_dim, args.vae_enc_dim),
        nn.ReLU(),
        nn.Linear(args.vae_enc_dim, args.latent_dim * 2),
    )
    encoder = GaussianEncoder(encoder_net)

    # ── Decoder (hypernetwork) ────────────────────────────────────────────────
    decoder_net = nn.Sequential(
        nn.Linear(args.latent_dim, args.vae_dec_dim),
        nn.ReLU(),
        nn.Linear(args.vae_dec_dim, args.vae_dec_dim),
        nn.ReLU(),
        nn.Linear(args.vae_dec_dim, args.vae_dec_dim),
        nn.ReLU(),
        nn.Linear(args.vae_dec_dim, inr.num_weights),
    )
    nn.init.zeros_(decoder_net[-1].bias)
    nn.init.normal_(decoder_net[-1].weight, std=0.01)

    # ── Prior ─────────────────────────────────────────────────────────────────
    if args.prior == "gaussian":
        prior = GaussianPrior(latent_dim=args.latent_dim)
    elif args.prior == "mog":
        prior = MoGPrior(latent_dim=args.latent_dim)
    else:
        raise ValueError(f"Unknown prior '{args.prior}'. Choose 'gaussian' or 'mog'.")
    print("MODULATION: ", getattr(args, "use_modulation", False))
    model = VAEINR(
        prior,
        encoder,
        decoder_net,
        inr,
        beta=1.0,
        prior_type=args.prior,
        use_modulation=getattr(args, "use_modulation", False),
    )

    return model


def _build_ndm_inr(args, data_config: dict) -> nn.Module:
    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]

    # ── INR ───────────────────────────────────────────────────────────────────
    inr = SirenINR(
        coord_dim=2,
        hidden_dim=args.inr_hidden_dim,
        n_hidden=args.inr_layers,
        out_dim=channels,
        output_activation="tanh",
        omega_0=args.omega_0,
    )
    weight_dim = inr.num_weights
    print(f"  Siren INR   : hidden={args.inr_hidden_dim}  layers={args.inr_layers}  out_dim={channels}  weights={weight_dim}")

    # ── Noise predictor ───────────────────────────────────────────────────────
    network = build_noise_predictor(
        variant=getattr(args, "predictor_variant", "mlp"),
        weight_dim=weight_dim,
        hidden_dim=args.noise_hidden_dim,
        n_blocks=args.noise_n_blocks,
        t_embed_dim=args.noise_t_embed,
        chunk_size=getattr(args, "transformer_chunk_size", 32),
        d_model=getattr(args, "transformer_d_model", 256),
        n_heads=getattr(args, "transformer_n_heads", 8),
        n_layers=getattr(args, "transformer_n_layers", 4),
        d_ff=getattr(args, "transformer_d_ff", 1024),
        dropout=getattr(args, "transformer_dropout", 0.1),
    )

    # ── Weight encoder + model — chosen by args.ndm_variant ──────────────────
    use_static = getattr(args, "ndm_variant", "temporal") == "static"

    encoder = build_encoder(
        variant=getattr(args, "encoder_variant", "mlp"),
        temporal=not use_static,
        data_dim=data_dim,
        weight_dim=weight_dim,
        img_size=img_size,
        channels=channels,
        hidden_dims=getattr(args, "f_phi_hidden", [128, 128]),
        t_embed_dim=getattr(args, "f_phi_t_embed", 32),
        base_ch=getattr(args, "cnn_base_ch", 32),
        n_blocks=getattr(args, "cnn_n_blocks", 4),
        enc_patch_size=getattr(args, "enc_patch_size", 4),
        enc_embed_dim=getattr(args, "enc_embed_dim", 128),
        enc_n_heads=getattr(args, "enc_n_heads", 4),
        enc_n_blocks=getattr(args, "enc_n_blocks", 4),
        enc_mlp_ratio=getattr(args, "enc_mlp_ratio", 4.0),
        enc_dropout=getattr(args, "enc_dropout", 0.1),
        noise_t_embed=getattr(args, "noise_t_embed", 16),  # shared with predictor
    )

    if use_static:
        model = NDMStaticINR(
            network=network,
            W=encoder,
            inr=inr,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            T=args.T,
            sigma_tilde_factor=args.sigma_tilde,
            data_dim=data_dim,
            img_size=img_size,
            use_modulation=getattr(args, "use_modulation", False),
        )
        variant_label = "Static  W(x)"
    else:
        model = NDMTemporalINR(
            network=network,
            F_phi=encoder,
            inr=inr,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            T=args.T,
            sigma_tilde_factor=args.sigma_tilde,
            data_dim=data_dim,
            img_size=img_size,
            use_modulation=getattr(args, "use_modulation", False),
        )
        variant_label = "Temporal  F_phi(x,t)"

    encoder_params = sum(p.numel() for p in encoder.parameters())
    net_params = sum(p.numel() for p in network.parameters())
    print(f"    Variant   : {variant_label}")
    print(f"    Encoder   : {getattr(args, 'encoder_variant', 'mlp')}  |  Predictor : {getattr(args, 'predictor_variant', 'mlp')}")
    print(f"    ε_θ       : {net_params:,}  |  Encoder : {encoder_params:,}")
    return model


# =============================================================================
# Factory functions
# =============================================================================
def build_encoder(
    variant: str,
    data_dim: int,
    weight_dim: int,
    img_size: int,
    channels: int,
    temporal: bool,
    # MLP kwargs
    hidden_dims: list = None,  # noqa: RUF013
    t_embed_dim: int = 64,
    # CNN kwargs
    base_ch: int = 32,
    n_blocks: int = 4,
    # Transformer kwargs
    enc_patch_size: int = 4,
    enc_embed_dim: int = 128,
    enc_n_heads: int = 4,
    enc_n_blocks: int = 4,
    enc_mlp_ratio: float = 4.0,
    enc_dropout: float = 0.1,
    noise_t_embed: int = 128,  # shared with predictor
) -> nn.Module:
    """
    Build a weight encoder.

    Parameters
    ----------
    variant   : "mlp" or "cnn"
    temporal  : if True build the time-dependent F_phi(x,t) version,
                otherwise the time-independent W(x) version
    """
    if variant == "mlp":
        if temporal:
            print("\nBuilding MLP Temporal Weight Encoder F_phi(x,t) …")
            print(f"  data_dim={data_dim}  weight_dim={weight_dim}  hidden_dims={hidden_dims}  t_embed_dim={t_embed_dim}")
            return MLPTemporalWeightEncoder(
                data_dim=data_dim,
                weight_dim=weight_dim,
                hidden_dims=hidden_dims,
                t_embed_dim=t_embed_dim,
            )
        else:
            print("Building MLP Static Weight Encoder W(x) …")
            print(f"  data_dim={data_dim}  weight_dim={weight_dim}  hidden_dims={hidden_dims}")
            return MLPStaticWeightEncoder(
                data_dim=data_dim,
                weight_dim=weight_dim,
                hidden_dims=hidden_dims,
            )
    elif variant == "cnn":
        if temporal:
            return CNNTemporalWeightEncoder(
                data_dim=data_dim,
                img_size=img_size,
                channels=channels,
                weight_dim=weight_dim,
                base_ch=base_ch,
                n_blocks=n_blocks,
                t_embed_dim=t_embed_dim,
            )
        else:
            return CNNStaticWeightEncoder(
                data_dim=data_dim,
                img_size=img_size,
                channels=channels,
                weight_dim=weight_dim,
                base_ch=base_ch,
                n_blocks=n_blocks,
            )
    elif variant == "transformer":
        if temporal:
            return TransformerTemporalWeightEncoder(
                data_dim=data_dim,
                img_size=img_size,
                channels=channels,
                weight_dim=weight_dim,
                patch_size=enc_patch_size,
                embed_dim=enc_embed_dim,
                n_heads=enc_n_heads,
                n_blocks=enc_n_blocks,
                mlp_ratio=enc_mlp_ratio,
                t_embed_dim=noise_t_embed,  # shared with predictor
                dropout=enc_dropout,
            )
        else:
            return TransformerStaticWeightEncoder(
                data_dim=data_dim,
                img_size=img_size,
                channels=channels,
                weight_dim=weight_dim,
                patch_size=enc_patch_size,
                embed_dim=enc_embed_dim,
                n_heads=enc_n_heads,
                n_blocks=enc_n_blocks,
                mlp_ratio=enc_mlp_ratio,
                dropout=enc_dropout,
            )
    else:
        raise ValueError(f"Unknown encoder variant '{variant}'. Choose 'mlp' or 'cnn'.")


def build_noise_predictor(
    variant: str,
    weight_dim: int,
    # MLP kwargs
    hidden_dim: int = 512,
    n_blocks: int = 4,
    t_embed_dim: int = 128,
    # Transformer kwargs
    chunk_size: int = 32,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 1024,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Build a noise predictor.

    Parameters
    ----------
    variant : "mlp" or "transformer"
    """
    if variant == "mlp":
        print("\nBuilding MLP Noise Predictor ε_θ(x,t) …")
        print(f"  weight_dim={weight_dim}  hidden_dim={hidden_dim}  n_blocks={n_blocks}  t_embed_dim={t_embed_dim}")
        return NoisePredictor(
            weight_dim=weight_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            t_embed_dim=t_embed_dim,
        )
    elif variant == "transformer":
        return TransformerNoisePredictor(
            weight_dim=weight_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            t_embed_dim=t_embed_dim,
        )
    else:
        raise ValueError(f"Unknown noise predictor variant '{variant}'. Choose 'mlp' or 'transformer'.")


def _build_ndm_transinr(args, data_config: dict):
    """
    Build NDMTransInr:  TransInrEncoder as W(x) + NDM diffusion in weight space.

    Required args fields
    --------------------
    # TransInr / SIREN
    trans_dim            : int   transformer embedding dim       (e.g. 256)
    trans_n_head         : int   attention heads                 (e.g. 8)
    trans_head_dim       : int   dims per head                   (e.g. 32)
    trans_ff_dim         : int   feedforward dim                 (e.g. 512)
    trans_enc_depth      : int   encoder depth                   (e.g. 4)
    trans_dec_depth      : int   decoder depth                   (e.g. 4)
    trans_patch_size     : int   patch size for ImageTokenizer   (e.g. 4)
    trans_n_groups       : int   wtoken groups                   (e.g. 8)
    trans_update_strategy: str   "normalize" | "scale" | "identity"
    inr_hidden_dim       : int   SIREN hidden dim                (e.g. 256)
    inr_layers           : int   SIREN depth (total layers)      (e.g. 5)

    # Noise predictor
    predictor_variant    : str   "mlp" | "transformer"
    noise_hidden_dim     : int
    noise_n_blocks       : int
    noise_t_embed        : int
    transformer_chunk_size, transformer_d_model, transformer_n_heads,
    transformer_n_layers, transformer_d_ff, transformer_dropout  (if transformer)

    # Diffusion schedule
    beta_1, beta_T, T, sigma_tilde

    # Dataset (from data_config)
    channels, img_size, data_dim
    """
    from src.models.NDM_INR import NDMTransInr
    from src.models.trans_inr_encoder import TransInrEncoder
    from src.utility.model_builders import build_noise_predictor

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]

    # ── TransInrEncoder config dicts ─────────────────────────────────────────
    dim = getattr(args, "trans_dim", 256)
    n_head = getattr(args, "trans_n_head", 8)
    head_dim = getattr(args, "trans_head_dim", 32)
    ff_dim = getattr(args, "trans_ff_dim", 512)
    enc_depth = getattr(args, "trans_enc_depth", 4)
    dec_depth = getattr(args, "trans_dec_depth", 4)
    patch_size = getattr(args, "trans_patch_size", 4)
    n_groups = getattr(args, "trans_n_groups", 8)
    update_strat = getattr(args, "trans_update_strategy", "normalize")

    inr_hidden = getattr(args, "inr_hidden_dim", 256)
    inr_layers = getattr(args, "inr_layers", 5)

    tokenizer_cfg = {
        "target": "src.models.trans_inr_helpers.ImageTokenizer",
        "params": {
            "in_channels": channels,
            "image_size": img_size,
            "patch_size": patch_size,
            "n_head": n_head,
            "head_dim": head_dim,
            # dim injected by TransInrEncoder via extra_args
        },
    }

    inr_cfg = {
        "target": "src.models.trans_inr_helpers.SIREN",
        "params": {
            "depth": inr_layers,
            "in_dim": 2,
            "out_dim": channels,
            "hidden_dim": inr_hidden,
        },
    }

    transformer_cfg = {
        "target": "src.models.trans_inr_helpers.Transformer",
        "params": {
            "dim": dim,
            "encoder_depth": enc_depth,
            "decoder_depth": dec_depth,
            "n_head": n_head,
            "head_dim": head_dim,
            "ff_dim": ff_dim,
        },
    }

    encoder = TransInrEncoder(
        tokenizer=tokenizer_cfg,
        inr=inr_cfg,
        n_groups=n_groups,
        transformer=transformer_cfg,
        update_strategy=update_strat,
    )
    print_encoder_stats(encoder)  # not model
    weight_dim = encoder.weight_dim

    # ── Noise predictor ───────────────────────────────────────────────────────
    network = build_noise_predictor(
        variant=getattr(args, "predictor_variant", "mlp"),
        weight_dim=weight_dim,
        hidden_dim=getattr(args, "noise_hidden_dim", 512),
        n_blocks=getattr(args, "noise_n_blocks", 4),
        t_embed_dim=getattr(args, "noise_t_embed", 128),
        chunk_size=getattr(args, "transformer_chunk_size", 32),
        d_model=getattr(args, "transformer_d_model", 256),
        n_heads=getattr(args, "transformer_n_heads", 8),
        n_layers=getattr(args, "transformer_n_layers", 4),
        d_ff=getattr(args, "transformer_d_ff", 1024),
        dropout=getattr(args, "transformer_dropout", 0.1),
    )
    print_noise_predictor_stats(network)
    # ── Coordinate grid for SIREN queries ────────────────────────────────────
    # Shape: (img_size, img_size, 2),  range [-1, 1]
    from src.models.trans_inr import make_coord_grid

    coord_grid = make_coord_grid((img_size, img_size), (-1, 1))  # (H, W, 2)

    # ── Assemble model ────────────────────────────────────────────────────────
    model = NDMTransInr(
        network=network,
        encoder=encoder,
        coord_grid=coord_grid,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        sigma_tilde_factor=args.sigma_tilde,
        data_dim=data_dim,
        img_size=img_size,
    )

    return model


def _build_ndm_temporal_transinr(args, data_config: dict):
    """
    Build NDMTemporalTransInr:
        TransInrEncoder as F(x, t) + NDM diffusion in weight space.
    """

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]

    # ── TransInrEncoder config ────────────────────────────────────────────────
    encoder_dim = getattr(args, "encoder_trans_dim", 256)
    encoder_n_head = getattr(args, "encoder_trans_n_head", 8)
    encoder_head_dim = getattr(args, "encoder_trans_head_dim", 32)
    encoder_ff_dim = getattr(args, "encoder_trans_ff_dim", 512)
    encoder_enc_depth = getattr(args, "encoder_trans_enc_depth", 4)
    encoder_dec_depth = getattr(args, "encoder_trans_dec_depth", 4)
    encoder_patch_size = getattr(args, "encoder_trans_patch_size", 4)
    encoder_n_groups = getattr(args, "encoder_trans_n_groups", 8)
    encoder_update_strat = getattr(args, "encoder_trans_update_strategy", "scale")
    inr_hidden = getattr(args, "inr_hidden_dim", 256)
    inr_layers = getattr(args, "inr_layers", 5)

    tokenizer_cfg = {
        "target": "src.models.trans_inr_helpers.ImageTokenizer",
        "params": {
            "in_channels": channels,
            "image_size": img_size,
            "patch_size": encoder_patch_size,
            "n_head": encoder_n_head,
            "head_dim": encoder_head_dim,
        },
    }

    inr_cfg = {
        "target": "src.models.trans_inr_helpers.SIREN",
        "params": {
            "depth": inr_layers,
            "in_dim": 2,
            "out_dim": channels,
            "hidden_dim": inr_hidden,
            "out_bias": 0.5,
        },
    }

    transformer_cfg = {
        "target": "src.models.trans_inr_helpers.Transformer",
        "params": {
            "dim": encoder_dim,
            "encoder_depth": encoder_enc_depth,
            "decoder_depth": encoder_dec_depth,
            "n_head": encoder_n_head,
            "head_dim": encoder_head_dim,
            "ff_dim": encoder_ff_dim,
        },
    }

    encoder = TransInrTemporalEncoder(
        tokenizer=tokenizer_cfg,
        inr=inr_cfg,
        n_groups=encoder_n_groups,
        transformer=transformer_cfg,
        update_strategy=encoder_update_strat,
        in_channels=channels,
        img_size=img_size,
        time_freq_dim=getattr(args, "encoder_time_freq_dim", 16),
    )

    weight_dim = encoder.weight_dim
    print_encoder_stats(encoder, mode="Temporal")  # not model

    # ── TransInrNoisePredictor config ─────────────────────────────────────────
    # Since we are now Encoder-only, we combine the depths or pick the max.
    # Let's combine them to maintain the total layer count.
    noise_predictor_depth = getattr(args, "noise_predictor_depth", 4)

    network = TransInrNoisePredictor(
        weight_dim=weight_dim,
        dim=getattr(args, "noise_predictor_dim", 256),
        depth=noise_predictor_depth,
        n_head=getattr(args, "noise_predictor_n_head", 8),
        head_dim=getattr(args, "noise_predictor_head_dim", 32),
        ff_dim=getattr(args, "noise_predictor_ff_dim", 1024),
        chunk_size=getattr(args, "noise_predictor_chunk_size", 32),
        t_embed_dim=getattr(args, "noise_predictor_t_embed", 128),
        dropout=getattr(args, "dropout", 0.0),
    )

    print_noise_predictor_stats(network)

    from src.models.NDM_TemporalTransInr import NDMTemporalTransInr

    # ── Coordinate grid ───────────────────────────────────────────────────────
    coord_grid = make_coord_grid((img_size, img_size), (-1, 1))  # (H, W, 2)

    # ── Assemble ──────────────────────────────────────────────────────────────
    model = NDMTemporalTransInr(
        NoisePredictor=network,
        WeightEncoder=encoder,
        coord_grid=coord_grid,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        sigma_tilde_factor=args.sigma_tilde,
        data_dim=data_dim,
        img_size=img_size,
    )
    return model


def _build_ndm_static_transinr(args, data_config: dict):
    """
    Build NDMStaticTransInr:
        TransInrEncoder as W(x) + NDM diffusion in weight space.
    """
    from src.models.NDM_StaticTransInr import NDMStaticTransInr
    from src.models.trans_inr_encoder import TransInrEncoder, TransInrNoisePredictor

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]

    # ── TransInrEncoder config ────────────────────────────────────────────────
    encoder_dim = getattr(args, "encoder_trans_dim", 256)
    encoder_n_head = getattr(args, "encoder_trans_n_head", 8)
    encoder_head_dim = getattr(args, "encoder_trans_head_dim", 32)
    encoder_ff_dim = getattr(args, "encoder_trans_ff_dim", 512)
    encoder_enc_depth = getattr(args, "encoder_trans_enc_depth", 4)
    encoder_dec_depth = getattr(args, "encoder_trans_dec_depth", 4)
    encoder_patch_size = getattr(args, "encoder_trans_patch_size", 4)
    encoder_n_groups = getattr(args, "encoder_trans_n_groups", 8)
    encoder_update_strat = getattr(args, "encoder_trans_update_strategy", "scale")
    inr_hidden = getattr(args, "inr_hidden_dim", 256)
    inr_layers = getattr(args, "inr_layers", 5)

    tokenizer_cfg = {
        "target": "src.models.trans_inr_helpers.ImageTokenizer",
        "params": {
            "in_channels": channels,
            "image_size": img_size,
            "patch_size": encoder_patch_size,
            "n_head": encoder_n_head,
            "head_dim": encoder_head_dim,
        },
    }

    inr_cfg = {
        "target": "src.models.trans_inr_helpers.MLP_INR",
        "params": {
            "depth": inr_layers,
            "in_dim": 2,
            "out_dim": channels,
            "hidden_dim": inr_hidden,
            "out_bias": 0.5,
        },
    }

    transformer_cfg = {
        "target": "src.models.trans_inr_helpers.Transformer",
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
        img_size=img_size,
    )

    weight_dim = encoder.weight_dim
    print_encoder_stats(encoder)

    # ── TransInrNoisePredictor config ─────────────────────────────────────────
    # Since we are now Encoder-only, we combine the depths or pick the max.
    # Let's combine them to maintain the total layer count.
    noise_predictor_depth = getattr(args, "noise_predictor_depth", 4)

    network = TransInrNoisePredictor(
        weight_dim=weight_dim,
        dim=getattr(args, "noise_predictor_dim", 256),
        depth=noise_predictor_depth,
        n_head=getattr(args, "noise_predictor_n_head", 8),
        head_dim=getattr(args, "noise_predictor_head_dim", 32),
        ff_dim=getattr(args, "noise_predictor_ff_dim", 1024),
        chunk_size=getattr(args, "noise_predictor_chunk_size", 32),
        t_embed_dim=getattr(args, "noise_predictor_t_embed", 128),
        dropout=getattr(args, "dropout", 0.0),
    )

    print_noise_predictor_stats(network)

    # ── Coordinate grid ───────────────────────────────────────────────────────
    coord_grid = make_coord_grid((img_size, img_size), (-1, 1))  # (H, W, 2)

    # ── Assemble ──────────────────────────────────────────────────────────────
    model = NDMStaticTransInr(
        NoisePredictor=network,
        WeightEncoder=encoder,
        coord_grid=coord_grid,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        sigma_tilde_factor=args.sigma_tilde,
        data_dim=data_dim,
        img_size=img_size,
    )
    return model


if __name__ == "__main__":
    import sys

    sys.path.append(".")

    import types

    import torch

    from src.utility.dataset_builders import build_dataset

    # ── Minimal args for each model ───────────────────────────────────────────
    COMMON = {
        "inr_hidden_dim": 32,
        "inr_layers": 3,
        "latent_dim": 16,
        "vae_enc_dim": 64,
        "vae_dec_dim": 64,
        "prior": "gaussian",
        "f_phi_type": "mlp",
        "f_phi_hidden": [128, 128],
        "f_phi_t_embed": 32,
        "base_channels": 16,
        "noise_hidden_dim": 128,
        "noise_n_blocks": 2,
        "noise_t_embed": 32,
        "T": 100,
        "beta_1": 1e-4,
        "beta_T": 2e-2,
        "sigma_tilde": 1.0,
    }

    MODELS = ["ndm", "inr_vae", "ndm_inr"]

    # Use a tiny MNIST subset for speed
    print("\nLoading dataset …")
    dataset, data_config = build_dataset("mnist", data_root="data/", subset_frac=0.01)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, list | tuple) else batch  # (4, 784)

    all_passed = True

    for model_name in MODELS:
        print(f"\n{'=' * 50}")
        print(f"  Testing model: {model_name.upper()}")
        print(f"{'=' * 50}")
        try:
            args = types.SimpleNamespace(model=model_name, **COMMON)
            model = build_model(args, data_config)
            model.eval()

            with torch.no_grad():
                # ── Forward pass ──────────────────────────────────────────
                if model_name == "inr_vae":
                    # inr_vae expects (image_flat, coords, pixels)
                    img_size = data_config["img_size"]
                    lin = torch.linspace(-1, 1, img_size)
                    gr, gc = torch.meshgrid(lin, lin, indexing="ij")
                    coords = torch.stack([gr.flatten(), gc.flatten()], dim=-1)  # (784, 2)
                    coords = coords.unsqueeze(0).expand(4, -1, -1)  # (4, 784, 2)
                    pixels = x.unsqueeze(-1).expand(-1, -1, 1)  # (4, 784, 1)
                    loss, l_diff, l_prior, l_rec = model(x, coords, pixels)

                else:
                    loss, l_diff, l_prior, l_rec = model.loss(x)

            # ── Checks ────────────────────────────────────────────────────
            checks = {
                "loss is scalar": loss.shape == torch.Size([]),
                "loss is finite": loss.isfinite().item(),
                "l_diff is finite": l_diff.isfinite().item(),
                "l_prior is finite": l_prior.isfinite().item(),
                "l_rec is finite": l_rec.isfinite().item(),
            }

            for desc, ok in checks.items():
                status = "✓" if ok else "✗"
                print(f"  {status} {desc}")
                if not ok:
                    all_passed = False

            print(f"  loss={loss.item():.4f}  diff={l_diff.item():.4f}  prior={l_prior.item():.4f}  rec={l_rec.item():.4f}")

        except Exception as e:
            print(f"  ✗ FAILED with exception: {e}")
            all_passed = False

    print(f"\n{'=' * 50}")
    print(f"  {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print(f"{'=' * 50}\n")
