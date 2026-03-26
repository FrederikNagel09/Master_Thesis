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

import torch.nn as nn

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
    else:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: 'ndm', 'inr_vae', 'ndm_inr'.")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Model   : {args.model.upper()}  | parameters={total:,}")
    return model


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
    dataset_name = data_config.get("dataset", "mnist").lower()
    C, H, W = shape_map.get(dataset_name, (1, 28, 28))  # noqa: N806

    def _make_attention_unet(identity_constraint: bool) -> nn.Module:
        unet = UNetModel(
            image_size=H,
            in_channels=C,
            model_channels=getattr(args, "base_channels", 32),  # 256
            out_channels=C,
            num_res_blocks=getattr(args, "num_res_blocks", 2),  # 3
            attention_resolutions=getattr(args, "attention_resolutions", (4,)),  # (16,8)
            channel_mult=getattr(args, "channel_mult", (1, 2, 4)),  # (1,2,2,2)
            num_heads=getattr(args, "num_heads", 4),  # 4
            num_head_channels=getattr(args, "num_head_channels", 64),  # 64
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
    from src.models.INR import INR
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
    from src.models.NDM_INR import INR, NeuralDiffusionModelINR, NoisePredictor, WeightEncoder

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    data_dim = data_config["data_dim"]

    # ── INR ───────────────────────────────────────────────────────────────────
    inr = INR(coord_dim=2, hidden_dim=args.inr_hidden_dim, n_hidden=args.inr_layers, out_dim=channels, output_activation="tanh")
    weight_dim = inr.num_weights
    print(f"    INR   : hidden={args.inr_hidden_dim}  layers={args.inr_layers}  out_dim={channels}  weights={weight_dim}")

    # ── Weight encoder F_phi ──────────────────────────────────────────────────
    f_phi = WeightEncoder(
        data_dim=data_dim,
        weight_dim=weight_dim,
        hidden_dims=args.f_phi_hidden,
        t_embed_dim=args.f_phi_t_embed,
    )

    # ── Noise predictor ───────────────────────────────────────────────────────
    network = NoisePredictor(
        weight_dim=weight_dim,
        hidden_dim=args.noise_hidden_dim,
        n_blocks=args.noise_n_blocks,
        t_embed_dim=args.noise_t_embed,
    )

    model = NeuralDiffusionModelINR(
        network=network,
        F_phi=f_phi,
        inr=inr,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        sigma_tilde_factor=args.sigma_tilde,
        data_dim=data_dim,
        img_size=img_size,
        use_modulation=getattr(args, "use_modulation", False),
    )

    fphi_params = sum(p.numel() for p in f_phi.parameters())
    net_params = sum(p.numel() for p in network.parameters())
    print(f"    ε_θ   : {net_params:,}  |  F_phi : {fphi_params:,}")
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
