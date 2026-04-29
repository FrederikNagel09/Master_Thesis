import argparse

# =============================================================================
# Default argparser (reference / standalone use)
# =============================================================================


def get_default_parser() -> argparse.ArgumentParser:
    """
    Returns a parser with all recognised arguments and their defaults.
    Import this in your main script, add model-specific args, then call
    run_training(parser.parse_args()).
    """
    p = argparse.ArgumentParser(description="Universal training launcher")

    # ── Identity ──────────────────────────────────────────────────────────────
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=("'ndm' | 'inr_vae' | 'ndm_inr' | 'ndm_transinr' | " "'ndm_temporal_transinr' | 'ndm_static_mlpinr'"),
    )
    p.add_argument("--dataset", type=str, default="mnist", help="'mnist' | 'cifar10' | 'celeba'")
    p.add_argument("--device", type=str, default="cuda")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_root", type=str, default="data/")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--subset_frac", type=float, default=1.0)
    p.add_argument("--single_class", action="store_true")
    p.add_argument("--single_class_label", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=20)
    p.add_argument("--resume", type=str, default=None, help="Path to a weights.pt checkpoint to resume from")
    p.add_argument("--deactivate_progress_bar", type=bool, default=False)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=5_000)
    p.add_argument("--peak_lr", type=float, default=None)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_1", type=float, default=1e-4)
    p.add_argument("--beta_T", type=float, default=2e-2)
    p.add_argument("--sigma_tilde", type=float, default=1.0)

    # ── INR ───────────────────────────────────────────────────────────────────
    p.add_argument("--inr_hidden_dim", type=int, default=32)
    p.add_argument("--inr_layers", type=int, default=3)
    p.add_argument("--use_modulation", type=bool, default=False)
    p.add_argument("--omega_0", type=float, default=30.0)

    # ── VAE ───────────────────────────────────────────────────────────────────
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--vae_enc_dim", type=int, default=256)
    p.add_argument("--vae_dec_dim", type=int, default=256)
    p.add_argument("--prior", type=str, default="gaussian")

    # ── F_phi / noise predictor ───────────────────────────────────────────────
    p.add_argument("--f_phi_type", type=str, default="mlp")
    p.add_argument("--f_phi_hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--f_phi_t_embed", type=int, default=64)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--noise_hidden_dim", type=int, default=512)
    p.add_argument("--noise_n_blocks", type=int, default=4)
    p.add_argument("--noise_t_embed", type=int, default=128)

    # ── NDM-INR Transformer / CNN settings ───────────────────────────────────
    p.add_argument("--encoder_variant", default="mlp", choices=["mlp", "cnn", "transformer"])
    p.add_argument("--predictor_variant", default="mlp", choices=["mlp", "transformer"])
    # Predictor transformer args
    p.add_argument("--transformer_chunk_size", type=int, default=32)
    p.add_argument("--transformer_d_model", type=int, default=256)
    p.add_argument("--transformer_n_heads", type=int, default=8)
    p.add_argument("--transformer_n_layers", type=int, default=4)
    p.add_argument("--transformer_d_ff", type=int, default=1024)
    p.add_argument("--transformer_dropout", type=float, default=0.1)
    # CNN encoder args
    p.add_argument("--cnn_base_ch", type=int, default=32)
    p.add_argument("--cnn_n_blocks", type=int, default=4)
    # Transformer encoder args  (enc_ prefix to avoid collision with predictor)
    p.add_argument("--enc_patch_size", type=int, default=4)
    p.add_argument("--enc_embed_dim", type=int, default=128)
    p.add_argument("--enc_n_heads", type=int, default=4)
    p.add_argument("--enc_n_blocks", type=int, default=4)
    p.add_argument("--enc_mlp_ratio", type=float, default=4.0)
    p.add_argument("--enc_dropout", type=float, default=0.1)

    # ─── NDM-INR────────────────────────────────────────────────────────────────
    p.add_argument("--ndm_variant", type=str, default="temporal", choices=["temporal", "static"])

    # ── UNet NDM-specific args ───────────────────────────────────────────────
    p.add_argument("--use_attention_unet", action="store_true")
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--attention_resolutions", type=int, nargs="+", default=[4])
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num_heads_channels", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--prior_scaling", type=float, default=1.0)

    # ── TransInr WeightEncoder args ─────────────────────────────────────────────────
    p.add_argument("--encoder_trans_dim", type=int, default=256)
    p.add_argument("--encoder_trans_n_head", type=int, default=8)
    p.add_argument("--encoder_trans_head_dim", type=int, default=32)
    p.add_argument("--encoder_trans_ff_dim", type=int, default=512)
    p.add_argument("--encoder_trans_enc_depth", type=int, default=4)
    p.add_argument("--encoder_trans_dec_depth", type=int, default=4)
    p.add_argument("--encoder_trans_patch_size", type=int, default=4)
    p.add_argument("--encoder_trans_n_groups", type=int, default=8)
    p.add_argument("--encoder_trans_update_strategy", type=str, default="normalize", choices=["normalize", "scale", "identity"])
    p.add_argument("--encoder_trans_t_embed_dim", type=int, default=128)
    p.add_argument("--encoder_time_freq_dim", type=int, default=128)

    # ── TransInr encoder args ─────────────────────────────────────────────────
    p.add_argument("--noise_predictor_dim", type=int, default=256)
    p.add_argument("--noise_predictor_n_head", type=int, default=8)
    p.add_argument("--noise_predictor_head_dim", type=int, default=32)
    p.add_argument("--noise_predictor_ff_dim", type=int, default=1024)
    p.add_argument("--noise_predictor_t_embed_dim", type=int, default=128)
    p.add_argument("--noise_predictor_depth", type=int, default=4)
    p.add_argument("--noise_predictor_dropout", type=float, default=0.1)

    return p
