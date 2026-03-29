"""
config_hyperparameters.py
Defines which hyperparameter sections exist and which models use which sections.

To add a new model:
    1. Add any new arg keys to the relevant section in SECTIONS (or create a new section).
    2. Add the model name and its list of section names to MODEL_SECTIONS.

To add a new hyperparameter:
    1. Add it to the appropriate section in SECTIONS.
    2. Make sure it exists as an arg in get_default_parser() in run_training.py.
"""

# =============================================================================
# All available sections and the arg keys they contain
# =============================================================================

SECTIONS: dict[str, list[str]] = {
    "training": [
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "subset_frac",
        "single_class",
        "single_class_label",
        "log_every_n_steps",
    ],
    "scheduler": [
        "use_scheduler",
        "warmup_steps",
        "peak_lr",
    ],
    "diffusion": [
        "T",
        "beta_1",
        "beta_T",
        "sigma_tilde",
        "prior_scaling",  # NDM prior scaling factor
    ],
    "inr": [
        "inr_hidden_dim",
        "inr_layers",
        "use_modulation",
    ],
    "vae": [
        "latent_dim",
        "vae_enc_dim",
        "vae_dec_dim",
        "prior",
    ],
    "f_phi": [
        "f_phi_type",
        "f_phi_hidden",
        "f_phi_t_embed",
        "base_channels",
    ],
    "noise_predictor": [
        "noise_hidden_dim",
        "noise_n_blocks",
        "noise_t_embed",
    ],
    "model_arch": [
        # Attention UNet (NDM)
        "use_attention_unet",
        "num_res_blocks",
        "channel_mult",
        "num_heads",
        "num_heads_channels",
        "attention_resolutions",
        # NDM-INR encoder / predictor variants
        "ndm_variant",
        "encoder_variant",
        "predictor_variant",
        # Transformer noise predictor
        "transformer_chunk_size",
        "transformer_d_model",
        "transformer_n_heads",
        "transformer_n_layers",
        "transformer_d_ff",
        "transformer_dropout",
        # CNN encoder
        "cnn_base_ch",
        "cnn_n_blocks",
    ],
}

# =============================================================================
# Which sections each model uses
# "training" should always be included for every model
# =============================================================================

MODEL_SECTIONS: dict[str, list[str]] = {
    "ndm": ["training", "scheduler", "diffusion", "f_phi", "model_arch"],
    "inr_vae": ["training", "inr", "vae"],
    "ndm_inr": ["training", "scheduler", "diffusion", "inr", "f_phi", "noise_predictor", "model_arch"],
}
