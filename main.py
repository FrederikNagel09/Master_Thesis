"""
main.py
- Main training script

######################### NDM Training ########################################
python main.py \
    --run_name ndm_mlp_mnist \
    --model ndm \
    --dataset mnist \
    --epochs 10 \
    --batch_size 128 \
    --lr 2e-4 \
    --T 1000 \
    --f_phi_type mlp \
    --f_phi_hidden 256 512 256 \
    --base_channels 32 \
    --f_phi_t_embed 16 \
    --sigma_tilde 1.0 \
    --use_scheduler \
    --warmup_steps 50 \
    --log_every_n_steps 20 \
    --subset_frac 0.1

python main.py \
    --run_name ndm_unet_mnist \
    --model ndm \
    --dataset mnist \
    --epochs 10 \
    --batch_size 128 \
    --lr 2e-4 \
    --T 1000 \
    --f_phi_type unet \
    --base_channels 32 \
    --sigma_tilde 1.0 \
    --use_scheduler \
    --warmup_steps 100 \
    --log_every_n_steps 20 \
    --subset_frac 0.1

# Resume a previous run
python main.py \
    --run_name ndm_mlp_mnist \
    --model ndm \
    --dataset mnist \
    --epochs 100 \
    --resume src/train_results/ndm_mlp_mnist/weights.pt \
    --lr 2e-4 \
    --T 1000 \
    --f_phi_type mlp \
    --f_phi_hidden 512 512 512 \
    --f_phi_t_embed 64

######################### INR-VAE Training ####################################
python main.py \
    --run_name inr_vae_cifar10 \
    --model inr_vae \
    --dataset mnist \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --latent_dim 64 \
    --prior gaussian \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --vae_enc_dim 512 \
    --weight_decay 1e-4 \
    --vae_dec_dim 512 \
    --subset_frac 0.25

python main.py \
    --run_name inr_vae_mog_mnist \
    --model inr_vae \
    --dataset mnist \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-3 \
    --latent_dim 128 \
    --prior mog \
    --inr_hidden_dim 64 \
    --inr_layers 3 \
    --vae_enc_dim 512 \
    --vae_dec_dim 512

######################### NDM-INR Training ####################################
python main.py \
    --run_name ndm_inr_mlp_mnist \
    --model ndm_inr \
    --dataset mnist \
    --epochs 10 \
    --batch_size 128 \
    --lr 3e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --f_phi_hidden 512 512 512 \
    --f_phi_t_embed 64 \
    --noise_hidden_dim 512 \
    --noise_n_blocks 4 \
    --noise_t_embed 128 \
    --warmup_steps 50 \
    --grad_clip 1.0 \
    --weight_decay 1e-5 \
    --log_every_n_steps 20 \
    --subset_frac 0.1
"""

import sys

sys.path.append(".")

from src.utility.general import _get_device
from src.utility.parser_util import get_default_parser
from src.utility.run_training import run_training


def main():
    parser = get_default_parser()
    args = parser.parse_args()

    # Use best available device if not explicitly set
    if args.device is None:
        args.device = _get_device()

    run_training(args)


if __name__ == "__main__":
    main()
