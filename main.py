"""
main.py
- Main training script
######################### Training ############################################
python main.py \
    --run_name my_run \                  # required, unique name for this run
    --model ndm \                        # ndm | inr_vae | ndm_inr
    --dataset mnist \                    # mnist | cifar10 | celeba
    --device mps \                       # cuda | mps | cpu (auto-detected if omitted)
    --epochs 100 \                       # number of epochs to train
    --batch_size 128 \                   # batch size
    --lr 1e-4 \                          # learning rate
    --weight_decay 0.0 \                 # L2 regularisation (0 = disabled)
    --grad_clip 1.0 \                    # max gradient norm (0 = disabled)
    --log_every_n_steps 20 \             # how often to log running averages to history
    --subset_frac 1.0 \                  # fraction of dataset to use (1.0 = all)
    --single_class \                     # flag: train on one class only
    --single_class_label 1 \             # which class label to keep (default: 1)
    --num_workers 0 \                    # dataloader worker processes
    --use_scheduler \                    # flag: enable warmup+decay LR schedule
    --warmup_steps 5000 \                # steps for linear LR warmup
    --peak_lr 1e-4 \                     # LR at top of warmup (defaults to lr)
    --T 1000 \                           # diffusion timesteps
    --beta_1 1e-4 \                      # noise schedule start value
    --beta_T 2e-2 \                      # noise schedule end value
    --sigma_tilde 1.0 \                  # 1.0 = stochastic DDPM | 0.0 = DDIM
    --latent_dim 128 \                   # VAE latent space dimension
    --prior gaussian \                   # gaussian | mog
    --vae_enc_dim 512 \                  # VAE encoder hidden dim
    --vae_dec_dim 512 \                  # VAE decoder hidden dim
    --inr_hidden_dim 32 \                # INR hidden layer width
    --inr_layers 3 \                     # INR number of hidden layers
    --f_phi_type mlp \                   # mlp | unet
    --f_phi_hidden 512 512 512 \         # F_phi MLP hidden dims (ignored for unet)
    --f_phi_t_embed 64 \                 # F_phi time embedding dim
    --base_channels 64 \                 # base channels for UNet F_phi / network
    --noise_hidden_dim 512 \             # noise predictor hidden dim
    --noise_n_blocks 4 \                 # noise predictor number of residual blocks
    --noise_t_embed 128 \                # noise predictor time embedding dim
    --resume src/train_results/.../weights/weights.pt  # path to checkpoint (omit to train from scratch)


######################### NDM Training ########################################
python main.py \
    --run_name ndm_unet_mnist \          # required, unique name for this run
    --model ndm \                        # ndm | inr_vae | ndm_inr
    --dataset mnist \                    # mnist | cifar10 | celeba
    --epochs 300 \                       # number of epochs to train
    --batch_size 128 \                   # batch size
    --lr 1e-3 \                          # learning rate
    --weight_decay 0.0 \                 # L2 regularisation (0 = disabled)
    --grad_clip 1.0 \                    # max gradient norm (0 = disabled)
    --log_every_n_steps 50 \             # how often to log to history
    --subset_frac 1.0 \                  # fraction of dataset to use (1.0 = all)
    --use_scheduler \                    # flag: enable warmup+decay LR schedule
    --warmup_steps 30000 \                # steps for linear LR warmup
    --peak_lr 2e-4 \                     # LR at top of warmup (defaults to lr)
    --T 1000 \                           # diffusion timesteps
    --beta_1 1e-4 \                      # noise schedule start
    --beta_T 2e-2 \                      # noise schedule end
    --sigma_tilde 1.0 \                  # 1.0 = stochastic DDPM | 0.0 = DDIM
    --f_phi_type unet \                   # mlp | unet
    --f_phi_t_embed 128 \                 # F_phi time embedding dim
    --base_channels 64 \                 # base channels for UNet F_phi / network

######################### INR-VAE Training ####################################
python main.py \
    --run_name inr_vae_mnist \           # required, unique name for this run
    --model inr_vae \                    # ndm | inr_vae | ndm_inr
    --dataset mnist \                    # mnist | cifar10 | celeba
    --epochs 300 \                       # number of epochs to train
    --batch_size 128 \                   # batch size
    --lr 1e-3 \                          # learning rate
    --weight_decay 0.0 \                 # L2 regularisation (0 = disabled)
    --grad_clip 1.0 \                    # max gradient norm (0 = disabled)
    --log_every_n_steps 50 \             # how often to log to history
    --subset_frac 1.0 \                  # fraction of dataset to use (1.0 = all)
    --latent_dim 128 \                   # VAE latent space dimension
    --prior mog \                   # gaussian | mog
    --vae_enc_dim 512 \                  # encoder hidden dim
    --vae_dec_dim 512 \                  # decoder hidden dim
    --inr_hidden_dim 64 \                # INR hidden layer width
    --inr_layers 3 \                     # INR number of hidden layers

######################### NDM-INR Training ####################################
python main.py \
    --run_name ndm_inr_mnist \           # required, unique name for this run
    --model ndm_inr \                    # ndm | inr_vae | ndm_inr
    --dataset mnist \                    # mnist | cifar10 | celeba
    --epochs 500 \                       # number of epochs to train
    --batch_size 128 \                   # batch size
    --lr 1e-3 \                          # learning rate
    --weight_decay 1e-5 \                # L2 regularisation (0 = disabled)
    --grad_clip 1.0 \                    # max gradient norm (0 = disabled)
    --log_every_n_steps 50 \             # how often to log to history
    --subset_frac 1.0 \                  # fraction of dataset to use (1.0 = all)
    --use_scheduler \                    # flag: enable warmup+decay LR schedule
    --warmup_steps 30000 \                # steps for linear LR warmup
    --peak_lr 1e-3 \                     # LR at top of warmup (defaults to lr)
    --T 1000 \                           # diffusion timesteps
    --beta_1 1e-4 \                      # noise schedule start
    --beta_T 2e-2 \                      # noise schedule end
    --sigma_tilde 1.0 \                  # 1.0 = stochastic DDPM | 0.0 = DDIM
    --inr_hidden_dim 64 \                # INR hidden layer width
    --inr_layers 3 \                     # INR number of hidden layers
    --f_phi_hidden 256 512 512 256 \         # F_phi hidden dims
    --f_phi_t_embed 128 \                 # F_phi time embedding dim
    --noise_hidden_dim 512 \             # noise predictor hidden dim
    --noise_n_blocks 4 \                 # noise predictor number of blocks
    --noise_t_embed 128 \                # noise predictor time embedding dim
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
