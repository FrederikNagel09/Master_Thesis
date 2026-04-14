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
    --num_heads 4 \                      # number of attention heads (UNet)
    --num_head_channels 64 \             # channels per attention head (UNet)
    --num_res_blocks 2 \                 # number of residual blocks per UNet level
    --attention_resolutions 4 \          # which UNet levels to apply attention at (e.g. 4 = 1/16 resolution)
    --channel_mult 1 2 4 \               # channel multiplier for each UNet level (e.g. 1 2 4 = [base, 2*base, 4*base])
    --use_attention_unet True\           # flag: use UNet architecture for F_phi in NDM (default MLP)
    --use_modulation False  \            # flag: use learnable base modulation in VAE-INR decoder and NDM-INR      
    --ndm_variant temporal \              # ndm_inr variant: 'temporal' or 'static' (only relevant if --model ndm_inr)
    --deactivate_progress_bar \              # flag: disable tqdm progress bar (e.g. for logging to file)
    --resume src/train_results/.../weights/weights.pt  # path to checkpoint (omit to train from scratch)


######################### NDM Training ########################################
python main.py \
    --run_name ndm_unet_mnist_no_scaling \
    --model ndm \
    --dataset mnist \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --prior_scaling 1.0 \
    --grad_clip 1.0 \
    --log_every_n_steps 50 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 30000 \
    --peak_lr 2e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --f_phi_type unet \
    --f_phi_t_embed 128 \
    --base_channels 32 

python main.py \
    --run_name ndm_attention_CIFAR_full \
    --model ndm \
    --dataset cifar10 \
    --epochs 1000 \
    --batch_size 128 \
    --lr 4e-4 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --log_every_n_steps 50 \
    --prior_scaling 1.0 \
    --use_scheduler \
    --warmup_steps 45000 \
    --peak_lr 4e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --base_channels 256 \
    --use_attention_unet \
    --num_res_blocks 2 \
    --channel_mult 1 2 2 2 \
    --num_heads 4 \
    --num_heads_channels 64 \
    --attention_resolutions 16

    --resume src/train_results/ndm_attention_CIFAR_full/.../weights/weights.pt    

######################### INR-VAE Training ####################################
python main.py \
    --run_name vae_inr_mnist_modulation \
    --model inr_vae \
    --dataset mnist \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --log_every_n_steps 10 \
    --subset_frac 1.0 \
    --latent_dim 128 \
    --prior gaussian \
    --vae_enc_dim 512 \
    --vae_dec_dim 512 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --use_modulation true




######################### TRANSFORMER ENCODER NDM-INR Training ####################################
python main.py \
    --run_name ndm_static_transinr_v1_new  \
    --model ndm_static_transinr\
    --dataset mnist \
    --epochs 20 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 50 \
    --subset_frac 1.0 \
    --peak_lr 1e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 4 \
    --trans_dim 256 \
    --trans_n_head 4 \
    --trans_head_dim 32 \
    --trans_ff_dim 256 \
    --trans_enc_depth 6 \
    --trans_dec_depth 6 \
    --trans_patch_size 4 \
    --trans_n_groups 16 \
    --trans_update_strategy scale \
    --predictor_variant transformer \
    --transformer_chunk_size 128 \
    --transformer_d_model 256 \
    --transformer_n_heads 8 \
    --transformer_n_layers 8 \
    --transformer_d_ff 256 \
    --transformer_dropout 0.1 \
    --noise_t_embed 256


    
python main.py \
    --run_name ndm_static_transinr_v1_new  \
    --model ndm_static_transinr\
    --dataset mnist \
    --epochs 20 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 50 \
    --subset_frac 1.0 \
    --peak_lr 1e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 4 \
    --trans_dim 256 \
    --trans_n_head 4 \
    --trans_head_dim 32 \
    --trans_ff_dim 256 \
    --trans_enc_depth 6 \
    --trans_dec_depth 6 \
    --trans_patch_size 4 \
    --trans_n_groups 16 \
    --trans_update_strategy scale \
    --predictor_variant mlp \
    --noise_hidden_dim 256 \
    --noise_n_blocks 6 \
    --noise_t_embed 512 

"""

import warnings

warnings.filterwarnings("ignore")
import sys  # noqa: E402

sys.path.append(".")

from src.utility.general import _get_device  # noqa: E402
from src.utility.parser_util import get_default_parser  # noqa: E402
from src.utility.run_training import run_training  # noqa: E402


def main():
    parser = get_default_parser()
    args = parser.parse_args()

    # Use best available device if not explicitly set
    if args.device is None:
        args.device = _get_device()

    run_training(args)


if __name__ == "__main__":
    main()
