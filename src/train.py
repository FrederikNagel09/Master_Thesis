"""
train.py
- Main training script

###### Basic INR Training ######
python src/train.py \
    --model basic_inr \
    --epochs 30 \
    --index 5 \
    --name test \
    --batch_size 32 \
    --lr 1e-4 \
    --h1 32 \
    --h2 32 \
    --h3 32 \
    --dataset mnist \
    --omega_0 20.0 

    
###### MLP Hypernet INR Training ######    
python src/train.py \
    --model hypernet_inr \
    --name full_train_40E \
    --epochs 40 \
    --batch_size 32 \
    --lr 1e-4 \
    --h1 32 \
    --h2 32 \
    --h3 32 \
    --dataset mnist \
    --omega_0 20.0 \
    --hyper_h 256 \
    --subset_frac 1

    ###### NDM Training ######
python src/train.py \
    --model ndm \
    --name quick_test \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4 \
    --T 500 \
    --fphi_ch 16 \
    --denoiser_ch 32 \
    --time_emb_dim 128 \
    --sample_every 4 \
    --subset_frac 0.5

    ###### VAE Training ######
python src/train.py \
    --model vae \
    --name vae_MoG \
    --prior mog \
    --epochs 75 \
    --batch_size 128 \
    --lr 1e-3 \
    --latent_dim 16 \
    --device mps \
    --hidden_dims 256 512 256 \
    --subset_frac 1

python src/train.py \
    --model vae_inr_hypernet \
    --name inr_vae_mog \
    --epochs 5 \
    --prior gaussian \
    --batch_size 64 \
    --lr 1e-3 \
    --latent_dim 128\
    --inr_hidden_dim 64 \
    --inr_layers 3 \
    --inr_out_dim 1 \
    --vae_enc_dim 512 \
    --vae_dec_dim 512 \
    --device mps \
    --subset_frac 0.1 

"""

import sys

sys.path.append(".")

from src.utils.parser_utils import parse_args_training
from src.utils.run_training_utils import (
    run_inr_vae_training,
    run_training_inr_mlp_hypernet,
    run_training_ndm,
    run_training_siren_inr,
    run_vae_training,
)

if __name__ == "__main__":
    args = parse_args_training()

    if args.model == "basic_inr":
        run_training_siren_inr(args)
    elif args.model == "hypernet_inr":
        run_training_inr_mlp_hypernet(args)
    elif args.model == "ndm":
        run_training_ndm(args)
    elif args.model == "vae":
        run_vae_training(args)
    elif args.model == "vae_inr_hypernet":
        run_inr_vae_training(args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
