#!/bin/bash
#BSUB -J Latent-Diffusion-CIFAR10      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1440                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/Latent-Diffusion-CIFAR10.out                        # Standard output redirection
#BSUB -e src/outputs/Latent-Diffusion-CIFAR10.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py\
    --run_name Latent-Diffusion-CIFAR10 \
    --epochs 200 \
    --batch_size 64 \
    --subset_frac 1.0 \
    --normalize False \
    --do_scaling False \
    --do_latent_recon False \
    --probablistic True \
    --stop_gradient_flow False \
    --n_fid_samples 16 \
    --model latent_inr_diffusion \
    --dataset cifar10 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 100 \
    --lambda_kl 1.0 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --inr_hidden_dim 256 \
    --inr_layers 5 \
    --latent_dim 32 \
    --latent_size 16 \
    --latent_patch_size 2 \
    --latent_enc_hidden_dim 42 \
    --pred_d_model 512 \
    --pred_n_heads 8 \
    --pred_n_layers 6 \
    --pred_d_ff 2048 \
    --pred_t_embed_dim 512 \
    --dec_trans_dim 256 \
    --dec_trans_n_head 8 \
    --dec_trans_head_dim 64 \
    --dec_trans_ff_dim 2048 \
    --dec_trans_enc_depth 6 \
    --dec_trans_dec_depth 6 \
    --dec_trans_n_groups 64 \
    --dec_trans_update_strategy scale \
    --resume /zhome/66/4/156534/Master_Thesis/src/train_results/Latent-Diffusion-CIFAR10/weights/weights.pt