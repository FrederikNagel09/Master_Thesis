#!/bin/bash
#BSUB -J vae-inr-cifar10-v100                    # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 800                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=1GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/vae-inr-cifar10-v100.out                        # Standard output redirection
#BSUB -e src/outputs/vae-inr-cifar10-v100.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/src/scripts/VAE_Baseline_Training.py \
    --run_name vae-inr-cifar10-v100 \
    --dataset cifar10 \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 1.0 \
    --lambda_kl_max 1.0 \
    --kl_warmup_frac 0.4 \
    --latent_dim 64 \
    --latent_size 12 \
    --latent_patch_size 2 \
    --latent_enc_hidden_dim 16\
    --dec_trans_dim 256 \
    --dec_trans_n_head 8 \
    --dec_trans_head_dim 128 \
    --dec_trans_ff_dim 1024 \
    --dec_trans_enc_depth 4 \
    --dec_trans_dec_depth 4 \
    --dec_trans_n_groups 1 \
    --dec_trans_update_strategy scale \
    --inr_hidden_dim 256 \
    --inr_layers 5