#!/bin/bash
#BSUB -J Weight-Diffusion-Probabilistic-CIFAR10      # Job name
#BSUB -q gpua100                          # Queue to submit the job to
#BSUB -W 4320                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/Weight-Diffusion-Probabilistic-CIFAR10.out                        # Standard output redirection
#BSUB -e src/outputs/Weight-Diffusion-Probabilistic-CIFAR10.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py\
    --run_name Weight-Diffusion-Probabilistic-CIFAR10 \
    --epochs 500 \
    --batch_size 64 \
    --subset_frac 1.0 \
    --normalize False\
    --probablistic True \
    --stop_gradient_flow True \
    --n_fid_samples 16 \
    --model weight_inr_diffusion\
    --dataset cifar10 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 2 \
    --peak_lr 1e-4 \
    --lambda_kl 1.0 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 256 \
    --inr_layers 5 \
    --encoder_trans_dim 256 \
    --encoder_trans_n_head 8 \
    --encoder_trans_head_dim 64 \
    --encoder_trans_ff_dim 2048 \
    --encoder_trans_enc_depth 7 \
    --encoder_trans_dec_depth 7 \
    --encoder_trans_patch_size 4 \
    --encoder_trans_n_groups 64 \
    --encoder_trans_update_strategy scale \
    --predictor_variant transformer \
    --noise_predictor_dim 512 \
    --noise_predictor_n_head 8 \
    --noise_predictor_head_dim 64 \
    --noise_predictor_ff_dim 2048 \
    --noise_predictor_depth 7 \
    --noise_predictor_dropout 0.1 \
    --noise_predictor_chunk_size 256 \
    --noise_predictor_t_embed_dim 512