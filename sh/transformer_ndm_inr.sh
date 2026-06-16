#!/bin/bash
#BSUB -J Latent-Diffusion-Probabilistic-Small      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1200                             # Wall time limit (6 hours)
#BSUB -n 8                                 # Request 8 cores
#BSUB -R "rusage[mem=1GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/Latent-Diffusion-Probabilistic-Small.out                        # Standard output redirection
#BSUB -e src/outputs/Latent-Diffusion-Probabilistic-Small.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py\
    --run_name Latent-Diffusion-Probabilistic-Small \
    --epochs 400 \
    --batch_size 128 \
    --subset_frac 1.0 \
    --normalize False\
    --do_scaling False \
    --do_latent_recon False \
    --probablistic True\
    --stop_gradient_flow False \
    --n_fid_samples 4096 \
    --model latent_inr_diffusion \
    --dataset mnist \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 100 \
    --lambda_kl 1.0 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --inr_hidden_dim 128 \
    --inr_layers 3 \
    --latent_dim 1 \
    --latent_size 14 \
    --latent_patch_size 1 \
    --latent_enc_hidden_dim 20\
    --pred_d_model 128 \
    --pred_n_heads 8 \
    --pred_n_layers 6 \
    --pred_d_ff 1024 \
    --pred_t_embed_dim 128 \
    --dec_trans_dim 128 \
    --dec_trans_n_head 8 \
    --dec_trans_head_dim 32 \
    --dec_trans_ff_dim 1024 \
    --dec_trans_enc_depth 4 \
    --dec_trans_dec_depth 4 \
    --dec_trans_n_groups 32 \
    --dec_trans_update_strategy scale