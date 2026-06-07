#!/bin/bash
#BSUB -J Latent-Diffusion-Probabilistic-cifar10       # Job name
#BSUB -q gpua100                          # Queue to submit the job to
#BSUB -W 1500                             # Wall time limit (6 hours)
#BSUB -n 8                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/Latent-Diffusion-Probabilistic-cifar10.out                        # Standard output redirection
#BSUB -e src/outputs/Latent-Diffusion-Probabilistic-cifar10.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py\
    --run_name Latent-Diffusion-Probabilistic-cifar10 \
    --model latent_inr_diffusion \
    --dataset cifar10 \
    --epochs 780 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 100 \
    --subset_frac 1.0 \
    --normalize False\
    --do_scaling True \
    --do_latent_recon False \
    --probablistic True \
    --lambda_kl 1.0 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 5e-2 \
    --inr_hidden_dim 256 \
    --inr_layers 5 \
    --latent_dim 64 \
    --latent_size 12 \
    --latent_patch_size 2 \
    --latent_enc_hidden_dim 16\
    --pred_d_model 256 \
    --pred_n_heads 8 \
    --pred_n_layers 6 \
    --pred_d_ff 1024 \
    --pred_t_embed_dim 256 \
    --dec_trans_dim 256 \
    --dec_trans_n_head 8 \
    --dec_trans_head_dim 128 \
    --dec_trans_ff_dim 1024 \
    --dec_trans_enc_depth 4 \
    --dec_trans_dec_depth 4 \
    --dec_trans_n_groups 1 \
    --dec_trans_update_strategy scale \
    --resume /zhome/66/4/156534/Master_Thesis/src/train_results/Latent-Diffusion-Probabilistic-cifar10/weights/weights.pt