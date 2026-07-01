#!/bin/bash
#BSUB -J Latent-two_stage_convergence      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1440                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/Latent-two_stage_convergence.out                        # Standard output redirection
#BSUB -e src/outputs/Latent-two_stage_convergence.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/two-stage-latent-training.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/train_two_stage.py\
    --run_name Latent-two_stage_convergence \
    --mode convergence \
    --ldm_config /zhome/66/4/156534/Master_Thesis/src/train_results/latent-diffusion/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.000001 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 10 \
    --vae_patience 15 \
    --vae_delta 0.01 \
    --ddpm_check_every 15 \
    --ddpm_patience 15 \
    --ddpm_delta 0.01 \
    --ddpm_max_epochs 1000 \
    --vae_max_epochs 1000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 4096 \
    --fid_batch_size 128