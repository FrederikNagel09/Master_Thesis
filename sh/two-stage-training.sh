#!/bin/bash
#BSUB -J latent_two_stage_fixed      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1000                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/latent_two_stage_fixed.out                        # Standard output redirection
#BSUB -e src/outputs/latent_two_stage_fixed.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/two-stage-training.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/two-stage-training.py\
    --run_name latent_two_stage_fixed \
    --mode fixed \
    --ldm_config src/train_results/latent-diffusion-1/metadata/config.json \
    --total_epochs 450 \
    --vae_epochs 150 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 4096 \
    --fid_batch_size 1024