#!/bin/bash
#BSUB -J wd_two_stage_convergence      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1000                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/wd_two_stage_convergence.out                        # Standard output redirection
#BSUB -e src/outputs/wd_two_stage_convergence.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/two-stage-weight-training.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/two-stage-weight-training.py\
    --run_name wd_two_stage_convergence \
    --mode convergence \
    --wd_config src/train_results/weight-diffusion-TEST/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 5 \
    --vae_patience 10 \
    --vae_delta 1e-4 \
    --ddpm_check_every 5 \
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 4096 \
    --fid_batch_size 1024