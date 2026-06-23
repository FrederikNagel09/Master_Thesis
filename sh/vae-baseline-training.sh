#!/bin/bash
#BSUB -J VAE_Baseline            # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 700                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/VAE_Baseline.out                        # Standard output redirection
#BSUB -e src/outputs/VAE_Baseline.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/VAE_Baseline_Training.py \
    --run_name VAE_Baseline \
    --ldm_config src/train_results/latent-diffusion-1/metadata/config.json \
    --epochs 450 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --n_fid_samples 4096 \
    --fid_batch_size 1024