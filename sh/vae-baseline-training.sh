#!/bin/bash
#BSUB -J vae_baselinea10           # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 800                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/vae_baselinea10.out                        # Standard output redirection
#BSUB -e src/outputs/vae_baselinea10.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/VAE_Baseline_Training.py \
    --run_name vae_baselinea10 \
    --ldm_config src/train_results/Latent-Diffusion-Probabilistic-1616/metadata/config.json \
    --epochs 400 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 1.0 \
    --lambda_kl_max 1.0 \
    --n_fid_samples 4096 \
    --fid_batch_size 1024