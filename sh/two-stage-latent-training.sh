#!/bin/bash
#BSUB -J Latent-two_stage_fixed-VOXEL      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1440                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/Latent-two_stage_fixed-VOXEL.out                        # Standard output redirection
#BSUB -e src/outputs/Latent-two_stage_fixed-VOXEL.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/two-stage-latent-training.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/train_two_stage.py\
    --run_name Latent-two_stage_fixed-VOXEL \
    --mode fixed \
    --ldm_config /zhome/66/4/156534/Master_Thesis/src/train_results/latent-probability-3D-data/metadata/config.json \
    --total_epochs 700 \
    --vae_epochs 350 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 4096 \
    --fid_batch_size 128