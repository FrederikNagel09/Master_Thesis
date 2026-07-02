#!/bin/bash
#BSUB -J vae_3d_baseline_1.0_newLoss           # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 1400                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/vae_3d_baseline_1.0_newLoss.out                        # Standard output redirection
#BSUB -e src/outputs/vae_3d_baseline_1.0_newLoss.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/vae-baseline-training.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/VAE_Baseline_Training_3D.py \
    --run_name vae_3d_baseline_1.0_newLoss \
    --ldm_config /zhome/66/4/156534/Master_Thesis/src/train_results/latent-diffusion-VOXEL-newLoss/metadata/config.json \
    --epochs 1400 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --subset_frac 1.0 \
    --lambda_kl_max 1.0 \
    --kl_warmup_frac 0.4 \
    --n_eval_samples 16 \
    --eval_batch_size 16