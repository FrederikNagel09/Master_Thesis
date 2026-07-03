#!/bin/bash
#BSUB -J VOXEL-Weight-Fixed-TEST     # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1440                             # Wall time limit (6 hours)
#BSUB -n 6                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/VOXEL-Weight-Fixed-TEST.out                        # Standard output redirection
#BSUB -e src/outputs/VOXEL-Weight-Fixed-TEST.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/two-stage-weight-training.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/scripts/two_stage_weight_training.py \
    --run_name VOXEL-Weight-Converge-TEST \
    --mode convergence \
    --stage vae \
    --wd_config /zhome/66/4/156534/Master_Thesis/src/train_results/VOXEL-Weight-Diffusion-TEST-test/metadata/config.json \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.000001 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 10 \
    --vae_patience 20 \
    --vae_delta 1e-4 \
    --ddpm_check_every 10 \
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --vae_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 16 \
    --fid_batch_size 16