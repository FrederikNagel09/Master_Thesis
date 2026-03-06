#!/bin/bash
#BSUB -J diffusion_hypernet               # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 240                               # Wall time limit (4 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=1GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/inr_ddpm_hypernetwork/output.out
#BSUB -e src/inr_ddpm_hypernetwork/output.err

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# Run training
python /zhome/66/4/156534/Master_Thesis/src/inr_ddpm_hypernetwork/run_training.py \
        --name diffusion_hyper_run \
        --epochs 1000 \
        --batch_size 32 \
        --lr 1e-4 \
        --inr_h 32 \
        --hyper_h 256 \
        --unet_channels 32 \
        --T 1000 \
        --lambda_denoise 0.5