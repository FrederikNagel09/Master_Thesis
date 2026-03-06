#!/bin/bash
#BSUB -J ddpm_training                     # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 60                                # Wall time limit
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=512MB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/denoising_diffusion/output.out                        # Standard output redirection
#BSUB -e src/denoising_diffusion/output.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# Run training
python /zhome/66/4/156534/Master_Thesis/src/denoising_diffusion/run_training.py \
    --num_epochs 30 \
    --batch_size 64 \
    --lr 1e-3