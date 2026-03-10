#!/bin/bash
#BSUB -J INR_mog                        # Job name
#BSUB -q gpuv100                         # Queue to submit the job to
#BSUB -W 40                               # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/vae_mog_output.out                        # Standard output redirection
#BSUB -e src/outputs/vae_mog_output.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/train.py \
    --model vae \
    --name vae_MoG \
    --prior mog \
    --epochs 75 \
    --batch_size 128 \
    --lr 1e-3 \
    --latent_dim 16 \
    --device cuda \
    --hidden_dims 256 512 256 \
    --subset_frac 1

