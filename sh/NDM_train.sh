#!/bin/bash
#BSUB -J ndm_unet_Final                       # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 200                               # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=1GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ndm_unet_Final.out                        # Standard output redirection
#BSUB -e src/outputs/ndm_unet_Final.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/train.py \
    --model ndm \
    --name ndm_unet_Final \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --T 1000 \
    --device cuda \
    --subset_frac 1.0 \
    --f_phi_type unet \
    --f_phi_hidden 512 512 512 \
    --f_phi_t_embed 32 \
    --sigma_tilde 1.0 \
    --log_every_n_steps 50
