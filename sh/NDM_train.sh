#!/bin/bash
#BSUB -J ndm_mlp_fix_forward                       # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 200                               # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ndm_mlp_fix_forward.out                        # Standard output redirection
#BSUB -e src/outputs/ndm_mlp_fix_forward.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/train.py \
    --model ndm \
    --name ndm_mlp_fix_forward \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --T 1000 \
    --device cuda \
    --subset_frac 1.0 \
    --f_phi_type mlp \
    --f_phi_hidden 256 256 256 \
    --f_phi_t_embed 16 \
    --sigma_tilde 1.0 \
    --log_every_n_steps 50
