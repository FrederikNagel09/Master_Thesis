#!/bin/bash
#BSUB -J fid_score                       # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 100                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/fid_score.out                        # Standard output redirection
#BSUB -e src/outputs/fid_score.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---

python /zhome/66/4/156534/Master_Thesis/src/scripts/calculate_vae_inr_fid.py