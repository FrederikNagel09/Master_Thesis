#!/bin/bash
#BSUB -J ddpm                       # Job name
#BSUB -q gpuv100                         # Queue to submit the job to
#BSUB -W 40                               # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ddpm_output.out                        # Standard output redirection
#BSUB -e src/outputs/ddpm_output.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/sample.py \
    --config_path /zhome/66/4/156534/Master_Thesis/src/results/ddpm/experiments/ddpm_full_run_09-03-17:00.json \
    --grid_size 5