#!/bin/bash
#BSUB -J weight-diffusion-submodules                       # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 1440                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/weight-diffusion-submodules.out                        # Standard output redirection
#BSUB -e src/outputs/weight-diffusion-submodules.err                        # Standard error redirection

# Activate virtual environment
# bsub < sh/eval_submodules.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---

python /zhome/66/4/156534/Master_Thesis/src/scripts/eval_submodels.py \
    --config_path src/train_results/weight-diffusion/metadata/config.json \
    --weights_dir src/train_results/weight-diffusion/weights \
    --n_fid_samples 10000 