#!/bin/bash
#BSUB -J fid_score_unet_ndm_scale                       # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 450                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/fid_score_unet_ndm_scale.out                        # Standard output redirection
#BSUB -e src/outputs/fid_score_unet_ndm_scale.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---

python /zhome/66/4/156534/Master_Thesis/src/scripts/compute_FID.py \
    --model ndm \
    --config /zhome/66/4/156534/Master_Thesis/src/train_results/ndm_unet_mnist_no_scaling/metadata/config.json \
    --n 50000