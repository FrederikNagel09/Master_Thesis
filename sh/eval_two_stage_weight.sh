#!/bin/bash
#BSUB -J wd_two_stage_convergence                       # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 1440                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/wd_two_stage_convergence.out                        # Standard output redirection
#BSUB -e src/outputs/wd_two_stage_convergence.err                        # Standard error redirection

# Activate virtual environment
# bsub < sh/eval_two_stage_weight.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---

python /zhome/66/4/156534/Master_Thesis/src/scripts/eval_two_stage_weight.py \
    --run_dir src/train_results/wd_two_stage_convergence \
    --wd_config src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_wd_config.json \
    --n_fid_samples 10000 \
    --fid_batch_size 1024