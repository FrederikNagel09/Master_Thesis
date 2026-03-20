#!/bin/bash
#BSUB -J ndm_inr_time                     # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 300                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=512MB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ndm_rec_scale.out                        # Standard output redirection
#BSUB -e src/outputs/ndm_rec_scale.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/train_ndm_inr_time.py