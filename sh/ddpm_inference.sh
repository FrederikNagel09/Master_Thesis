#!/bin/bash
#BSUB -J diffusion_hypernet               # Job name
#BSUB -q hpc                          # Queue to submit the job to
#BSUB -W 20                               # Wall time limit (4 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=1GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/inr_ddpm_hypernetwork/output.out
#BSUB -e src/inr_ddpm_hypernetwork/output.err

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# Run training
python /zhome/66/4/156534/Master_Thesis/src/inr_ddpm_hypernetwork/inference.py \
        --unet_weights  /zhome/66/4/156534/Master_Thesis/src/inr_ddpm_hypernetwork/weights/diffusion_hyper_run_inr32_hyper256_unet32_unet.pth \
        --hyper_weights /zhome/66/4/156534/Master_Thesis/src/inr_ddpm_hypernetwork/weights/diffusion_hyper_run_inr32_hyper256_unet32_hypernet.pth \
        --height 512 --width 512 \
        --inr_h 32 --hyper_h 256 --unet_channels 32 \
        --mode generate