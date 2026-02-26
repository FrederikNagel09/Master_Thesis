#!/bin/bash
#BSUB -J latent_ndm_inference              # Job name
#BSUB -q gpua10                            # Queue to submit the job to
#BSUB -W 30                                # Wall time limit (30 minutes)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/neural_latent_diffusion/inference.out
#BSUB -e src/neural_latent_diffusion/inference.err

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# Run inference
python /zhome/66/4/156534/Master_Thesis/src/neural_latent_diffusion/inference.py \
    --config src/neural_latent_diffusion/weights/latent_ndm_mnist_config.json \
    --n_images 16 \
    --out_path src/neural_latent_diffusion/results/inference_grid.png