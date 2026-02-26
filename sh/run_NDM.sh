#!/bin/bash
#BSUB -J latent_ndm_training               # Job name
#BSUB -q gpua10                            # Queue to submit the job to
#BSUB -W 240                               # Wall time limit (4 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/neural_latent_diffusion/output.out
#BSUB -e src/neural_latent_diffusion/output.err

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# Run training
python /zhome/66/4/156534/Master_Thesis/src/neural_latent_diffusion/run_training.py \
    --T 1000 \
    --beta_start 1e-4 \
    --beta_end 0.02 \
    --latent_channels 4 \
    --latent_size 8 \
    --ndm_hidden_dim 512 \
    --ndm_num_layers 4 \
    --ndm_time_dim 256 \
    --img_size 32 \
    --unet_channels 128 \
    --time_dim 256 \
    --batch_size 256 \
    --lr 2e-4 \
    --num_epochs_vae 15 \
    --num_epochs_ndm 50 \
    --beta_kl 1e-4 \
    --lambda_vae 0.05 \
    --experiment_name latent_ndm_mnist