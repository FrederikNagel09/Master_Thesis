#!/bin/bash
#BSUB -J latent_ndm                        # Job name
#BSUB -q gpua10                            # Queue to submit the job to
#BSUB -W 50                               # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/neural_latent_diffusion/output.out
#BSUB -e src/neural_latent_diffusion/output.err

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
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
    --batch_size 64 \
    --lr 5e-5 \
    --num_epochs_vae 20 \
    --num_epochs_ndm 50 \
    --beta_kl 1e-4 \
    --lambda_vae 0.05 \
    --experiment_name latent_ndm_mnist

# --- Inference (only runs if training succeeded) ---
python /zhome/66/4/156534/Master_Thesis/src/neural_latent_diffusion/inference.py \
    --config src/neural_latent_diffusion/weights/latent_ndm_mnist_config.json \
    --n_images 16 \
    --out_path src/neural_latent_diffusion/results/inference_grid.png
