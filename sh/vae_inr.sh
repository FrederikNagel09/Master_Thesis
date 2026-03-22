#!/bin/bash
#BSUB -J INR_VAE_mog                        # Job name
#BSUB -q gpuv100                         # Queue to submit the job to
#BSUB -W 400                               # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=1GB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/vae_inr_mog_output.out                        # Standard output redirection
#BSUB -e src/outputs/vae_inr_mog_output.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/train.py \
    --model vae_inr_hypernet \
    --name vae_inr_MoG_Final \
    --epochs 300 \
    --prior mog \
    --batch_size 128 \
    --lr 1e-3 \
    --latent_dim 128\
    --inr_hidden_dim 64 \
    --inr_layers 3 \
    --inr_out_dim 1 \
    --vae_enc_dim 512 \
    --vae_dec_dim 512 \
    --device cuda \
    --subset_frac 1.0