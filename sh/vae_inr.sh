#!/bin/bash
#BSUB -J INR_VAE_gauss                        # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 20                               # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/vae_inr_gauss_output.out                        # Standard output redirection
#BSUB -e src/outputs/vae_inr_gauss_output.err                        # Standard error redirection

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/train.py \
    --model vae_inr_hypernet \
    --name inr_vae_gauss \
    --epochs 2 \
    --prior gaussian \
    --batch_size 64 \
    --lr 1e-3 \
    --latent_dim 128\
    --inr_hidden_dim 64 \
    --inr_layers 3 \
    --inr_out_dim 1 \
    --vae_enc_dim 512 \
    --vae_dec_dim 512 \
    --device cuda \
    --subset_frac 1

