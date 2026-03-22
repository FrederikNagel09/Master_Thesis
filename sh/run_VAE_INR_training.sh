#!/bin/bash
#BSUB -J vae_inr_mnist                    # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 160                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=512MB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/vae_inr_mnist.out                        # Standard output redirection
#BSUB -e src/outputs/vae_inr_mnist.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py \
    --run_name vae_inr_mnist \
    --model inr_vae \
    --dataset mnist \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --log_every_n_steps 50 \
    --subset_frac 1.0 \
    --latent_dim 128 \
    --prior mog \
    --vae_enc_dim 512 \
    --vae_dec_dim 512 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \