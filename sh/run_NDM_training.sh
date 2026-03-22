#!/bin/bash
#BSUB -J ndm_unet_mnist                     # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 450                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=512MB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ndm_unet_mnist.out                        # Standard output redirection
#BSUB -e src/outputs/ndm_unet_mnist.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py \
    --run_name ndm_unet_mnist \
    --model ndm \
    --dataset mnist \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --log_every_n_steps 50 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 30000 \
    --peak_lr 2e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --f_phi_type unet \
    --f_phi_t_embed 128 \
    --base_channels 32 \
    