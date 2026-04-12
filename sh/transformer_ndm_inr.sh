#!/bin/bash
#BSUB -J ndm_transinr_mnist_v4                  # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 100                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=512MB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ndm_transinr_mnist_v4.out                        # Standard output redirection
#BSUB -e src/outputs/ndm_transinr_mnist_v4.err                        # Standard error redirection
##BSUB -N                                   # send email when job finishes
##BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py \
    --run_name ndm_transinr_mnist_v4\
    --model ndm_transinr \
    --dataset mnist \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --warmup_steps 2000 \
    --peak_lr 3e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 64 \
    --inr_layers 4 \
    --trans_dim 256 \
    --trans_n_head 8 \
    --trans_head_dim 32 \
    --trans_ff_dim 512 \
    --trans_enc_depth 4 \
    --trans_dec_depth 4 \
    --trans_patch_size 4 \
    --trans_n_groups 8 \
    --trans_update_strategy scale \
    --predictor_variant mlp \
    --noise_hidden_dim 256 \
    --noise_n_blocks 3 \
    --noise_t_embed 256