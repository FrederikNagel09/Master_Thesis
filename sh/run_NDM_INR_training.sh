#!/bin/bash
#BSUB -J ndm_inr_CNN-MLP_Static_ADD_SIREN15                   # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 150                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=512MB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ndm_inr_CNN-MLP_Static_ADD_SIREN15.out                        # Standard output redirection
#BSUB -e src/outputs/ndm_inr_CNN-MLP_Static_ADD_SIREN15.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py \
    --run_name ndm_inr_CNN-MLP_Static_ADD_SIREN15 \
    --model ndm_inr \
    --ndm_variant static \
    --encoder_variant cnn \
    --predictor_variant mlp \
    --dataset mnist \
    --use_modulation True \
    --epochs 200 \
    --batch_size 128 \
    --lr 5e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 10000 \
    --peak_lr 5e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --omega_0 15.0 \
    --inr_layers 3 \
    --cnn_base_ch 128 \
    --cnn_n_blocks 3 \
    --noise_hidden_dim 512 \
    --noise_n_blocks 4 \
    --noise_t_embed 128