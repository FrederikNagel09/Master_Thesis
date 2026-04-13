#!/bin/bash
#BSUB -J ndm_temporal_transinr_v3                 # Job name
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 100                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=512MB]"                 # Request 1 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/ndm_temporal_transinr_v3.out                        # Standard output redirection
#BSUB -e src/outputs/ndm_temporal_transinr_v3.err                        # Standard error redirection
##BSUB -N                                   # send email when job finishes
##BSUB -B                                   # Send email when job begins

# Activate virtual environment
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py \
    --run_name ndm_temporal_transinr_v3\
    --model ndm_temporal_transinr \
    --dataset mnist \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --peak_lr 3e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --trans_dim 256 \
    --trans_n_head 8 \
    --trans_head_dim 32 \
    --trans_ff_dim 512 \
    --trans_enc_depth 4 \
    --trans_dec_depth 4 \
    --trans_patch_size 4 \
    --trans_n_groups 8 \
    --trans_t_embed_dim 128 \
    --trans_update_strategy scale \
    --predictor_variant transformer \
    --transformer_chunk_size 32 \
    --transformer_d_model 256 \
    --transformer_n_heads 8 \
    --transformer_n_layers 6 \
    --transformer_d_ff 512 \
    --transformer_dropout 0.0 \
    --noise_t_embed 128