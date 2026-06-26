#!/bin/bash
#BSUB -J weight-diffusion-VOXEL      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1440                             # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/weight-diffusion-VOXEL.out                        # Standard output redirection
#BSUB -e src/outputs/weight-diffusion-VOXEL.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/weight-training.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py\
    --run_name weight-diffusion-VOXEL \
    --epochs 400 \
    --batch_size 64 \
    --subset_frac 1.0 \
    --n_fid_samples 4096 \
    --stop_gradient_flow True \
    --normalize True \
    --model weight_inr_diffusion\
    --dataset shapenet_voxels \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 200 \
    --T 100 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 256 \
    --inr_layers 5 \
    --encoder_trans_dim 128 \
    --encoder_trans_n_head 8 \
    --encoder_trans_head_dim 64 \
    --encoder_trans_ff_dim 1024 \
    --encoder_trans_enc_depth 4 \
    --encoder_trans_dec_depth 4 \
    --encoder_trans_patch_size 8 \
    --encoder_trans_n_groups 32 \
    --encoder_trans_update_strategy scale \
    --predictor_variant transformer \
    --noise_predictor_dim 128 \
    --noise_predictor_n_head 8 \
    --noise_predictor_head_dim 32 \
    --noise_predictor_ff_dim 1024 \
    --noise_predictor_depth 8 \
    --noise_predictor_dropout 0.1 \
    --noise_predictor_chunk_size 64 \
    --noise_predictor_t_embed_dim 128