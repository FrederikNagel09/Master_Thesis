#!/bin/bash
#BSUB -J latent-diffusion-VOXEL-newLoss      # Job name
#BSUB -q gpuv100                          # Queue to submit the job to
#BSUB -W 1440                             # Wall time limit (6 hours)
#BSUB -n 6                                 # Request 8 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/latent-diffusion-VOXEL-newLoss.out                        # Standard output redirection
#BSUB -e src/outputs/latent-diffusion-VOXEL-newLoss.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/3d_data.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/main.py\
    --run_name latent-diffusion-VOXEL-newLoss \
    --epochs 700 \
    --batch_size 64 \
    --subset_frac 1.0 \
    --n_fid_samples 128 \
    --model latent_inr_diffusion \
    --dataset shapenet_voxels \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 1 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --inr_hidden_dim 128 \
    --inr_layers 6 \
    --latent_dim 64 \
    --latent_size 8 \
    --latent_patch_size 1 \
    --latent_enc_hidden_dim 64\
    --pred_d_model 128 \
    --pred_n_heads 8 \
    --pred_n_layers 8 \
    --pred_d_ff 1024 \
    --pred_t_embed_dim 128 \
    --dec_trans_dim 128 \
    --dec_trans_n_head 8 \
    --dec_trans_head_dim 32 \
    --dec_trans_ff_dim 1024 \
    --dec_trans_enc_depth 6 \
    --dec_trans_dec_depth 6 \
    --dec_trans_n_groups 32 \
    --dec_trans_update_strategy scale \
    --resume /zhome/66/4/156534/Master_Thesis/src/train_results/latent-diffusion-VOXEL-newLoss/weights/weights.pt