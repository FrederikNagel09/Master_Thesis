# Fixed-budget mode (version 1):
python src/scripts/train_two_stage.py \
    --run_name TESTING-two_stage_fixed-MNIST \
    --mode fixed \
    --ldm_config src/train_results/latent-diffusion/metadata/config.json \
    --total_epochs 2 \
    --vae_epochs 1 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.000001 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 128 \
    --fid_batch_size 128

# Convergence mode (version 2):
python src/scripts/train_two_stage.py \
    --run_name TESTING-two_stage_convergence-MNIST \
    --mode convergence \
    --ldm_config src/train_results/TESTING-latent-diffusion-MNIST/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 1 \
    --vae_patience 5 \
    --vae_delta 10 \
    --ddpm_check_every 1 \
    --ddpm_patience 10 \
    --ddpm_delta 1e-1 \
    --ddpm_max_epochs 2 \
    --vae_max_epochs 1 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 128 \
    --fid_batch_size 128

# 3D ShapeNet voxels mode (convergence):
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_two_stage.py \
    --run_name VOXEL-Latent-Converge-TEST \
    --mode convergence \
    --ldm_config src/train_results/latent-diffusion-VOXEL-newLoss/metadata/config.json \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.000001 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 10 \
    --vae_patience 20 \
    --vae_delta 1e-4 \
    --ddpm_check_every 10 \
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --vae_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 16 \
    --fid_batch_size 16

# 3D ShapeNet voxels mode (fixed):
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_two_stage.py \
    --run_name VOXEL-Latent-Fixed-TEST \
    --mode fixed \
    --ldm_config src/train_results/latent-diffusion-VOXEL-newLoss/metadata/config.json \
    --total_epochs 1400 \
    --vae_epochs 560 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.000001 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 16 \
    --fid_batch_size 16