# Fixed-budget mode (version 1):
python src/scripts/train_two_stage.py \
    --run_name two_stage_fixed \
    --mode fixed \
    --ldm_config src/train_results/Latent-Diffusion-Probabilistic-1612/metadata/config.json \
    --total_epochs 15 \
    --vae_epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 32 \
    --fid_batch_size 32

# Convergence mode (version 2):
python src/scripts/train_two_stage.py \
    --run_name two_stage_convergence \
    --mode convergence \
    --ldm_config src/train_results/Latent-Diffusion-Probabilistic-1612/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 5 \
    --vae_patience 10 \
    --vae_delta 1e-4 \
    --ddpm_check_every 5 \
    --ddpm_patience 30 \
    --ddpm_delta 7e-5 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 1024 \
    --fid_batch_size 64

# Skip-VAE mode (re-run DDPM only, VAE files left untouched):
python src/scripts/train_two_stage.py \
    --run_name two_stage_convergence \
    --mode convergence \
    --skip_vae \
    --vae_weights src/results/two_stage_convergence/two_stage_convergence_vae_weights.pt \
    --ldm_config src/train_results/latent-diffusion-4/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --ddpm_check_every 5 \
    --ddpm_patience 30 \
    --ddpm_delta 5e-5 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 128 \
    --fid_batch_size 64

# 3D ShapeNet voxels mode (convergence):
python src/scripts/train_two_stage.py \
    --run_name two_stage_shapenet \
    --mode convergence \
    --ldm_config src/train_results/latent-probability-3D-data/metadata/config.json \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 5 \
    --vae_patience 10 \
    --vae_delta 1e-4 \
    --ddpm_check_every 5 \
    --ddpm_patience 30 \
    --ddpm_delta 7e-5 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 64 \
    --fid_batch_size 16

# 3D ShapeNet voxels mode (fixed):
python src/scripts/train_two_stage.py \
    --run_name two_stage_shapenet_fixed \
    --mode fixed \
    --ldm_config src/train_results/latent-probability-3D-data/metadata/config.json \
    --total_epochs 10 \
    --vae_epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 64 \
    --fid_batch_size 16