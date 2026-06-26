# Fixed-budget mode:
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_fixed \
    --mode fixed \
    --wd_config src/train_results/weight-diffusion-TEST/metadata/config.json \
    --total_epochs 300 \
    --vae_epochs 150 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 256 \
    --fid_batch_size 64

# Convergence mode:
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_convergence \
    --mode convergence \
    --wd_config src/train_results/weight-Diffusion/metadata/config.json \
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
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 4096 \
    --fid_batch_size 1024

# Skip-encoder mode (re-run diffusion only, encoder files left untouched):
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_convergence \
    --mode convergence \
    --skip_vae \
    --encoder_weights src/train_results/wd_two_stage_convergence/wd_two_stage_convergence_encoder_weights.pt \
    --wd_config src/train_results/weight-Diffusion/metadata/config.json \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --ddpm_check_every 5 \
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 256 \
    --fid_batch_size 64

# 3D ShapeNet voxels mode (convergence):
python src/scripts/two-stage-weight-training.py \
    --run_name wd_two_stage_shapenet \
    --mode convergence \
    --wd_config src/train_results/weight-diffusion-shapenet/metadata/config.json \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --lambda_kl_max 0.1 \
    --kl_warmup_frac 0.4 \
    --vae_check_every 5 \
    --vae_patience 10 \
    --vae_delta 1e-4 \
    --ddpm_check_every 5 \
    --ddpm_patience 20 \
    --ddpm_delta 1e-4 \
    --ddpm_max_epochs 2000 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --n_fid_samples 128 \
    --fid_batch_size 16