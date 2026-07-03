#!/bin/bash
#BSUB -J final_eval_latent_fixed                       # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 400                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/final_eval_latent_fixed.out                        # Standard output redirection
#BSUB -e src/outputs/final_eval_latent_fixed.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/final_evals/l_fix.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/utility/unified_results_eval.py \
    --model_type latent \
    --config_path_2d src/train_results/Latent-two_stage_fixed/Latent-two_stage_fixed_ldm_config.json \
    --weights_path_2d src/train_results/Latent-two_stage_fixed/Latent-two_stage_fixed_ldm_checkpoint.pt \
    --config_path_3d src/train_results/VOXEL-Latent-Fixed-TEST/VOXEL-Latent-Fixed-TEST_ldm_config.json \
    --weights_path_3d src/train_results/VOXEL-Latent-Fixed-TEST/VOXEL-Latent-Fixed-TEST_ldm_checkpoint.pt \
    --run_name latent_fixed_suite \
    --n_metric_samples 10000 --metric_batch_size 512 --n_pca_samples 5000