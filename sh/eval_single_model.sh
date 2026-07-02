#!/bin/bash
#BSUB -J latent-two-stage-convergence                       # Job name
#BSUB -q gpuv100                           # Queue to submit the job to
#BSUB -W 300                              # Wall time limit (6 hours)
#BSUB -n 4                                 # Request 4 cores
#BSUB -R "rusage[mem=2GB]"                 # Request 2 GB of memory per core
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -o src/outputs/latent-two-stage-convergence.out                        # Standard output redirection
#BSUB -e src/outputs/latent-two-stage-convergence.err                        # Standard error redirection
#BSUB -N                                   # send email when job finishes
#BSUB -B                                   # Send email when job begins

# Activate virtual environment
# bsub < sh/eval_single_model.sh
source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

# --- Phase 1+2+3: Training ---
python /zhome/66/4/156534/Master_Thesis/src/utility/unified_results_eval.py \
    --model_type latent \
    --config_path_2d src/train_results/Latent-two_stage_convergence/Latent-two_stage_convergence_ldm_config.json \
    --weights_path_2d src/train_results/Latent-two_stage_convergence/Latent-two_stage_convergence_ldm_checkpoint.pt \
    --config_path_3d src/train_results/Latent-two_stage_convergence-VOXEL/Latent-two_stage_convergence-VOXEL_ldm_config.json \
    --weights_path_3d src/train_results/Latent-two_stage_convergence-VOXEL/Latent-two_stage_convergence-VOXEL_ldm_checkpoint.pt \
    --run_name latent_converged_suite \
    --n_metric_samples 5120 --metric_batch_size 64 --n_pca_samples 2024