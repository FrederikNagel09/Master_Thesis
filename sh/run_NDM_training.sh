#!/bin/bash
#BSUB -J ndm_attention_cifar2
#BSUB -q gpuv100
#BSUB -W 1200
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o src/outputs/ndm_attention_unet_cifar_test_2_1.out
#BSUB -e src/outputs/ndm_attention_unet_cifar_test_2_1.err
##BSUB -N

source /zhome/66/4/156534/Master_Thesis/.venv/bin/activate

python /zhome/66/4/156534/Master_Thesis/main.py \
    --run_name ndm_attention_CIFAR_TEST2 \
    --model ndm \
    --dataset cifar10 \
    --epochs 200 \
    --batch_size 128 \
    --lr 4e-4 \
    --weight_decay 0.0 \
    --subset_frac 1.0 \
    --grad_clip 1.0 \
    --log_every_n_steps 50 \
    --prior_scaling 1.0 \
    --use_scheduler \
    --warmup_steps 4500 \
    --peak_lr 4e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --base_channels 64 \
    --use_attention_unet \
    --num_res_blocks 2 \
    --channel_mult 1 2 2 2 \
    --num_heads 4 \
    --num_heads_channels 64 \
    --resume /zhome/66/4/156534/Master_Thesis/src/train_results/ndm_attention_CIFAR_TEST2/weights/weights.pt \
    --attention_resolutions 16