# MLP based
Encoder   : mlp  |  Predictor : mlp
    ε_θ       : 2,207,169  |  Encoder : 2,076,865
  Model   : NDM_INR  | parameters=4,284,034

################# STATIC #################
python main.py \
    --run_name ndm_inr_MLP_STATIC \
    --model ndm_inr \
    --ndm_variant static \
    --encoder_variant mlp \
    --predictor_variant mlp \
    --dataset mnist \
    --use_modulation True \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 50 \
    --peak_lr 1e-3 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --f_phi_hidden 512 512 512 \
    --f_phi_t_embed 128 \
    --noise_hidden_dim 256 \
    --noise_n_blocks 6 \
    --noise_t_embed 128

################# Temporal #################
python main.py \
    --run_name ndm_inr_MLP_Temporal \
    --model ndm_inr \
    --ndm_variant temporal \
    --encoder_variant mlp \
    --predictor_variant mlp \
    --dataset mnist \
    --use_modulation True \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 50 \
    --peak_lr 1e-3 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --f_phi_hidden 512 512 512 \
    --f_phi_t_embed 128 \
    --noise_hidden_dim 256 \
    --noise_n_blocks 6 \
    --noise_t_embed 128



# MLP / cnn based
  Siren INR   : hidden=32  layers=3  out_dim=1  weights=2241
    Variant   : Static  W(x)
    Encoder   : cnn  |  Predictor : mlp
    ε_θ       : 4,767,169  |  Encoder : 4,304,323
  Model   : NDM_INR  | parameters=9,073,733

################# STATIC  #################
python main.py \
    --run_name ndm_inr_CNN-MLP_Static \
    --model ndm_inr \
    --ndm_variant static \
    --encoder_variant cnn \
    --predictor_variant mlp \
    --dataset mnist \
    --use_modulation True \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 5000 \
    --peak_lr 1e-4 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --cnn_base_ch 128 \
    --cnn_n_blocks 3 \
    --noise_hidden_dim 512 \
    --noise_n_blocks 4 \
    --noise_t_embed 128



################# Temporal #################
python main.py \
    --run_name ndm_inr_CNN_Temporal \
    --model ndm_inr \
    --ndm_variant temporal \
    --encoder_variant cnn \
    --predictor_variant transformer \
    --dataset mnist \
    --use_modulation True \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 50 \
    --peak_lr 1e-3 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --cnn_base_ch 64 \
    --cnn_n_blocks 4 \
    --transformer_chunk_size 32 \
    --transformer_d_model 256 \
    --transformer_n_heads 8 \
    --transformer_n_layers 6 \
    --transformer_d_ff 512 \
    --transformer_dropout 0.2 \
    --noise_t_embed 128


# Transformer / Transformer based 
Variant   : Static  W(x)
    Encoder   : transformer  |  Predictor : transformer
    ε_θ       : 3,263,776  |  Encoder : 4,462,659
  Model   : NDM_INR  | parameters=7,728,676
  
################# STATIC  #################
python main.py \
    --run_name ndm_inr_TRANS_Static \
    --model ndm_inr \
    --ndm_variant static \
    --encoder_variant transformer \
    --predictor_variant transformer \
    --dataset mnist \
    --use_modulation True \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 50 \
    --peak_lr 1e-3 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --enc_patch_size 4 \
    --enc_embed_dim 256 \
    --enc_n_heads 4 \
    --enc_n_blocks 4 \
    --enc_mlp_ratio 4.0 \
    --enc_dropout 0.1 \
    --transformer_chunk_size 32 \
    --transformer_d_model 256 \
    --transformer_n_heads 8 \
    --transformer_n_layers 6 \
    --transformer_d_ff 512 \
    --transformer_dropout 0.2 \
    --noise_t_embed 128

################# TEMPORAL #################
python main.py \
    --run_name ndm_inr_TRANS_Temporal \
    --model ndm_inr \
    --ndm_variant temporal \
    --encoder_variant transformer \
    --predictor_variant transformer \
    --dataset mnist \
    --use_modulation True \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_every_n_steps 20 \
    --subset_frac 1.0 \
    --use_scheduler \
    --warmup_steps 50 \
    --peak_lr 1e-3 \
    --T 1000 \
    --beta_1 1e-4 \
    --beta_T 2e-2 \
    --sigma_tilde 1.0 \
    --inr_hidden_dim 32 \
    --inr_layers 3 \
    --enc_patch_size 4 \
    --enc_embed_dim 256 \
    --enc_n_heads 4 \
    --enc_n_blocks 4 \
    --enc_mlp_ratio 4.0 \
    --enc_dropout 0.1 \
    --transformer_chunk_size 32 \
    --transformer_d_model 256 \
    --transformer_n_heads 8 \
    --transformer_n_layers 6 \
    --transformer_d_ff 512 \
    --transformer_dropout 0.2 \
    --noise_t_embed 128