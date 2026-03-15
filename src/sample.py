"""
python src/sample.py \
        --config_path src/results/hypernet_inr/experiments/full_train_40E_03-03-09:32.json \
        --height 1024 \
        --width 1024 

        
python src/sample.py \
    --config_path src/results/ndm/experiments/ndm_unet_15-03-07:37.json \
    --grid_size 6 
"""

import sys

sys.path.append(".")

from src.utils.parser_utils import parse_args_sample, parse_config_vars
from src.utils.run_inference_utils import (
    run_inference_ddpm,
    run_inference_inr_mlp_hypernet,
    run_inference_inr_vae,
    run_inference_ndm,
    run_inference_siren_inr,
    run_inference_vae,
)

if __name__ == "__main__":
    args = parse_args_sample()

    config = parse_config_vars(args.config_path)

    if config["model"] == "basic_inr":
        run_inference_siren_inr(args, config)
    elif config["model"] == "hypernet_inr":
        run_inference_inr_mlp_hypernet(args, config)
    elif config["model"] == "ndm":
        run_inference_ndm(args, config)
    elif config["model"] == "vae":
        run_inference_vae(args, config)
    elif config["model"] == "vae_inr_hypernet":
        run_inference_inr_vae(args, config)
    elif config["model"] == "ddpm":
        run_inference_ddpm(args, config)
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
