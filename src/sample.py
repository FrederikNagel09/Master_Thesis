"""


usage:

python src/sample.py \
        --config_path src/results/basic_inr/experiments/test_02-03-20:26.json \
        --height 512 \
        --width 512 
"""

import argparse
import sys

sys.path.append(".")

from src.utils.parser_utils import parse_config_vars
from src.utils.run_inference_utils import run_inference_siren_inr


def parse_args():
    parser = argparse.ArgumentParser(description="INR inference — reconstruct an image at any resolution.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the experiment directory.")
    parser.add_argument("--height", type=int, required=True, help="Output image height in pixels.")
    parser.add_argument("--width", type=int, required=True, help="Output image width in pixels.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    config = parse_config_vars(args.config_path)

    if config["model"] == "basic_inr":
        run_inference_siren_inr(args, config)

    """
    if args.model == "ddpm":
        run_training_ddpm(args)
    elif args.model == "siren_inr":
        run_training_siren_inr(args)
    elif args.model == "inr_mlp_hypernet":
        run_training_inr_mlp_hypernet(args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    """
