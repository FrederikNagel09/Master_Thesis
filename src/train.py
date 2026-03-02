"""
python src/train.py \
    --model basic_inr \
    --index 5 \
    --name test \
    --epochs 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --h1 20 \
    --h2 20 \
    --h3 20 \
    --dataset mnist \
    --omega_0 20.0
"""

import sys

sys.path.append(".")

from src.utils.parser_utils import parse_args_basic_inr
from src.utils.run_training_utils import (
    run_training_siren_inr,
)

if __name__ == "__main__":
    args = parse_args_basic_inr()

    if args.model == "basic_inr":
        run_training_siren_inr(args)

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
