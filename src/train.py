"""
python src/train.py \
    --model basic_inr \
    --epochs 30 \
    --index 5 \
    --name test \
    --batch_size 32 \
    --lr 1e-4 \
    --h1 32 \
    --h2 32 \
    --h3 32 \
    --dataset mnist \
    --omega_0 20.0 

python src/train.py \
    --model hypernet_inr \
    --name test \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --h1 32 \
    --h2 32 \
    --h3 32 \
    --dataset mnist \
    --omega_0 20.0 \
    --hyper_h 256 \
    --subset_frac 0.1
"""

import sys

sys.path.append(".")

from src.utils.parser_utils import parse_args_training
from src.utils.run_training_utils import (
    run_training_inr_mlp_hypernet,
    run_training_siren_inr,
)

if __name__ == "__main__":
    args = parse_args_training()

    if args.model == "basic_inr":
        run_training_siren_inr(args)
    elif args.model == "hypernet_inr":
        run_training_inr_mlp_hypernet(args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

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
