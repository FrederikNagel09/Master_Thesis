import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore", message="The operator 'aten::im2col'")

sys.path.append(".")

from src.scripts.train_transINR_mnist import VAE  # noqa: E402
from src.utility.evaluation import run_evaluation  # noqa: E402

# =============================================================================
#  EVALUATION CONFIG
# =============================================================================
CHECKPOINT_PATH = "src/trained_models/vae_transINR_weights.pt"
DATA_ROOT = "./data"
OUTPUT_PATH = "src/results/vae_evaluation.png"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model_from_checkpoint(checkpoint_path, device):
    """Loads a VAE model and its config from a saved .pt file."""
    print(f"[load] Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = VAE(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, config


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Could not find checkpoint at {CHECKPOINT_PATH}")
    else:
        model, config = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

        # Then evaluate
        run_evaluation(model, config, OUTPUT_PATH, DEVICE)
