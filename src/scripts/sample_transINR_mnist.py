import os
import sys
import warnings

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T  # noqa: N812
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message="The operator 'aten::im2col'")

sys.path.append(".")

from src.scripts.train_transINR_mnist import VAE  # noqa: E402

# =============================================================================
#  EVALUATION CONFIG
# =============================================================================
CHECKPOINT_PATH = "src/train_results/vae_latest.pt"
DATA_ROOT = "./data"
OUTPUT_PATH = "src/results/vae_evaluation.png"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def run_evaluation():
    # 1. Load Checkpoint and Rebuild Model
    print(f"[load] Loading checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = ckpt["config"]

    model = VAE(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 2. Prepare Data for Original & Reconstructions
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    val_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=36, shuffle=True)

    # Get a batch of 36 images
    originals, _ = next(iter(val_loader))
    originals = originals.to(DEVICE)

    # 3. Process Reconstructions
    # We pass originals through the full VAE (Encoder -> TransInr)
    reconstructions, _, _ = model(originals)

    # 4. Process Samples
    # We sample from the prior: z ~ N(0, I)
    z_shape = (36, config["latent_chan"], config["latent_res"], config["latent_res"])
    z_prior = torch.randn(z_shape).to(DEVICE)
    samples = model.trans_inr(z_prior)

    # 5. Plotting (Stitching images for a perfect 6x6 grid)
    _, axes = plt.subplots(1, 3, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.3)  # Space between the three main categories

    titles = ["Original MNIST", "Reconstructions", "Samples"]
    data_sources = [originals, reconstructions, samples]

    for ax, title, data in zip(axes, titles, data_sources, strict=False):
        # 1. Create the grid
        grid_img = torchvision.utils.make_grid(data, nrow=6, padding=0)

        # 2. Rescale from [-1, 1] to [0, 1] manually to avoid RGB clipping issues
        grid_img = (grid_img + 1.0) / 2.0
        grid_img = grid_img.clamp(0, 1)

        # 3. Convert to numpy and permute to (H, W, C)
        grid_np = grid_img.permute(1, 2, 0).cpu().numpy()

        # 4. Display (No vmin/vmax needed now as it's 0-1 RGB)
        ax.imshow(grid_np)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[done] Evaluation saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Could not find checkpoint at {CHECKPOINT_PATH}")
    else:
        run_evaluation()
