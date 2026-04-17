import os

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T  # noqa: N812
from torch.utils.data import DataLoader


@torch.no_grad()
def run_evaluation(model, config, output_path, device, data_root="./data"):
    """Performs reconstruction and sampling using a model instance."""
    model.eval()  # Ensure model is in eval mode

    # 1. Prepare Data for Original & Reconstructions
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    val_set = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=36, shuffle=True)

    # Get a batch of 36 images
    originals, _ = next(iter(val_loader))
    originals = originals.to(device)

    # 2. Process Reconstructions
    reconstructions, _, _ = model(originals)

    # 3. Process Samples
    z_shape = (36, config["latent_chan"], config["latent_res"], config["latent_res"])
    z_prior = torch.randn(z_shape).to(device)
    samples = model.trans_inr(z_prior)

    # 4. Plotting (Stitching images for a perfect 6x6 grid)
    _, axes = plt.subplots(1, 3, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.3)

    titles = ["Original MNIST", "Reconstructions", "Samples"]
    data_sources = [originals, reconstructions, samples]

    for ax, title, data in zip(axes, titles, data_sources, strict=False):
        grid_img = torchvision.utils.make_grid(data, nrow=6, padding=0)
        grid_img = (grid_img + 1.0) / 2.0
        grid_img = grid_img.clamp(0, 1)
        grid_np = grid_img.permute(1, 2, 0).cpu().numpy()

        ax.imshow(grid_np)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()  # Important to close plot to free up memory
    print(f"[done] Evaluation saved to {output_path}")
