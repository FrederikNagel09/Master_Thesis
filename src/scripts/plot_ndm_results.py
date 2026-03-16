"""
NDM Results Script
------------------
Hardcode the paths to your MLP and UNet config JSON files below.
Produces two plots (one per model), each with 3 rows:
  Row 1 — 8 real MNIST images
  Row 2 — those same images passed through F_phi (data transformation)
  Row 3 — 8 images sampled from the full NDM

Saved to: src/results/general/
"""

import json
import os
import sys

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# =============================================================================
# HARDCODE YOUR CONFIG PATHS HERE
# =============================================================================
MLP_CONFIG_PATH = "Master_Thesis/src/results/ndm/experiments/ndm_mlp_Final_10-03-19:13.json"

UNET_CONFIG_PATH = "/zhome/66/4/156534/Master_Thesis/src/results/ndm/experiments/ndm_unet_full_final_smallT_15-03-20:50.json"
# =============================================================================

N_IMAGES = 8
OUT_DIR = "src/results/ndm/samples"


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_model(config: dict):
    from src.models.ndm import MLPTransformation, NeuralDiffusionModel, UnetNDM, UNetTransformation

    if config.get("f_phi_type", "mlp") == "mlp":
        F_phi = MLPTransformation(  # noqa: N806
            data_dim=28 * 28,
            hidden_dims=config.get("f_phi_hidden", [512, 512, 512]),
            t_embed_dim=config.get("f_phi_t_embed", 32),
        )
    else:
        F_phi = UNetTransformation()  # noqa: N806

    model = NeuralDiffusionModel(
        network=UnetNDM(),
        F_phi=F_phi,
        T=config["T"],
        sigma_tilde_factor=config.get("sigma_tilde", 1.0),
    )

    device = config.get("device", "cpu")
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model.to(device)
    model.eval()
    return model, device


def get_mnist_samples(n: int, single_class: bool = True) -> torch.Tensor:
    """Return n randomly selected MNIST images, flattened and normalised to [-1, 1]."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )
    single_class = False  # set to False for all classes
    dataset = datasets.MNIST("data/", train=False, download=True, transform=transform)

    indices = [i for i, (_, label) in enumerate(dataset) if label == 1] if single_class else list(range(len(dataset)))

    selected = torch.tensor(indices)[torch.randperm(len(indices))[:n]]
    images = torch.stack([dataset[i][0] for i in selected])  # (n, 784)
    return images


def to_grid(tensors: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of (n, 784) or (n, 1, 28, 28) tensors to a single
    horizontal strip numpy array of shape (28, n*28).
    """
    if tensors.dim() == 2:
        tensors = tensors.view(-1, 1, 28, 28)
    imgs = ((tensors * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
    return np.concatenate([imgs[i, 0] for i in range(len(imgs))], axis=1)  # (28, n*28)


def make_plot(config: dict, out_dir: str):
    f_phi_type = config.get("f_phi_type", "mlp").upper()
    model_label = f"NDM ({f_phi_type} transformation)"
    device = config.get("device", "cpu")

    print(f"\nBuilding model: {model_label}")
    model, device = build_model(config)

    # ── Row 1: real MNIST images ──────────────────────────────────────────────
    print("Sampling MNIST images...")
    real = get_mnist_samples(N_IMAGES).to(device)  # (8, 784)
    print(f"real.min(): {real.min()}")
    print(f"real.max(): {real.max()}")
    # ── Row 2: F_phi transformation across t=0 → t=1 ─────────────────────────
    print("Running data transformation F_phi across time...")
    # Each of the 8 images is transformed at a different t, evenly spaced 0→1,

    print("=======================================")

    # Show transformation at t = 0, T/4, T/2, 3T/4, T for the first image only
    # Show F_phi transformation at t=T for all real images
    t_T = torch.ones(N_IMAGES, 1, device=device)  # t=1 for all  # noqa: N806
    with torch.no_grad():
        transformed = model.F_phi(real, t_T)  # (8, 784)

    print(f"transformed.min(): {transformed.min()}")
    print(f"transformed.max(): {transformed.max()}")
    # ── Row 3: full NDM samples ────────────────────────────────────────────────
    print("Sampling from full NDM...")
    with torch.no_grad():
        samples = model.sample((N_IMAGES, 28 * 28))  # (8, 784)

    print(f"samples.min(): {samples.min()}")
    print(f"samples.max(): {samples.max()}")
    # ── Build strips ──────────────────────────────────────────────────────────
    strip_real = to_grid(real)
    strip_transformed = to_grid(transformed)
    strip_samples = to_grid(samples)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(N_IMAGES * 1.5, 6))
    fig.subplots_adjust(hspace=0.4)

    rows = [
        (strip_real, f"Real MNIST images"),  # noqa: F541
        (strip_transformed, f"F_phi transformation  t=T  [{f_phi_type}]"),
        (strip_samples, f"Full NDM samples  [{f_phi_type}]"),
    ]

    for ax, (strip, title) in zip(axes, rows):  # noqa: B905
        ax.imshow(strip, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6, loc="left")

    os.makedirs(out_dir, exist_ok=True)
    run_name = config.get("name", f"ndm_{f_phi_type.lower()}")
    out_path = os.path.join(out_dir, f"results_{run_name}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for label, config_path in [(("UNet", UNET_CONFIG_PATH))]:  # "MLP", MLP_CONFIG_PATH,
        if not os.path.exists(config_path):
            print(f"[SKIP] {label} config not found at: {config_path}")
            continue
        config = load_config(config_path)
        make_plot(config, OUT_DIR)

    print(f"\nAll plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
