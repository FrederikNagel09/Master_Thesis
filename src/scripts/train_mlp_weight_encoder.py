"""
train_mlp_weight_encoder.py

Trains MLPStaticWeightEncoder on MNIST.
Pipeline per batch:
    1. Encode image -> flat weights via MLP
    2. Inflate flat weights -> param dict
    3. Set params on SIREN
    4. Decode coord grid -> reconstructed image
    5. MSE loss vs. original image

python src/scripts/train_mlp_weight_encoder.py \
    --inr_hidden_dim 16 \
    --inr_layers 3 \
    --f_phi_hidden 512 512 512 \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-4
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(".")
from src.models.NDM_INR import MLPStaticWeightEncoder

# ── Coord grid ────────────────────────────────────────────────────────────────


def make_coord_grid(shape: tuple[int, int], range: tuple[float, float]) -> torch.Tensor:
    """
    Build a 2D coordinate grid.

    Args:    shape (H, W), range (min, max) for both axes
    Returns: coords (H, W, 2) in [range[0], range[1]]
    """
    h, w = shape
    ys = torch.linspace(range[0], range[1], h)
    xs = torch.linspace(range[0], range[1], w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)


# ── Decode helper ─────────────────────────────────────────────────────────────


def decode(encoder: MLPStaticWeightEncoder, flat_weights: torch.Tensor, coord_grid: torch.Tensor) -> torch.Tensor:
    """
    Decode flat weights to image via SIREN.

    Args:
        encoder      MLPStaticWeightEncoder
        flat_weights (B, weight_dim)
        coord_grid   (H, W, 2)
    Returns:
        recon (B, C, H, W) in [-1, 1] due to SIREN tanh output
    """
    B = flat_weights.shape[0]  # noqa: N806
    H, W, _ = coord_grid.shape  # noqa: N806

    param_dict = encoder.inflate(flat_weights)
    encoder.inr.set_params(param_dict)

    # (B, H, W, 2)
    coords = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)
    # (B, H, W, C)
    out = encoder.inr(coords)
    # (B, C, H, W)
    return out.permute(0, 3, 1, 2)


# ── Training loop ─────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    """
    Main training loop.

    Args:    args  parsed CLI arguments
    Returns: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # Normalize to [-1, 1] to match SIREN tanh output range
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    val_set = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    img_size = 28
    channels = 1
    data_dim = img_size * img_size * channels

    inr_cfg = {
        "target": "src.models.trans_inr_helpers.SIREN",
        "params": {
            "depth": args.inr_layers,
            "in_dim": 2,
            "out_dim": channels,
            "hidden_dim": args.inr_hidden_dim,
            "out_bias": 0.5,
        },
    }

    encoder = MLPStaticWeightEncoder(
        inr=inr_cfg,
        data_dim=data_dim,
        hidden_dims=args.f_phi_hidden,
        in_channels=channels,
        img_size=img_size,
    ).to(device)

    print(f"weight_dim : {encoder.weight_dim}")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    coord_grid = make_coord_grid((img_size, img_size), (-1, 1)).to(device)  # (H, W, 2)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    # ── Epochs ────────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        encoder.train()
        train_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)  # (B, 1, H, W)

            flat_weights = encoder(imgs)  # (B, weight_dim)
            recon = decode(encoder, flat_weights, coord_grid)  # (B, 1, H, W)

            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # ── Validation ────────────────────────────────────────────────────────
        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                flat_weights = encoder(imgs)
                recon = decode(encoder, flat_weights, coord_grid)
                val_loss += criterion(recon, imgs).item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, "best_encoder.pt")
            torch.save(encoder.state_dict(), ckpt_path)
            print(f"  -> Saved best checkpoint (val_loss={best_val_loss:.6f})")

    # Always save final weights
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "final_encoder.pt"))
    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLPStaticWeightEncoder on MNIST")

    parser.add_argument("--inr_hidden_dim", type=int, default=16)
    parser.add_argument("--inr_layers", type=int, default=3)
    parser.add_argument("--f_phi_hidden", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--save_dir", type=str, default="checkpoints/mlp_encoder")

    args = parser.parse_args()
    train(args)
