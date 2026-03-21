"""
MNIST UNet Classifier + FID Score
Usage:
    python src/scripts/mnist_classifier.py --train
    python src/scripts/mnist_classifier.py --train --epochs 10 --batch_size 64 --lr 0.001
"""

import argparse
import json
import os
import sys

sys.path.append(".")

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg
from src.models.mnist_classifier import UNetClassifier

# ─────────────────────────────────────────────
# FID Score
# ─────────────────────────────────────────────


class FIDScore:
    """
    Computes FID using the trained UNet classifier's bottleneck features.

    Usage:
        fid = FIDScore(weights_path='src/classifier/weights.pth',
                       config_path='src/classifier/config.json')
        score = fid.compute(real_images, generated_images)
        # images: torch.Tensor (N, 1, 28, 28) in [0, 1]  OR  np.ndarray same shape
    """

    def __init__(self, weights_path="src/classifier/weights.pth", config_path="src/classifier/config.json", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps")

        with open(config_path) as f:
            cfg = json.load(f)

        self.model = UNetClassifier(
            num_classes=cfg["num_classes"],
            base_ch=cfg["base_ch"],
            dropout=0.0,  # no dropout at eval
        ).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Normalize((0.1307,), (0.3081,))

    @torch.no_grad()
    def _get_features(self, images, batch_size=256):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self.device)

        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            # Normalize
            batch = self.transform(batch)
            feats.append(self.model.get_features(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    @staticmethod
    def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):  # noqa: ARG004
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    def compute(self, real_images, generated_images):
        """
        Args:
            real_images:      (N, 1, 28, 28) tensor or array, values in [0, 1]
            generated_images: (M, 1, 28, 28) tensor or array, values in [0, 1]
        Returns:
            fid_score: float  (lower is better, 0 = identical distributions)
        """
        f_real = self._get_features(real_images)
        f_gen = self._get_features(generated_images)

        mu1, sigma1 = f_real.mean(0), np.cov(f_real, rowvar=False)
        mu2, sigma2 = f_gen.mean(0), np.cov(f_gen, rowvar=False)

        return self._frechet_distance(mu1, sigma1, mu2, sigma2)

    @torch.no_grad()
    def predict(self, image):
        """
        Classify a single image or batch.
        Args:
            image: (1, 28, 28) or (B, 1, 28, 28) tensor/array in [0, 1]
        Returns:
            label:      int 0-9  (or list of ints for a batch)
            confidence: float in [0, 1]  (or list for a batch)

        Example — applying your own unknown threshold:
            label, conf = fid.predict(img)
            display = str(label) if conf >= 0.7 else "?"
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = self.transform(image.to(self.device))
        probs = F.softmax(self.model(image), dim=-1)
        confidence, label = probs.max(dim=-1)
        if label.numel() == 1:
            return label.item(), confidence.item()
        return label.tolist(), confidence.tolist()


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────


def get_dataloaders(batch_size=128, seed=42):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform_test)

    indices = np.arange(len(full))
    labels = np.array(full.targets)

    # 90% train, 5% val, 5% test-split from train set
    train_idx, val_idx = train_test_split(indices, test_size=0.10, random_state=seed, stratify=labels)

    train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}  |  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────


def train_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    correct = total = 0
    pbar = tqdm(loader, desc="  Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss = F.cross_entropy(model(x), y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pred = model(x).argmax(1) if not scaler else model(x).argmax(1)  # noqa: F841
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

        pbar.set_postfix(acc=f"{correct / total * 100:.1f}%")

    scheduler.step()
    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    eval_bar = tqdm(loader, desc="  Evaluating", leave=False)
    for x, y in eval_bar:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
        eval_bar.set_postfix(acc=f"{correct / total * 100:.1f}%")
    return correct / total


def save_plot(train_accs, val_accs, test_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    epochs = range(1, len(train_accs) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [a * 100 for a in train_accs], label="Train Accuracy", linewidth=2)
    ax.plot(epochs, [a * 100 for a in val_accs], label="Val Accuracy", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"MNIST classifier training | test acc = {test_acc * 100:.2f}%", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"\n{'=' * 50}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"{'=' * 50}\n")

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size)

    model = UNetClassifier(num_classes=10, base_ch=32, dropout=0.15).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    train_accs, val_accs = [], []
    best_val, best_state = 0.0, None
    print("====== Starting training... ======")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_bar:  # noqa: B007
        train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_acc = evaluate(model, val_loader, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        epoch_bar.set_postfix(train=f"{train_acc * 100:.2f}%", val=f"{val_acc * 100:.2f}%", best=f"{best_val * 100:.2f}%")
    print("====== Training complete! ======\n")
    # Test with best weights
    model.load_state_dict(best_state)
    print("Evaluating on test set...")
    test_acc = evaluate(model, test_loader, device)
    print(f"\n  ✓ Test Accuracy: {test_acc * 100:.2f}%")

    # Save weights
    os.makedirs("src/results/classifier", exist_ok=True)
    torch.save(best_state, "src/results/classifier/weights.pth")
    print("  Weights saved → src/results/classifier/weights.pth")

    # Save config
    cfg = {
        "num_classes": 10,
        "base_ch": 32,
        "dropout": 0.15,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "weight_decay": 1e-4,
        "total_params": total_params,
        "test_accuracy": round(test_acc * 100, 4),
        "best_val_accuracy": round(best_val * 100, 4),
        "data_split_seed": 42,
        "train_pct": 0.90,
        "val_pct": 0.05,
        "test_pct": 0.05,
    }
    with open("src/results/classifier/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("  Config saved  → src/results/classifier/config.json")

    # Save plot
    save_plot(train_accs, val_accs, test_acc, "src/results/general/MNIST_Classifier_training.png")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        print("Pass --train to start training.")
        print("Example: python mnist_classifier.py --train --epochs 30")
