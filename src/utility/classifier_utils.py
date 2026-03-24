import json
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from src.configs.results_config import (
    CACHE_DIR,
    CACHE_PATH,
    CLASSIFIER_CONFIG,
    CLASSIFIER_WEIGHTS,
)
from src.models.MNIST_classifier import UNetClassifier


def _load_classifier(device: str) -> UNetClassifier:
    with open(CLASSIFIER_CONFIG) as f:
        cfg = json.load(f)
    model = UNetClassifier(
        num_classes=cfg["num_classes"],
        base_ch=cfg["base_ch"],
    ).to(device)
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
    model.eval()
    return model


# =============================================================================
# Inception
# =============================================================================


def _get_inception(device: str):
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()
    return inception


@torch.no_grad()
def _inception_features(
    images_01: torch.Tensor,
    inception,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """images_01: (N, 1, H, W) or (N, 3, H, W) in [0,1]. Returns (N, 2048)."""
    from torchvision.transforms.functional import resize

    all_feats = []
    for i in tqdm(range(0, len(images_01), batch_size), desc="    Inception features", leave=False):
        batch = images_01[i : i + batch_size]  # keep on CPU for resize
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = resize(batch, [299, 299], antialias=True)  # resize on CPU
        batch = batch.to(device)  # move to device after resize
        feats = inception(batch)[0].squeeze(-1).squeeze(-1).cpu().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats)


@torch.no_grad()
def _mnist_features(
    images_01: torch.Tensor,
    classifier: UNetClassifier,
    device: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features (N, D), predicted_labels (N,))."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    all_feats, all_preds = [], []
    for i in range(0, len(images_01), batch_size):
        batch = normalize(images_01[i : i + batch_size].to(device))
        all_feats.append(classifier.get_features(batch).cpu().numpy())
        all_preds.append(classifier(batch).argmax(1).cpu().numpy())
    return np.concatenate(all_feats), np.concatenate(all_preds)


# =============================================================================
# Real MNIST feature cache
# =============================================================================


def _load_or_compute_real_features(
    classifier: UNetClassifier,
    inception,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mnist_features, inception_features) for the real MNIST train set.
    Computes once and caches to CACHE_PATH; loads from cache on subsequent runs.
    """
    if os.path.exists(CACHE_PATH):
        print("  Loading cached real MNIST features …")
        data = np.load(CACHE_PATH)
        return data["mnist_features"], data["inception_features"]

    print("  Computing real MNIST features (first run — will be cached) …")
    mnist = datasets.MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=512, shuffle=False)

    all_imgs = []
    for x, _ in tqdm(loader, desc="    Loading MNIST", leave=False):
        all_imgs.append(x)
    real_images = torch.cat(all_imgs)  # (60000, 1, 28, 28)

    print("    Extracting MNIST classifier features …")
    mnist_feats, _ = _mnist_features(real_images, classifier, device)

    print("    Extracting Inception features …")
    inception_feats = _inception_features(real_images, inception, device)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(CACHE_PATH, mnist_features=mnist_feats, inception_features=inception_feats)
    print(f"  Real features cached → {CACHE_PATH}")

    return mnist_feats, inception_feats
