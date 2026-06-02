"""
dataset_builder.py
Builds datasets and returns a (dataset, data_config) pair.

data_config is always:
    {
        "channels": int,
        "img_size":  int,
        "data_dim":  int,   # channels * img_size * img_size
    }

Supported dataset names (args.dataset):
    "mnist"     - 28x28 greyscale
    "cifar10"   - 32x32 RGB
    "celeba"    - 64x64 RGB
"""

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
# =============================================================================
# Public API
# =============================================================================

def build_dataset(
    dataset_name: str,
    data_root: str = "data/",
    subset_frac: float = 1.0,
    single_class: bool = False,
    single_class_label: list[int] = [3, 8, 5],  # noqa: B006
    val_frac: float = 0.05,
) -> tuple[Dataset, Dataset, dict]:
    """
    Build train/val datasets and return them together with a data_config dict.

    Parameters
    ----------
    dataset_name        : One of "mnist", "cifar10", "celeba".
    data_root           : Root directory for torchvision downloads.
    subset_frac         : Fraction of the (possibly filtered) dataset to keep.
    single_class        : If True, keep only samples with single_class_label.
    single_class_label  : Class label to keep when single_class=True.
    val_frac            : Fraction of data to use as validation set.

    Returns
    -------
    train_dataset : Dataset for training.
    val_dataset   : Dataset for validation.
    data_config   : Dict with keys "channels", "img_size", "data_dim".
    """
    name = dataset_name.lower()

    if name == "mnist":
        dataset, data_config = _build_mnist(data_root)
    elif name == "cifar10":
        dataset, data_config = _build_cifar10(data_root)
    elif name == "celeba":
        dataset, data_config = _build_celeba(data_root)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: 'mnist', 'cifar10', 'celeba'.")

    # ── Optional single-class filtering ──────────────────────────────────────
    if single_class:
        indices = [i for i, (_, label) in enumerate(dataset) if label in single_class_label]
        dataset = Subset(dataset, indices)
        print(f"  Single-class filter: keeping label={single_class_label} ({len(dataset)} samples)")

    # ── Optional subset ───────────────────────────────────────────────────────
    if subset_frac < 1.0:
        n = int(len(dataset) * subset_frac)
        dataset = Subset(dataset, range(n))

    # ── Train / val split ────────────────────────────────────────────────────
    n_total = len(dataset)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    print(
        f"  Dataset : {dataset_name.upper()}  "
        f"| train={n_train:,}  "
        f"| val={n_val:,}  "
        f"| channels={data_config['channels']}  "
        f"| img_size={data_config['img_size']}  "
        f"| data_dim={data_config['data_dim']}"
    )

    return train_dataset, val_dataset, data_config
# =============================================================================
# Per-dataset builders
# =============================================================================


def _base_transform(img_size: int, channels: int):
    """Shared transform: dequantise → scale to [-1, 1] → flatten."""
    steps = [
        transforms.Resize(img_size),
        transforms.ToTensor(),  # [0, 1]
        transforms.Lambda(lambda x: x + __import__("torch").rand(x.shape) / 255),  # dequantise
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),  # → [-1, 1]
        transforms.Lambda(lambda x: x.flatten()),  # → (data_dim,)
    ]
    if channels == 1:
        steps.insert(0, transforms.Grayscale(num_output_channels=1))
    return transforms.Compose(steps)


def _build_mnist(data_root: str) -> tuple[Dataset, dict]:
    img_size, channels = 28, 1
    transform = _base_transform(img_size, channels)
    dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    data_config = {"channels": channels, "img_size": img_size, "data_dim": channels * img_size**2}
    return dataset, data_config


def _build_cifar10(data_root: str) -> tuple[Dataset, dict]:
    img_size, channels = 32, 3
    transform = _base_transform(img_size, channels)
    dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
    data_config = {"channels": channels, "img_size": img_size, "data_dim": channels * img_size**2}
    return dataset, data_config


def _build_celeba(data_root: str) -> tuple[Dataset, dict]:
    img_size, channels = 64, 3
    transform = _base_transform(img_size, channels)
    dataset = datasets.CelebA(data_root, split="train", download=True, transform=transform)
    data_config = {"channels": channels, "img_size": img_size, "data_dim": channels * img_size**2}
    return dataset, data_config


if __name__ == "__main__":
    import sys

    sys.path.append(".")

    from torch.utils.data import DataLoader

    DATASETS = ["mnist", "cifar10"]
    EXPECTED = {
        "mnist": {"channels": 1, "img_size": 28, "data_dim": 784},
        "cifar10": {"channels": 3, "img_size": 32, "data_dim": 3072},
    }

    all_passed = True

    for name in DATASETS:
        print(f"\n{'=' * 50}")
        print(f"  Testing dataset: {name.upper()}")
        print(f"{'=' * 50}")
        try:
            dataset, data_config = build_dataset(
                dataset_name=name,
                data_root="data/",
                subset_frac=0.01,  # tiny subset for speed
            )

            # ── Check data_config fields ──────────────────────────────────
            expected = EXPECTED[name]
            for key, val in expected.items():
                actual = data_config[key]
                status = "✓" if actual == val else "✗"
                print(f"  {status} data_config['{key}'] = {actual}  (expected {val})")
                if actual != val:
                    all_passed = False

            # ── Check a batch comes out with the right shape ──────────────
            loader = DataLoader(dataset, batch_size=8, shuffle=False)
            batch = next(iter(loader))
            x = batch[0] if isinstance(batch, list | tuple) else batch

            expected_shape = (8, data_config["data_dim"])
            shape_ok = tuple(x.shape) == expected_shape
            status = "✓" if shape_ok else "✗"
            print(f"  {status} batch shape = {tuple(x.shape)}  (expected {expected_shape})")
            if not shape_ok:
                all_passed = False

            # ── Check value range ─────────────────────────────────────────
            lo, hi = x.min().item(), x.max().item()
            range_ok = lo >= -2.0 and hi <= 2.0  # [-1,1] after transform + dequant
            status = "✓" if range_ok else "✗"
            print(f"  {status} value range = [{lo:.3f}, {hi:.3f}]  (expected ~ [-1, 1])")
            if not range_ok:
                all_passed = False

            print(f"  Dataset size after subset: {len(dataset)}")

        except Exception as e:
            print(f"  ✗ FAILED with exception: {e}")
            all_passed = False

    print(f"\n{'=' * 50}")
    print(f"  {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print(f"{'=' * 50}\n")
