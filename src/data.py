"""Data splits and loaders for CIFAR-10.

Split (seed 42): 45,000 train / 5,000 calibration from the CIFAR-10 train set,
plus the untouched 10,000-image CIFAR-10 test set.

Train loader uses augmentation. Calibration and test loaders do NOT.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
N_TRAIN, N_CALIB = 45_000, 5_000
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Two different transforms: augmented (train only) and plain (calib + test).
TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
PLAIN_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_split_indices(seed: int = SEED):
    """Return (train_idx, calib_idx) as numpy arrays. Same result every call."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N_TRAIN + N_CALIB, generator=g).numpy()
    return perm[:N_TRAIN], perm[N_TRAIN:]


def save_calib_indices(path="results/metrics/calib_idx.npy"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _, calib_idx = get_split_indices()
    np.save(path, calib_idx)
    return path


def get_train_loader(root="data", batch_size=128, num_workers=2):
    train_idx, _ = get_split_indices()
    ds = datasets.CIFAR10(root, train=True, download=True, transform=TRAIN_TF)
    g = torch.Generator().manual_seed(SEED)  # reproducible shuffling
    return DataLoader(Subset(ds, train_idx.tolist()), batch_size=batch_size,
                      shuffle=True, num_workers=num_workers, generator=g)


def get_calib_loader(root="data", batch_size=256, num_workers=2):
    _, calib_idx = get_split_indices()
    ds = datasets.CIFAR10(root, train=True, download=True, transform=PLAIN_TF)
    return DataLoader(Subset(ds, calib_idx.tolist()), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


def get_test_loader(root="data", batch_size=256, num_workers=2):
    ds = datasets.CIFAR10(root, train=False, download=True, transform=PLAIN_TF)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def get_raw_test_data(root="data"):
    """Test images as UNNORMALIZED float tensors in [0,1], shape (10000,3,32,32),
    plus labels. Used in Step 5: shifts are applied to these, then normalized."""
    ds = datasets.CIFAR10(root, train=False, download=True,
                          transform=transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor(ds.targets)
    return x, y


def normalize(x):
    """Normalize a [0,1] image batch with the CIFAR-10 mean/std."""
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s
