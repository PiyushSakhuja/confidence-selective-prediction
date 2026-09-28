"""Controlled distribution shifts for CIFAR-10 test images.

Shifts are applied to RAW images in [0, 1] (before normalization), then clipped
to [0, 1]. Saved tensors are therefore un-normalized; inference.py normalizes
them right before feeding the model.

Severity values are locked in the README (section 4).
"""
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src.data import get_raw_test_data
from src.utils import set_seed

SEED = 42
SHIFT_TYPES = ["noise", "blur", "brightness", "rotation"]

# severity 1..4 (index 0 = severity 1)
SEVERITY_PARAMS = {
    "noise":      [0.02, 0.04, 0.06, 0.09],
    "blur":       [0.4, 0.7, 1.0, 1.5],
    "brightness": [1.2, 1.4, 1.6, 1.8],
    "rotation":   [5, 12, 20, 35],        # unchanged
}

def apply_shift(images, shift_type, severity, seed=SEED):
    """images: float tensor (N,3,H,W) in [0,1]. Returns shifted images in [0,1].
    severity 0 returns the images unchanged."""
    if severity == 0:
        return images.clone()
    if shift_type not in SEVERITY_PARAMS:
        raise ValueError(f"unknown shift_type: {shift_type}")
    if severity not in (1, 2, 3, 4):
        raise ValueError("severity must be 0-4")
    p = SEVERITY_PARAMS[shift_type][severity - 1]

    if shift_type == "noise":
        g = torch.Generator().manual_seed(seed)
        out = images + p * torch.randn(images.shape, generator=g)
    elif shift_type == "blur":
        k = 2 * math.ceil(3 * p) + 1          # odd kernel size covering ~3 sigma
        out = TF.gaussian_blur(images, kernel_size=[k, k], sigma=[p, p])
    elif shift_type == "brightness":
        out = TF.adjust_brightness(images, p)
    else:  # rotation
        out = TF.rotate(images, float(p), interpolation=InterpolationMode.BILINEAR, fill=0.0)
    return out.clamp(0.0, 1.0)


def generate_all(data_root="/content/data", out_dir="data/shifted_test", drive_dir=None):
    """Create and save all 17 variants (clean + 4 shifts x 4 severities)."""
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)
    x, _ = get_raw_test_data(data_root)
    print("raw test images:", tuple(x.shape), "range", x.min().item(), x.max().item())

    variants = [("clean", 0)] + [(s, sev) for s in SHIFT_TYPES for sev in (1, 2, 3, 4)]
    for shift_type, sev in variants:
        shifted = apply_shift(x, shift_type, sev)
        path = f"{out_dir}/{shift_type}_sev{sev}.pt"
        torch.save(shifted.half(), path)      # float16 to save space
        if drive_dir:
            os.makedirs(f"{drive_dir}/{out_dir}", exist_ok=True)
            shutil.copy(path, f"{drive_dir}/{out_dir}/")
        print(f"saved {path}  mean={shifted.mean():.3f}")
    print(f"done: {len(variants)} variants")


def make_examples_figure(data_root="/content/data", idx=0,
                         fig_path="results/figures/shift_examples.png"):
    """4 rows (shift types) x 5 columns (severity 0-4) for one test image."""
    x, _ = get_raw_test_data(data_root)
    img = x[idx:idx + 1]
    fig, axes = plt.subplots(4, 5, figsize=(10, 8.4))
    for r, s in enumerate(SHIFT_TYPES):
        for sev in range(5):
            shifted = apply_shift(img, s, sev)[0].permute(1, 2, 0).numpy()
            ax = axes[r, sev]
            ax.imshow(shifted)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("clean" if sev == 0 else f"severity {sev}")
            if sev == 0:
                ax.set_ylabel(s, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


if __name__ == "__main__":
    # quick self-test: severity 0 is identity, outputs stay in [0,1]
    t = torch.rand(4, 3, 32, 32)
    assert torch.equal(apply_shift(t, "noise", 0), t)
    for s in SHIFT_TYPES:
        for sev in (1, 4):
            o = apply_shift(t, s, sev)
            assert o.shape == t.shape and o.min() >= 0 and o.max() <= 1
    print("shifts self-test passed")
