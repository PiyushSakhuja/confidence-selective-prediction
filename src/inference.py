"""Step 6: run the trained model on the calibration set and all 17 test variants.

Outputs (schema locked in README section 5):
  results/metrics/master_predictions.csv   test set only, 17 variants x 10,000 = 170,000 rows
  results/metrics/calib_predictions.csv    calibration set only, 5,000 rows
  results/metrics/logits_{shift}_sev{n}.npy   raw logits, shape (N, 10)
  results/metrics/logits_calib_sev0.npy       calibration logits

confidence_calibrated is left as NaN here; Step 7 (temperature scaling) fills it.
"""
import os
import shutil

import numpy as np
import pandas as pd
import torch

from src.data import get_calib_loader, get_raw_test_data, get_split_indices, normalize
from src.model import SmallCNN
from src.shifts import SHIFT_TYPES
from src.utils import get_device, set_seed

COLUMNS = ["shift_type", "severity", "true_label", "pred_label", "correct",
           "confidence_raw", "confidence_calibrated", "sample_id"]


def load_model(ckpt_path, device):
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()                      # CRITICAL: dropout off, batchnorm uses running stats
    assert not model.training
    return model


@torch.no_grad()
def logits_from_batches(model, batches, device):
    """batches: iterable of already-normalized image tensors. Returns (N,10) numpy."""
    out = []
    for x in batches:
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def build_frame(logits, labels, shift_type, severity, sample_ids):
    probs = torch.softmax(torch.from_numpy(logits), dim=1)
    conf, pred = probs.max(dim=1)
    labels = np.asarray(labels)
    pred = pred.numpy()
    return pd.DataFrame({
        "shift_type": shift_type,
        "severity": severity,
        "true_label": labels.astype(int),
        "pred_label": pred.astype(int),
        "correct": pred == labels,
        "confidence_raw": conf.numpy().astype(float),
        "confidence_calibrated": np.nan,          # filled in Step 7
        "sample_id": np.asarray(sample_ids).astype(int),
    })[COLUMNS]


def _save_logits(logits, name, out_dir, drive_dir):
    path = f"{out_dir}/{name}"
    np.save(path, logits.astype(np.float32))
    if drive_dir:
        os.makedirs(f"{drive_dir}/{out_dir}", exist_ok=True)
        shutil.copy(path, f"{drive_dir}/{out_dir}/")


def run_inference(ckpt_path="models/cnn_best.pt", data_root="/content/data",
                  shifted_dir="data/shifted_test", out_dir="results/metrics",
                  drive_dir=None, batch_size=500):
    set_seed(42)
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt_path, device)

    # ---- calibration set (clean, from CIFAR-10 train set) ----
    calib_loader = get_calib_loader(data_root)          # shuffle=False, no augmentation
    _, calib_idx = get_split_indices()                  # same order as the loader
    calib_labels = np.concatenate([y.numpy() for _, y in calib_loader])
    calib_logits = logits_from_batches(model, (x for x, _ in calib_loader), device)
    calib_df = build_frame(calib_logits, calib_labels, "clean", 0, calib_idx)
    calib_df.to_csv(f"{out_dir}/calib_predictions.csv", index=False)
    _save_logits(calib_logits, "logits_calib_sev0.npy", out_dir, drive_dir)
    print(f"calibration: n={len(calib_df)}  acc={calib_df.correct.mean():.4f}")

    # ---- test variants (clean + 16 shifted) ----
    _, test_labels = get_raw_test_data(data_root)
    test_labels = test_labels.numpy()
    sample_ids = np.arange(len(test_labels))
    variants = [("clean", 0)] + [(s, sev) for s in SHIFT_TYPES for sev in (1, 2, 3, 4)]

    frames = []
    for shift_type, sev in variants:
        x = torch.load(f"{shifted_dir}/{shift_type}_sev{sev}.pt").float()   # [0,1], un-normalized
        batches = (normalize(x[i:i + batch_size]) for i in range(0, len(x), batch_size))
        logits = logits_from_batches(model, batches, device)
        df = build_frame(logits, test_labels, shift_type, sev, sample_ids)
        frames.append(df)
        _save_logits(logits, f"logits_{shift_type}_sev{sev}.npy", out_dir, drive_dir)
        print(f"{shift_type:10s} sev{sev}: acc={df.correct.mean():.4f}  "
              f"mean_conf={df.confidence_raw.mean():.4f}")

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(f"{out_dir}/master_predictions.csv", index=False)
    if drive_dir:
        for f in ("master_predictions.csv", "calib_predictions.csv"):
            shutil.copy(f"{out_dir}/{f}", f"{drive_dir}/{out_dir}/")
    print("master rows:", len(master), "| calib rows:", len(calib_df))
    return master, calib_df
