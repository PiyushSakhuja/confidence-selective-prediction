"""Step 7: temperature scaling (Guo et al. 2017).

Fit ONE scalar T on the calibration logits by minimizing NLL (grid search),
then use that same T for every variant. Fills `confidence_calibrated` in
calib_predictions.csv and master_predictions.csv, saves temperature_value.txt,
and makes the before/after reliability diagram.

T is never refit on shifted data.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils import set_seed

MET = "results/metrics"


def fit_temperature(logits, labels, grid=None):
    """Grid search T in [0.5, 5.0] (step 0.01) minimizing NLL on calibration data."""
    grid = np.arange(0.5, 5.0 + 1e-9, 0.01) if grid is None else grid
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    nlls = [F.cross_entropy(logits / float(t), labels).item() for t in grid]
    best = int(np.argmin(nlls))
    return float(grid[best]), float(nlls[best]), float(nlls[np.argmin(np.abs(grid - 1.0))])


def calibrated_confidence(logits, T):
    probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float32) / T, dim=1)
    return probs.max(dim=1).values.numpy()


def ece_score(correct, conf, n_bins=10):
    """Standard binned ECE (equal-width bins). B's metrics.py has its own version."""
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def _bin_stats(correct, conf, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        centers.append((lo + hi) / 2)
        accs.append(np.asarray(correct, float)[m].mean() if m.any() else np.nan)
        counts.append(int(m.sum()))
    return np.array(centers), np.array(accs), np.array(counts)


def reliability_diagram(correct, conf_raw, conf_cal, T, fig_path, title_set="clean test set"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, conf, name in [(axes[0], conf_raw, "Before (raw)"),
                           (axes[1], conf_cal, f"After (T = {T:.2f})")]:
        c, a, n = _bin_stats(correct, conf)
        ok = ~np.isnan(a)
        ax.bar(c[ok], a[ok], width=0.1, edgecolor="black", alpha=0.8, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence",
               title=f"{name}\nECE = {ece_score(correct, conf):.3f}")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle(f"Reliability diagram on {title_set}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def run_temperature_scaling(met_dir=MET, fig_path="results/figures/reliability_diagram_before_after.png",
                            drive_dir=None):
    set_seed(42)
    calib = pd.read_csv(f"{met_dir}/calib_predictions.csv")
    master = pd.read_csv(f"{met_dir}/master_predictions.csv")

    # ---- fit T on calibration logits ONLY ----
    calib_logits = np.load(f"{met_dir}/logits_calib_sev0.npy")
    assert len(calib_logits) == len(calib)
    T, nll_T, nll_1 = fit_temperature(calib_logits, calib.true_label.values)
    with open(f"{met_dir}/temperature_value.txt", "w") as f:
        f.write(f"{T:.2f}\n")
    print(f"T = {T:.2f}   calib NLL: {nll_1:.4f} (T=1) -> {nll_T:.4f} (T={T:.2f})")

    # ---- calibration set ----
    calib["confidence_calibrated"] = calibrated_confidence(calib_logits, T)
    assert (calib_logits.argmax(1) == calib.pred_label.values).all()
    calib.to_csv(f"{met_dir}/calib_predictions.csv", index=False)
    print(f"calib ECE: {ece_score(calib.correct, calib.confidence_raw):.4f} -> "
          f"{ece_score(calib.correct, calib.confidence_calibrated):.4f}")

    # ---- every test variant, same T ----
    master["confidence_calibrated"] = np.nan
    for (shift, sev), g in master.groupby(["shift_type", "severity"]):
        logits = np.load(f"{met_dir}/logits_{shift}_sev{sev}.npy")
        assert len(logits) == len(g), f"row mismatch for {shift} sev{sev}"
        assert (logits.argmax(1) == g.pred_label.values).all(), f"order mismatch {shift} sev{sev}"
        master.loc[g.index, "confidence_calibrated"] = calibrated_confidence(logits, T)
    assert master.confidence_calibrated.notna().all()
    master.to_csv(f"{met_dir}/master_predictions.csv", index=False)

    summary = (master.groupby(["shift_type", "severity"])
               .apply(lambda g: pd.Series({
                   "acc": g.correct.mean(),
                   "conf_raw": g.confidence_raw.mean(),
                   "conf_cal": g.confidence_calibrated.mean(),
                   "ece_raw": ece_score(g.correct, g.confidence_raw),
                   "ece_cal": ece_score(g.correct, g.confidence_calibrated)}))
               .round(4))
    print(summary)

    # ---- reliability diagram on the clean TEST set (held out from the fit) ----
    clean = master[(master.shift_type == "clean") & (master.severity == 0)]
    reliability_diagram(clean.correct.values, clean.confidence_raw.values,
                        clean.confidence_calibrated.values, T, fig_path)

    if drive_dir:
        os.makedirs(f"{drive_dir}/{met_dir}", exist_ok=True)
        os.makedirs(f"{drive_dir}/results/figures", exist_ok=True)
        for f in ("calib_predictions.csv", "master_predictions.csv", "temperature_value.txt"):
            shutil.copy(f"{met_dir}/{f}", f"{drive_dir}/{met_dir}/")
        shutil.copy(fig_path, f"{drive_dir}/results/figures/")
    return T, master, calib
