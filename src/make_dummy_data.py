"""
make_dummy_data.py

Generates a fake master_predictions.csv matching the agreed schema, so
Person B can build and test selection.py / metrics.py / plots.py
BEFORE Person A's real model pipeline is ready.

Schema:
    shift_type            str    'clean','noise','blur','brightness','rotation'
    severity               int    0-4  (0 = clean, only used with shift_type='clean')
    true_label             int    0-9
    pred_label             int    0-9
    correct                bool
    confidence_raw          float  softmax max-prob, in [0.1, 1.0]
    confidence_calibrated  float  after temperature scaling, in [0.1, 1.0]
    sample_id               int    index into the original 10,000-image test set

Design choices (so metrics behave realistically even on fake data):
- As severity increases, accuracy decreases and confidence drops slightly.
- confidence_calibrated is a mildly "softened" version of confidence_raw
  (temperature scaling typically pulls overconfident probs down a bit),
  with some noise so it isn't a trivial linear transform.
- 'clean' condition (severity=0) is generated once and reused as the
  severity=0 baseline for every shift_type, matching how the real
  pipeline will work (severity 0 = identity transform for all shifts).
"""

import numpy as np
import pandas as pd

# ---- Config (match README.md interface contract) ----
SEED = 42
N_TEST = 10000                     # real test set size
N_DUMMY_PER_CONDITION = 800        # smaller for fast local iteration
SHIFT_TYPES = ["noise", "blur", "brightness", "rotation"]
SEVERITIES = [1, 2, 3, 4]          # severity 0 handled separately as 'clean'
N_CLASSES = 10

OUTPUT_PATH = "results/metrics/dummy_predictions.csv"


def make_dummy_data(n_test=N_TEST, n_per_condition=N_DUMMY_PER_CONDITION, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []

    # sample_ids are shared across conditions (same underlying test images
    # get shifted differently), matching the real pipeline's structure
    sample_ids = rng.choice(n_test, size=n_per_condition, replace=False)

    def generate_condition(shift_type, severity, base_acc, base_conf_mean):
        """Generate one (shift_type, severity) block of rows."""
        true_labels = rng.integers(0, N_CLASSES, size=n_per_condition)

        # correctness probability degrades with severity (baked into base_acc)
        correct = rng.random(n_per_condition) < base_acc

        # predicted label: correct -> matches true label; incorrect -> random wrong label
        pred_labels = true_labels.copy()
        wrong_mask = ~correct
        n_wrong = wrong_mask.sum()
        if n_wrong > 0:
            offsets = rng.integers(1, N_CLASSES, size=n_wrong)
            pred_labels[wrong_mask] = (true_labels[wrong_mask] + offsets) % N_CLASSES

        # confidence_raw: correct predictions get higher confidence on average
        # (mimics a reasonably calibrated-ish but overconfident classifier)
        conf_raw = np.where(
            correct,
            np.clip(rng.normal(base_conf_mean + 0.15, 0.08, n_per_condition), 0.1, 0.999),
            np.clip(rng.normal(base_conf_mean - 0.10, 0.10, n_per_condition), 0.1, 0.999),
        )

        # confidence_calibrated: mild softening toward 1/N_CLASSES + a bit of noise
        # (rough stand-in for what temperature scaling tends to do)
        shrink = 0.85
        conf_calibrated = shrink * conf_raw + (1 - shrink) * (1.0 / N_CLASSES)
        conf_calibrated += rng.normal(0, 0.02, n_per_condition)
        conf_calibrated = np.clip(conf_calibrated, 0.1, 0.999)

        return pd.DataFrame({
            "shift_type": shift_type,
            "severity": severity,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "correct": correct,
            "confidence_raw": conf_raw,
            "confidence_calibrated": conf_calibrated,
            "sample_id": sample_ids,
        })

    # --- Clean baseline (severity 0), used once, shared logically across shift types ---
    rows.append(generate_condition("clean", 0, base_acc=0.80, base_conf_mean=0.70))

    # --- Calibration set: separate synthetic block, same distribution as clean test ---
    calib_ids = rng.choice(n_test, size=n_per_condition, replace=False)  # pretend disjoint set
    calib_block = generate_condition("calibration", 0, base_acc=0.80, base_conf_mean=0.70)
    calib_block["sample_id"] = calib_ids
    rows.append(calib_block)

    # --- Shifted conditions: accuracy and confidence degrade with severity ---
    for shift_type in SHIFT_TYPES:
        for severity in SEVERITIES:
            # accuracy drops roughly linearly with severity, floor around 0.30
            base_acc = max(0.80 - 0.11 * severity, 0.30)
            base_conf_mean = max(0.70 - 0.06 * severity, 0.25)
            rows.append(generate_condition(shift_type, severity, base_acc, base_conf_mean))

    df = pd.concat(rows, ignore_index=True)
    return df


if __name__ == "__main__":
    import os

    df = make_dummy_data()

    os.makedirs("results/metrics", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df)} rows -> {OUTPUT_PATH}")
    print("\nRows per (shift_type, severity):")
    print(df.groupby(["shift_type", "severity"]).size())
    print("\nAccuracy per (shift_type, severity) [sanity check - should decrease with severity]:")
    print(df.groupby(["shift_type", "severity"])["correct"].mean().round(3))
    print("\nMean confidence_raw per (shift_type, severity):")
    print(df.groupby(["shift_type", "severity"])["confidence_raw"].mean().round(3))