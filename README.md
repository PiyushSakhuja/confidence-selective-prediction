# Evaluating Confidence-Based Selective Prediction Under Controlled Distribution Shifts

## Research Question

Does calibrating a selection threshold on clean data, via a fixed vs coverage-controlled criterion, preserve reliable selective prediction as test data shifts?

## Overview

A single SmallCNN trained on CIFAR-10 is evaluated under four synthetic shifts (Gaussian noise, Gaussian blur, brightness increase, rotation) at four severity levels each. Two selection policies are compared:

1. **Fixed confidence threshold** (chosen once on clean calibration data)
2. **Coverage-controlled threshold** (threshold that hits a target coverage on clean calibration data)

Both thresholds are applied unchanged to shifted test data. Temperature scaling is studied as a secondary experiment.

**Metrics:** accuracy, coverage, selective accuracy, selective risk, risk-coverage curve, AURC, ECE.

## Project Structure and Ownership

```text
src/
  utils.py, data.py, model.py, train.py        Person A
  shifts.py, inference.py, temp_scaling.py     Person A
  selection.py, metrics.py, plots.py           Person B
tests/                                         Person B (unit tests)
notebooks/                                     Person A (Colab driver)
data/shifted_test/                             generated tensors (NOT in git)
models/cnn_best.pt                             checkpoint (NOT in git)
results/metrics/                               CSVs, txt, logits .npy
results/figures/                               PNG figures
paper/                                         shared writing
```

**Ownership rule:** only the owner edits a file. If you need a change in someone else's file, message them.

**Large files** (`.pt` checkpoints, shifted tensors, logits `.npy`) are shared through Google Drive, not git.

## Interface Contract (LOCKED)

Any change to this section must be announced to the other person BEFORE it is made.

### 1. Reproducibility
- Random seed: `42`, everywhere (split, model init, noise generation).
- Every script calls `set_seed(42)` from `src/utils.py` at the top.

### 2. Dataset Split

| Set | Size | Source | Purpose |
|---|---:|---|---|
| Train | 45,000 | CIFAR-10 train | Model training |
| Calibration | 5,000 | CIFAR-10 train | Temperature fitting, threshold selection |
| Test | 10,000 | CIFAR-10 test | Final evaluation only |

- Split: `torch.utils.data.random_split` (or equivalent index permutation) with `torch.Generator().manual_seed(42)`.
- Calibration indices saved to `results/metrics/calib_idx.npy`.
- The test set is never used to pick a threshold, temperature, or hyperparameter.

### 3. Model and Training (fixed)
- Architecture: SmallCNN. Three Conv(3x3)-BN-ReLU-MaxPool blocks (32, 64, 128 channels), Flatten, FC(2048->256), ReLU, Dropout(0.3), FC(256->10).
- Adam, lr 1e-3, cross-entropy, batch size 128, about 20 epochs.
- Best checkpoint chosen by calibration accuracy, saved to `models/cnn_best.pt`.
- Normalization: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616).

### 4. Distribution Shifts

Shifts are applied to raw images in the [0, 1] range, output clipped to [0, 1], and THEN normalized. Applied to the 10,000 test images only.

| Severity | Noise (sigma) | Blur (sigma) | Brightness (factor) | Rotation (degrees) |
|---:|---:|---:|---:|---:|
| 0 (clean) | none | none | none | none |
| 1 | 0.02 | 0.5 | 1.3 | 5 |
| 2 | 0.05 | 1.0 | 1.6 | 12 |
| 3 | 0.08 | 1.5 | 2.0 | 20 |
| 4 | 0.12 | 2.5 | 2.5 | 35 |

Implementation details:
- Noise: add `N(0, sigma^2)` per pixel, seeded with 42.
- Blur: `GaussianBlur` with `kernel_size = 2*ceil(3*sigma)+1`.
- Brightness: `adjust_brightness` (increase only).
- Rotation: `rotate`, bilinear interpolation, black fill (0).

If any value changes after the visual check, Person A updates this table and messages Person B.

### 5. Prediction Files

Both files share the same columns.

| File | Contents | Rows |
|---|---|---:|
| `results/metrics/master_predictions.csv` | Test set only: clean + 16 shifted variants | 170,000 |
| `results/metrics/calib_predictions.csv` | Calibration set only (clean) | 5,000 |

| Column | Type | Notes |
|---|---|---|
| shift_type | str | `clean`, `noise`, `blur`, `brightness`, `rotation` |
| severity | int | 0-4 (0 = clean) |
| true_label | int | 0-9 |
| pred_label | int | 0-9 |
| correct | bool | `pred_label == true_label` |
| confidence_raw | float | Max softmax probability |
| confidence_calibrated | float | Max softmax of `logits / T` |
| sample_id | int | Index into original CIFAR-10 test set (train set for `calib_predictions.csv`) |

Notes:
- In `master_predictions.csv`, the clean test set is `shift_type = clean`, `severity = 0`.
- In `calib_predictions.csv`, every row is `shift_type = clean`, `severity = 0`.
- Thresholds and temperature come ONLY from `calib_predictions.csv`.

### 6. Logits
- Stored separately, one file per variant: `results/metrics/logits_{shift_type}_sev{n}.npy`, shape `(N, 10)`.
- Row order matches the row order of that variant in its CSV (test variants: `sample_id` 0 to 9999).
- Clean test logits: `logits_clean_sev0.npy`. Calibration logits: `logits_calib_sev0.npy`.

### 7. Other Files Person A Hands Over
- `results/metrics/temperature_value.txt` (single number T)
- `results/metrics/training_log.csv`
- `results/figures/training_curves.png`
- `results/figures/shift_examples.png`
- `results/figures/reliability_diagram_before_after.png`

### 8. Compatibility Rules
- Column names and types above must not change silently.
- One temperature T, fitted on calibration logits, is used for every variant. Never refit on shifted data.
- Thresholds are computed only from `calib_predictions.csv`, then applied unchanged to all test variants.
- Coverage drift under shift is a measured result, not something to correct.
