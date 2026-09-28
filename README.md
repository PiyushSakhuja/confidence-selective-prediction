# Evaluating Confidence-Based Selective Prediction Under Controlled Distribution Shifts

## Research Question

How does confidence-based selective prediction behave when a classifier is evaluated under controlled distribution shifts?

## Overview

This project evaluates confidence-based selective prediction using a SmallCNN trained on CIFAR-10.

The experiment studies:

- Gaussian noise
- Gaussian blur
- Brightness shift
- Rotation

under multiple severity levels.

Two selection policies are compared:

1. Fixed confidence threshold
2. Coverage-controlled threshold calibrated on clean data

Confidence calibration is additionally studied using temperature scaling.

## Dataset

CIFAR-10:

- 45,000 training samples
- 5,000 calibration samples
- 10,000 test samples

## Evaluation Metrics

- Accuracy
- Coverage
- Selective Accuracy
- Selective Risk
- Risk-Coverage Curve
- AURC
- ECE

## Project Structure

```text
src/        Source code
tests/      Unit tests
data/       Local datasets
models/     Local model checkpoints
results/    Experimental results
paper/      Research paper
notebooks/  Exploratory notebooks


## Data split

- Seed: 42
- Train: 45,000 / Calibration: 5,000 / Test: 10,000
- Split via torch.utils.data.random_split(seed=42) on original CIFAR-10 train set

## Shift severities
- Noise (sigma): 0.02, 0.05, 0.08, 0.12
- Blur (sigma): 0.5, 1.0, 1.5, 2.5
- Brightness (factor): 1.3, 1.6, 2.0, 2.5
- Rotation (degrees): 5, 12, 20, 35
- Severity 0 = clean (identity) for all

## master_predictions.csv schema
| column | type | notes |
|---|---|---|
| shift_type | str | 'clean','noise','blur','brightness','rotation' |
| severity | int | 0-4 |
| true_label | int | 0-9 |
| pred_label | int | 0-9 |
| correct | bool | |
| confidence_raw | float | softmax max-prob |
| confidence_calibrated | float | after temp scaling |
| sample_id | int | index into original test set, for joining if needed |

## Logits storage
- Separate .npy file: logits_{shift_type}_sev{n}.npy, shape (10000, 10)
- Indexed by same order as sample_id in the CSV