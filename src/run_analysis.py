"""
run_analysis.py

Orchestration script: loads master_predictions.csv (dummy or real, same
schema), computes thresholds from calibration data, then computes every
metric for every (shift_type, severity, confidence_type) condition.

CRITICAL RULE: all thresholds are computed ONLY from the 'calibration'
rows. They are then applied UNCHANGED to every other condition (including
'clean' severity=0 test data and every shifted condition). Never recompute
a threshold from test or shifted data.

Usage:
    python3 run_analysis.py --input results/metrics/dummy_predictions.csv
    python3 run_analysis.py --input results/metrics/master_predictions.csv

Outputs:
    results/metrics/full_results.csv
        Long-format table: one row per
        (shift_type, severity, confidence_type, policy, target_coverage)
        with achieved coverage, selective accuracy/risk, AURC, ECE.

    results/metrics/rc_curves/{shift_type}_sev{severity}_{confidence_type}.npz
        Full risk-coverage curve arrays (coverages, risks) per condition,
        needed for plotting (not just the AURC scalar).
"""

import argparse
import os

import numpy as np
import pandas as pd

from selection import fixed_threshold, coverage_controlled_thresholds
from metrics import (
    coverage,
    selective_accuracy,
    selective_risk,
    risk_coverage_curve,
    aurc,
    ece,
)

CONFIDENCE_COLUMNS = {
    "raw": "confidence_raw",
    "calibrated": "confidence_calibrated",
}

FIXED_TARGET_COVERAGE = 0.90
COVERAGE_CONTROLLED_TARGETS = [0.70, 0.80, 0.90, 0.95]

RC_CURVE_DIR = "results/metrics/rc_curves"
OUTPUT_PATH = "results/metrics/full_results.csv"


def load_data(input_path):
    df = pd.read_csv(input_path)

    required_cols = {
        "shift_type", "severity", "true_label", "pred_label",
        "correct", "confidence_raw", "confidence_calibrated", "sample_id",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # normalize 'correct' to bool in case it was saved as 0/1 or string
    df["correct"] = df["correct"].astype(bool)

    if "calibration" not in df["shift_type"].unique():
        raise ValueError(
            "No rows with shift_type == 'calibration' found. "
            "Thresholds must be computed from a dedicated calibration set."
        )

    return df


def compute_thresholds(calib_df):
    """
    Compute both policies' thresholds, separately for raw and calibrated
    confidence, using ONLY the calibration rows.

    Returns
    -------
    dict with structure:
        {
          "raw": {
              "fixed": <float>,
              "coverage_controlled": {0.70: <float>, 0.80: <float>, ...}
          },
          "calibrated": { ... same structure ... }
        }
    """
    thresholds = {}
    for conf_type, conf_col in CONFIDENCE_COLUMNS.items():
        calib_conf = calib_df[conf_col].values
        thresholds[conf_type] = {
            "fixed": fixed_threshold(calib_conf, target_coverage=FIXED_TARGET_COVERAGE),
            "coverage_controlled": coverage_controlled_thresholds(
                calib_conf, COVERAGE_CONTROLLED_TARGETS
            ),
        }
    return thresholds


def compute_condition_metrics(group, conf_col, thresholds_for_conf_type, shift_type, severity, conf_type):
    """
    Compute all metrics for one (shift_type, severity, confidence_type)
    test condition. Returns a list of result-row dicts (one per policy
    setting) plus the raw risk-coverage curve arrays for saving.
    """
    correct = group["correct"].values
    conf = group[conf_col].values

    rows = []

    # --- Threshold-free metrics (primary evaluation) ---
    rc_cov, rc_risk = risk_coverage_curve(correct, conf)
    a = aurc(rc_cov, rc_risk)
    e = ece(correct, conf, n_bins=10)
    raw_accuracy = float(np.mean(correct))  # no-abstention accuracy, for reference

    # --- Fixed threshold policy ---
    tau_fixed = thresholds_for_conf_type["fixed"]
    rows.append({
        "shift_type": shift_type,
        "severity": severity,
        "confidence_type": conf_type,
        "policy": "fixed",
        "target_coverage": FIXED_TARGET_COVERAGE,
        "threshold": tau_fixed,
        "achieved_coverage": coverage(conf, tau_fixed),
        "selective_accuracy": selective_accuracy(correct, conf, tau_fixed),
        "selective_risk": selective_risk(correct, conf, tau_fixed),
        "no_abstention_accuracy": raw_accuracy,
        "aurc": a,
        "ece": e,
        "n_samples": len(correct),
    })

    # --- Coverage-controlled policy, one row per target coverage ---
    for target_c, tau_c in thresholds_for_conf_type["coverage_controlled"].items():
        rows.append({
            "shift_type": shift_type,
            "severity": severity,
            "confidence_type": conf_type,
            "policy": "coverage_controlled",
            "target_coverage": target_c,
            "threshold": tau_c,
            "achieved_coverage": coverage(conf, tau_c),
            "selective_accuracy": selective_accuracy(correct, conf, tau_c),
            "selective_risk": selective_risk(correct, conf, tau_c),
            "no_abstention_accuracy": raw_accuracy,
            "aurc": a,
            "ece": e,
            "n_samples": len(correct),
        })

    return rows, rc_cov, rc_risk


def run(input_path):
    df = load_data(input_path)
    calib_df = df[df.shift_type == "calibration"]
    test_df = df[df.shift_type != "calibration"]

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"  Calibration rows: {len(calib_df)}")
    print(f"  Test rows (all conditions): {len(test_df)}")

    thresholds = compute_thresholds(calib_df)
    print("\nThresholds computed from calibration data:")
    for conf_type, t in thresholds.items():
        print(f"  [{conf_type}] fixed (target={FIXED_TARGET_COVERAGE}): {t['fixed']:.4f}")
        for c, tau in t["coverage_controlled"].items():
            print(f"  [{conf_type}] coverage_controlled (target={c}): {tau:.4f}")

    os.makedirs(RC_CURVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_rows = []
    conditions = test_df.groupby(["shift_type", "severity"])
    print(f"\nProcessing {len(conditions)} (shift_type, severity) conditions "
          f"x {len(CONFIDENCE_COLUMNS)} confidence types...")

    for (shift_type, severity), group in conditions:
        for conf_type, conf_col in CONFIDENCE_COLUMNS.items():
            rows, rc_cov, rc_risk = compute_condition_metrics(
                group, conf_col, thresholds[conf_type], shift_type, severity, conf_type
            )
            all_rows.extend(rows)

            # save the full risk-coverage curve for plotting later
            curve_path = os.path.join(
                RC_CURVE_DIR, f"{shift_type}_sev{severity}_{conf_type}.npz"
            )
            np.savez(curve_path, coverages=rc_cov, risks=rc_risk)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved full results table -> {OUTPUT_PATH} ({len(results_df)} rows)")
    print(f"Saved {len(conditions) * len(CONFIDENCE_COLUMNS)} risk-coverage curve files -> {RC_CURVE_DIR}/")

    return results_df


def print_sanity_summary(results_df):
    """Quick printed checks so a broken pipeline is obvious immediately."""
    print("\n=== Sanity summary ===")

    print("\nAURC (threshold-free, should increase with severity per shift_type):")
    aurc_summary = (
        results_df[results_df.policy == "fixed"]  # aurc is same regardless of policy row, just dedupe
        .drop_duplicates(subset=["shift_type", "severity", "confidence_type"])
        .pivot_table(index=["shift_type", "severity"], columns="confidence_type", values="aurc")
    )
    print(aurc_summary.round(4))

    print("\nAchieved coverage under FIXED policy (raw confidence) "
          "-- watch how far this drifts from the 0.90 calibration target as severity increases:")
    fixed_cov = results_df[
        (results_df.policy == "fixed") & (results_df.confidence_type == "raw")
    ][["shift_type", "severity", "achieved_coverage", "selective_accuracy"]]
    print(fixed_cov.sort_values(["shift_type", "severity"]).to_string(index=False))

    # flag anything obviously broken
    n_nan = results_df["selective_accuracy"].isna().sum()
    if n_nan > 0:
        print(f"\nWARNING: {n_nan} rows have NaN selective_accuracy "
              f"(threshold accepted zero samples in that condition) -- inspect these.")
    else:
        print("\nNo NaN selective_accuracy values -- OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/metrics/dummy_predictions.csv",
        help="Path to master_predictions.csv (dummy or real, same schema)",
    )
    args = parser.parse_args()

    results_df = run(args.input)
    print_sanity_summary(results_df)