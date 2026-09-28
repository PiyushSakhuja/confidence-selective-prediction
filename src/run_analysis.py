"""
run_analysis.py

Real evaluation pipeline for:
"Evaluating Confidence-Based Selective Prediction Under Controlled
Distribution Shifts"

Data layout:

results/metrics/
    calib_predictions.csv       # CLEAN calibration set (5000 rows)
    master_predictions.csv      # TEST set (170000 rows)

CRITICAL RULE:
Thresholds are computed ONLY from calib_predictions.csv.
They are then frozen and applied unchanged to master_predictions.csv.

Outputs:
    results/metrics/thresholds.csv
    results/metrics/full_results.csv
    results/metrics/rc_curves/*.npz

Run from project root:
    python src/run_analysis.py

Or from src:
    python run_analysis.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from selection import (
    fixed_threshold,
    coverage_controlled_thresholds,
)

from metrics import (
    coverage,
    selective_accuracy,
    selective_risk,
    risk_coverage_curve,
    aurc,
    ece,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

CALIB_PATH = METRICS_DIR / "calib_predictions.csv"
MASTER_PATH = METRICS_DIR / "master_predictions.csv"

OUTPUT_PATH = METRICS_DIR / "full_results.csv"
THRESHOLDS_PATH = METRICS_DIR / "thresholds.csv"

RC_CURVE_DIR = METRICS_DIR / "rc_curves"


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

CONFIDENCE_COLUMNS = {
    "raw": "confidence_raw",
    "calibrated": "confidence_calibrated",
}

FIXED_TARGET_COVERAGE = 0.90

COVERAGE_CONTROLLED_TARGETS = [
    0.70,
    0.80,
    0.90,
    0.95,
]


# ---------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "shift_type",
    "severity",
    "true_label",
    "pred_label",
    "correct",
    "confidence_raw",
    "confidence_calibrated",
    "sample_id",
}


# ---------------------------------------------------------------------
# Load calibration data
# ---------------------------------------------------------------------

def load_calibration_data():
    """Load the dedicated clean calibration CSV."""

    if not CALIB_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n{CALIB_PATH}"
        )

    calib_df = pd.read_csv(CALIB_PATH)

    missing = REQUIRED_COLUMNS - set(calib_df.columns)

    if missing:
        raise ValueError(
            f"Calibration CSV is missing required columns: {missing}"
        )

    calib_df["correct"] = calib_df["correct"].astype(bool)

    return calib_df


# ---------------------------------------------------------------------
# Load master test data
# ---------------------------------------------------------------------

def load_master_data():
    """Load the 170,000-row test prediction CSV."""

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master predictions file not found:\n{MASTER_PATH}"
        )

    master_df = pd.read_csv(MASTER_PATH)

    missing = REQUIRED_COLUMNS - set(master_df.columns)

    if missing:
        raise ValueError(
            f"Master CSV is missing required columns: {missing}"
        )

    master_df["correct"] = master_df["correct"].astype(bool)

    return master_df


# ---------------------------------------------------------------------
# Compute thresholds
# ---------------------------------------------------------------------

def compute_thresholds(calib_df):
    """
    Compute thresholds ONLY from the clean calibration set.

    No test or shifted data is used here.
    """

    thresholds = {}

    for conf_type, conf_col in CONFIDENCE_COLUMNS.items():

        calib_conf = calib_df[conf_col].values

        thresholds[conf_type] = {
            "fixed": fixed_threshold(
                calib_conf,
                target_coverage=FIXED_TARGET_COVERAGE,
            ),

            "coverage_controlled": coverage_controlled_thresholds(
                calib_conf,
                COVERAGE_CONTROLLED_TARGETS,
            ),
        }

    return thresholds


# ---------------------------------------------------------------------
# Save thresholds
# ---------------------------------------------------------------------

def save_thresholds(thresholds):
    """Save all calibration-derived thresholds."""

    rows = []

    for conf_type, policy_data in thresholds.items():

        # Fixed 90% threshold
        rows.append({
            "confidence_type": conf_type,
            "policy": "fixed",
            "target_coverage": FIXED_TARGET_COVERAGE,
            "threshold": policy_data["fixed"],
        })

        # Coverage-controlled thresholds
        for target, threshold in policy_data[
            "coverage_controlled"
        ].items():

            rows.append({
                "confidence_type": conf_type,
                "policy": "coverage_controlled",
                "target_coverage": target,
                "threshold": threshold,
            })

    thresholds_df = pd.DataFrame(rows)

    thresholds_df.to_csv(
        THRESHOLDS_PATH,
        index=False,
    )

    return thresholds_df


# ---------------------------------------------------------------------
# Compute metrics for one condition
# ---------------------------------------------------------------------

def compute_condition_metrics(
    group,
    conf_col,
    thresholds_for_conf_type,
    shift_type,
    severity,
    conf_type,
):
    """
    Compute metrics for one:

        shift_type
        severity
        confidence_type

    Thresholds were already computed from calibration data.
    """

    correct = group["correct"].values
    conf = group[conf_col].values

    rows = []

    # -------------------------------------------------------------
    # Basic accuracy
    # -------------------------------------------------------------

    no_abstention_accuracy = float(
        np.mean(correct)
    )

    # -------------------------------------------------------------
    # Threshold-free metrics
    # -------------------------------------------------------------

    rc_cov, rc_risk = risk_coverage_curve(
        correct,
        conf,
    )

    aurc_value = aurc(
        rc_cov,
        rc_risk,
    )

    ece_value = ece(
        correct,
        conf,
        n_bins=10,
    )

    # -------------------------------------------------------------
    # Fixed threshold
    # -------------------------------------------------------------

    tau_fixed = thresholds_for_conf_type["fixed"]

    rows.append({
        "shift_type": shift_type,
        "severity": severity,
        "confidence_type": conf_type,
        "policy": "fixed",
        "target_coverage": FIXED_TARGET_COVERAGE,
        "threshold": tau_fixed,

        "achieved_coverage": coverage(
            conf,
            tau_fixed,
        ),

        "selective_accuracy": selective_accuracy(
            correct,
            conf,
            tau_fixed,
        ),

        "selective_risk": selective_risk(
            correct,
            conf,
            tau_fixed,
        ),

        "no_abstention_accuracy": no_abstention_accuracy,

        "aurc": aurc_value,

        "ece": ece_value,

        "n_samples": len(correct),
    })

    # -------------------------------------------------------------
    # Coverage-controlled thresholds
    # -------------------------------------------------------------

    for target_coverage, threshold in (
        thresholds_for_conf_type["coverage_controlled"].items()
    ):

        rows.append({
            "shift_type": shift_type,
            "severity": severity,
            "confidence_type": conf_type,
            "policy": "coverage_controlled",
            "target_coverage": target_coverage,
            "threshold": threshold,

            "achieved_coverage": coverage(
                conf,
                threshold,
            ),

            "selective_accuracy": selective_accuracy(
                correct,
                conf,
                threshold,
            ),

            "selective_risk": selective_risk(
                correct,
                conf,
                threshold,
            ),

            "no_abstention_accuracy": no_abstention_accuracy,

            "aurc": aurc_value,

            "ece": ece_value,

            "n_samples": len(correct),
        })

    return rows, rc_cov, rc_risk


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def run_analysis():

    print("=" * 70)
    print("SELECTIVE PREDICTION REAL-DATA EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load calibration
    # -------------------------------------------------------------

    calib_df = load_calibration_data()

    print("\nCalibration data:")
    print(f"  Path: {CALIB_PATH}")
    print(f"  Rows: {len(calib_df)}")

    print(
        f"  Shift types: "
        f"{calib_df['shift_type'].unique().tolist()}"
    )

    # Make sure calibration really is clean
    if not all(calib_df["shift_type"] == "clean"):
        raise ValueError(
            "Calibration file contains non-clean rows. "
            "Expected only clean calibration data."
        )

    # -------------------------------------------------------------
    # Load master test data
    # -------------------------------------------------------------

    master_df = load_master_data()

    print("\nMaster test data:")
    print(f"  Path: {MASTER_PATH}")
    print(f"  Rows: {len(master_df)}")

    # -------------------------------------------------------------
    # Compute thresholds ONLY from calibration
    # -------------------------------------------------------------

    thresholds = compute_thresholds(calib_df)

    print("\n" + "=" * 70)
    print("CALIBRATION-DERIVED THRESHOLDS")
    print("=" * 70)

    for conf_type, policy_data in thresholds.items():

        print(
            f"\n[{conf_type.upper()}]"
        )

        print(
            f"  Fixed 90%: "
            f"{policy_data['fixed']:.6f}"
        )

        for target, threshold in (
            policy_data["coverage_controlled"].items()
        ):
            print(
                f"  Target {target:.2f}: "
                f"{threshold:.6f}"
            )

    # Save thresholds
    thresholds_df = save_thresholds(thresholds)

    print(
        f"\nSaved thresholds -> {THRESHOLDS_PATH}"
    )

    # -------------------------------------------------------------
    # Create output directories
    # -------------------------------------------------------------

    RC_CURVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Evaluate all test conditions
    # -------------------------------------------------------------

    all_rows = []

    conditions = master_df.groupby(
        ["shift_type", "severity"]
    )

    print("\n" + "=" * 70)
    print("TEST CONDITIONS")
    print("=" * 70)

    print(
        f"\nFound {len(conditions)} "
        f"(shift_type, severity) conditions."
    )

    print(
        "Expected: 17 conditions "
        "(clean + 4 shifts × 4 severities)"
    )

    print()

    # -------------------------------------------------------------
    # Process each condition
    # -------------------------------------------------------------

    for (shift_type, severity), group in conditions:

        print(
            f"Processing "
            f"{shift_type:<12} "
            f"severity={severity}"
        )

        for conf_type, conf_col in CONFIDENCE_COLUMNS.items():

            rows, rc_cov, rc_risk = compute_condition_metrics(
                group=group,
                conf_col=conf_col,
                thresholds_for_conf_type=thresholds[conf_type],
                shift_type=shift_type,
                severity=severity,
                conf_type=conf_type,
            )

            all_rows.extend(rows)

            # -------------------------------------------------
            # Save risk-coverage curve
            # -------------------------------------------------

            curve_filename = (
                f"{shift_type}_sev{severity}_{conf_type}.npz"
            )

            curve_path = (
                RC_CURVE_DIR / curve_filename
            )

            np.savez(
                curve_path,
                coverages=rc_cov,
                risks=rc_risk,
            )

    # -------------------------------------------------------------
    # Create final results dataframe
    # -------------------------------------------------------------

    results_df = pd.DataFrame(all_rows)

    results_df = results_df.sort_values(
        [
            "shift_type",
            "severity",
            "confidence_type",
            "policy",
            "target_coverage",
        ]
    ).reset_index(drop=True)

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved full results -> {OUTPUT_PATH}"
    )

    print(
        f"Rows in full_results.csv: "
        f"{len(results_df)}"
    )

    print(
        f"Risk-coverage curves -> {RC_CURVE_DIR}"
    )

    return results_df


# ---------------------------------------------------------------------
# Sanity summary
# ---------------------------------------------------------------------

def print_sanity_summary(results_df):

    print("\n" + "=" * 70)
    print("SANITY SUMMARY")
    print("=" * 70)

    # -------------------------------------------------------------
    # Accuracy by condition
    # -------------------------------------------------------------

    print("\nNo-abstention accuracy:")

    accuracy_table = (
        results_df[
            [
                "shift_type",
                "severity",
                "confidence_type",
                "no_abstention_accuracy",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "shift_type",
                "severity",
                "confidence_type",
            ]
        )
    )

    print(
        accuracy_table.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Fixed raw coverage
    # -------------------------------------------------------------

    print(
        "\nFixed-threshold achieved coverage "
        "(RAW confidence):"
    )

    fixed_cov = results_df[
        (results_df["policy"] == "fixed")
        & (results_df["confidence_type"] == "raw")
    ][
        [
            "shift_type",
            "severity",
            "achieved_coverage",
            "selective_accuracy",
            "selective_risk",
        ]
    ]

    print(
        fixed_cov
        .sort_values(
            ["shift_type", "severity"]
        )
        .to_string(index=False)
    )

    # -------------------------------------------------------------
    # Coverage-controlled behavior
    # -------------------------------------------------------------

    print(
        "\nCoverage-controlled RAW confidence:"
    )

    cc_raw = results_df[
        (results_df["policy"] == "coverage_controlled")
        & (results_df["confidence_type"] == "raw")
    ][
        [
            "shift_type",
            "severity",
            "target_coverage",
            "achieved_coverage",
        ]
    ]

    print(
        cc_raw
        .sort_values(
            [
                "shift_type",
                "severity",
                "target_coverage",
            ]
        )
        .to_string(index=False)
    )

    # -------------------------------------------------------------
    # NaN check
    # -------------------------------------------------------------

    n_nan = results_df[
        "selective_accuracy"
    ].isna().sum()

    print()

    if n_nan > 0:
        print(
            f"WARNING: {n_nan} rows have "
            f"NaN selective_accuracy."
        )
    else:
        print(
            "No NaN selective_accuracy values. OK."
        )

    # -------------------------------------------------------------
    # Sample-count check
    # -------------------------------------------------------------

    unique_counts = (
        results_df["n_samples"]
        .unique()
    )

    print(
        f"\nUnique sample counts per condition: "
        f"{unique_counts.tolist()}"
    )

    if len(unique_counts) == 1 and unique_counts[0] == 10000:
        print(
            "All test conditions contain 10,000 samples. OK."
        )
    else:
        print(
            "WARNING: unexpected sample counts."
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    results_df = run_analysis()

    print_sanity_summary(
        results_df
    )