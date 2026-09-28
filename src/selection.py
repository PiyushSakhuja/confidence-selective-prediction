"""
selection.py

Implements the two selection (abstention) policies used to compare
"fixed threshold" vs "coverage-controlled threshold" selective prediction:

1. fixed_threshold(calib_confidences)
   - One threshold, calibrated once on CLEAN calibration data to hit
     ~90% coverage, then applied UNCHANGED to every shifted test condition.

2. coverage_controlled_thresholds(calib_confidences, target_coverages)
   - A threshold per target coverage level, each calibrated on CLEAN
     calibration data, then applied UNCHANGED to every shifted test
     condition. Coverage is expected to DRIFT away from the target as
     severity increases -- measuring that drift is part of the research
     question, so these thresholds are never re-fit on shifted data.

CRITICAL RULE (do not violate): both functions must only ever be called
with confidences from the CLEAN CALIBRATION set. Never pass test-set or
shifted-set confidences into these functions to "compute" a threshold --
that would leak test information into calibration and invalidate the
whole point of the comparison.
"""

import numpy as np


def fixed_threshold(calib_confidences, target_coverage=0.90):
    """
    Pick a single threshold from clean calibration confidences such that
    approximately `target_coverage` fraction of calibration samples are
    accepted (confidence >= threshold).

    Parameters
    ----------
    calib_confidences : array-like of float
        Confidence scores from the CLEAN calibration set only.
    target_coverage : float, default 0.90
        Desired coverage on calibration data used to pick the threshold.

    Returns
    -------
    float
        The fixed threshold tau. Apply this same value, unchanged, to
        every shifted test condition.
    """
    calib_confidences = np.asarray(calib_confidences, dtype=float)
    if calib_confidences.size == 0:
        raise ValueError("calib_confidences is empty")
    if not (0.0 < target_coverage <= 1.0):
        raise ValueError("target_coverage must be in (0, 1]")

    # To accept the top `target_coverage` fraction by confidence, the
    # threshold is the (1 - target_coverage) quantile.
    tau = np.quantile(calib_confidences, 1.0 - target_coverage)
    return float(tau)


def coverage_controlled_thresholds(calib_confidences, target_coverages):
    """
    Pick one threshold per requested target coverage level, each computed
    from clean calibration confidences only.

    Parameters
    ----------
    calib_confidences : array-like of float
        Confidence scores from the CLEAN calibration set only.
    target_coverages : list of float
        e.g. [0.70, 0.80, 0.90, 0.95]

    Returns
    -------
    dict
        {target_coverage: threshold}. Apply each threshold, unchanged,
        to every shifted test condition -- achieved coverage on shifted
        data will generally differ from the target; that drift is a
        primary result of this study, not something to correct for.
    """
    calib_confidences = np.asarray(calib_confidences, dtype=float)
    if calib_confidences.size == 0:
        raise ValueError("calib_confidences is empty")

    thresholds = {}
    for c in target_coverages:
        if not (0.0 < c <= 1.0):
            raise ValueError(f"target coverage {c} must be in (0, 1]")
        thresholds[c] = float(np.quantile(calib_confidences, 1.0 - c))
    return thresholds


# ---------------------------------------------------------------------
# Sanity checks / unit tests against dummy data
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    CALIB_PATH = PROJECT_ROOT / "results" / "metrics" / "calib_predictions.csv"

    calib_df = pd.read_csv(CALIB_PATH)
    print(f"Calibration set size: {len(calib_df)}")

    # --- Test 1: fixed_threshold basic sanity ---
    for conf_col in ["confidence_raw", "confidence_calibrated"]:
        calib_conf = calib_df[conf_col].values
        tau = fixed_threshold(calib_conf, target_coverage=0.90)
        assert 0.0 <= tau <= 1.0, "threshold out of expected [0,1] range"

        # verify it actually produces ~90% coverage ON THE CALIBRATION SET
        achieved_coverage = np.mean(calib_conf >= tau)
        print(f"[fixed_threshold] {conf_col}: tau={tau:.4f}, "
              f"achieved coverage on calib = {achieved_coverage:.3f} (target 0.90)")
        assert abs(achieved_coverage - 0.90) < 0.03, \
            "fixed_threshold coverage is off by more than expected quantile noise"

    # --- Test 2: coverage_controlled_thresholds monotonicity ---
    # Higher target coverage -> more lenient (LOWER) threshold, since we
    # need to accept more samples. This is the easy off-by-one bug to catch.
    calib_conf = calib_df["confidence_raw"].values
    targets = [0.70, 0.80, 0.90, 0.95]
    thresholds = coverage_controlled_thresholds(calib_conf, targets)

    print("\n[coverage_controlled_thresholds] confidence_raw:")
    for c in targets:
        achieved = np.mean(calib_conf >= thresholds[c])
        print(f"  target={c:.2f} -> tau={thresholds[c]:.4f}, "
              f"achieved coverage on calib = {achieved:.3f}")

    tau_values = [thresholds[c] for c in targets]
    assert all(tau_values[i] >= tau_values[i + 1] for i in range(len(tau_values) - 1)), \
        "BUG: threshold should DECREASE as target coverage INCREASES " \
        "(more lenient threshold needed to accept more samples)"

    # --- Test 3: empty / bad input handling ---
    try:
        fixed_threshold([], target_coverage=0.9)
        raise AssertionError("should have raised ValueError on empty input")
    except ValueError:
        pass

    try:
        fixed_threshold(calib_conf, target_coverage=1.5)
        raise AssertionError("should have raised ValueError on invalid target_coverage")
    except ValueError:
        pass

    print("\nAll selection.py sanity checks passed.")