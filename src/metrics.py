"""
metrics.py

Core evaluation metrics for the selective prediction study.

Threshold-based metrics (depend on a chosen threshold from selection.py):
    - coverage(confidences, threshold)
    - selective_accuracy(correct, confidences, threshold)
    - selective_risk(correct, confidences, threshold)

Threshold-free metrics (primary evaluation, per your research plan):
    - risk_coverage_curve(correct, confidences)  -> the curve itself
    - aurc(coverages, risks)                      -> area under that curve

Calibration metric:
    - ece(correct, confidences, n_bins=10)

All functions take plain numpy arrays / array-likes: `correct` is a
boolean (or 0/1) array, `confidences` is a float array in [0, 1].
"""

import numpy as np


def coverage(confidences, threshold):
    """Fraction of samples accepted (confidence >= threshold)."""
    confidences = np.asarray(confidences, dtype=float)
    return float(np.mean(confidences >= threshold))


def selective_accuracy(correct, confidences, threshold):
    """
    Accuracy among accepted samples only (confidence >= threshold).
    Returns np.nan if no samples are accepted at this threshold.
    """
    correct = np.asarray(correct, dtype=bool)
    confidences = np.asarray(confidences, dtype=float)
    mask = confidences >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(correct[mask]))


def selective_risk(correct, confidences, threshold):
    """1 - selective_accuracy. NaN if nothing is accepted."""
    acc = selective_accuracy(correct, confidences, threshold)
    if np.isnan(acc):
        return float("nan")
    return 1.0 - acc


def risk_coverage_curve(correct, confidences):
    """
    Compute the full risk-coverage curve by sorting samples from most to
    least confident, and at each cutoff k (accepting the top-k most
    confident samples) computing coverage = k/N and risk = 1 - accuracy
    among those top-k accepted samples.

    Parameters
    ----------
    correct : array-like of bool/int
    confidences : array-like of float

    Returns
    -------
    coverages : np.ndarray, shape (N,)
        Coverage values, evenly spaced from 1/N to 1.0.
    risks : np.ndarray, shape (N,)
        Risk (1 - cumulative selective accuracy) at each coverage level.
    """
    correct = np.asarray(correct, dtype=float)  # float so cumsum is clean
    confidences = np.asarray(confidences, dtype=float)
    n = len(correct)
    if n == 0:
        raise ValueError("empty input to risk_coverage_curve")

    # sort by confidence descending -> most confident samples accepted first
    order = np.argsort(-confidences)
    correct_sorted = correct[order]

    cum_correct = np.cumsum(correct_sorted)
    ks = np.arange(1, n + 1)

    coverages = ks / n
    risks = 1.0 - (cum_correct / ks)

    return coverages, risks


def aurc(coverages, risks):
    """
    Area under the risk-coverage curve, via trapezoidal integration.
    Lower AURC = better (risk stays low across more of the coverage range).
    """
    coverages = np.asarray(coverages, dtype=float)
    risks = np.asarray(risks, dtype=float)
    # integrates risks over coverages; curve must be sorted by coverage
    # ascending, which risk_coverage_curve already guarantees.
    # np.trapz was renamed to np.trapezoid in newer numpy; support both.
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(risks, coverages))


def ece(correct, confidences, n_bins=10):
    """
    Expected Calibration Error via equal-width confidence binning.

    For each bin, computes |avg_confidence - avg_accuracy| and takes a
    weighted average across bins (weighted by number of samples in bin).
    Empty bins are skipped (contribute 0 weight).
    """
    correct = np.asarray(correct, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    n = len(confidences)
    if n == 0:
        raise ValueError("empty input to ece")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            # include the right edge (confidence == 1.0) in the last bin
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        bin_acc = np.mean(correct[mask])
        bin_conf = np.mean(confidences[mask])
        total += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(total)


# ---------------------------------------------------------------------
# Sanity checks / unit tests
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    print("=== Hand-constructed edge case tests ===\n")

    # --- coverage() ---
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert coverage(conf, 0.7) == 0.6, "coverage() basic case failed"  # 3/5 >= 0.7
    print("coverage() basic case: OK")

    # --- selective_accuracy() / selective_risk() ---
    correct = np.array([True, True, False, True, False])
    acc = selective_accuracy(correct, conf, 0.7)  # top 3: [T,T,F] -> 2/3
    assert abs(acc - 2 / 3) < 1e-9, "selective_accuracy() basic case failed"
    print(f"selective_accuracy() basic case: OK ({acc:.4f} == 2/3)")

    risk = selective_risk(correct, conf, 0.7)
    assert abs(risk - 1 / 3) < 1e-9, "selective_risk() basic case failed"
    print(f"selective_risk() basic case: OK ({risk:.4f} == 1/3)")

    # empty-selection edge case
    nan_acc = selective_accuracy(correct, conf, 1.5)  # nothing has conf >= 1.5
    assert np.isnan(nan_acc), "selective_accuracy() should return NaN when nothing accepted"
    print("selective_accuracy() empty-selection edge case: OK (returns NaN)")

    # --- risk_coverage_curve() + aurc(): constant-risk case ---
    # If accuracy is IDENTICAL regardless of confidence rank (confidence
    # carries no information), risk should be ~constant across coverage,
    # and AURC should equal that constant risk.
    rng = np.random.default_rng(0)
    n = 5000
    correct_const = rng.random(n) < 0.8          # 80% accuracy, independent of...
    conf_uninformative = rng.random(n)            # ...random confidence
    cov, risk_curve = risk_coverage_curve(correct_const, conf_uninformative)
    a = aurc(cov, risk_curve)
    print(f"\nUninformative-confidence case: mean risk ~ {np.mean(risk_curve):.4f}, "
          f"AURC = {a:.4f} (expect both close to 0.20 = 1 - 0.8)")
    assert abs(a - 0.20) < 0.02, "AURC should be close to (1 - accuracy) when confidence is uninformative"

    # --- risk_coverage_curve(): informative-confidence case ---
    # Construct data where high confidence really does mean more likely
    # correct. Risk at low coverage (only most-confident accepted) should
    # be LOWER than risk at full coverage.
    conf_informative = rng.random(n)
    correct_informative = rng.random(n) < conf_informative  # correctness tracks confidence
    cov2, risk2 = risk_coverage_curve(correct_informative, conf_informative)
    risk_at_low_coverage = risk2[int(0.1 * n)]   # ~10% coverage
    risk_at_full_coverage = risk2[-1]             # 100% coverage
    print(f"\nInformative-confidence case: risk@~10% coverage = {risk_at_low_coverage:.4f}, "
          f"risk@100% coverage = {risk_at_full_coverage:.4f}")
    assert risk_at_low_coverage < risk_at_full_coverage, \
        "BUG: risk at low coverage should be lower than at full coverage when confidence is informative"
    print("risk_coverage_curve() informative case: OK (risk increases as coverage increases)")

    # --- ece(): perfectly calibrated case ---
    # Construct confidences that exactly equal per-sample accuracy probability,
    # binned average should match, so ECE should be near 0.
    n2 = 20000
    conf_cal = rng.uniform(0.1, 0.99, n2)
    correct_cal = rng.random(n2) < conf_cal  # accuracy matches confidence by construction
    e = ece(correct_cal, conf_cal, n_bins=10)
    print(f"\nPerfectly-calibrated case: ECE = {e:.4f} (expect close to 0)")
    assert e < 0.03, "ECE should be small for near-perfectly-calibrated confidences"

    # --- ece(): badly miscalibrated case (always overconfident) ---
    conf_over = np.full(n2, 0.95)
    correct_over = rng.random(n2) < 0.60  # true accuracy only 60%, but confidence claims 95%
    e_over = ece(correct_over, conf_over, n_bins=10)
    print(f"Overconfident case: ECE = {e_over:.4f} (expect close to 0.35 = |0.95 - 0.60|)")
    assert abs(e_over - 0.35) < 0.03

    print("\n=== Full pipeline test against dummy_predictions.csv ===\n")
    df = pd.read_csv("results/metrics/dummy_predictions.csv")

    from selection import fixed_threshold

    calib_df = df[df.shift_type == "calibration"]
    tau = fixed_threshold(calib_df["confidence_raw"].values, target_coverage=0.90)

    print(f"{'shift_type':<12} {'sev':>3} {'coverage':>9} {'sel_acc':>8} "
          f"{'sel_risk':>9} {'AURC':>7} {'ECE':>7}")

    results = []
    for (shift_type, severity), group in df.groupby(["shift_type", "severity"]):
        if shift_type == "calibration":
            continue
        c = group["correct"].values
        conf = group["confidence_raw"].values

        cov_val = coverage(conf, tau)
        sel_acc = selective_accuracy(c, conf, tau)
        sel_risk = selective_risk(c, conf, tau)
        rc_cov, rc_risk = risk_coverage_curve(c, conf)
        a = aurc(rc_cov, rc_risk)
        e = ece(c, conf, n_bins=10)

        results.append((shift_type, severity, a))
        print(f"{shift_type:<12} {severity:>3} {cov_val:>9.3f} {sel_acc:>8.3f} "
              f"{sel_risk:>9.3f} {a:>7.3f} {e:>7.3f}")

    # sanity check: AURC should generally increase with severity for each shift type
    print("\nChecking AURC increases with severity per shift type...")
    by_shift = {}
    for shift_type, severity, a in results:
        by_shift.setdefault(shift_type, {})[severity] = a
    for shift_type, sev_dict in by_shift.items():
        sevs_sorted = sorted(sev_dict.keys())
        aurcs_sorted = [sev_dict[s] for s in sevs_sorted]
        increasing = all(aurcs_sorted[i] <= aurcs_sorted[i + 1] + 0.02  # small slack for noise
                          for i in range(len(aurcs_sorted) - 1))
        status = "OK" if increasing else "CHECK (not monotonic - inspect dummy data / real data)"
        print(f"  {shift_type}: {[round(a,3) for a in aurcs_sorted]} -> {status}")

    print("\nAll metrics.py sanity checks passed.")