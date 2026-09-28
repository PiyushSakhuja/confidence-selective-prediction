"""
plots.py

Generates all figures and tables for the paper from the outputs of
run_analysis.py:
    - results/metrics/full_results.csv
    - results/metrics/rc_curves/{shift_type}_sev{severity}_{confidence_type}.npz

Figures produced (matching the research plan):
    Fig 2 - accuracy vs severity (no abstention)
    Fig 3 - risk-coverage curves (PRIMARY) - grid, one subplot per shift type
    Fig 4 - AURC vs severity (PRIMARY)
    Fig 5 - achieved coverage vs target coverage under shift (coverage drift)
    Fig 6 - ECE vs severity, raw vs calibrated confidence

Tables produced:
    Table 1 - main results summary (accuracy, AURC, ECE per condition)
    Table 2 - coverage-controlled policy behavior

Usage:
    python3 plots.py
    (reads from results/metrics/, writes to results/figures/ and
    results/metrics/tables/)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

RESULTS_CSV = "results/metrics/full_results.csv"
RC_CURVE_DIR = "results/metrics/rc_curves"
FIGURE_DIR = "results/figures"
TABLE_DIR = "results/metrics/tables"

SHIFT_ORDER = ["noise", "blur", "brightness", "rotation"]
SEVERITIES_TO_PLOT_IN_RC = [0, 2, 4]  # keep risk-coverage subplots readable

DPI = 300


def _ensure_dirs():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def _load_results():
    df = pd.read_csv(RESULTS_CSV)
    return df


def _load_rc_curve(shift_type, severity, confidence_type):
    path = os.path.join(RC_CURVE_DIR, f"{shift_type}_sev{severity}_{confidence_type}.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["coverages"], data["risks"]


# ---------------------------------------------------------------------
# Fig 2: accuracy vs severity (no abstention)
# ---------------------------------------------------------------------
def plot_accuracy_vs_severity(df, save_path=None):
    # one row per (shift_type, severity, confidence_type) has the same
    # no_abstention_accuracy repeated across policy rows -> dedupe
    plot_df = df[df.confidence_type == "raw"].drop_duplicates(
        subset=["shift_type", "severity"]
    )
    # include clean as severity 0 baseline for every shift type line
    clean_row = plot_df[plot_df.shift_type == "clean"]
    clean_acc = clean_row["no_abstention_accuracy"].values[0] if len(clean_row) else None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for shift_type in SHIFT_ORDER:
        sub = plot_df[plot_df.shift_type == shift_type].sort_values("severity")
        if sub.empty:
            continue
        severities = [0] + sub["severity"].tolist()
        accs = ([clean_acc] if clean_acc is not None else []) + sub["no_abstention_accuracy"].tolist()
        ax.plot(severities, accs, marker="o", label=shift_type)

    ax.set_xlabel("Shift severity")
    ax.set_ylabel("Accuracy (no abstention)")
    ax.set_title("Model accuracy under increasing distribution shift")
    ax.set_xticks(sorted(plot_df["severity"].unique().tolist() + [0]))
    ax.legend(title="Shift type")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ---------------------------------------------------------------------
# Fig 3: risk-coverage curves (PRIMARY)
# ---------------------------------------------------------------------
def plot_risk_coverage_curves(confidence_type="raw", save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, shift_type in enumerate(SHIFT_ORDER):
        ax = axes[i]
        for severity in SEVERITIES_TO_PLOT_IN_RC:
            if severity == 0:
                cov, risk = _load_rc_curve("clean", 0, confidence_type)
                label = "clean (sev 0)"
            else:
                cov, risk = _load_rc_curve(shift_type, severity, confidence_type)
                label = f"sev {severity}"
            if cov is None:
                continue
            ax.plot(cov, risk, label=label, linewidth=1.5)

        ax.set_title(shift_type)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk")
        ax.legend(fontsize=8)

    fig.suptitle(f"Risk-coverage curves by shift type ({confidence_type} confidence)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ---------------------------------------------------------------------
# Fig 4: AURC vs severity (PRIMARY)
# ---------------------------------------------------------------------
def plot_aurc_vs_severity(df, save_path=None):
    plot_df = df.drop_duplicates(subset=["shift_type", "severity", "confidence_type"])
    clean_aurc = {
        conf_type: plot_df[(plot_df.shift_type == "clean") & (plot_df.confidence_type == conf_type)]["aurc"].values
        for conf_type in ["raw", "calibrated"]
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, conf_type in zip(axes, ["raw", "calibrated"]):
        for shift_type in SHIFT_ORDER:
            sub = plot_df[(plot_df.shift_type == shift_type) & (plot_df.confidence_type == conf_type)].sort_values("severity")
            if sub.empty:
                continue
            severities = [0] + sub["severity"].tolist()
            aurcs = (list(clean_aurc[conf_type]) if len(clean_aurc[conf_type]) else []) + sub["aurc"].tolist()
            ax.plot(severities, aurcs, marker="o", label=shift_type)
        ax.set_xlabel("Shift severity")
        ax.set_title(f"{conf_type} confidence")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("AURC (lower is better)")

    fig.suptitle("Area under the risk-coverage curve vs. shift severity")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ---------------------------------------------------------------------
# Fig 5: coverage drift (achieved vs target, coverage-controlled policy)
# ---------------------------------------------------------------------
def plot_coverage_drift(df, confidence_type="raw", save_path=None):
    sub_df = df[
        (df.policy == "coverage_controlled") & (df.confidence_type == confidence_type)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, shift_type in enumerate(SHIFT_ORDER):
        ax = axes[i]
        shift_sub = sub_df[sub_df.shift_type == shift_type]
        for target_c in sorted(shift_sub["target_coverage"].unique()):
            tc_sub = shift_sub[shift_sub.target_coverage == target_c].sort_values("severity")
            if tc_sub.empty:
                continue
            ax.plot(
                tc_sub["severity"], tc_sub["achieved_coverage"],
                marker="o", label=f"target={target_c:.2f}"
            )
            # reference line at the target itself
            ax.axhline(target_c, linestyle="--", linewidth=0.7, alpha=0.4)

        ax.set_title(shift_type)
        ax.set_xlabel("Shift severity")
        ax.set_ylabel("Achieved coverage")
        ax.legend(fontsize=7)

    fig.suptitle(f"Coverage drift under shift: calibration-time target vs. achieved ({confidence_type} confidence)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ---------------------------------------------------------------------
# Fig 6: ECE vs severity, raw vs calibrated
# ---------------------------------------------------------------------
def plot_ece_vs_severity(df, save_path=None):
    plot_df = df.drop_duplicates(subset=["shift_type", "severity", "confidence_type"])

    fig, axes = plt.subplots(1, len(SHIFT_ORDER), figsize=(16, 4), sharey=True)

    for ax, shift_type in zip(axes, SHIFT_ORDER):
        for conf_type in ["raw", "calibrated"]:
            sub = plot_df[
                (plot_df.shift_type == shift_type) & (plot_df.confidence_type == conf_type)
            ].sort_values("severity")
            clean_row = plot_df[(plot_df.shift_type == "clean") & (plot_df.confidence_type == conf_type)]
            severities = ([0] if len(clean_row) else []) + sub["severity"].tolist()
            eces = (clean_row["ece"].tolist() if len(clean_row) else []) + sub["ece"].tolist()
            ax.plot(severities, eces, marker="o", label=conf_type)
        ax.set_title(shift_type)
        ax.set_xlabel("Severity")

    axes[0].set_ylabel("ECE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Expected Calibration Error vs. severity: raw vs. temperature-scaled confidence")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def make_table1_summary(df, save_path=None):
    """Table 1: shift x severity -> accuracy, AURC, ECE (raw + calibrated)."""
    dedup = df.drop_duplicates(subset=["shift_type", "severity", "confidence_type"])

    table = dedup.pivot_table(
        index=["shift_type", "severity"],
        columns="confidence_type",
        values=["no_abstention_accuracy", "aurc", "ece"],
    )
    table = table.round(4)

    if save_path:
        table.to_csv(save_path)
    return table


def make_table2_coverage_controlled(df, save_path=None):
    """Table 2: coverage-controlled policy behavior across conditions."""
    sub = df[df.policy == "coverage_controlled"][
        ["shift_type", "severity", "confidence_type", "target_coverage",
         "achieved_coverage", "selective_accuracy", "selective_risk"]
    ].sort_values(["shift_type", "severity", "confidence_type", "target_coverage"])
    sub = sub.round(4)

    if save_path:
        sub.to_csv(save_path, index=False)
    return sub


# ---------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------
if __name__ == "__main__":
    _ensure_dirs()
    df = _load_results()

    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")
    print("Generating figures...")

    plot_accuracy_vs_severity(df, save_path=f"{FIGURE_DIR}/fig2_accuracy_vs_severity.png")
    print("  Fig 2 saved: accuracy_vs_severity")

    plot_risk_coverage_curves("raw", save_path=f"{FIGURE_DIR}/fig3_risk_coverage_curves_raw.png")
    plot_risk_coverage_curves("calibrated", save_path=f"{FIGURE_DIR}/fig3_risk_coverage_curves_calibrated.png")
    print("  Fig 3 saved: risk_coverage_curves (raw + calibrated)")

    plot_aurc_vs_severity(df, save_path=f"{FIGURE_DIR}/fig4_aurc_vs_severity.png")
    print("  Fig 4 saved: aurc_vs_severity")

    plot_coverage_drift(df, "raw", save_path=f"{FIGURE_DIR}/fig5_coverage_drift_raw.png")
    print("  Fig 5 saved: coverage_drift")

    plot_ece_vs_severity(df, save_path=f"{FIGURE_DIR}/fig6_ece_vs_severity.png")
    print("  Fig 6 saved: ece_vs_severity")

    print("\nGenerating tables...")
    t1 = make_table1_summary(df, save_path=f"{TABLE_DIR}/table1_summary.csv")
    print(f"  Table 1 saved ({len(t1)} rows)")

    t2 = make_table2_coverage_controlled(df, save_path=f"{TABLE_DIR}/table2_coverage_controlled.csv")
    print(f"  Table 2 saved ({len(t2)} rows)")

    print("\nAll figures and tables generated successfully.")
    print(f"Figures: {FIGURE_DIR}/")
    print(f"Tables:  {TABLE_DIR}/")