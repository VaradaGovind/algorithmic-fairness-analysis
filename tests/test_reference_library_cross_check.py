# ============================================================================
# REFERENCE LIBRARY CROSS-CHECK TEST SUITE & REPORT GENERATOR
# Compares internal metric results against external reference implementations
# (Fairlearn and scikit-learn), asserts numerical tolerance, and generates
# a machine-readable cross-check report (docs/reference_library_cross_check.json).
# ============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
import fairlearn.metrics as flm

from src.fairness_evaluator import (
    brier_score_loss_safe,
    evaluate_fairness_defensively,
    safe_demographic_parity_difference,
    safe_equalized_odds_difference,
)

REPORT_PATH = Path(__file__).resolve().parents[1] / "docs" / "reference_library_cross_check.json"


def test_reference_library_cross_check_and_report():
    """
    Evaluates internal metrics against Fairlearn and scikit-learn on a standard fixture.
    Asserts strict absolute tolerance (< 1e-6) and outputs docs/reference_library_cross_check.json.
    """
    # Deterministic standard fixture
    rng = np.random.RandomState(42)
    n = 300
    y_true = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    # Model predictions with moderate noise
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0.1, 0.4, size=n), 0.01, 0.99)
    y_pred = (y_prob >= 0.50).astype(int)
    groups = rng.choice(["Subgroup_Alpha", "Subgroup_Beta"], size=n, p=[0.60, 0.40])
    sensitive_series = pd.Series(groups)

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive=sensitive_series,
        compute_uncertainty=False,
    )

    records = []

    def check_metric(name, internal_val, ref_val, tol=1e-6):
        diff = abs(float(internal_val) - float(ref_val))
        passed = diff <= tol
        records.append({
            "metric": name,
            "internal_result": round(float(internal_val), 8),
            "reference_result": round(float(ref_val), 8),
            "absolute_difference": round(diff, 10),
            "tolerance": tol,
            "pass_fail": "PASS" if passed else "FAIL",
            "reference_library": "fairlearn" if "parity" in name.lower() or "odds" in name.lower() else "scikit-learn",
        })
        assert passed, f"Metric '{name}' mismatch: internal {internal_val} vs ref {ref_val} (diff={diff} > {tol})"

    # 1. Performance Metrics (vs scikit-learn)
    check_metric("Accuracy", suite.metrics["Accuracy"].value, skm.accuracy_score(y_true, y_pred))
    check_metric("Balanced Accuracy", suite.metrics["Balanced Accuracy"].value, skm.balanced_accuracy_score(y_true, y_pred))
    check_metric("Precision", suite.metrics["Precision"].value, skm.precision_score(y_true, y_pred))
    check_metric("Recall", suite.metrics["Recall"].value, skm.recall_score(y_true, y_pred))
    check_metric("F1-Score", suite.metrics["F1-Score"].value, skm.f1_score(y_true, y_pred))
    check_metric("Brier Score", suite.metrics["Brier Score"].value, skm.brier_score_loss(y_true, y_prob))

    # 2. Fairness Disparity Metrics (vs Fairlearn)
    fl_dpd = flm.demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_series)
    check_metric("Demographic Parity Difference", suite.metrics["Demographic Parity Diff"].value, fl_dpd)

    fl_eod = flm.equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_series)
    check_metric("Equalized Odds Difference", suite.metrics["Equalized Odds Diff"].value, fl_eod)

    # 3. Subgroup Conditional Selection Rates (vs Fairlearn)
    mask_a = (sensitive_series == "Subgroup_Alpha")
    fl_sel_a = flm.selection_rate(y_true[mask_a], y_pred[mask_a])
    check_metric("Selection Rate (Subgroup_Alpha)", suite.subgroup_metrics["Subgroup_Alpha"]["predicted_positive_rate"], fl_sel_a)

    mask_b = (sensitive_series == "Subgroup_Beta")
    fl_sel_b = flm.selection_rate(y_true[mask_b], y_pred[mask_b])
    check_metric("Selection Rate (Subgroup_Beta)", suite.subgroup_metrics["Subgroup_Beta"]["predicted_positive_rate"], fl_sel_b)

    # 4. Subgroup True Positive Rates (vs Fairlearn)
    fl_tpr_a = flm.true_positive_rate(y_true[mask_a], y_pred[mask_a])
    check_metric("TPR (Subgroup_Alpha)", suite.subgroup_metrics["Subgroup_Alpha"]["TPR"], fl_tpr_a)

    fl_tpr_b = flm.true_positive_rate(y_true[mask_b], y_pred[mask_b])
    check_metric("TPR (Subgroup_Beta)", suite.subgroup_metrics["Subgroup_Beta"]["TPR"], fl_tpr_b)

    # Write machine-readable report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "report_title": "Independent Metric Reference Library Cross-Check Report",
        "evaluation_standard": "POL-STD-2026.1",
        "sample_size": n,
        "tolerance": 1e-6,
        "total_metrics_evaluated": len(records),
        "all_passed": all(r["pass_fail"] == "PASS" for r in records),
        "methodological_notice": (
            "Independent implementation cross-check verifying mathematical agreement between "
            "internal defensive metrics and external reference libraries (Fairlearn and scikit-learn). "
            "Neither library is treated as infallible ground truth; agreement confirms algorithmic equivalence."
        ),
        "metrics_cross_check": records,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    assert REPORT_PATH.exists()
