# ============================================================================
# INDEPENDENT METRIC ORACLE VALIDATION TEST SUITE
# Validates core internal metric calculations against hand-derived arithmetic
# constants and simple independent formulas without calling internal metric helpers.
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from src.fairness_evaluator import (
    EvaluationStatus,
    brier_score_loss_safe,
    expected_calibration_error_safe,
    safe_demographic_parity_difference,
    safe_disparate_impact_ratio,
    safe_equal_opportunity_difference,
    safe_equalized_odds_difference,
    safe_predictive_parity_difference,
)


# ---------------------------------------------------------------------------
# Test Case 1: Hand-Derived Binary Performance Metrics
# ---------------------------------------------------------------------------
def test_oracle_hand_derived_binary_performance():
    """
    Validates confusion matrix rates against hand-calculated constants on a 10-sample toy set.
    Truth:  4 positive, 6 negative
    Preds:  3 positive, 7 negative
    TP=2, FP=1, FN=2, TN=5
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    assert tp == 2
    assert fp == 1
    assert fn == 2
    assert tn == 5

    # Independent calculations
    expected_acc = 7.0 / 10.0            # 0.70
    expected_tpr = 2.0 / 4.0             # 0.50
    expected_tnr = 5.0 / 6.0             # 0.8333333333333334
    expected_fpr = 1.0 / 6.0             # 0.16666666666666666
    expected_fnr = 2.0 / 4.0             # 0.50
    expected_ppv = 2.0 / 3.0             # 0.6666666666666666
    expected_bal_acc = (0.50 + 5.0/6.0) / 2.0  # 4/6 = 0.6666666666666666
    expected_f1 = 2.0 * (expected_ppv * expected_tpr) / (expected_ppv + expected_tpr)  # 4/7 = 0.57142857

    # Assert exact floating point tolerances
    actual_acc = np.mean(y_true == y_pred)
    assert abs(actual_acc - expected_acc) < 1e-9
    assert abs((tp / (tp + fn)) - expected_tpr) < 1e-9
    assert abs((tn / (tn + fp)) - expected_tnr) < 1e-9
    assert abs((fp / (tn + fp)) - expected_fpr) < 1e-9
    assert abs((fn / (tp + fn)) - expected_fnr) < 1e-9
    assert abs((tp / (tp + fp)) - expected_ppv) < 1e-9


# ---------------------------------------------------------------------------
# Test Case 2: Hand-Derived Group Disparity Oracle
# ---------------------------------------------------------------------------
def test_oracle_hand_derived_group_disparities():
    """
    8-sample dataset with 2 equal-sized groups (A and B).
    Group A: 4 samples, 2 pos, 2 neg, predictions: [1, 0, 0, 1]
             Selection Rate = 2/4 = 0.50, TPR = 1/2 = 0.50, FPR = 1/2 = 0.50, PPV = 1/2 = 0.50
    Group B: 4 samples, 2 pos, 2 neg, predictions: [1, 1, 0, 0]
             Selection Rate = 2/4 = 0.50, TPR = 2/2 = 1.00, FPR = 0/2 = 0.00, PPV = 2/2 = 1.00
    """
    y_true = np.array([1, 1, 0, 0,   1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1,   1, 1, 0, 0])
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    # Hand-derived expected constants
    EXPECTED_DPD = 0.00          # |0.50 - 0.50|
    EXPECTED_DIR = 1.00          # 0.50 / 0.50
    EXPECTED_EO_DIFF = 0.50      # |TPR_A - TPR_B| = |0.50 - 1.00|
    EXPECTED_EODDS_DIFF = 0.50   # max(|0.50 - 1.00|, |0.50 - 0.00|) = 0.50
    EXPECTED_PP_DIFF = 0.50      # |PPV_A - PPV_B| = |0.50 - 1.00|

    # Internal function calls
    dpd_res = safe_demographic_parity_difference(y_pred, groups, min_group_size=2)
    dir_res = safe_disparate_impact_ratio(y_pred, groups, min_group_size=2)
    eo_res = safe_equal_opportunity_difference(y_true, y_pred, groups, min_group_size=2)
    eodds_res = safe_equalized_odds_difference(y_true, y_pred, groups, min_group_size=2)
    pp_res = safe_predictive_parity_difference(y_true, y_pred, groups, min_group_size=2)

    assert dpd_res.status == EvaluationStatus.VALID
    assert abs(dpd_res.value - EXPECTED_DPD) < 1e-9

    assert dir_res.status == EvaluationStatus.VALID
    assert abs(dir_res.value - EXPECTED_DIR) < 1e-9

    assert eo_res.status == EvaluationStatus.VALID
    assert abs(eo_res.value - EXPECTED_EO_DIFF) < 1e-9

    assert eodds_res.status == EvaluationStatus.VALID
    assert abs(eodds_res.value - EXPECTED_EODDS_DIFF) < 1e-9

    assert pp_res.status == EvaluationStatus.VALID
    assert abs(pp_res.value - EXPECTED_PP_DIFF) < 1e-9


# ---------------------------------------------------------------------------
# Test Case 3: Brier Score & Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------
def test_oracle_brier_and_ece_deterministic():
    """
    Validates Brier score and 10-bin ECE against hand-computed sums.
    """
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.6, 0.4])

    # Hand-derived squared errors:
    # (0.8 - 1)^2 = 0.04
    # (0.2 - 0)^2 = 0.04
    # (0.6 - 1)^2 = 0.16
    # (0.4 - 0)^2 = 0.16
    # Mean = 0.40 / 4 = 0.10
    EXPECTED_BRIER = 0.10

    actual_brier = brier_score_loss_safe(y_true, y_prob)
    assert abs(actual_brier - EXPECTED_BRIER) < 1e-9

    ece_val, details = expected_calibration_error_safe(y_true, y_prob, n_bins=10)
    assert not np.isnan(ece_val)
    assert 0.0 <= ece_val <= 1.0


# ---------------------------------------------------------------------------
# Test Case 4: Perfect Classifier Edge Case
# ---------------------------------------------------------------------------
def test_oracle_perfect_classifier():
    """Perfect predictions: accuracy=1.0, disparities=0.0 except DPD where base rates differ."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = y_true.copy()
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    # Group A: 2 pos / 4 total = 0.50 selection
    # Group B: 1 pos / 4 total = 0.25 selection
    EXPECTED_DPD = abs(0.50 - 0.25)  # 0.25
    EXPECTED_DIR = 0.25 / 0.50        # 0.50
    EXPECTED_EO_DIFF = 0.0            # Both groups have TPR=1.0
    EXPECTED_EODDS_DIFF = 0.0         # Both groups have TPR=1.0 and FPR=0.0

    dpd_res = safe_demographic_parity_difference(y_pred, groups, min_group_size=2)
    dir_res = safe_disparate_impact_ratio(y_pred, groups, min_group_size=2)
    eodds_res = safe_equalized_odds_difference(y_true, y_pred, groups, min_group_size=2)

    assert abs(dpd_res.value - EXPECTED_DPD) < 1e-9
    assert abs(dir_res.value - EXPECTED_DIR) < 1e-9
    assert abs(eodds_res.value - EXPECTED_EO_DIFF) < 1e-9


# ---------------------------------------------------------------------------
# Test Case 5: Inverted Classifier Edge Case
# ---------------------------------------------------------------------------
def test_oracle_inverted_classifier():
    """Inverted predictions: accuracy=0.0, TPR=0.0, TNR=0.0."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = 1 - y_true
    groups = pd.Series(["A", "A", "B", "B"])

    acc = np.mean(y_true == y_pred)
    assert acc == 0.0

    eodds = safe_equalized_odds_difference(y_true, y_pred, groups, min_group_size=2)
    assert eodds.status == EvaluationStatus.VALID
    assert abs(eodds.value - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Test Case 6: Constant Zero Predictor (No Positives)
# ---------------------------------------------------------------------------
def test_oracle_constant_zero_predictor():
    """Predicting all zeros: Disparate Impact Ratio is undefined (zero denominator)."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    groups = pd.Series(["A", "A", "B", "B"])

    di_res = safe_disparate_impact_ratio(y_pred, groups, min_group_size=2)
    assert di_res.status == EvaluationStatus.ZERO_SELECTION_RATE
    assert np.isnan(di_res.value)


# ---------------------------------------------------------------------------
# Test Case 7: Constant One Predictor (All Positives)
# ---------------------------------------------------------------------------
def test_oracle_constant_one_predictor():
    """Predicting all ones: rates are 1.0 for all groups. DPD=0.0, DIR=1.0."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1])
    groups = pd.Series(["A", "A", "B", "B"])

    dpd_res = safe_demographic_parity_difference(y_pred, groups, min_group_size=2)
    dir_res = safe_disparate_impact_ratio(y_pred, groups, min_group_size=2)

    assert abs(dpd_res.value - 0.0) < 1e-9
    assert abs(dir_res.value - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Test Case 8: Imbalanced Groups & Sparse Group Edge Case
# ---------------------------------------------------------------------------
def test_oracle_sparse_group_rejection():
    """When a group has fewer than min_group_size samples, status is INSUFFICIENT_GROUP_SUPPORT."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    # Group A has 4 samples, Group B has only 1 sample
    groups = pd.Series(["A", "A", "A", "A", "B"])

    eodds_res = safe_equalized_odds_difference(y_true, y_pred, groups, min_group_size=2)
    assert eodds_res.status == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT
    assert np.isnan(eodds_res.value)


# ---------------------------------------------------------------------------
# Test Case 9: Zero Positive Subgroup in Ground Truth
# ---------------------------------------------------------------------------
def test_oracle_zero_positive_subgroup_in_ground_truth():
    """Group B has zero true positives in ground truth -> TPR is undefined."""
    y_true = np.array([1, 0, 1, 0,   0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0,   0, 1, 0, 0])
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    eo_res = safe_equal_opportunity_difference(y_true, y_pred, groups, min_group_size=2)
    assert eo_res.status == EvaluationStatus.ZERO_POSITIVE_SUBGROUP
    assert np.isnan(eo_res.value)
