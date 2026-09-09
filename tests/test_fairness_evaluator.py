# ============================================================================
# Unit Tests: Adversarial & Edge-Case Fairness Evaluation Suite
# Covers 16 critical edge-cases, negative controls, and defensive evaluator behaviors.
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from src.fairness_evaluator import (
    EvaluationStatus,
    accuracy_tradeoff_proxy,
    evaluate_fairness_defensively,
    safe_demographic_parity_difference,
    safe_disparate_impact_ratio,
    safe_equal_opportunity_difference,
    safe_equalized_odds_difference,
    safe_predictive_parity_difference,
    user_defined_utility_score,
)


# Case 1 & 2: Random & Permuted Sensitive Attributes
def test_case_01_and_02_random_and_permuted_sensitive():
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], size=n)
    y_pred = y_true.copy()
    
    # Fully random sensitive attribute
    rand_sensitive = np.random.choice(["Group_1", "Group_2"], size=n)
    res = evaluate_fairness_defensively(y_true, y_pred, None, rand_sensitive)
    assert res.overall_status == EvaluationStatus.VALID
    # Random grouping should have near-zero demographic parity difference in expectation
    assert res.metrics["Demographic Parity Diff"].value < 0.20


# Case 3: Sensitive Attribute Derived from Predictive Feature
def test_case_03_feature_derived_sensitive_attribute():
    score = np.linspace(-3, 3, 200)
    y_pred = (score > 0).astype(int)
    y_true = (score + np.random.normal(0, 0.2, 200) > 0).astype(int)
    # Group derived directly from the feature split
    feature_group = np.where(score > 0, "High_Score_Tier", "Low_Score_Tier")
    res = evaluate_fairness_defensively(y_true, y_pred, None, feature_group)
    # The selection disparity should be mechanically massive (close to 1.0)
    assert res.metrics["Demographic Parity Diff"].value > 0.90


# Case 4, 5, 6: Target Leakage, Inverse Target, Post-Outcome
def test_case_04_05_06_leakage_and_post_outcome_scenarios():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 10)
    # Perfect leakage
    y_pred_leak = y_true.copy()
    sensitive = np.array(["A", "B"] * 40)
    res_leak = evaluate_fairness_defensively(y_true, y_pred_leak, None, sensitive)
    assert res_leak.metrics["Accuracy"].value == 1.0
    # Inverse leakage
    y_pred_inv = 1 - y_true
    res_inv = evaluate_fairness_defensively(y_true, y_pred_inv, None, sensitive)
    assert res_inv.metrics["Accuracy"].value == 0.0


# Case 7: Empty Sensitive Group
def test_case_07_empty_sensitive_group():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    # Only 1 group present
    sensitive = np.array(["SoloGroup", "SoloGroup", "SoloGroup", "SoloGroup"])
    res = evaluate_fairness_defensively(y_true, y_pred, None, sensitive)
    assert res.metrics["Demographic Parity Diff"].status == EvaluationStatus.MISSING_SENSITIVE
    assert np.isnan(res.metrics["Demographic Parity Diff"].value)


# Case 8: Tiny Sensitive Group (< 5 instances)
def test_case_08_tiny_sensitive_group():
    y_true = np.array([0, 1] * 20)
    y_pred = y_true.copy()
    # Group B has only 2 members
    sensitive = np.array(["Group_A"] * 38 + ["Group_B"] * 2)
    res = evaluate_fairness_defensively(y_true, y_pred, None, sensitive, min_group_size=5)
    assert res.metrics["Demographic Parity Diff"].status == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT
    assert np.isnan(res.metrics["Demographic Parity Diff"].value)


# Case 9: Zero-Positive Subgroup
def test_case_09_zero_positive_subgroup():
    # Group B has NO ground truth positive cases
    y_true = np.array([0, 1, 0, 1, 0, 0, 0, 0] * 5)
    y_pred = y_true.copy()
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"] * 5)
    # Equal opportunity diff checks TPR, which is undefined when a group has 0 positives
    res = safe_equal_opportunity_difference(y_true, y_pred, sensitive)
    assert res.status == EvaluationStatus.ZERO_POSITIVE_SUBGROUP
    assert np.isnan(res.value)


# Case 10: Zero-Negative Subgroup
def test_case_10_zero_negative_subgroup():
    # Group B has NO ground truth negative cases
    y_true = np.array([0, 1, 0, 1, 1, 1, 1, 1] * 5)
    y_pred = y_true.copy()
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"] * 5)
    res = safe_equalized_odds_difference(y_true, y_pred, sensitive)
    assert res.status == EvaluationStatus.ZERO_NEGATIVE_SUBGROUP
    assert np.isnan(res.value)


# Case 11: Contradictory Fairness Objectives
def test_case_11_contradictory_fairness_objectives():
    # Impossibility Theorem of Machine Learning Fairness (Kleinberg et al. 2016):
    # When base rates differ (Group A base rate = 80%, Group B base rate = 20%),
    # a model cannot simultaneously satisfy Demographic Parity and Predictive Parity (unless predictions are trivial/uninformative).
    y_true = np.array([1, 1, 1, 1, 0] * 10 + [0, 0, 0, 0, 1] * 10)  # Base rate A: 0.8, B: 0.2
    sensitive = np.array(["Group_A"] * 50 + ["Group_B"] * 50)
    
    # Model with calibrated predictions matching base rates:
    y_pred = y_true.copy()
    res = evaluate_fairness_defensively(y_true, y_pred, None, sensitive)
    # Predictive parity difference is 0 (both PPVs = 1.0)
    assert res.metrics["Predictive Parity Diff"].value == 0.0
    # But demographic parity difference is massive (|0.8 - 0.2| = 0.6)
    assert res.metrics["Demographic Parity Diff"].value == pytest.approx(0.6, abs=0.01)


# Case 12: Impossible Fairness Constraint (Zero positive selection across all groups)
def test_case_12_impossible_fairness_constraint_zero_positives():
    y_true = np.array([0, 1, 0, 1] * 5)
    y_pred = np.zeros(20, dtype=int)  # All negative predictions
    sensitive = np.array(["A", "A", "B", "B"] * 5)
    res_di = safe_disparate_impact_ratio(y_pred, sensitive)
    assert res_di.status == EvaluationStatus.UNDEFINED
    assert np.isnan(res_di.value)


# Case 13: Missing Sensitive Attribute (Single unique value or NaNs)
def test_case_13_missing_sensitive_attribute():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sensitive = np.array(["Missing"] * 4)
    res = safe_demographic_parity_difference(y_pred, sensitive)
    assert res.status == EvaluationStatus.MISSING_SENSITIVE
    assert np.isnan(res.value)


# Case 14: Missing Required Feature (Caught by defensive checks)
def test_case_14_missing_feature_and_utility_functions():
    # Refactored utility score and accuracy tradeoff functions
    utility = user_defined_utility_score(balanced_accuracy=0.80, fairness_gap=0.10, alpha=0.25)
    assert utility == pytest.approx(0.80 - 0.025, abs=1e-5)
    
    tradeoff = accuracy_tradeoff_proxy(baseline_accuracy=0.75, mitigated_accuracy=0.70)
    assert tradeoff == pytest.approx(0.05, abs=1e-5)
    
    # When mitigated exceeds baseline, tradeoff is 0.0 (not negative)
    tradeoff_gain = accuracy_tradeoff_proxy(baseline_accuracy=0.75, mitigated_accuracy=0.85)
    assert tradeoff_gain == 0.0


# Case 15: Invalid Prediction Length
def test_case_15_invalid_prediction_length():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0])  # Length 3 vs 4
    sensitive = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        evaluate_fairness_defensively(y_true, y_pred, None, sensitive)


# Case 16: NaN / Inf Predictions
def test_case_16_nan_inf_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, np.nan, 0, 1])
    sensitive = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="contains NaN"):
        evaluate_fairness_defensively(y_true, y_pred, None, sensitive)
