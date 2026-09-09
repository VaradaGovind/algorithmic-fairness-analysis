# ============================================================================
# Test Suite: Evaluator Independence, Policy Engine, Calibration & Gaming Defense
# Validates that fairness evaluation is strictly decoupled from scenario DGP,
# tests the formal FairnessPolicy engine, asserts immunity to 8 adversarial
# gaming models, verifies probability calibration metrics, and checks stress cases.
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from src.canonical_benchmark import (
    MajorityClassClassifier,
    calculate_population_dgp_truth,
    compare_sample_to_population_truth,
    compute_paired_seed_differences,
    generate_controlled_synthetic_benchmark,
    run_canonical_evaluation,
    SplitRegime,
)
from src.fairness_evaluator import (
    EvaluationStatus,
    FairnessEvaluationSuite,
    FairnessPolicy,
    FairnessSuccessStatus,
    ModelDegeneracyStatus,
    PolicyDecision,
    PolicyEvaluationResult,
    brier_score_loss_safe,
    classify_fairness_success,
    detect_prediction_collapse,
    evaluate_fairness_defensively,
    evaluate_policy_compliance,
    expected_calibration_error_safe,
)


# ============================================================================
# 1. Evaluator Independence Tests
# ============================================================================

def test_evaluator_metadata_independence():
    """
    Asserts that the evaluation function produces strictly identical numerical metrics
    regardless of what metadata (dataset_name, model_name) is provided.
    The evaluator must not branch on scenario names or model labels.
    """
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.5, 200)
    y_pred = rng.binomial(1, 0.5, 200)
    y_prob = rng.uniform(0.1, 0.9, 200)
    sensitive = rng.choice(["GroupA", "GroupB"], size=200)

    suite1 = evaluate_fairness_defensively(
        y_true, y_pred, y_prob, sensitive,
        dataset_name="ArbitraryDataset1",
        model_name="ArbitraryModel1",
    )
    suite2 = evaluate_fairness_defensively(
        y_true, y_pred, y_prob, sensitive,
        dataset_name="DifferentDataset2",
        model_name="DifferentModel2",
    )

    for metric_name in ["Balanced Accuracy", "Equalized Odds Diff", "Demographic Parity Diff", "Brier Score", "ROC-AUC"]:
        v1 = suite1.metrics[metric_name].value
        v2 = suite2.metrics[metric_name].value
        np.testing.assert_allclose(v1, v2, err_msg=f"Metric {metric_name} varied with metadata!")


def test_evaluator_refuses_scenario_parameters():
    """
    Asserts that evaluate_fairness_defensively does NOT accept scenario names,
    DGP effect sizes (gamma), or ground-truth labels in its signature.
    """
    import inspect
    sig = inspect.signature(evaluate_fairness_defensively)
    forbidden_params = ["scenario", "scenario_name", "gamma", "ground_truth", "dgp_params"]
    for param in forbidden_params:
        assert param not in sig.parameters, f"Forbidden scenario parameter '{param}' found in evaluator signature!"


# ============================================================================
# 2. FairnessPolicy & Policy Compliance Tests
# ============================================================================

def test_fairness_policy_evaluation_compliant():
    """Asserts that a model with high balanced accuracy and small parity gap is POLICY_COMPLIANT."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0] * 20)
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0] * 20)  # Perfect predictor
    y_prob = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1] * 20)
    sensitive = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 20)

    policy = FairnessPolicy(
        min_balanced_accuracy=0.80,
        max_equalized_odds_diff=0.05,
        max_demographic_parity_diff=0.05,
    )
    suite = evaluate_fairness_defensively(y_true, y_pred, y_prob, sensitive, policy=policy)
    assert suite.policy_result.decision == PolicyDecision.POLICY_COMPLIANT
    assert suite.policy_result.is_useful is True
    assert suite.policy_result.is_fair is True
    assert suite.policy_result.is_degenerate is False


def test_fairness_policy_evaluation_noncompliant_due_to_disparity():
    """Asserts that a high-utility model with large parity gap is POLICY_NONCOMPLIANT."""
    y_true = np.array([1, 0, 1, 0] * 30)
    sensitive = np.array([0, 0, 1, 1] * 30)
    # Predict 80% positive for group 0, 20% positive for group 1
    y_pred = np.zeros(120, dtype=int)
    for i in range(120):
        if sensitive[i] == 0:
            y_pred[i] = 1 if i % 5 != 0 else 0
        else:
            y_pred[i] = 1 if i % 5 == 0 else 0
    y_prob = np.where(y_pred == 1, 0.8, 0.2)

    policy = FairnessPolicy(max_equalized_odds_diff=0.10, max_demographic_parity_diff=0.10)
    suite = evaluate_fairness_defensively(y_true, y_pred, y_prob, sensitive, policy=policy)
    assert suite.policy_result.decision == PolicyDecision.POLICY_NONCOMPLIANT
    assert suite.policy_result.is_fair is False


# ============================================================================
# 3. Multi-Metric Gaming Tests (8 Adversarial Prediction Policies)
# ============================================================================

def test_gaming_model_1_always_negative():
    """Adversarial Model 1: Constant negative (all 0s) must be flagged DEGENERATE."""
    y_true = np.random.binomial(1, 0.6, 200)
    y_pred = np.zeros(200, dtype=int)
    sensitive = np.random.choice([0, 1], size=200)

    suite = evaluate_fairness_defensively(y_true, y_pred, None, sensitive)
    assert suite.degeneracy_status == ModelDegeneracyStatus.DEGENERATE
    assert suite.policy_result.decision == PolicyDecision.POLICY_DEGENERATE
    assert suite.fairness_success_status != FairnessSuccessStatus.FAIR_AND_USEFUL


def test_gaming_model_2_always_positive():
    """Adversarial Model 2: Constant positive (all 1s) must be flagged DEGENERATE."""
    y_true = np.random.binomial(1, 0.6, 200)
    y_pred = np.ones(200, dtype=int)
    sensitive = np.random.choice([0, 1], size=200)

    suite = evaluate_fairness_defensively(y_true, y_pred, None, sensitive)
    assert suite.degeneracy_status == ModelDegeneracyStatus.DEGENERATE
    assert suite.policy_result.decision == PolicyDecision.POLICY_DEGENERATE
    assert suite.fairness_success_status != FairnessSuccessStatus.FAIR_AND_USEFUL


def test_gaming_model_3_random_prevalence_prediction():
    """Adversarial Model 3: Random guessing at prevalence rate has ~0.50 balanced accuracy; fails utility."""
    rng = np.random.default_rng(123)
    y_true = rng.binomial(1, 0.70, 500)
    y_pred = rng.binomial(1, 0.70, 500)
    sensitive = rng.choice([0, 1], size=500)

    policy = FairnessPolicy(min_balanced_accuracy=0.60)
    suite = evaluate_fairness_defensively(y_true, y_pred, None, sensitive, policy=policy)
    assert suite.policy_result.decision != PolicyDecision.POLICY_COMPLIANT
    assert suite.policy_result.is_useful is False


def test_gaming_model_4_group_constant_prediction():
    """Adversarial Model 4: Group-constant prediction (predicting 1 for G0, 0 for G1) is noncompliant."""
    sensitive = np.array([0] * 100 + [1] * 100)
    y_true = np.random.binomial(1, 0.5, 200)
    y_pred = np.where(sensitive == 0, 1, 0)

    suite = evaluate_fairness_defensively(y_true, y_pred, None, sensitive)
    assert suite.policy_result.decision != PolicyDecision.POLICY_COMPLIANT
    assert suite.metrics["Demographic Parity Diff"].value >= 0.90


def test_gaming_model_5_highly_imbalanced_prediction():
    """Adversarial Model 5: 99% positive prediction rate triggers prediction collapse."""
    y_true = np.random.binomial(1, 0.5, 300)
    y_pred = np.ones(300, dtype=int)
    y_pred[:3] = 0  # 99% positive
    sensitive = np.random.choice([0, 1], size=300)

    deg_status, msg = detect_prediction_collapse(y_pred)
    assert deg_status == ModelDegeneracyStatus.DEGENERATE
    assert "High" in msg or "entropy" in msg


def test_gaming_model_6_poor_minority_recall():
    """Adversarial Model 6: High accuracy overall, but minority group recall = 0."""
    # 90 instances in G0, 10 in G1
    s = np.array([0] * 90 + [1] * 10)
    y_true = np.array([1, 0] * 45 + [1] * 5 + [0] * 5)
    # Predict accurately for G0, but predict all 0s for G1
    y_pred = y_true.copy()
    y_pred[s == 1] = 0

    suite = evaluate_fairness_defensively(y_true, y_pred, None, s)
    g1_tpr = suite.subgroup_metrics["1"]["TPR"]
    assert g1_tpr == 0.0
    # Equal opportunity difference should reflect this severe discrepancy
    assert suite.metrics["Equal Opportunity Diff"].value >= 0.50
    assert suite.policy_result.decision != PolicyDecision.POLICY_COMPLIANT


def test_gaming_model_7_high_di_low_utility():
    """Adversarial Model 7: High Disparate Impact (ratio near 1.0) but balanced accuracy chance-level."""
    rng = np.random.default_rng(999)
    y_true = rng.binomial(1, 0.50, 400)
    # Coin flip prediction gives ~50% selection rate in both groups (DI ~ 1.0), but bal_acc ~ 0.50
    y_pred = rng.binomial(1, 0.50, 400)
    sensitive = rng.choice([0, 1], size=400)

    policy = FairnessPolicy(min_balanced_accuracy=0.60, min_disparate_impact_ratio=0.80)
    suite = evaluate_fairness_defensively(y_true, y_pred, None, sensitive, policy=policy)
    assert suite.metrics["Disparate Impact Ratio"].value >= 0.80
    assert suite.policy_result.decision != PolicyDecision.POLICY_COMPLIANT
    assert suite.policy_result.is_useful is False


def test_gaming_model_8_majority_class_baseline():
    """Adversarial Model 8: MajorityClassClassifier B0 is strictly noncompliant under standard policy."""
    clf = MajorityClassClassifier()
    X = np.zeros((100, 2))
    y = np.array([1] * 80 + [0] * 20)
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    s = np.array([0, 1] * 50)

    suite = evaluate_fairness_defensively(y, preds, probs[:, 1], s)
    assert suite.degeneracy_status == ModelDegeneracyStatus.DEGENERATE
    assert suite.policy_result.decision == PolicyDecision.POLICY_DEGENERATE


# ============================================================================
# 4. Probability Calibration Tests
# ============================================================================

def test_brier_score_safe_bounds():
    """Tests Brier score computation for perfect, worst, and uniform cases."""
    y_true = np.array([1, 0, 1, 0])
    perfect_prob = np.array([1.0, 0.0, 1.0, 0.0])
    worst_prob = np.array([0.0, 1.0, 0.0, 1.0])
    uniform_prob = np.array([0.5, 0.5, 0.5, 0.5])

    assert brier_score_loss_safe(y_true, perfect_prob) == 0.0
    assert brier_score_loss_safe(y_true, worst_prob) == 1.0
    assert brier_score_loss_safe(y_true, uniform_prob) == 0.25


def test_expected_calibration_error_bounds():
    """Tests Expected Calibration Error computation on perfectly calibrated probabilities."""
    # Probability equals label prevalence exactly in each bucket
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # 4 items with prob 0.0 (all 0s -> acc 0.0, diff 0.0)
    # 4 items with prob 1.0 (all 1s -> acc 1.0, diff 0.0)
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    ece, details = expected_calibration_error_safe(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)
    assert len(details) == 5


# ============================================================================
# 5. Paired-Seed & Population Recovery Logic Tests
# ============================================================================

def test_paired_seed_differences_calculation():
    """Asserts that compute_paired_seed_differences correctly calculates per-seed deltas."""
    df_dummy = pd.DataFrame([
        {"regime": "GROUP_HOLDOUT", "baseline_id": "B1", "seed": 42, "accuracy": 0.80, "balanced_accuracy": 0.80, "equalized_odds_diff": 0.15, "demographic_parity_diff": 0.15, "disparate_impact_ratio": 0.70},
        {"regime": "GROUP_HOLDOUT", "baseline_id": "B3", "seed": 42, "accuracy": 0.82, "balanced_accuracy": 0.81, "equalized_odds_diff": 0.05, "demographic_parity_diff": 0.08, "disparate_impact_ratio": 0.85},
        {"regime": "GROUP_HOLDOUT", "baseline_id": "B1", "seed": 43, "accuracy": 0.78, "balanced_accuracy": 0.79, "equalized_odds_diff": 0.17, "demographic_parity_diff": 0.16, "disparate_impact_ratio": 0.68},
        {"regime": "GROUP_HOLDOUT", "baseline_id": "B3", "seed": 43, "accuracy": 0.80, "balanced_accuracy": 0.80, "equalized_odds_diff": 0.07, "demographic_parity_diff": 0.09, "disparate_impact_ratio": 0.84},
    ])
    paired = compute_paired_seed_differences(df_dummy, baseline_a="B1", baseline_b="B3", regime="GROUP_HOLDOUT")
    assert not paired.empty
    eo_row = paired[paired["metric"] == "equalized_odds_diff"].iloc[0]
    # In both seeds, B3 has 0.10 lower EODiff (-0.10)
    assert eo_row["mean_paired_delta"] == pytest.approx(-0.10, abs=1e-4)


def test_population_dgp_truth_calculation():
    """Asserts that calculate_population_dgp_truth calculates valid population quantities."""
    truth = calculate_population_dgp_truth("SCENARIO_C_EXPLICIT_GROUP_EFFECT", "MODERATE_SIGNAL", n_mc=10000)
    assert 0.0 < truth["population_overall_prevalence"] < 1.0
    assert 0.0 < truth["population_p_y1_given_g0"] < 1.0
    assert 0.0 < truth["population_p_y1_given_g1"] < 1.0
    # In Scenario C, group 0 has higher success than group 1 due to negative gamma
    assert truth["population_base_rate_gap"] > 0.05
