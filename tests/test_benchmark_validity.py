# ============================================================================
# Unit Tests: Benchmark Validity, Anti-Degeneracy Guardrails, & Controlled Scenarios
# Covers Phase 16 specifications for Pass 3 Commercial-Readiness Upgrade.
# ============================================================================

import math
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.canonical_benchmark import (
    MajorityClassClassifier,
    generate_controlled_synthetic_benchmark,
    run_canonical_evaluation,
)
from src.fairness_evaluator import (
    EvaluationStatus,
    FairnessEvaluationSuite,
    FairnessSuccessStatus,
    MetricResult,
    ModelDegeneracyStatus,
    classify_fairness_success,
    detect_prediction_collapse,
    evaluate_fairness_defensively,
)
from src.firewall import LockedTestData, ModelSelectionFirewallViolation
from src.main import reweighing_weights
from src.split_utils import SplitRegime


# 1. Majority-Class Baseline Test
def test_majority_class_baseline_predicts_training_mode():
    clf = MajorityClassClassifier()
    # 70% class 1, 30% class 0
    y_train = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    X_train = np.random.randn(10, 2)
    clf.fit(X_train, y_train)

    assert clf.majority_class_ == 1
    X_test = np.random.randn(25, 2)
    preds = clf.predict(X_test)
    assert len(preds) == 25
    assert (preds == 1).all()


# 2. Prediction Collapse Detection Test
def test_detect_prediction_collapse_identifies_degenerate_states():
    # Case A: Constant 1s
    y_const = np.ones(100, dtype=int)
    status_a, msg_a = detect_prediction_collapse(y_const)
    assert status_a == ModelDegeneracyStatus.DEGENERATE
    assert "Trivial constant predictor" in msg_a

    # Case B: Extreme positive rate (e.g. 99% 1s)
    y_skewed = np.array([1] * 99 + [0])
    status_b, msg_b = detect_prediction_collapse(y_skewed)
    assert status_b == ModelDegeneracyStatus.DEGENERATE
    assert "Prediction collapse" in msg_b

    # Case C: Diverse predictions
    y_diverse = np.array([0, 1] * 50)
    status_c, msg_c = detect_prediction_collapse(y_diverse)
    assert status_c == ModelDegeneracyStatus.NORMAL


# 3. Both-Class Prediction Requirement for FAIR_AND_USEFUL
def test_both_class_prediction_requirement_for_useful_status():
    y_true = np.array([0, 1, 0, 1] * 25)
    y_pred_const = np.ones(100, dtype=int)
    sensitive = np.array(["GroupA", "GroupA", "GroupB", "GroupB"] * 25)

    suite = evaluate_fairness_defensively(y_true, y_pred_const, None, sensitive)
    assert suite.degeneracy_status == ModelDegeneracyStatus.DEGENERATE
    # Even though parity differences are zero (both groups 100% positive), status MUST NOT be FAIR_AND_USEFUL
    assert suite.fairness_success_status == FairnessSuccessStatus.FAIR_BUT_DEGENERATE


# 4. Sample Weights Application & Meaningful Variation
def test_sample_weight_application_and_variation():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    s = np.array(["A", "A", "B", "A", "B", "B", "B", "B"])
    weights = reweighing_weights(y, s)

    assert len(weights) == len(y)
    assert len(np.unique(weights)) > 1
    assert (weights > 0).all()


# 5. ThresholdOptimizer Effect Test
def test_threshold_optimizer_effect_with_signal():
    np.random.seed(42)
    n = 300
    s = np.random.choice(["A", "B"], size=n)
    # Feature x predicts y, with group A facing a penalty
    x = np.random.normal(0, 1, n)
    z = 1.5 * x - 1.2 * (s == "A")
    prob = 1 / (1 + np.exp(-z))
    y = (prob > 0.5).astype(int)

    from fairlearn.postprocessing import ThresholdOptimizer
    lr = LogisticRegression()
    lr.fit(x.reshape(-1, 1), y)
    to = ThresholdOptimizer(estimator=lr, constraints="equalized_odds", predict_method="predict_proba")
    to.fit(x.reshape(-1, 1), y, sensitive_features=s)

    preds_raw = lr.predict(x.reshape(-1, 1))
    preds_mitigated = to.predict(x.reshape(-1, 1), sensitive_features=s)
    # Mitigator should produce distinct predictions from raw baseline
    assert not np.array_equal(preds_raw, preds_mitigated)


# 6. Controlled Signal Regimes: Non-Zero Mutual Information
def test_controlled_signal_regimes_auc():
    for reg, min_auc in [("LOW_SIGNAL", 0.53), ("MODERATE_SIGNAL", 0.65), ("HIGH_SIGNAL", 0.78)]:
        bundle = generate_controlled_synthetic_benchmark(
            n_samples=600, seed=42, scenario="SCENARIO_A_NO_GROUP_EFFECT", signal_regime=reg
        )
        X = bundle["X"][["planned_distance", "planned_time", "stops_count", "vehicle_capacity"]]
        y = bundle["y"].to_numpy()
        lr = LogisticRegression()
        lr.fit(X, y)
        probs = lr.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        assert auc >= min_auc, f"Regime {reg} failed AUC threshold: {auc:.4f} < {min_auc}"


# 7. Scenario A: No Group Effect
def test_scenario_a_no_group_effect():
    bundle = generate_controlled_synthetic_benchmark(
        n_samples=1000, seed=42, scenario="SCENARIO_A_NO_GROUP_EFFECT", signal_regime="MODERATE_SIGNAL"
    )
    res = run_canonical_evaluation(bundle, regime=SplitRegime.IID_HOLDOUT, seed=42)
    b1 = [r for r in res if r.baseline_id == "B1_Clean_Unmitigated_Linear"][0]
    # Under independent grouping, demographic parity difference is small (sampling noise only)
    assert b1.suite.metrics["Demographic Parity Diff"].value < 0.15


# 8. Scenario C: Explicit Group Effect
def test_scenario_c_explicit_group_effect():
    bundle = generate_controlled_synthetic_benchmark(
        n_samples=1500, seed=42, scenario="SCENARIO_C_EXPLICIT_GROUP_EFFECT", signal_regime="MODERATE_SIGNAL"
    )
    res = run_canonical_evaluation(bundle, regime=SplitRegime.IID_HOLDOUT, seed=42)
    b1 = [r for r in res if r.baseline_id == "B1_Clean_Unmitigated_Linear"][0]
    # Direct group penalty causes measurable selection disparity
    assert b1.suite.metrics["Demographic Parity Diff"].value >= 0.10


# 9. Scenario D: Impossible Fairness Trade-Off
def test_scenario_d_impossible_fairness():
    bundle = generate_controlled_synthetic_benchmark(
        n_samples=1000, seed=42, scenario="SCENARIO_D_IMPOSSIBLE_FAIRNESS", signal_regime="HIGH_SIGNAL"
    )
    # Severe base rate shift between groups
    y = bundle["y"]
    s = bundle["sensitive"]
    rate_high = np.mean(y[s == "High_Complexity"])
    rate_std = np.mean(y[s == "Standard_Complexity"])
    assert abs(rate_high - rate_std) >= 0.20


# 10. Scenario E: Constant Predictor / Degeneracy Replica
def test_scenario_e_constant_predictor_triggers_degeneracy():
    bundle = generate_controlled_synthetic_benchmark(
        n_samples=800, seed=42, scenario="SCENARIO_E_CONSTANT_PREDICTOR", signal_regime="ZERO_SIGNAL"
    )
    res = run_canonical_evaluation(bundle, regime=SplitRegime.IID_HOLDOUT, seed=42)
    b0 = [r for r in res if r.baseline_id == "B0_Majority_Class_Baseline"][0]
    assert b0.suite.degeneracy_status == ModelDegeneracyStatus.DEGENERATE
    assert b0.suite.fairness_success_status == FairnessSuccessStatus.FAIR_BUT_DEGENERATE


# 11. Group Holdout Semantics Validation
def test_group_holdout_semantics_in_config():
    with open("configs/canonical_benchmark.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["datasets"]["delhivery_logistics"]["grouping_semantics"] == "TRIP_SEGMENT_ISOLATION"
    assert cfg["datasets"]["amazon_last_mile"]["grouping_semantics"] == "UNSEEN_STATION_HOLDOUT"


# 12. Fairness Success Status Classifier Logic
def test_fairness_success_classifier_logic():
    # Construct synthetic suite with degenerate model
    suite = FairnessEvaluationSuite(
        dataset_name="Test",
        model_name="M",
        sample_size=100,
        group_counts={"A": 50, "B": 50},
        metrics={
            "Balanced Accuracy": MetricResult("Balanced Accuracy", 0.50, EvaluationStatus.VALID),
            "Equalized Odds Diff": MetricResult("Equalized Odds Diff", 0.00, EvaluationStatus.VALID),
            "Demographic Parity Diff": MetricResult("Demographic Parity Diff", 0.00, EvaluationStatus.VALID),
        },
        overall_status=EvaluationStatus.VALID,
        fairness_gap=0.0,
        user_utility_score=0.50,
        degeneracy_status=ModelDegeneracyStatus.DEGENERATE,
    )
    status = classify_fairness_success(suite)
    assert status == FairnessSuccessStatus.FAIR_BUT_DEGENERATE


# 13. Per-Group Confusion Metrics
def test_per_group_confusion_metrics_populated():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    s = np.array(["G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2"])
    suite = evaluate_fairness_defensively(y_true, y_pred, None, s)

    assert "G1" in suite.subgroup_metrics
    assert "G2" in suite.subgroup_metrics
    assert "TPR" in suite.subgroup_metrics["G1"]
    assert "PPV" in suite.subgroup_metrics["G1"]
    assert "count" in suite.subgroup_metrics["G1"]
    assert suite.subgroup_metrics["G1"]["count"] == 4


# 14. Undefined Metric Defensive Handling
def test_undefined_metric_handling_on_zero_positive_subgroup():
    # Group B has zero positive ground-truth instances
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    s = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    suite = evaluate_fairness_defensively(y_true, y_pred, None, s)

    # TPR for B is undefined (0 positives)
    assert suite.metrics["TPR"].status == EvaluationStatus.VALID  # Global TPR is valid
    assert math.isnan(suite.subgroup_metrics["B"]["TPR"])


# 15. Firewall Prevents Test Set Tuning
def test_firewall_blocks_tuning_on_test_data():
    X = pd.DataFrame({"feat": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    s = pd.Series(["A", "B", "A"])
    locked = LockedTestData(X=X, y=y, sensitive=s, dataset_name="SealedTest")

    with pytest.raises(ModelSelectionFirewallViolation):
        locked.fit()
    with pytest.raises(ModelSelectionFirewallViolation):
        locked.tune()
    with pytest.raises(ModelSelectionFirewallViolation):
        locked.select()
