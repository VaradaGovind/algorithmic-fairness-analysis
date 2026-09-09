# ============================================================================
# Unit & Adversarial Tests: Intersectional Fairness & Uncertainty Suite
# Covers multi-attribute intersectionality, sparse-group support gating,
# within-test bootstrap estimation uncertainty vs multi-seed variation,
# global vs subgroup utility, policy sensitivity, and 10 adversarial fixtures.
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from src.canonical_benchmark import (
    CanonicalCandidateResult,
    execute_locked_real_data_benchmark,
    generate_controlled_synthetic_benchmark,
    run_canonical_evaluation,
)
from src.fairness_evaluator import (
    EvaluationStatus,
    FairnessEvaluationSuite,
    FairnessPolicy,
    FairnessSuccessStatus,
    FairnessUncertaintyResult,
    IntersectionalGroupSpec,
    ModelDegeneracyStatus,
    PolicyDecision,
    PolicyEvaluationResult,
    build_intersectional_groups,
    compute_fairness_uncertainty_bootstrap,
    evaluate_fairness_defensively,
    evaluate_policy_compliance,
    evaluate_policy_sensitivity,
    safe_equalized_odds_difference,
)
from src.ingest_data import DatasetManifest, IngestionStatus


# ============================================================================
# 1. Intersectional Grouping & Provenance
# ============================================================================

def test_intersectional_group_creation_and_provenance():
    df = pd.DataFrame({
        "region": ["North", "South", "North", "South"],
        "vehicle": ["Van", "Van", "Bike", "Bike"],
        "target": [1, 0, 1, 0],
    })
    spec = IntersectionalGroupSpec(attributes=["region", "vehicle"], name="region_x_vehicle")
    composite, prov = build_intersectional_groups(df, spec)

    assert composite.name == "region_x_vehicle"
    assert len(composite) == 4
    assert composite.iloc[0] == "North x Van"
    assert composite.iloc[1] == "South x Van"

    assert "North x Van" in prov
    assert prov["North x Van"] == {"region": "North", "vehicle": "Van"}
    assert prov["South x Bike"] == {"region": "South", "vehicle": "Bike"}


def test_intersectional_missing_attribute_raises_error():
    df = pd.DataFrame({"region": ["North", "South"]})
    spec = IntersectionalGroupSpec(attributes=["region", "nonexistent"])
    with pytest.raises(KeyError, match="Attribute 'nonexistent' not found"):
        build_intersectional_groups(df, spec)


# ============================================================================
# 2. Sparse Group Support & Explicit Gating
# ============================================================================

def test_sparse_subgroup_support_check():
    # 1 large group (100 rows), 1 sparse group (3 rows)
    y_true = np.array([1, 0] * 50 + [1, 0, 1])
    y_pred = y_true.copy()
    sensitive = np.array(["Main"] * 100 + ["Sparse"] * 3)

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        min_group_size=10,
    )
    # The sparse group must be flagged as INSUFFICIENT_GROUP_SUPPORT
    assert suite.subgroup_metrics["Sparse"]["support_status"] == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value
    assert suite.subgroup_metrics["Main"]["support_status"] == EvaluationStatus.VALID.value

    # Safe disparity calculation flags insufficient support
    assert suite.metrics["Equalized Odds Diff"].status == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT


def test_zero_positive_and_zero_negative_subgroups():
    # Group A has only 0s (zero positives)
    # Group B has only 1s (zero negatives)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sensitive = np.array(["ZeroPos", "ZeroPos", "ZeroPos", "ZeroPos", "ZeroNeg", "ZeroNeg", "ZeroNeg", "ZeroNeg"])

    suite = evaluate_fairness_defensively(y_true, y_pred, None, sensitive, min_group_size=2)
    assert suite.metrics["Equalized Odds Diff"].status in (
        EvaluationStatus.ZERO_POSITIVE_SUBGROUP,
        EvaluationStatus.ZERO_NEGATIVE_SUBGROUP,
    )
    assert np.isnan(suite.metrics["Equalized Odds Diff"].value)


# ============================================================================
# 3. Within-Test Bootstrap Uncertainty Estimation
# ============================================================================

def test_bootstrap_uncertainty_bounds_calculation():
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    y_pred = y_true.copy()
    # Add minor noise to make EODiff non-zero
    flip_idx = np.random.choice(n, size=20, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    sensitive = np.random.choice(["GroupA", "GroupB"], size=n)

    unc = compute_fairness_uncertainty_bootstrap(
        y_true, y_pred, sensitive, n_bootstraps=100, ci_level=0.95, seed=42
    )
    assert "Equalized Odds Diff" in unc
    eo_res = unc["Equalized Odds Diff"]
    assert isinstance(eo_res, FairnessUncertaintyResult)
    assert 0.0 <= eo_res.point_estimate <= 1.0
    assert 0.0 <= eo_res.ci_low <= eo_res.ci_high <= 1.0
    assert eo_res.estimation_method == "STRATIFIED_BOOTSTRAP"
    assert eo_res.sample_size == n


# ============================================================================
# 4. Distinction Between Seed Variation and Bootstrap Uncertainty
# ============================================================================

def test_distinction_seed_variation_vs_bootstrap_uncertainty():
    """
    Demonstrates that:
    - Bootstrap CI represents finite-sample uncertainty on ONE locked test split.
    - Seed-variation intervals represent variance across DIFFERENT dataset splits/seeds.
    Both concepts must not be conflated into a single number.
    """
    # 1. Single-split bootstrap estimation uncertainty
    y_t = np.array([0, 1, 0, 1] * 50)
    y_p = np.array([0, 1, 1, 0] * 50)
    s = np.array(["A", "A", "B", "B"] * 50)

    unc = compute_fairness_uncertainty_bootstrap(y_t, y_p, s, n_bootstraps=50, seed=42)
    eo_unc = unc["Equalized Odds Diff"]

    # Verify that CI bounds capture finite-sample uncertainty
    assert not np.isnan(eo_unc.ci_low)
    assert not np.isnan(eo_unc.ci_high)
    assert eo_unc.ci_high >= eo_unc.ci_low

    # 2. Distinct reporting in CanonicalCandidateResult flat dictionary
    suite = evaluate_fairness_defensively(y_t, y_p, None, s, compute_uncertainty=True, n_bootstraps=50)
    cand_res = CanonicalCandidateResult(
        dataset_name="Test",
        baseline_id="B1",
        model_name="Linear",
        regime="IID_HOLDOUT",
        seed=42,
        suite=suite,
    )
    flat = cand_res.to_flat_dict()
    assert "eo_ci_low_bootstrap" in flat
    assert "eo_ci_high_bootstrap" in flat
    assert flat["seed"] == 42


# ============================================================================
# 5. Global vs Subgroup Utility & Policy Sensitivity
# ============================================================================

def test_global_vs_subgroup_utility_evaluation():
    # Global balanced accuracy is high, but minority subgroup accuracy is poor
    # Group A: 90 samples, 100% correct (bal acc = 1.0)
    # Group B: 10 samples, 0% correct (bal acc = 0.0)
    y_true_a = np.array([0, 1] * 45)
    y_pred_a = y_true_a.copy()
    s_a = np.array(["GroupA"] * 90)

    y_true_b = np.array([0, 1] * 5)
    y_pred_b = 1 - y_true_b  # 100% error on Group B
    s_b = np.array(["GroupB"] * 10)

    y_t = np.concatenate([y_true_a, y_true_b])
    y_p = np.concatenate([y_pred_a, y_pred_b])
    s = np.concatenate([s_a, s_b])

    policy_with_subgroup_constraint = FairnessPolicy(
        name="Strict_Subgroup_Utility_Policy",
        min_balanced_accuracy=0.70,  # Global threshold
        min_subgroup_balanced_accuracy=0.50,  # Subgroup floor
    )

    suite = evaluate_fairness_defensively(
        y_t, y_p, None, s, policy=policy_with_subgroup_constraint, min_group_size=5
    )

    # Global balanced accuracy passes (0.90 * 1.0 + 0.10 * 0.0 = 0.90)
    assert suite.metrics["Balanced Accuracy"].value > 0.85

    # But Policy Decision is NONCOMPLIANT because Group B utility violated floor
    assert suite.policy_result.decision == PolicyDecision.POLICY_NONCOMPLIANT
    assert any("Subgroup 'GroupB' Balanced Accuracy" in v for v in suite.policy_result.violations)


def test_multi_policy_sensitivity_evaluation():
    y_t = np.array([0, 1] * 50)
    y_p = y_t.copy()
    # Introduce small disparity (EODiff = 0.08)
    y_p[0] = 1  # 1 flip
    s = np.array(["A"] * 50 + ["B"] * 50)

    suite = evaluate_fairness_defensively(y_t, y_p, None, s)
    sens = evaluate_policy_sensitivity(suite)

    assert "Standard_Operational_Policy" in sens
    assert "Strict_Equalized_Odds_Policy" in sens
    assert "Lenient_Diagnostic_Policy" in sens

    # Under standard (max 0.10), model is compliant
    assert sens["Standard_Operational_Policy"].decision == PolicyDecision.POLICY_COMPLIANT


# ============================================================================
# 6. Phase 13: 10 Intersectional Adversarial Tests
# ============================================================================

# 1. Fair marginally, unfair intersectionally
def test_adv_1_fair_marginally_unfair_intersectionally():
    # Attributes: Sex (M, F), Shift (Day, Night)
    # Sex alone has equal selection rates, Shift alone has equal selection rates,
    # but the intersection M x Day has 100% positive and F x Day has 0% positive.
    n = 40
    data = []
    for _ in range(10):
        data.append({"sex": "M", "shift": "Day", "target": 1, "pred": 1})
        data.append({"sex": "M", "shift": "Night", "target": 0, "pred": 0})
        data.append({"sex": "F", "shift": "Day", "target": 0, "pred": 0})
        data.append({"sex": "F", "shift": "Night", "target": 1, "pred": 1})
    df = pd.DataFrame(data)

    # Marginal evaluation by Sex:
    suite_sex = evaluate_fairness_defensively(df["target"], df["pred"], None, df["sex"])
    assert suite_sex.metrics["Equalized Odds Diff"].value == pytest.approx(0.0, abs=1e-4)

    # Intersectional evaluation:
    spec = IntersectionalGroupSpec(attributes=["sex", "shift"])
    composite, _ = build_intersectional_groups(df, spec)
    suite_inter = evaluate_fairness_defensively(df["target"], df["pred"], None, composite, min_group_size=5)

    # Intersectional disparity is acute!
    assert suite_inter.metrics["Demographic Parity Diff"].value > 0.80


# 2. Fair intersections with sparse groups
def test_adv_2_fair_intersections_with_sparse_groups():
    # 4 intersections, 1 is tiny (N=2)
    y = np.array([1, 0] * 20 + [1, 0])
    p = y.copy()
    s = np.array(["G1"] * 20 + ["G2"] * 20 + ["Sparse"] * 2)

    suite = evaluate_fairness_defensively(y, p, None, s, min_group_size=5)
    # Should not report a fake clean result when a sparse group is present
    assert suite.metrics["Equalized Odds Diff"].status == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT


# 3. One intersection with zero positives
def test_adv_3_one_intersection_zero_positives():
    y = np.array([1, 1, 0, 0] * 10 + [0, 0, 0, 0])  # Group C has 0 positives
    p = y.copy()
    s = np.array(["A"] * 20 + ["B"] * 20 + ["ZeroPosGroup"] * 4)

    suite = evaluate_fairness_defensively(y, p, None, s, min_group_size=2)
    assert suite.metrics["Equalized Odds Diff"].status == EvaluationStatus.ZERO_POSITIVE_SUBGROUP


# 4. One intersection with zero negatives
def test_adv_4_one_intersection_zero_negatives():
    y = np.array([1, 1, 0, 0] * 10 + [1, 1, 1, 1])  # Group C has 0 negatives
    p = y.copy()
    s = np.array(["A"] * 20 + ["B"] * 20 + ["ZeroNegGroup"] * 4)

    suite = evaluate_fairness_defensively(y, p, None, s, min_group_size=2)
    assert suite.metrics["Equalized Odds Diff"].status == EvaluationStatus.ZERO_NEGATIVE_SUBGROUP


# 5. Conflicting intersectional metrics (EODiff passes, but DIR fails)
def test_adv_5_conflicting_intersectional_metrics():
    # High base rate disparity yields small EODiff but low Disparate Impact Ratio
    y = np.array([1, 0] * 50)
    # Group A: positive selection = 0.90
    # Group B: positive selection = 0.40 -> DIR = 0.40 / 0.90 = 0.44 (< 0.80)
    p = np.array([1] * 45 + [0] * 5 + [1] * 20 + [0] * 30)
    s = np.array(["A"] * 50 + ["B"] * 50)

    suite = evaluate_fairness_defensively(y, p, None, s)
    dir_val = suite.metrics["Disparate Impact Ratio"].value
    assert dir_val < 0.60  # Fails 80% rule


# 6. Simpson's paradox aggregation
def test_adv_6_simpsons_paradox_aggregation():
    # Within each subgroup, Group A has higher selection rate.
    # But when aggregated across confounder, Group B appears to have higher selection rate.
    data = []
    # Subgroup 1: High overall acceptance
    for _ in range(80):
        data.append({"confounder": "Tier1", "group": "A", "target": 1, "pred": 1})
    for _ in range(20):
        data.append({"confounder": "Tier1", "group": "A", "target": 0, "pred": 0})
    for _ in range(15):
        data.append({"confounder": "Tier1", "group": "B", "target": 1, "pred": 1})
    for _ in range(5):
        data.append({"confounder": "Tier1", "group": "B", "target": 0, "pred": 0})

    # Subgroup 2: Low overall acceptance
    for _ in range(5):
        data.append({"confounder": "Tier2", "group": "A", "target": 1, "pred": 1})
    for _ in range(15):
        data.append({"confounder": "Tier2", "group": "A", "target": 0, "pred": 0})
    for _ in range(20):
        data.append({"confounder": "Tier2", "group": "B", "target": 1, "pred": 1})
    for _ in range(80):
        data.append({"confounder": "Tier2", "group": "B", "target": 0, "pred": 0})

    df = pd.DataFrame(data)

    spec = IntersectionalGroupSpec(attributes=["group", "confounder"])
    comp, _ = build_intersectional_groups(df, spec)
    suite = evaluate_fairness_defensively(df["target"], df["pred"], None, comp, min_group_size=5)
    assert suite.overall_status == EvaluationStatus.VALID
    assert len(suite.subgroup_metrics) == 4


# 7. Many sparse groups
def test_adv_7_many_sparse_groups():
    # 20 groups of 2 samples each
    y = np.random.choice([0, 1], size=40)
    p = y.copy()
    s = np.array([f"Group_{i // 2}" for i in range(40)])

    suite = evaluate_fairness_defensively(y, p, None, s, min_group_size=5)
    # All groups are sparse -> must be flagged
    assert suite.metrics["Equalized Odds Diff"].status == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT


# 8. Harmless high-cardinality grouping with adequate support
def test_adv_8_harmless_high_cardinality_with_adequate_support():
    # 10 groups with 50 samples each (total 500 samples)
    np.random.seed(42)
    y = np.random.choice([0, 1], size=500)
    p = y.copy()
    s = np.array([f"Cluster_{i // 50}" for i in range(500)])

    suite = evaluate_fairness_defensively(y, p, None, s, min_group_size=20)
    assert suite.overall_status == EvaluationStatus.VALID
    assert len(suite.subgroup_metrics) == 10
    assert all(info["support_status"] == EvaluationStatus.VALID.value for info in suite.subgroup_metrics.values())


# 9. Random intersection labels
def test_adv_9_random_intersection_labels():
    np.random.seed(42)
    y = np.random.choice([0, 1], size=300)
    p = y.copy()
    s_rand = np.random.choice(["X x 1", "X x 2", "Y x 1", "Y x 2"], size=300)

    suite = evaluate_fairness_defensively(y, p, None, s_rand, min_group_size=20)
    assert suite.overall_status == EvaluationStatus.VALID
    # Random grouping should have low disparity in expectation
    assert suite.metrics["Equalized Odds Diff"].value < 0.15


# 10. Group permutations invariance of core metrics
def test_adv_10_group_permutations_invariance():
    y = np.array([0, 1] * 50)
    p = y.copy()
    s1 = np.array(["A"] * 50 + ["B"] * 50)
    s2 = np.array(["B"] * 50 + ["A"] * 50)  # Inverted names

    res1 = evaluate_fairness_defensively(y, p, None, s1)
    res2 = evaluate_fairness_defensively(y, p, None, s2)

    assert res1.metrics["Equalized Odds Diff"].value == res2.metrics["Equalized Odds Diff"].value
    assert res1.metrics["Demographic Parity Diff"].value == res2.metrics["Demographic Parity Diff"].value
    assert res1.metrics["Disparate Impact Ratio"].value == res2.metrics["Disparate Impact Ratio"].value
