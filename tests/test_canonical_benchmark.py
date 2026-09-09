# ============================================================================
# Unit Tests: Canonical Benchmark, Feature Contracts, Split Isolation & Firewall
# ============================================================================

import numpy as np
import pandas as pd
import pytest
import yaml

from src.canonical_benchmark import (
    CanonicalCandidateResult,
    aggregate_canonical_statistics,
    generate_clean_synthetic_benchmark,
    run_canonical_evaluation,
)
from src.firewall import (
    LockedTestData,
    LockedPolicy,
    ModelSelectionFirewallViolation,
    PolicyLockError,
    verify_no_test_leakage_in_args,
)
from src.split_utils import (
    GroupLeakageError,
    SplitRegime,
    TemporalLeakageError,
    group_aware_3way_split,
    stratified_3way_split,
    temporal_3way_split,
)


def test_feature_contract_yaml_disallows_post_outcome():
    """Verify that benchmark_features.yaml classifies actual_time and related variables as forbidden."""
    with open("configs/benchmark_features.yaml", "r") as f:
        contract = yaml.safe_load(f)

    delhivery_feats = {
        f["feature_name"]: f for f in contract["delhivery_logistics"]["features"]
    }
    assert delhivery_feats["actual_time"]["classification"] == "POST_OUTCOME"
    assert not delhivery_feats["actual_time"]["included_in_canonical_benchmark"]
    assert delhivery_feats["segment_actual_time"]["classification"] == "POST_OUTCOME"
    assert not delhivery_feats["segment_actual_time"]["included_in_canonical_benchmark"]
    assert delhivery_feats["osrm_distance"]["classification"] == "PRE_DISPATCH_VALID"
    assert delhivery_feats["osrm_distance"]["included_in_canonical_benchmark"]

    amazon_feats = {
        f["feature_name"]: f for f in contract["amazon_last_mile"]["features"]
    }
    assert amazon_feats["invalid_sequence_score"]["classification"] == "REVIEW_REQUIRED"
    assert not amazon_feats["invalid_sequence_score"]["included_in_canonical_benchmark"]


def test_group_aware_split_isolation():
    """Verify group-aware 3-way split guarantees zero group overlap across splits."""
    np.random.seed(42)
    n = 200
    groups = np.random.choice([f"trip_{i}" for i in range(20)], size=n)
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], size=n))
    s = pd.Series(np.random.choice([0, 1], size=n))

    result = group_aware_3way_split(X, y, s, groups=groups, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    # Verify sets of groups are disjoint
    assert len(result.train_groups.intersection(result.val_groups)) == 0
    assert len(result.train_groups.intersection(result.test_groups)) == 0
    assert len(result.val_groups.intersection(result.test_groups)) == 0

    # Verify method raises no error
    result.verify_isolation()


def test_group_aware_split_leakage_detection():
    """Verify that injecting an overlapping group triggers GroupLeakageError."""
    n = 30
    groups = [f"g_{i % 5}" for i in range(n)]
    X = pd.DataFrame({"feat": np.arange(n)})
    y = pd.Series([0, 1] * 15)
    s = pd.Series([1, 0] * 15)

    result = group_aware_3way_split(X, y, s, groups=groups, seed=42)
    # Artificially inject group overlap
    result.val_groups.add(list(result.train_groups)[0])

    with pytest.raises(GroupLeakageError):
        result.verify_isolation()


def test_temporal_split_order():
    """Verify temporal 3-way split maintains strict chronological progression."""
    n = 100
    timestamps = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame({"feat": np.arange(n)})
    y = pd.Series([0, 1] * 50)
    s = pd.Series([1, 0] * 50)

    result = temporal_3way_split(X, y, s, timestamps=timestamps, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    assert result.regime == SplitRegime.TEMPORAL_HOLDOUT
    assert len(result.X_train) == 60
    assert len(result.X_val) == 20
    assert len(result.X_test) == 20

    # Ensure train indices are strictly before val indices, which are strictly before test indices
    assert result.X_train.index.max() < result.X_val.index.min()
    assert result.X_val.index.max() < result.X_test.index.min()


def test_firewall_lock_blocks_fitting():
    """Verify LockedTestData blocks attempts to fit, tune, or select models."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    s = pd.Series([0, 1, 0])
    locked = LockedTestData(X=df, y=y, sensitive=s, dataset_name="test_data")

    with pytest.raises(ModelSelectionFirewallViolation, match="Cannot call .fit()"):
        locked.fit()

    with pytest.raises(ModelSelectionFirewallViolation, match="Cannot perform tuning"):
        locked.tune()

    with pytest.raises(ModelSelectionFirewallViolation, match="Cannot perform model selection"):
        locked.select()


def test_firewall_blocks_passing_locked_data_to_tuner():
    """Verify verify_no_test_leakage_in_args catches LockedTestData passed to model routines."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    s = pd.Series([0, 1, 0])
    locked = LockedTestData(X=df, y=y, sensitive=s, dataset_name="test_data")

    with pytest.raises(ModelSelectionFirewallViolation):
        verify_no_test_leakage_in_args(locked)

    with pytest.raises(ModelSelectionFirewallViolation):
        verify_no_test_leakage_in_args(valid_arg=123, holdout=locked)


def test_locked_policy_frozen_state():
    """Verify LockedPolicy enforces frozen state."""
    policy = LockedPolicy(model_name="test", estimator=None, is_frozen=False)

    with pytest.raises(PolicyLockError):
        policy.predict(np.array([[1, 2]]))


def test_synthetic_benchmark_dataset_generation():
    """Verify clean synthetic benchmark generator creates valid pre-dispatch data."""
    bundle = generate_clean_synthetic_benchmark(n_samples=150, seed=42)

    assert "actual_time" not in bundle["X"].columns
    assert "segment_actual_time" not in bundle["X"].columns
    assert "planned_distance" in bundle["X"].columns
    assert "station_code" in bundle["X"].columns
    assert len(bundle["y"]) == 150
    assert len(bundle["sensitive"]) == 150
    assert len(bundle["groups"]) == 150


def test_canonical_baseline_hierarchy_execution():
    """Verify canonical baseline hierarchy executes and produces complete metric suites."""
    bundle = generate_clean_synthetic_benchmark(n_samples=250, seed=42)
    results = run_canonical_evaluation(bundle, regime=SplitRegime.GROUP_HOLDOUT, seed=42, alpha=0.25)

    baseline_ids = [r.baseline_id for r in results]
    assert "B0_Majority_Class_Baseline" in baseline_ids
    assert "B1_Clean_Unmitigated_Linear" in baseline_ids
    assert "B2_Fairness_Reweighted" in baseline_ids
    assert "B3_Validation_Threshold_Mitigated" in baseline_ids
    assert "B4_Proposed_Tuned_XGB" in baseline_ids

    for res in results:
        flat = res.to_flat_dict()
        assert "accuracy" in flat
        assert "balanced_accuracy" in flat
        assert "demographic_parity_diff" in flat
        assert "equal_opportunity_diff" in flat
        assert "equalized_odds_diff" in flat
        assert "predictive_parity_diff" in flat
        assert "disparate_impact_ratio" in flat
        assert "accuracy_tradeoff_proxy" in flat
        assert "utility_score" in flat
        assert "TP" in flat
        assert "TN" in flat
        assert "FP" in flat
        assert "FN" in flat
        assert "TPR" in flat
        assert "TNR" in flat
        assert "degeneracy_status" in flat
        assert "fairness_success_status" in flat
        assert not np.isnan(flat["balanced_accuracy"])


def test_aggregate_canonical_statistics():
    """Verify multi-seed statistical aggregator computes mean, std, CI95, and bounds."""
    bundle = generate_clean_synthetic_benchmark(n_samples=150, seed=42)
    res1 = run_canonical_evaluation(bundle, regime=SplitRegime.GROUP_HOLDOUT, seed=42)
    res2 = run_canonical_evaluation(bundle, regime=SplitRegime.GROUP_HOLDOUT, seed=43)

    flat_rows = [r.to_flat_dict() for r in res1 + res2]
    df_raw = pd.DataFrame(flat_rows)
    stats = aggregate_canonical_statistics(df_raw)

    assert len(stats) > 0
    assert "accuracy_mean" in stats.columns
    assert "accuracy_std" in stats.columns
    assert "accuracy_ci95_low" in stats.columns
    assert "accuracy_ci95_high" in stats.columns
    assert "fairness_gap_mean" in stats.columns
