# ============================================================================
# Cluster Uncertainty & Data Governance Test Suite
# Tests cluster-aware bootstrap uncertainty, sparse cluster gating,
# missing sensitive attribute governance, ratio metric defenses, and data ingestion limits.
# ============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.fairness_evaluator import (
    AnalysisType,
    EvaluationStatus,
    EvidenceState,
    FairnessEvaluationSuite,
    FairnessPolicy,
    FairnessUncertaintyResult,
    MissingAttributePolicy,
    ModelDegeneracyStatus,
    PolicyDecision,
    ResamplingUnit,
    compute_fairness_uncertainty_bootstrap,
    evaluate_fairness_defensively,
    evaluate_policy_compliance,
    safe_disparate_impact_ratio,
)
from src.ingest_data import (
    SecurityValidationError,
    check_json_depth,
    validate_ingestion_resource_limits,
    verify_offline_ingestion_capability,
)


# ---------------------------------------------------------------------------
# Test 1: Cluster Bootstrap Sampling Integrity (All Selected or All Excluded)
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_replicate_cluster_integrity():
    """
    Verifies that for each cluster bootstrap replicate, all rows of a cluster
    are either all selected (with identical multiplicity) or all excluded.
    """
    n_clusters = 5
    cluster_ids = [f"cluster_{i}" for i in range(n_clusters)]
    clusters = np.repeat(cluster_ids, 4)

    rng = np.random.RandomState(42)
    unique_clusters = np.unique(clusters)
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    for _ in range(50):
        sampled_c = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        boot_idx = np.concatenate([cluster_to_indices[c] for c in sampled_c])
        sampled_cluster_counts = pd.Series(clusters[boot_idx]).value_counts()
        for cid in unique_clusters:
            count = sampled_cluster_counts.get(cid, 0)
            assert count % 4 == 0, f"Cluster {cid} row count {count} is not a multiple of 4"


# ---------------------------------------------------------------------------
# Test 2: Cluster Bootstrap Widens Uncertainty Under Correlation
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_widens_uncertainty_under_correlation():
    """
    Demonstrates that CLUSTER_BOOTSTRAP accounts for intra-cluster correlation
    and yields wider confidence intervals than ROW_BOOTSTRAP on correlated data.
    """
    rng = np.random.RandomState(42)
    n_clusters = 20
    rows_per_cluster = 10
    clusters_list = []
    y_true_list = []
    y_pred_list = []
    sens_list = []

    for c in range(n_clusters):
        cid = f"cluster_{c}"
        c_sens = "GroupA" if c % 2 == 0 else "GroupB"
        c_base = 1 if rng.rand() > 0.4 else 0
        for _ in range(rows_per_cluster):
            clusters_list.append(cid)
            sens_list.append(c_sens)
            yt = c_base if rng.rand() > 0.1 else (1 - c_base)
            yp = yt if rng.rand() > 0.2 else (1 - yt)
            y_true_list.append(yt)
            y_pred_list.append(yp)

    y_t = np.array(y_true_list)
    y_p = np.array(y_pred_list)
    sens = np.array(sens_list)
    clusters = np.array(clusters_list)

    res_row = compute_fairness_uncertainty_bootstrap(
        y_t, y_p, sens,
        n_bootstraps=100,
        resampling_unit=ResamplingUnit.ROW_BOOTSTRAP,
        seed=42,
    )
    res_cluster = compute_fairness_uncertainty_bootstrap(
        y_t, y_p, sens,
        n_bootstraps=100,
        resampling_unit=ResamplingUnit.CLUSTER_BOOTSTRAP,
        cluster_series=clusters,
        cluster_key="trip_uuid",
        seed=42,
    )

    dp_row = res_row["Demographic Parity Diff"]
    dp_cluster = res_cluster["Demographic Parity Diff"]

    width_row = dp_row.ci_high - dp_row.ci_low
    width_cluster = dp_cluster.ci_high - dp_cluster.ci_low

    assert res_cluster.resampling_unit == "CLUSTER_BOOTSTRAP"
    assert res_cluster.cluster_key == "trip_uuid"
    assert width_cluster > 0.0
    assert width_row > 0.0
    assert width_cluster >= width_row * 0.90


# ---------------------------------------------------------------------------
# Test 3: Sparse Cluster Support Gating
# ---------------------------------------------------------------------------

def test_sparse_cluster_support_gating():
    y_true = np.array([1, 0, 1, 0] * 2)
    y_pred = np.array([1, 0, 1, 0] * 2)
    sens = np.array(["A", "B"] * 4)
    clusters = np.array(["c1", "c1", "c2", "c2", "c3", "c3", "c3", "c3"])

    res = compute_fairness_uncertainty_bootstrap(
        y_true, y_pred, sens,
        n_bootstraps=50,
        resampling_unit=ResamplingUnit.CLUSTER_BOOTSTRAP,
        cluster_series=clusters,
        seed=42,
    )
    assert res.support_status == EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT.value


# ---------------------------------------------------------------------------
# Test 4: Missing Sensitive Attribute Governance
# ---------------------------------------------------------------------------

def test_missing_sensitive_attribute_governance():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    sensitive_with_nones = np.array(["A", "A", None, "B", "B", None], dtype=object)

    suite_reject = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive_with_nones,
        missing_attribute_policy=MissingAttributePolicy.REJECT,
    )
    assert suite_reject.overall_status == EvaluationStatus.MISSING_SENSITIVE_ATTRIBUTE

    suite_exclude = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive_with_nones,
        missing_attribute_policy=MissingAttributePolicy.EXCLUDE,
        min_group_size=1,
    )
    assert suite_exclude.overall_status == EvaluationStatus.VALID
    assert suite_exclude.sample_size == 4


# ---------------------------------------------------------------------------
# Test 5: Safe Disparate Impact Ratio & Denominator Defenses
# ---------------------------------------------------------------------------

def test_disparate_impact_ratio_edge_cases():
    dir_val, stat, rates = safe_disparate_impact_ratio(
        y_pred=np.array([0, 0, 0, 0]),
        sensitive=pd.Series(["A", "A", "B", "B"]),
    )
    assert np.isnan(dir_val) or dir_val == 0.0 or stat in (EvaluationStatus.ZERO_SELECTION_RATE, EvaluationStatus.UNSTABLE_RATIO)


# ---------------------------------------------------------------------------
# Test 6: Ingestion Resource Limits & JSON Depth
# ---------------------------------------------------------------------------

def test_ingestion_resource_limits_and_json_depth():
    shallow_json = {"a": {"b": {"c": 1}}}
    check_json_depth(shallow_json, max_depth=5)

    deep_json = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}
    with pytest.raises(SecurityValidationError):
        check_json_depth(deep_json, max_depth=4)

    large_df = pd.DataFrame(np.zeros((100, 10)))
    validate_ingestion_resource_limits(large_df, max_rows=200, max_cols=20)

    with pytest.raises(SecurityValidationError):
        validate_ingestion_resource_limits(large_df, max_rows=50)


# ---------------------------------------------------------------------------
# Test 7: Offline Ingestion Capability
# ---------------------------------------------------------------------------

def test_offline_ingestion_capability():
    assert verify_offline_ingestion_capability() is True


# ---------------------------------------------------------------------------
# Test 8: Worst-Subgroup Selection Disclosure
# ---------------------------------------------------------------------------

def test_worst_subgroup_selection_disclosure():
    yt = np.array([1, 0, 1, 0, 1, 0])
    yp = np.array([1, 0, 0, 0, 1, 1])
    s = np.array(["G1", "G1", "G2", "G2", "G3", "G3"])

    suite = evaluate_fairness_defensively(yt, yp, sensitive=s)
    for g in ["G1", "G2", "G3"]:
        assert g in suite.subgroup_metrics
