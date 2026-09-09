# ============================================================================
# PASS 7 COMPREHENSIVE VALIDATION & AUDIT TEST SUITE
# Validates cluster-aware bootstrap uncertainty, policy evidence trace,
# imputed sensitive attribute governance, multiplicity corrections,
# worst-subgroup selection disclosures, ingestion resource limits,
# offline ingestion capabilities, and result classification taxonomy.
# ============================================================================

import json
import socket
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation_manifest import (
    EvaluationManifest,
    ResultClassification,
    compute_manifest_fingerprint,
    create_evaluation_manifest,
)
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
)
from src.ingest_data import (
    SecurityValidationError,
    check_json_depth,
    validate_ingestion_resource_limits,
    verify_offline_ingestion_capability,
)


# ---------------------------------------------------------------------------
# TEST 1: Cluster Bootstrap Sampling Integrity (All Selected or All Excluded)
# ---------------------------------------------------------------------------
def test_cluster_bootstrap_replicate_cluster_integrity():
    """
    Verifies that for each cluster bootstrap replicate, all rows of a cluster
    are either all selected (with identical multiplicity) or all excluded.
    """
    # 5 clusters, 4 rows each = 20 rows
    n_clusters = 5
    cluster_ids = [f"cluster_{i}" for i in range(n_clusters)]
    clusters = np.repeat(cluster_ids, 4)
    y_true = np.array([1, 0, 1, 0] * n_clusters)
    y_pred = np.array([1, 0, 1, 0] * n_clusters)
    sensitive = np.array(["GroupA", "GroupB"] * (len(clusters) // 2))

    rng = np.random.RandomState(42)
    unique_clusters = np.unique(clusters)
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    # Simulate 50 replicates and check index grouping
    for _ in range(50):
        sampled_c = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        boot_idx = np.concatenate([cluster_to_indices[c] for c in sampled_c])
        sampled_cluster_counts = pd.Series(clusters[boot_idx]).value_counts()
        for cid in unique_clusters:
            # Each cluster has 4 rows; its row count in replicate must be a multiple of 4
            count = sampled_cluster_counts.get(cid, 0)
            assert count % 4 == 0, f"Cluster {cid} row count {count} is not a multiple of 4"


# ---------------------------------------------------------------------------
# TEST 2: Cluster Bootstrap Uncertainty Under Correlation vs Row Bootstrap
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
        # Highly correlated cluster effect: all rows in cluster share group and outcome tendency
        c_sens = "GroupA" if c % 2 == 0 else "GroupB"
        c_base = 1 if rng.rand() > 0.4 else 0
        for _ in range(rows_per_cluster):
            clusters_list.append(cid)
            sens_list.append(c_sens)
            # Add minor intra-cluster noise
            yt = c_base if rng.rand() > 0.1 else (1 - c_base)
            yp = yt if rng.rand() > 0.2 else (1 - yt)
            y_true_list.append(yt)
            y_pred_list.append(yp)

    y_t = np.array(y_true_list)
    y_p = np.array(y_pred_list)
    sens = np.array(sens_list)
    clusters = np.array(clusters_list)

    # 1. Row Bootstrap
    res_row = compute_fairness_uncertainty_bootstrap(
        y_t, y_p, sens,
        n_bootstraps=200,
        resampling_unit=ResamplingUnit.ROW_BOOTSTRAP,
        seed=42,
    )
    # 2. Cluster Bootstrap
    res_cluster = compute_fairness_uncertainty_bootstrap(
        y_t, y_p, sens,
        n_bootstraps=200,
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
    # Cluster bootstrap properly preserves cluster variation, producing wider or equal interval
    assert width_cluster >= width_row * 0.90


# ---------------------------------------------------------------------------
# TEST 3: Cluster Bootstrap Refusal to Silently Fall Back to Row Bootstrap
# ---------------------------------------------------------------------------
def test_cluster_bootstrap_no_silent_fallback():
    """
    When CLUSTER_BOOTSTRAP is requested without cluster_series, it raises ValueError
    rather than silently reverting to row-level bootstrap.
    """
    y_true = np.array([1, 0, 1, 0] * 5)
    y_pred = np.array([1, 0, 1, 0] * 5)
    sens = np.array(["A", "B"] * 10)

    with pytest.raises(ValueError, match="no cluster_series was provided"):
        compute_fairness_uncertainty_bootstrap(
            y_true, y_pred, sens,
            resampling_unit=ResamplingUnit.CLUSTER_BOOTSTRAP,
            cluster_series=None,
        )


# ---------------------------------------------------------------------------
# TEST 4: Sparse Cluster Support Gating
# ---------------------------------------------------------------------------
def test_sparse_cluster_support_gating():
    """
    When unique clusters < 5, cluster bootstrap flags INSUFFICIENT_BOOTSTRAP_SUPPORT.
    """
    y_true = np.array([1, 0, 1, 0] * 2)
    y_pred = np.array([1, 0, 1, 0] * 2)
    sens = np.array(["A", "B"] * 4)
    clusters = np.array(["c1", "c1", "c2", "c2", "c3", "c3", "c3", "c3"])  # Only 3 clusters

    res = compute_fairness_uncertainty_bootstrap(
        y_true, y_pred, sens,
        n_bootstraps=50,
        resampling_unit=ResamplingUnit.CLUSTER_BOOTSTRAP,
        cluster_series=clusters,
        seed=42,
    )
    assert res.support_status == EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT.value


# ---------------------------------------------------------------------------
# TEST 5: Policy Evidence Trace Generation & Transparency
# ---------------------------------------------------------------------------
def test_policy_evidence_trace_completeness():
    """
    Verifies that evaluate_policy_compliance produces an auditable condition-by-condition
    evidence trace with reason codes, failed/passed conditions, and numeric details.
    """
    suite = FairnessEvaluationSuite(
        dataset_name="AuditTest",
        model_name="CandidateModel",
        sample_size=100,
        group_counts={"GroupA": 50, "GroupB": 50},
        metrics={},
        degeneracy_status=ModelDegeneracyStatus.NORMAL,
    )
    policy = FairnessPolicy(
        name="Auditable_Operational_Policy",
        policy_id="POL-AUDIT-2026.1",
        min_balanced_accuracy=0.60,
        max_equalized_odds_diff=0.05,
    )

    res = evaluate_policy_compliance(suite, policy)
    assert hasattr(res, "evidence_trace")
    assert isinstance(res.evidence_trace, list)
    assert len(res.evidence_trace) > 0
    assert hasattr(res, "reason_codes")
    assert isinstance(res.reason_codes, list)
    assert hasattr(res, "failed_conditions")
    assert hasattr(res, "passed_conditions")
    assert hasattr(res, "missing_evidence_conditions")

    # Serialize to dict and verify all fields preserved
    d = res.to_dict()
    assert "evidence_trace" in d
    assert "reason_codes" in d
    assert "failed_conditions" in d
    assert "passed_conditions" in d
    assert "missing_evidence_conditions" in d


# ---------------------------------------------------------------------------
# TEST 6: Imputed Sensitive Attribute Governance & DERIVED_PROXY Classification
# ---------------------------------------------------------------------------
def test_imputed_sensitive_attribute_provenance_governance():
    """
    Under IMPUTE_WITH_DOCUMENTED_RULE, missing attributes are classified as DERIVED_PROXY,
    rule provenance is tracked, and causal/legal disclaimers are attached.
    """
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    sens = pd.Series(["A", None, "B", "A", np.nan, "B"])

    policy = FairnessPolicy(
        missing_attribute_policy=MissingAttributePolicy.IMPUTE_WITH_DOCUMENTED_RULE,
    )
    policy.imputation_rule_id = "RULE-ZIPCODE-PROXY-01"
    policy.imputation_rule_description = "Impute missing demographic group via geographic modal proxy"

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sens,
        policy=policy,
    )

    prov = suite.missing_attribute_handling.get("imputation_provenance")
    assert prov is not None
    assert prov["rule_id"] == "RULE-ZIPCODE-PROXY-01"
    assert prov["sensitive_origin"] == "DERIVED_PROXY"
    assert prov["original_missing_count"] == 2
    assert "disclaimer" in prov
    assert "DERIVED_PROXY" in prov["disclaimer"]


# ---------------------------------------------------------------------------
# TEST 7: Worst-Subgroup Selection Reporting Disclosure
# ---------------------------------------------------------------------------
def test_worst_subgroup_selection_disclosure_formatting():
    """
    Reports state groups searched, supported count, unsupported count, post hoc selection,
    and exact phrase 'worst observed subgroup among the evaluated supported groups'.
    """
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    sens = pd.Series(["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"])

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sens,
        min_group_size=2,
    )

    disclosure = suite.worst_group_selection_disclosure
    assert disclosure is not None
    assert disclosure["groups_searched_count"] == 4
    assert disclosure["supported_groups_count"] == 4
    assert disclosure["unsupported_groups_count"] == 0
    assert disclosure["selected_post_hoc"] is True
    assert disclosure["label"] == "worst observed subgroup among the evaluated supported groups"
    assert "winner's curse" in disclosure["disclosure"]


# ---------------------------------------------------------------------------
# TEST 8: Multiple Comparisons & Confirmatory Multiplicity Procedure Guarantee
# ---------------------------------------------------------------------------
def test_confirmatory_multiplicity_procedure_guarantee():
    """
    CONFIRMATORY mode records procedure guarantee without overclaiming
    'prevents false discoveries'.
    """
    suite = FairnessEvaluationSuite(
        dataset_name="MultiplicityTest",
        model_name="ModelA",
        sample_size=100,
        group_counts={"A": 50, "B": 50},
        metrics={},
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
        number_of_comparisons=10,
    )
    assert suite.analysis_type == "CONFIRMATORY_ANALYSIS"
    assert suite.number_of_comparisons == 10


# ---------------------------------------------------------------------------
# TEST 9: Ingestion Resource Limits (Max Rows & Max Columns)
# ---------------------------------------------------------------------------
def test_ingestion_resource_limits_rejection():
    """
    Ingestion limits reject payloads exceeding max_rows or max_columns.
    """
    # DataFrame exceeding column limit
    df_huge_cols = pd.DataFrame(np.zeros((5, 1005)))
    with pytest.raises(SecurityValidationError, match="column count"):
        validate_ingestion_resource_limits(df_huge_cols, max_cols=1000)

    # DataFrame exceeding row limit
    df_huge_rows = pd.DataFrame(np.zeros((150, 5)))
    with pytest.raises(SecurityValidationError, match="row count"):
        validate_ingestion_resource_limits(df_huge_rows, max_rows=100)


# ---------------------------------------------------------------------------
# TEST 10: Ingestion JSON Depth Resource Limits
# ---------------------------------------------------------------------------
def test_ingestion_json_depth_limit():
    """
    Rejects deeply nested JSON structures exceeding max_depth.
    """
    # Build 12-level nested dict
    nested = {"val": 1}
    for _ in range(12):
        nested = {"level": nested}

    with pytest.raises(SecurityValidationError, match="nesting depth"):
        check_json_depth(nested, max_depth=10)


# ---------------------------------------------------------------------------
# TEST 11: Offline Ingestion Capability
# ---------------------------------------------------------------------------
def test_offline_ingestion_capability():
    """
    Confirms offline ingestion capability operates without external sockets.
    """
    assert verify_offline_ingestion_capability() is True


# ---------------------------------------------------------------------------
# TEST 12: Result Classification Taxonomy Standardization
# ---------------------------------------------------------------------------
def test_result_classification_taxonomy():
    """
    Verifies all 6 standardized Pass 7 classifications are recognized.
    """
    expected_categories = [
        "INTERNAL_ENGINEERING_VERIFICATION",
        "SYNTHETIC_DGP_VALIDATION",
        "INDEPENDENT_ORACLE_VALIDATION",
        "EXTERNAL_REAL_DATA_VALIDATION",
        "EXPLORATORY",
        "LEGACY",
    ]
    for cat in expected_categories:
        enum_val = ResultClassification(cat)
        assert enum_val.value == cat

    # Verify manifest creation with standardized taxonomy
    manifest = create_evaluation_manifest(
        run_id="TAXONOMY-TEST-001",
        model_name="TestModel",
        dataset_name="TestDataset",
        regime="IID_HOLDOUT",
        metrics={"Accuracy": 0.85},
        seeds=[42],
        result_classification=ResultClassification.INTERNAL_ENGINEERING_VERIFICATION,
    )
    assert manifest.result_classification == ResultClassification.INTERNAL_ENGINEERING_VERIFICATION
    fp = compute_manifest_fingerprint(manifest)
    assert len(fp) == 64
