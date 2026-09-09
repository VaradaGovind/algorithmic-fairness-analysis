# ============================================================================
# Regression Test Suite - Pass 8: Evaluation Units, Product Schema & Privacy
# Tests statistical evaluation units, entity aggregation, inverse weighting,
# statistical mismatch guards, Draft-07 audit result schema, and report privacy.
# ============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.audit_result_schema import (
    AuditDataset,
    AuditEvaluationUnitAlignment,
    AuditGlobalPerformance,
    AuditPolicyCompliance,
    AuditPrivacyGovernance,
    AuditResult,
    AuditScope,
    AuditSensitiveAttribute,
    SensitiveAttributeLineage,
    build_audit_result_from_suite,
    compute_canonical_audit_fingerprint,
    validate_audit_result_dict,
)
from src.evaluation_manifest import (
    EvaluationManifest,
    ResultClassification,
    create_evaluation_manifest,
)
from src.fairness_evaluator import (
    AggregationStrategy,
    EvaluationStatus,
    EvaluationUnitLevel,
    EvidenceState,
    FairnessEvaluationSuite,
    FairnessPolicy,
    PolicyDecision,
    PolicyType,
    ReportingPrivacyMode,
    ResamplingUnit,
    aggregate_to_evaluation_unit,
    compute_entity_weights,
    evaluate_fairness_defensively,
    govern_evaluation_reporting,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Test 1: Evaluation Unit Registry Loading & Validation
# ---------------------------------------------------------------------------

def test_evaluation_unit_registry_loading():
    config_path = PROJECT_ROOT / "configs" / "evaluation_units.yaml"
    assert config_path.exists(), f"Configuration file {config_path} missing."

    with open(config_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    assert "version" in registry
    assert "evaluation_units" in registry

    datasets = registry["evaluation_units"]
    required_datasets = ["delhivery_logistics", "amazon_last_mile", "adult_income", "synthetic_logistics"]
    for ds in required_datasets:
        assert ds in datasets, f"Dataset '{ds}' missing from evaluation_units.yaml"
        ds_cfg = datasets[ds]
        assert "natural_decision_unit" in ds_cfg
        assert "raw_record_unit" in ds_cfg
        assert "evaluation_unit_key" in ds_cfg
        assert "historical_unit_mismatch" in ds_cfg

    # Verify Delhivery specifics: Raw is segment, decision is trip
    dlh = datasets["delhivery_logistics"]
    assert dlh["natural_decision_unit"] == "TRIP_LEVEL"
    assert dlh["raw_record_unit"] == "SEGMENT_LEVEL"
    assert dlh["historical_unit_mismatch"]["historical_evaluated_unit"] == "SEGMENT_LEVEL"


# ---------------------------------------------------------------------------
# Test 2: Row vs. Entity Metric Divergence Fixture
# ---------------------------------------------------------------------------

def test_row_vs_entity_metric_divergence_fixture():
    """
    Constructs a scenario where 1 trip in Group A has 20 segments that all pass,
    while 9 trips in Group A have 1 segment each and all fail.
    Row-level: 20 pass, 9 fail -> 20/29 = 68.9% success.
    Trip-level: 1 pass, 9 fail -> 1/10 = 10% success.
    Demonstrates mathematically that row evaluation misrepresents entity outcome.
    """
    # Group A: 1 big trip with 20 passing segments (all y_true=1, y_pred=1)
    # Plus 9 single-segment failing trips (y_true=0, y_pred=0)
    trip_ids_a = ["trip_A_big"] * 20 + [f"trip_A_{i}" for i in range(1, 10)]
    y_true_a = [1] * 20 + [0] * 9
    y_pred_a = [1] * 20 + [0] * 9
    group_a = ["Group_A"] * 29

    # Group B: 10 trips of 2 segments each (all y_true=1, y_pred=1)
    trip_ids_b = []
    for i in range(10):
        trip_ids_b.extend([f"trip_B_{i}"] * 2)
    y_true_b = [1] * 20
    y_pred_b = [1] * 20
    group_b = ["Group_B"] * 20

    all_trips = np.array(trip_ids_a + trip_ids_b)
    all_yt = np.array(y_true_a + y_true_b)
    all_yp = np.array(y_pred_a + y_pred_b)
    all_groups = np.array(group_a + group_b)

    # 1. Unaggregated Row-level evaluation
    suite_row = evaluate_fairness_defensively(
        y_true=all_yt,
        y_pred=all_yp,
        sensitive=all_groups,
        entity_series=all_trips,
        evaluation_unit_level=EvaluationUnitLevel.ROW_LEVEL,
        aggregation_strategy=AggregationStrategy.NONE,
    )
    # Row level positive prediction rate in Group A: 20/29 = 0.6897
    row_sub = suite_row.subgroup_metrics
    rate_a_row = row_sub["Group_A"]["predicted_positive_rate"]
    assert pytest.approx(rate_a_row, 0.01) == 20 / 29

    # 2. Trip-level aggregated evaluation (ANY_FAILURE strategy)
    suite_trip = evaluate_fairness_defensively(
        y_true=all_yt,
        y_pred=all_yp,
        sensitive=all_groups,
        entity_series=all_trips,
        evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
        aggregation_strategy=AggregationStrategy.ANY_FAILURE,
    )
    trip_sub = suite_trip.subgroup_metrics
    # Trip level positive rate in Group A: exactly 1 passing trip / 10 total trips = 0.10
    rate_a_trip = trip_sub["Group_A"]["predicted_positive_rate"]
    assert pytest.approx(rate_a_trip, 0.01) == 0.10

    # The two rates diverge dramatically
    assert abs(rate_a_row - rate_a_trip) > 0.50
    assert suite_trip.sample_size == 20  # 10 trips in A + 10 trips in B


# ---------------------------------------------------------------------------
# Test 3: High-Volume Entity Domination Guard (Inverse-Frequency Weighting)
# ---------------------------------------------------------------------------

def test_high_volume_entity_domination_guard():
    """
    Verifies that ENTITY_WEIGHTED strategy assigns 1/N_entity weight to each row,
    preventing high-volume entities from dominating global accuracy and rates.
    """
    # Entity 1 has 90 rows (all y_true=1, y_pred=1)
    # Entities 2 to 11 have 1 row each (all y_true=0, y_pred=1 -> 10 mistakes)
    entities = ["ENT_DOMINANT"] * 90 + [f"ENT_SMALL_{i}" for i in range(10)]
    y_true = np.array([1] * 90 + [0] * 10)
    y_pred = np.array([1] * 100)
    sensitive = np.array(["G1"] * 50 + ["G2"] * 50)

    # Unweighted row evaluation: accuracy = 90 / 100 = 0.90
    suite_unweighted = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        entity_series=entities,
        aggregation_strategy=AggregationStrategy.NONE,
    )
    assert pytest.approx(suite_unweighted.accuracy, 0.01) == 0.90

    # Entity-weighted evaluation:
    # ENT_DOMINANT gets weight 1.0 (90 rows * 1/90 = 1.0)
    # The 10 small entities get weight 1.0 each (10 rows * 1/1 = 10.0)
    # Total entity weight = 11.0. Correct entity count = 1.0. Accuracy = 1/11 = 0.0909
    suite_weighted = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        entity_series=entities,
        aggregation_strategy=AggregationStrategy.ENTITY_WEIGHTED,
    )
    assert pytest.approx(suite_weighted.accuracy, 0.01) == 1.0 / 11.0


# ---------------------------------------------------------------------------
# Test 4: Aggregation Strategies Determinism
# ---------------------------------------------------------------------------

def test_aggregation_strategies():
    # 2 trips, each with 3 segments
    # Trip 1: preds = [1, 0, 1], trues = [1, 1, 1]
    # Trip 2: preds = [0, 0, 0], trues = [0, 0, 0]
    entities = np.array(["T1", "T1", "T1", "T2", "T2", "T2"])
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    sensitive = np.array(["A", "A", "A", "B", "B", "B"])

    # Strategy: ANY_FAILURE (pred is 1 only if ALL segments are 1)
    yt_af, yp_af, s_af, _, _ = aggregate_to_evaluation_unit(
        y_true, y_pred, sensitive, entities, strategy=AggregationStrategy.ANY_FAILURE
    )
    # T1 has a 0 prediction, so any failure makes it 0!
    assert yp_af[0] == 0
    assert yp_af[1] == 0

    # Strategy: MAJORITY_VOTE (T1 has two 1s and one 0 -> majority is 1)
    yt_mv, yp_mv, s_mv, _, _ = aggregate_to_evaluation_unit(
        y_true, y_pred, sensitive, entities, strategy=AggregationStrategy.MAJORITY_VOTE
    )
    assert yp_mv[0] == 1
    assert yp_mv[1] == 0

    # Strategy: ALL_SUCCESS (pred is 1 if any segment succeeded)
    yt_as, yp_as, s_as, _, _ = aggregate_to_evaluation_unit(
        y_true, y_pred, sensitive, entities, strategy=AggregationStrategy.ALL_SUCCESS
    )
    assert yp_as[0] == 1
    assert yp_as[1] == 0


# ---------------------------------------------------------------------------
# Test 5: Compute Entity Weights
# ---------------------------------------------------------------------------

def test_entity_weights_computation():
    entities = ["A", "A", "A", "B"]
    weights = compute_entity_weights(entities)
    # Raw inverse frequencies: A=1/3, A=1/3, A=1/3, B=1.0. Sum = 2.0.
    # Normalized to mean 1.0 (sum=4.0): A = (1/3)/(2/4) = 2/3. B = 1.0/(2/4) = 2.0.
    assert len(weights) == 4
    assert pytest.approx(weights[0], 1e-5) == 2.0 / 3.0
    assert pytest.approx(weights[1], 1e-5) == 2.0 / 3.0
    assert pytest.approx(weights[2], 1e-5) == 2.0 / 3.0
    assert pytest.approx(weights[3], 1e-5) == 2.0
    assert pytest.approx(np.sum(weights), 1e-5) == 4.0


# ---------------------------------------------------------------------------
# Test 6: Statistical Unit Mismatch Guard
# ---------------------------------------------------------------------------

def test_statistical_unit_mismatch_guard():
    # Repeated entities evaluated as TRIP_LEVEL with ROW_BOOTSTRAP requested
    entities = np.array(["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4"])
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    # 1. Must raise ValueError when row bootstrap is used without override
    with pytest.raises(ValueError) as exc_info:
        evaluate_fairness_defensively(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
            entity_series=entities,
            evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
            compute_uncertainty=True,
            resampling_unit=ResamplingUnit.ROW_BOOTSTRAP,
            allow_row_bootstrap_override=False,
        )
    assert "Statistical unit mismatch" in str(exc_info.value)
    assert "violates i.i.d. sampling assumptions" in str(exc_info.value)

    # 2. Must succeed with warning when override is explicitly granted
    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        entity_series=entities,
        evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
        compute_uncertainty=True,
        resampling_unit=ResamplingUnit.ROW_BOOTSTRAP,
        allow_row_bootstrap_override=True,
    )
    assert any("Statistical unit mismatch override" in w for w in suite.warnings)


# ---------------------------------------------------------------------------
# Test 7: Cluster Bootstrap Entity Alignment
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_entity_alignment():
    # Entities act as bootstrap clusters
    entities = np.array(["C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4", "C5", "C5", "C6", "C6"])
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    sensitive = np.array(["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"])

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        cluster_series=entities,
        evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
        compute_uncertainty=True,
        resampling_unit=ResamplingUnit.CLUSTER_BOOTSTRAP,
        n_bootstraps=50,
    )
    assert suite.uncertainty_intervals is not None
    assert "Equalized Odds Diff" in suite.uncertainty_intervals
    eo_unc = suite.uncertainty_intervals["Equalized Odds Diff"]
    assert eo_unc["resampling_unit"] == "CLUSTER_BOOTSTRAP"


# ---------------------------------------------------------------------------
# Test 8: Intersectional Entity Fairness
# ---------------------------------------------------------------------------

def test_intersectional_entity_fairness():
    # Intersectional grouping at entity level
    entities = np.array([f"E_{i}" for i in range(20)])
    y_true = np.array([1, 0] * 10)
    y_pred = np.array([1, 0] * 10)
    # 2 attributes: Gender and AgeGroup
    df_sens = pd.DataFrame({
        "Gender": ["M"] * 10 + ["F"] * 10,
        "Age": ["Young"] * 5 + ["Senior"] * 5 + ["Young"] * 5 + ["Senior"] * 5,
    })

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_sens,
        entity_series=entities,
        evaluation_unit_level=EvaluationUnitLevel.INDIVIDUAL_LEVEL,
    )
    assert len(suite.group_counts) == 4
    assert "M_x_Young" in suite.group_counts
    assert "F_x_Senior" in suite.group_counts
    assert suite.overall_status == EvaluationStatus.VALID


# ---------------------------------------------------------------------------
# Test 9: Audit Result Schema Complete Validation
# ---------------------------------------------------------------------------

def test_audit_result_schema_validation():
    yt = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    yp = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 5)
    s = np.array(["Alpha"] * 25 + ["Beta"] * 25)

    suite = evaluate_fairness_defensively(
        y_true=yt,
        y_pred=yp,
        sensitive=s,
        evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
        evaluation_unit_key="trip_id",
    )

    audit_res = build_audit_result_from_suite(
        suite=suite,
        purpose="Production logistics risk audit",
        dataset_name="delhivery_logistics",
        raw_observation_unit="WAYPOINT_SEGMENT",
        unit_key="trip_uuid",
        records_per_unit_mean=4.2,
        records_per_unit_max=18,
        divergence_from_row_level=True,
        sensitive_attribute_name="origin_hub_tier",
        sensitive_lineage=SensitiveAttributeLineage.OBSERVED,
    )

    audit_dict = audit_res.to_dict()
    is_valid, errors = validate_audit_result_dict(audit_dict)
    assert is_valid is True, f"Validation failed with errors: {errors}"
    assert len(errors) == 0
    assert len(audit_dict) == 20
    assert len(audit_res.audit_fingerprint) == 64


# ---------------------------------------------------------------------------
# Test 10: Audit Result JSON Serialization & Roundtrip
# ---------------------------------------------------------------------------

def test_audit_result_json_canonical_roundtrip(tmp_path):
    yt = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    yp = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    s = np.array(["G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2"])
    suite = evaluate_fairness_defensively(yt, yp, sensitive=s)

    audit_res = build_audit_result_from_suite(suite)
    fp_orig = audit_res.audit_fingerprint

    # Save to disk and reload
    save_file = tmp_path / "test_audit_result.json"
    audit_res.save_json(save_file)
    assert save_file.exists()

    loaded = AuditResult.load_json(save_file)
    assert loaded.audit_id == audit_res.audit_id
    assert loaded.audit_fingerprint == fp_orig
    assert loaded.dataset["dataset_name"] == audit_res.dataset["dataset_name"]


# ---------------------------------------------------------------------------
# Test 11: Privacy Governance - FULL_INTERNAL Mode
# ---------------------------------------------------------------------------

def test_privacy_mode_full_internal():
    yt = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    yp = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    # Very small group of 2 samples
    s = np.array(["Common", "Common", "Common", "Common", "Common", "Common", "Common", "Common", "Rare", "Rare"])

    suite = evaluate_fairness_defensively(
        yt, yp, sensitive=s,
        reporting_privacy_mode=ReportingPrivacyMode.FULL_INTERNAL,
        min_reporting_population=5,
    )
    assert "Rare" in suite.subgroup_metrics
    assert suite.subgroup_metrics["Rare"]["count"] == 2
    assert suite.privacy_suppression_metadata["suppression_applied"] is False
    assert suite.privacy_suppression_metadata["disclosed_groups_count"] == 2


# ---------------------------------------------------------------------------
# Test 12: Privacy Governance - REDACTED_EXTERNAL Mode
# ---------------------------------------------------------------------------

def test_privacy_mode_redacted_external():
    yt = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    yp = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    # Group 'Common' has 8, 'Rare' has 2
    s = np.array(["Common", "Common", "Common", "Common", "Common", "Common", "Common", "Common", "Rare", "Rare"])

    suite = evaluate_fairness_defensively(
        yt, yp, sensitive=s,
        reporting_privacy_mode=ReportingPrivacyMode.REDACTED_EXTERNAL,
        min_reporting_population=5,
        mask_privacy_labels=True,
    )
    # Rare group (N=2 < 5) must be suppressed!
    assert "Rare" not in suite.subgroup_metrics
    # Masked label must be applied to the remaining group
    assert "REDACTED_GROUP_001" in suite.subgroup_metrics
    assert suite.privacy_suppression_metadata["suppression_applied"] is True
    assert suite.privacy_suppression_metadata["suppressed_groups_count"] == 1
    assert suite.privacy_suppression_metadata["disclosed_groups_count"] == 1
    assert suite.privacy_suppression_metadata["labels_masked"] is True


# ---------------------------------------------------------------------------
# Test 13: Privacy Governance - SUMMARY_ONLY Mode
# ---------------------------------------------------------------------------

def test_privacy_mode_summary_only():
    yt = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    yp = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    s = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    suite = evaluate_fairness_defensively(
        yt, yp, sensitive=s,
        reporting_privacy_mode=ReportingPrivacyMode.SUMMARY_ONLY,
    )
    # All subgroup breakdowns suppressed
    assert suite.subgroup_metrics == {}
    assert suite.calibration_by_subgroup == {}
    # Global metrics preserved
    assert suite.accuracy == 1.0
    assert suite.privacy_suppression_metadata["suppressed_groups_count"] == 2
    assert suite.privacy_suppression_metadata["disclosed_groups_count"] == 0


# ---------------------------------------------------------------------------
# Test 14: Privacy Governance Disclaimer Invariant
# ---------------------------------------------------------------------------

def test_privacy_governance_disclaimer_invariant():
    """
    Enforces that privacy governance must explicitly state report privacy,
    and never claim differential privacy.
    """
    yt = np.array([1, 0, 1, 0])
    yp = np.array([1, 0, 1, 0])
    s = np.array(["A", "A", "B", "B"])
    suite = evaluate_fairness_defensively(yt, yp, sensitive=s)
    audit_res = build_audit_result_from_suite(suite)

    # Valid disclaimer
    is_valid, _ = validate_audit_result_dict(audit_res.to_dict())
    assert is_valid is True

    # Tamper with disclaimer to falsely claim differential privacy
    bad_dict = audit_res.to_dict()
    bad_dict["privacy_governance"]["disclaimer"] = "Guaranteed differential privacy epsilon=0.1"
    is_valid_bad, errs = validate_audit_result_dict(bad_dict)
    assert is_valid_bad is False
    assert any("must explicitly distinguish report privacy from differential privacy" in e for e in errs)


# ---------------------------------------------------------------------------
# Test 15: Sensitive Attribute Lineage & Proxy Disclaimers
# ---------------------------------------------------------------------------

def test_sensitive_attribute_lineage_and_proxy_rules():
    # Observed lineage
    attr_obs = AuditSensitiveAttribute(attribute_name="race", lineage="OBSERVED")
    assert attr_obs.lineage == "OBSERVED"

    # Derived proxy lineage automatically attaches non-causal disclaimer
    attr_proxy = AuditSensitiveAttribute(
        attribute_name="zip_proxy",
        lineage="DERIVED_PROXY",
        proxy_derivation_rule="Mapped from 5-digit postal code to median income quartile",
    )
    assert attr_proxy.lineage == "DERIVED_PROXY"
    assert "do not support causal or legal claims" in attr_proxy.legal_disclaimer

    # Validation catches missing proxy derivation rule on proxy lineage
    yt = np.array([1, 0, 1, 0])
    yp = np.array([1, 0, 1, 0])
    s = np.array(["Z1", "Z1", "Z2", "Z2"])
    suite = evaluate_fairness_defensively(yt, yp, sensitive=s)
    audit_res = build_audit_result_from_suite(
        suite,
        sensitive_attribute_name="zip_proxy",
        sensitive_lineage=SensitiveAttributeLineage.DERIVED_PROXY,
    )
    audit_dict = audit_res.to_dict()
    # Strip legal disclaimer and proxy rule to test enforcement
    audit_dict["sensitive_attributes"][0]["legal_disclaimer"] = None
    audit_dict["sensitive_attributes"][0]["proxy_derivation_rule"] = None
    is_valid, errs = validate_audit_result_dict(audit_dict)
    assert is_valid is False
    assert any("must document proxy_derivation_rule or legal_disclaimer" in e for e in errs)


# ---------------------------------------------------------------------------
# Test 16: Policy Hierarchy Policy Types
# ---------------------------------------------------------------------------

def test_policy_hierarchy_policy_types():
    # Verify PolicyType enum
    assert PolicyType.ANALYTICAL.value == "ANALYTICAL"
    assert PolicyType.ORGANIZATIONAL.value == "ORGANIZATIONAL"
    assert PolicyType.REGULATORY_MAPPING.value == "REGULATORY_MAPPING"

    policy_reg = FairnessPolicy(
        name="EU_AI_Act_HighRisk_Logistics",
        policy_type=PolicyType.REGULATORY_MAPPING,
        effective_date="2026-08-01",
        description="EU AI Act Annex III high-risk dispatch evaluation",
        min_balanced_accuracy=0.60,
        max_equalized_odds_diff=0.08,
    )
    assert policy_reg.policy_type == PolicyType.REGULATORY_MAPPING

    yt = np.array([1, 0, 1, 0] * 5)
    yp = np.array([1, 0, 1, 0] * 5)
    s = np.array(["A"] * 10 + ["B"] * 10)

    suite = evaluate_fairness_defensively(yt, yp, sensitive=s, policy=policy_reg)
    assert suite.policy_result is not None
    assert suite.policy_result.policy_type == PolicyType.REGULATORY_MAPPING

    audit_res = build_audit_result_from_suite(suite)
    assert audit_res.policy_compliance["policy_type"] == "REGULATORY_MAPPING"
    is_val, _ = validate_audit_result_dict(audit_res.to_dict())
    assert is_val is True


# ---------------------------------------------------------------------------
# Test 17: External Real Data Pending Status Integrity (No Fabrication)
# ---------------------------------------------------------------------------

def test_external_real_data_pending_status_integrity():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    delhivery_file = raw_dir / "delhivery_data.csv"
    amazon_file = raw_dir / "route_data.json"
    adult_file = raw_dir / "adult.data"

    # Verify that raw proprietary files are NOT checked in
    # (they are staged externally or downloaded out-of-band)
    for raw_f in [delhivery_file, amazon_file, adult_file]:
        assert not raw_f.exists(), (
            f"Proprietary customer data file {raw_f} found in repository! "
            f"External data must remain segregated."
        )

    # Verify EvaluationManifest handles EXTERNAL_PENDING_STAGING properly
    manifest = create_evaluation_manifest(
        dataset_name="delhivery_logistics",
        result_classification=ResultClassification.EXTERNAL_PENDING_STAGING,
        evidence_state="EXTERNAL_PENDING_STAGING",
        evaluation_unit_level="TRIP_LEVEL",
        evaluation_unit_key="trip_uuid",
    )
    assert manifest.result_classification == ResultClassification.EXTERNAL_PENDING_STAGING
    assert manifest.evidence_state == "EXTERNAL_PENDING_STAGING"
    assert manifest.evaluation_unit_level == "TRIP_LEVEL"
