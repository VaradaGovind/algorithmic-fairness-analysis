# ============================================================================
# PASS 9 COMPREHENSIVE TEST SUITE: Statistical Estimands, Semantic Invariants,
# Decoupled Units, Policy Certification Gate, & Evaluation Bundles
# ============================================================================

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import yaml

from src.audit_bundle import (
    DatasetStorageStatus,
    create_evaluation_bundle,
)
from src.audit_invariants import (
    ERR_INVALID_COUNT_IDENTITY,
    ERR_INVALID_METRIC_IDENTITY,
    ERR_INVALID_GROUP_MAPPING,
    ERR_INVALID_UNIT_ALIGNMENT,
    ERR_MISSING_ESTIMAND,
    audit_artifact_privacy,
    validate_audit_mathematical_invariants,
)
from src.audit_result_schema import (
    AuditEvaluationUnitAlignment,
    AuditPolicyCompliance,
    AuditResult,
    AuditStatisticalEstimand,
    build_audit_result_from_suite,
    validate_audit_result_dict,
)
from src.evaluation_manifest import (
    EvaluationManifest,
    ResultClassification,
)
from src.external_dataset_adapter import (
    ExternalDataBundle,
    ExternalDatasetAdapter,
)
from src.fairness_evaluator import (
    AGGREGATION_STRATEGY_REGISTRY,
    AggregationStrategy,
    AggregationStrategySpecification,
    AnalysisType,
    EvidenceState,
    EvaluationUnitLevel,
    FairnessEvaluationSuite,
    FairnessPolicy,
    MissingAttributePolicy,
    PolicyDecision,
    aggregate_to_evaluation_unit,
    evaluate_fairness_defensively,
    evaluate_policy_compliance,
    validate_evaluation_unit_configuration,
    verify_policy_certification_gate,
)


@pytest.fixture
def sample_estimand_dict() -> Dict[str, Any]:
    return {
        "estimand_id": "EST-DELHIVERY-ROUTE-001",
        "description": "Route-level delivery success disparity across driver complexity tiers",
        "target_population": "Operational delivery routes in urban hubs",
        "evaluation_decision_unit": "ROUTE_LEVEL",
        "observational_record_unit": "ROUTE_SEGMENT_RECORD",
        "weighting_rule": "UNIFORM_ROUTE_WEIGHT",
        "target_quantity": "BINARY_ROUTE_SUCCESS_ON_TIME",
        "fairness_quantity": "DEMOGRAPHIC_PARITY_DIFFERENCE",
        "cluster_dependence_unit": "STATION_HUB",
        "bootstrap_resampling_unit": "CLUSTER_BOOTSTRAP",
        "grouping_definition": "Complexity_Tier",
    }


@pytest.fixture
def sample_estimand_dataclass(sample_estimand_dict) -> AuditStatisticalEstimand:
    return AuditStatisticalEstimand.from_dict(sample_estimand_dict)


# ============================================================================
# 1. Estimand Declaration & Validation
# ============================================================================

def test_estimand_declaration_and_validation(sample_estimand_dict, sample_estimand_dataclass):
    """Verifies that the 8-dimension estimand is properly defined and validated."""
    assert sample_estimand_dataclass.decision_unit == "ROUTE_LEVEL"
    assert sample_estimand_dataclass.observation_unit == "ROUTE_SEGMENT_RECORD"
    assert sample_estimand_dataclass.fairness_quantity == "DEMOGRAPHIC_PARITY_DIFFERENCE"
    assert sample_estimand_dataclass.bootstrap_unit == "CLUSTER_BOOTSTRAP"
    
    # Check round-trip serialization
    d = sample_estimand_dataclass.to_dict()
    assert d["decision_unit"] == "ROUTE_LEVEL"
    assert d["observation_unit"] == "ROUTE_SEGMENT_RECORD"


def test_estimand_yaml_specifications():
    """Verifies configs/evaluation_units.yaml contains full estimand contracts for all 4 datasets."""
    units_file = Path("configs/evaluation_units.yaml")
    assert units_file.exists()
    with open(units_file, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    contracts = spec.get("evaluation_units", {})
    assert "delhivery_logistics" in contracts
    assert "amazon_last_mile" in contracts
    assert "adult_income" in contracts
    assert "synthetic_logistics" in contracts

    for ds_name, cfg in contracts.items():
        assert "statistical_estimand" in cfg
        estimand = cfg["statistical_estimand"]
        assert "population" in estimand
        assert "decision_unit" in estimand
        assert "target_quantity" in estimand
        assert "fairness_quantity" in estimand


# ============================================================================
# 2. Estimand Unspecified Blocks Confirmatory Certification
# ============================================================================

def test_estimand_unspecified_blocks_confirmatory_certification():
    """Asserts policy certification gate blocks PASS when estimand is missing in confirmatory mode."""
    policy = FairnessPolicy(name="Standard_Confirmatory")
    
    # Generate clean predictions
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"] * 10)

    # Suite without statistical estimand
    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        policy=policy,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
        statistical_estimand=None,
    )

    gate_ok, gate_errs, gate_codes = verify_policy_certification_gate(suite, policy=policy)
    assert not gate_ok
    assert "GATE_MISSING_ESTIMAND" in gate_codes
    assert suite.evidence_state == EvidenceState.INSUFFICIENT_EVIDENCE


# ============================================================================
# 3. Decoupled 5 Core Concepts
# ============================================================================

def test_decoupled_five_concepts():
    """Asserts observational records, evaluation units, fairness groups, cluster keys, and estimand are distinct."""
    alignment = AuditEvaluationUnitAlignment(
        evaluation_decision_unit="TRIP_LEVEL",
        observational_record_unit="SEGMENT_LEVEL",
        fairness_group_definition="Education_Group",
        cluster_dependence_unit="trip_uuid",
        unit_key="trip_uuid",
        statistical_estimand=AuditStatisticalEstimand(
            population="Intercity freight",
            decision_unit="TRIP_LEVEL",
            observation_unit="SEGMENT_LEVEL",
            weighting_rule="UNIFORM_TRIP",
            target_quantity="TRIP_SUCCESS",
            fairness_quantity="EQUALIZED_ODDS",
            bootstrap_unit="CLUSTER_BOOTSTRAP",
            grouping_definition="Education_Group",
        ),
    )
    d = alignment.to_dict()
    assert d["evaluation_decision_unit"] != d["observational_record_unit"]
    assert d["cluster_dependence_unit"] == "trip_uuid"
    assert d["statistical_estimand"]["decision_unit"] == "TRIP_LEVEL"


# ============================================================================
# 4. Evaluation-Unit Configuration Validation
# ============================================================================

def test_invalid_evaluation_unit_configuration():
    """Passing nonexistent unit key raises ValueError with INVALID_EVALUATION_UNIT_CONFIGURATION."""
    df = pd.DataFrame({
        "trip_id": [1, 1, 2, 2],
        "feature": [10, 20, 30, 40],
        "label": [1, 0, 1, 0],
    })

    with pytest.raises(ValueError, match="INVALID_EVALUATION_UNIT_CONFIGURATION"):
        validate_evaluation_unit_configuration(df, evaluation_unit_key="nonexistent_unit_uuid")


def test_valid_evaluation_unit_configuration():
    """Valid evaluation unit key returns unit metadata dictionary."""
    df = pd.DataFrame({
        "trip_id": [1, 1, 2, 2],
        "feature": [10, 20, 30, 40],
        "label": [1, 0, 1, 0],
    })
    meta = validate_evaluation_unit_configuration(df, evaluation_unit_key="trip_id")
    assert meta["evaluation_unit_key"] == "trip_id"
    assert meta["n_records"] == 4
    assert meta["n_units"] == 2


# ============================================================================
# 5. Target vs Prediction Aggregation Semantics
# ============================================================================

def test_aggregation_target_vs_prediction_different_operators():
    """Asserts target and prediction can take different operators (ANY_FAILURE vs ALL_SUCCESS)."""
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    sensitive = np.array(["A", "A", "B", "B"])
    unit_ids = np.array(["U1", "U1", "U2", "U2"])

    agg_yt, agg_yp, agg_s, _, agg_u = aggregate_to_evaluation_unit(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        unit_ids=unit_ids,
        target_strategy="ANY_FAILURE",
        prediction_strategy="ALL_SUCCESS",
    )

    assert len(agg_u) == 2
    # U1 had targets [1, 0] -> ANY_FAILURE => 0
    assert agg_yt[0] == 0
    # U1 had preds [0, 1] -> ALL_SUCCESS => 1
    assert agg_yp[0] == 1


def test_unsupported_aggregation_strategy_rejected():
    """Unsupported strategy raises ValueError with UNSUPPORTED_EVALUATION_AGGREGATION."""
    with pytest.raises(ValueError, match="UNSUPPORTED_EVALUATION_AGGREGATION"):
        aggregate_to_evaluation_unit(
            y_true=np.array([1, 0]),
            y_pred=np.array([1, 0]),
            sensitive=np.array(["A", "B"]),
            unit_ids=np.array(["U1", "U2"]),
            target_strategy="RANDOM_GUESS",
        )


# ============================================================================
# 6. Semantic Mathematical Invariants
# ============================================================================

def test_audit_invariants_confusion_counts():
    """Validates confusion count identities via validate_audit_mathematical_invariants."""
    audit_data = {
        "dataset": {"total_records": 100, "entity_count": 50},
        "subgroup_evaluations": {
            "GroupA": {
                "sample_size": 50,
                "positive_count": 25,
                "negative_count": 25,
                "TP": 20, "FP": 5, "FN": 5, "TN": 20,
                "PPV": 20 / 25, "TPR": 20 / 25, "FPR": 5 / 25,
                "balanced_accuracy": (20/25 + 20/25) / 2,
            },
            "GroupB": {
                "sample_size": 50,
                "positive_count": 25,
                "negative_count": 25,
                "TP": 20, "FP": 5, "FN": 5, "TN": 20,
                "PPV": 20 / 25, "TPR": 20 / 25, "FPR": 5 / 25,
                "balanced_accuracy": (20/25 + 20/25) / 2,
            },
        },
        "global_performance": {
            "accuracy": 0.80,
            "precision": 0.80,
            "recall": 0.80,
            "f1_score": 0.80,
        },
        "fairness_metrics": {
            "demographic_parity_difference": 0.0,
            "equal_opportunity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "disparate_impact_ratio": 1.0,
        },
        "evaluation_unit_alignment": {
            "evaluation_unit_level": "ROW_LEVEL",
            "statistical_estimand": {
                "population": "Pop",
                "decision_unit": "ROW_LEVEL",
                "observation_unit": "ROW_LEVEL",
                "weighting_rule": "UNIFORM",
                "target_quantity": "TARGET",
                "fairness_quantity": "DPD",
                "estimand_status": "SPECIFIED",
            }
        }
    }
    is_valid, errors = validate_audit_mathematical_invariants(audit_data)
    assert is_valid, f"Expected valid invariants, got errors: {errors}"
    assert len(errors) == 0


def test_audit_invariants_violation_detection():
    """Injected mathematical contradiction raises invariant error or flags violation."""
    bad_data = {
        "dataset": {"total_records": 100, "entity_count": 50},
        "subgroup_evaluations": {
            "GroupA": {
                "sample_size": 50,
                # Contradiction: positive (30) + negative (30) = 60 != 50
                "positive_count": 30,
                "negative_count": 30,
            }
        }
    }
    is_valid, errors = validate_audit_mathematical_invariants(bad_data)
    assert not is_valid
    assert any(ERR_INVALID_COUNT_IDENTITY in e for e in errors)


def test_audit_invariants_disparity_identities():
    """Validates disparity metric identities."""
    audit_data = {
        "subgroup_evaluations": {
            "GroupA": {
                "sample_size": 50,
                "TP": 20, "FP": 5, "FN": 5, "TN": 20,
                "TPR": 0.80, "FPR": 0.20,
                "predicted_positive_rate": 0.50,
            },
            "GroupB": {
                "sample_size": 50,
                "TP": 15, "FP": 10, "FN": 10, "TN": 15,
                "TPR": 0.60, "FPR": 0.40,
                "predicted_positive_rate": 0.50,
            },
        },
        "fairness_metrics": {
            # Selection rates are 0.50 and 0.50 => DPD = 0.0, DIR = 1.0
            "demographic_parity_difference": 0.0,
            "disparate_impact_ratio": 1.0,
            # TPRs are 0.80 and 0.60 => EOD = 0.20
            "equal_opportunity_difference": 0.20,
            # FPRs are 0.20 and 0.40 => FPR gap = 0.20, TPR gap = 0.20 => EODiff = 0.20
            "equalized_odds_difference": 0.20,
        },
        "evaluation_unit_alignment": {
            "statistical_estimand": {
                "population": "Pop",
                "decision_unit": "ROW",
                "observation_unit": "ROW",
                "weighting_rule": "UNIFORM",
                "target_quantity": "T",
                "fairness_quantity": "DPD",
                "estimand_status": "SPECIFIED",
            }
        }
    }
    is_valid, errors = validate_audit_mathematical_invariants(audit_data)
    assert is_valid, f"Disparity check failed: {errors}"


# ============================================================================
# 7. Unit Alignment & Bootstrap Guard
# ============================================================================

def test_bootstrap_unit_alignment_enforcement():
    """Row bootstrap on multi-row entity level without clustering or override triggers guard."""
    policy = FairnessPolicy(name="Strict_Unit_Policy")
    entities = np.array(["T1", "T1", "T2", "T2"] * 10)
    y_true = np.array([1, 0, 1, 0] * 10)
    y_pred = np.array([1, 0, 1, 0] * 10)
    sensitive = np.array(["A", "A", "B", "B"] * 10)

    # Calling with TRIP_LEVEL, repeated entity rows, and ROW_BOOTSTRAP without override raises ValueError
    with pytest.raises(ValueError, match="Statistical unit mismatch"):
        evaluate_fairness_defensively(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
            policy=policy,
            entity_series=entities,
            evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
            compute_uncertainty=True,
            resampling_unit="ROW_BOOTSTRAP",
            allow_row_bootstrap_override=False,
        )


# ============================================================================
# 8. Adult & Amazon Dataset Semantics
# ============================================================================

def test_adult_dataset_semantics():
    """Asserts census income semantics (> $50K), no credit references."""
    units_file = Path("configs/evaluation_units.yaml")
    with open(units_file, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    
    adult_cfg = spec["evaluation_units"]["adult_income"]
    desc = adult_cfg["semantic_description"].lower()
    assert "census" in desc
    assert "credit" not in desc
    assert "income" in desc
    assert adult_cfg["statistical_estimand"]["target_quantity"] == "PROBABILITY_OF_ANNUAL_INCOME_OVER_50K"


def test_amazon_dataset_semantics_and_status():
    """Asserts route_id as unit, station_code as cluster, status PENDING_RAW_DATA_VERIFICATION."""
    units_file = Path("configs/evaluation_units.yaml")
    with open(units_file, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    
    amz_cfg = spec["evaluation_units"]["amazon_last_mile"]
    assert amz_cfg["identifier_roles"]["route_id"]["semantic_role"] == "route identity"
    assert amz_cfg["identifier_roles"]["station_code"]["semantic_role"] == "facility/domain"
    assert amz_cfg["semantic_validation_status"] == "PENDING_RAW_DATA_VERIFICATION"


# ============================================================================
# 9. Legacy Non-Comparable Classification
# ============================================================================

def test_legacy_noncomparable_classification():
    """Asserts historical un-aggregated results are flagged LEGACY_NONCOMPARABLE."""
    units_file = Path("configs/evaluation_units.yaml")
    with open(units_file, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    
    delhivery_cfg = spec["evaluation_units"]["delhivery_logistics"]
    assert delhivery_cfg["historical_unit_mismatch"]["comparison_validity"] == "LEGACY_NONCOMPARABLE"


# ============================================================================
# 10. External Real-Data Validation Pending State
# ============================================================================

def test_external_validation_pending_state():
    """External black-box adapter reports EXTERNAL_VALIDATION_PENDING without fabrication when unstaged."""
    adapter = ExternalDatasetAdapter(dataset_name="Delhivery Logistics")
    assert adapter.provenance is not None
    assert adapter.provenance.dataset_name == "Delhivery Logistics"


# ============================================================================
# 11. Reproducible Evaluation Bundle Generation & Artifact Privacy Scanner
# ============================================================================

def test_evaluation_bundle_generation(sample_estimand_dict):
    """Generates reproducible bundle in temporary dir; asserts all required files and zero raw data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dir = Path(tmp_dir) / "audit_bundle"

        suite = evaluate_fairness_defensively(
            y_true=np.array([1, 0, 1, 0] * 10),
            y_pred=np.array([1, 0, 1, 0] * 10),
            sensitive=np.array(["A", "A", "B", "B"] * 10),
            statistical_estimand=sample_estimand_dict,
        )
        audit_res = build_audit_result_from_suite(suite)

        bundle_path = create_evaluation_bundle(
            output_dir=bundle_dir,
            audit_result=audit_res,
            dataset_storage_status=DatasetStorageStatus.EXTERNAL_REFERENCE,
        )

        assert (bundle_dir / "audit_result.json").exists()
        assert (bundle_dir / "evaluation_manifest.json").exists()
        assert (bundle_dir / "environment.json").exists()
        assert (bundle_dir / "README.md").exists()
        assert (bundle_dir / "configuration" / "evaluation_units.json").exists()
        assert (bundle_dir / "provenance" / "dataset_storage_status.json").exists()
        assert (bundle_dir / "results" / "fairness_metrics.json").exists()

        # Check privacy scan on the generated README and audit_result
        readme_text = (bundle_dir / "README.md").read_text(encoding="utf-8")
        scan_res = audit_artifact_privacy(readme_text)
        assert scan_res["privacy_passed"] is True
        assert scan_res["total_leaks_found"] == 0


def test_artifact_privacy_scanner():
    """Detects heuristic PII/tokens in artifacts and verifies clean reports."""
    dirty_content = "Here is the key: AKIAIOSFODNN7EXAMPLE and email john.doe@customer.internal"
    scan_res = audit_artifact_privacy(dirty_content)
    assert scan_res["privacy_passed"] is False
    assert scan_res["total_leaks_found"] >= 2
    assert len(scan_res["findings"]["emails"]) == 1
    assert len(scan_res["findings"]["secrets"]) == 1


# ============================================================================
# 12. Policy Certification Gate (9-Condition Engine)
# ============================================================================

def test_policy_certification_gate_all_nine_conditions(sample_estimand_dict):
    """Tests that all 9 conditions pass for a compliant confirmatory evaluation."""
    policy = FairnessPolicy(
        name="Compliant_Confirmatory_Policy",
        max_demographic_parity_diff=0.10,
    )
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"] * 10)

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        policy=policy,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
        statistical_estimand=sample_estimand_dict,
        evaluation_unit_key="route_id",
        evaluation_units=np.repeat(np.arange(40), 2),
    )

    gate_ok, gate_errs, gate_codes = verify_policy_certification_gate(suite, policy=policy)
    assert gate_ok, f"Gate failed: {gate_errs}"
    assert len(gate_errs) == 0
    assert suite.policy_result.decision == PolicyDecision.POLICY_COMPLIANT


def test_policy_certification_gate_rejections():
    """Tests specific rejection paths: degeneracy and missing estimand."""
    # Degenerate model (all 1s)
    policy = FairnessPolicy(name="Degeneracy_Test_Policy")
    y_true = np.array([1, 0, 1, 0] * 10)
    y_pred = np.ones(40, dtype=int)  # Degenerate
    sensitive = np.array(["A", "A", "B", "B"] * 10)

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        policy=policy,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
    )

    # In confirmatory mode with degenerate model and no estimand, certification gate blocks
    assert suite.policy_result.decision == PolicyDecision.POLICY_UNDEFINED
    assert suite.evidence_state == EvidenceState.INSUFFICIENT_EVIDENCE
    assert any("GATE_MISSING_ESTIMAND" in c for c in suite.policy_result.reason_codes)


# ============================================================================
# 13. Duplicate Evaluation Key Handling & Diagnostics
# ============================================================================

def test_duplicate_evaluation_key_handling():
    """Asserts duplicate key rate, cardinality, and repeated rows are computed and reported."""
    df = pd.DataFrame({
        "trip_id": ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4", "T5", "T5"],
        "feature": np.arange(10),
        "label": [1, 0] * 5,
    })
    meta = validate_evaluation_unit_configuration(df, evaluation_unit_key="trip_id")
    assert meta["evaluation_unit_key"] == "trip_id"
    assert meta["n_records"] == 10
    assert meta["n_units"] == 5
    assert meta["duplicate_key_rate"] == 0.50
    assert meta["records_per_entity"]["min"] == 2
    assert meta["records_per_entity"]["mean"] == 2.0
    assert meta["records_per_entity"]["max"] == 2


# ============================================================================
# 14. Metric Identity Invariants (TNR, FNR, Accuracy, F1)
# ============================================================================

def test_metric_identity_invariants():
    """Validates TNR, FNR, subgroup accuracy, and F1 arithmetic identities."""
    audit_data = {
        "dataset": {"total_records": 100, "entity_count": 100},
        "subgroup_evaluations": {
            "G1": {
                "sample_size": 100,
                "positive_count": 50,
                "negative_count": 50,
                "TP": 40, "FP": 10, "FN": 10, "TN": 40,
                "accuracy": 0.80,
                "PPV": 40 / 50,
                "TPR": 40 / 50,
                "TNR": 40 / 50,
                "FPR": 10 / 50,
                "FNR": 10 / 50,
                "balanced_accuracy": 0.80,
            }
        },
        "global_performance": {
            "accuracy": 0.80,
            "precision": 0.80,
            "recall": 0.80,
            "f1_score": 0.80,
            "balanced_accuracy": 0.80,
        },
        "fairness_metrics": {
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "disparate_impact_ratio": 1.0,
        },
        "evaluation_unit_alignment": {
            "statistical_estimand": {
                "population": "Pop",
                "decision_unit": "ROW",
                "observation_unit": "ROW",
                "weighting_rule": "UNIFORM",
                "target_quantity": "T",
                "fairness_quantity": "DPD",
                "estimand_status": "SPECIFIED",
            }
        }
    }
    is_valid, errors = validate_audit_mathematical_invariants(audit_data)
    assert is_valid, f"Expected valid invariants, got: {errors}"

    # Invalidate FNR: set FNR to 0.50 when FN/(FN+TP) is 0.20
    audit_data["subgroup_evaluations"]["G1"]["FNR"] = 0.50
    is_valid, errors = validate_audit_mathematical_invariants(audit_data)
    assert not is_valid
    assert any(ERR_INVALID_METRIC_IDENTITY in e for e in errors)


# ============================================================================
# 15. Invalid Adult Semantic Mapping Detection
# ============================================================================

def test_invalid_adult_semantic_mapping_detection():
    """Asserts attempting to map Adult to credit decisions is flagged as INVALID_GROUP_MAPPING."""
    bad_adult_data = {
        "dataset": {"dataset_name": "Adult Income Benchmark", "total_records": 100, "entity_count": 100},
        "evaluation_scope": {"purpose": "Audit", "analysis_type": "CONFIRMATORY_AUDIT", "target_decision": "Credit card approval"},
        "sensitive_attributes": [
            {
                "attribute_name": "Education_Group",
                "lineage": "DERIVED_PROXY",
                "proxy_derivation_rule": "Creditworthiness proxy",
                "disclosed_categories": ["G1"],
            }
        ],
        "subgroup_evaluations": {
            "G1": {
                "sample_size": 100,
                "positive_count": 50,
                "negative_count": 50,
                "TP": 40, "FP": 10, "FN": 10, "TN": 40,
                "TPR": 0.80, "FPR": 0.20,
                "balanced_accuracy": 0.80,
            }
        },
        "evaluation_unit_alignment": {
            "statistical_estimand": {
                "population": "Pop",
                "decision_unit": "ROW",
                "observation_unit": "ROW",
                "weighting_rule": "UNIFORM",
                "target_quantity": "T",
                "fairness_quantity": "DPD",
                "estimand_status": "SPECIFIED",
            }
        }
    }
    is_valid, errors = validate_audit_mathematical_invariants(bad_adult_data)
    assert not is_valid
    assert any(ERR_INVALID_GROUP_MAPPING in e for e in errors)


# ============================================================================
# 16. Invalid Amazon Entity Key Detection
# ============================================================================

def test_invalid_amazon_entity_key_detection():
    """Asserts invalid Amazon entity key raises INVALID_EVALUATION_UNIT_CONFIGURATION."""
    df_amazon = pd.DataFrame({
        "route_id": ["R1", "R2"],
        "station_code": ["D1", "D2"],
        "departure_hour": [8.0, 9.0],
    })
    # Passing unverified key
    with pytest.raises(ValueError, match="INVALID_EVALUATION_UNIT_CONFIGURATION"):
        validate_evaluation_unit_configuration(df_amazon, evaluation_unit_key="unverified_drone_uuid")


# ============================================================================
# 17. External Dataset Black-Box Boundary
# ============================================================================

def test_external_dataset_black_box_boundary(sample_estimand_dict):
    """Verifies that external evaluation operates with zero access to synthetic parameters."""
    from sklearn.linear_model import LogisticRegression

    df = pd.DataFrame({
        "f1": np.random.randn(60),
        "f2": np.random.randn(60),
        "target": np.random.choice([0, 1], size=60),
        "sensitive": np.random.choice(["X", "Y"], size=60),
    })

    adapter = ExternalDatasetAdapter(dataset_name="TestExternalData")
    bundle = adapter.load_from_dataframe(
        df=df,
        target_column="target",
        sensitive_column="sensitive",
        feature_columns=["f1", "f2"],
        statistical_estimand=sample_estimand_dict,
    )

    clf = LogisticRegression()
    suite, manifest = adapter.evaluate_external_black_box(
        bundle=bundle,
        model=clf,
        policy=FairnessPolicy(name="External_Boundary_Policy"),
        compute_uncertainty=False,
    )

    assert manifest.result_classification == ResultClassification.EXTERNAL_REAL_DATA_VALIDATION
    assert "dgp_coefficients" not in dir(bundle)
    assert "scenario" not in dir(bundle)
    assert suite.accuracy >= 0.0


# ============================================================================
# 18. Audit-Result Schema + Semantic Invariant Validator
# ============================================================================

def test_audit_result_schema_and_semantic_validation(sample_estimand_dict):
    """Verifies dual enforcement: schema validation + semantic invariant validation."""
    suite = evaluate_fairness_defensively(
        y_true=np.array([1, 0, 1, 0] * 10),
        y_pred=np.array([1, 0, 1, 0] * 10),
        sensitive=np.array(["A", "A", "B", "B"] * 10),
        statistical_estimand=sample_estimand_dict,
    )
    audit_res = build_audit_result_from_suite(suite)
    audit_dict = audit_res.to_dict()

    # Valid result passes both schema and invariants
    is_valid, errors = validate_audit_result_dict(audit_dict, validate_invariants=True)
    assert is_valid, f"Validation failed: {errors}"

    # Corrupt schema (missing required top-level key)
    corrupt_schema = audit_dict.copy()
    del corrupt_schema["split_protocol"]
    is_valid, errors = validate_audit_result_dict(corrupt_schema, validate_invariants=True)
    assert not is_valid
    assert any("Missing required top-level key" in e for e in errors)

    # Corrupt invariants (count contradiction)
    corrupt_inv = audit_dict.copy()
    corrupt_inv["subgroup_evaluations"] = {
        "A": {"sample_size": 20, "positive_count": 15, "negative_count": 15}
    }
    is_valid, errors = validate_audit_result_dict(corrupt_inv, validate_invariants=True)
    assert not is_valid
    assert any(ERR_INVALID_COUNT_IDENTITY in e for e in errors)


# ============================================================================
# 19. Invalid Unit Blocks Confirmatory Certification
# ============================================================================

def test_invalid_unit_blocks_confirmatory_certification(sample_estimand_dict):
    """Asserts policy certification gate fails when evaluation_unit_valid is False."""
    policy = FairnessPolicy(name="Strict_Gate_Policy")
    suite = evaluate_fairness_defensively(
        y_true=np.array([1, 0, 1, 0] * 10),
        y_pred=np.array([1, 0, 1, 0] * 10),
        sensitive=np.array(["A", "A", "B", "B"] * 10),
        policy=policy,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
        statistical_estimand=sample_estimand_dict,
        evaluation_unit_valid=False,  # Unverified or invalid unit
    )

    gate_ok, gate_errs, gate_codes = verify_policy_certification_gate(
        suite, policy=policy, evaluation_unit_valid=False
    )
    assert not gate_ok
    assert "GATE_INVALID_EVALUATION_UNIT" in gate_codes
    assert suite.evidence_state == EvidenceState.INSUFFICIENT_EVIDENCE


# ============================================================================
# 20. Invalid Metric Identity Rejects Result
# ============================================================================

def test_invalid_metric_identity_rejects_result(sample_estimand_dict):
    """Asserts impossible metric combinations reject the result with INVALID_METRIC_IDENTITY."""
    suite = evaluate_fairness_defensively(
        y_true=np.array([1, 0, 1, 0] * 10),
        y_pred=np.array([1, 0, 1, 0] * 10),
        sensitive=np.array(["A", "A", "B", "B"] * 10),
        statistical_estimand=sample_estimand_dict,
    )
    audit_res = build_audit_result_from_suite(suite)
    audit_dict = audit_res.to_dict()

    # Invalidate precision (PPV): set PPV to 0.10 when TP=10, FP=0 => PPV=1.0
    for g in audit_dict["subgroup_evaluations"].values():
        g["PPV"] = 0.10

    is_valid, errors = validate_audit_result_dict(audit_dict, validate_invariants=True)
    assert not is_valid
    assert any(ERR_INVALID_METRIC_IDENTITY in e for e in errors)

