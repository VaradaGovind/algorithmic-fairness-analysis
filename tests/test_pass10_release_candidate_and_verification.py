# ============================================================================
# Pass 10 Release-Candidate & Independent Verification Test Suite
# ============================================================================

import json
import shutil
import warnings
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.fairness_evaluator import (
    AnalysisType,
    AuditExecutionStatus,
    AuditTaxonomyState,
    EvaluationStatus,
    EvaluationUnitLevel,
    EvidenceStatus,
    FairnessPolicy,
    HumanReviewBoundary,
    HumanReviewReasonCode,
    HUMAN_REVIEW_REASON_CODE_ALIASES,
    ModelDegeneracyStatus,
    ModelStatus,
    PolicyDecision,
    PolicyStatus,
    canonicalize_human_review_reason_code,
    determine_human_review_boundary,
    evaluate_fairness_defensively,
    map_to_audit_taxonomy,
    validate_evaluation_unit_configuration,
    verify_policy_decision_gate,
    verify_policy_certification_gate,
)
from src.audit_result_schema import (
    AuditResult,
    SensitiveAttributeLineage,
    build_audit_result_from_suite,
    compute_canonical_audit_fingerprint,
    validate_audit_result_dict,
)
from src.audit_bundle import (
    BUNDLE_CONFIGURATION_MISMATCH,
    BUNDLE_CROSS_ARTIFACT_MISMATCH,
    BUNDLE_FINGERPRINT_MISMATCH,
    BUNDLE_INVARIANTS_INVALID,
    BUNDLE_MANIFEST_MISMATCH,
    BUNDLE_MISSING_FILE,
    BUNDLE_PRIVACY_VIOLATION,
    BUNDLE_RAW_DATA_DETECTED,
    BUNDLE_SCHEMA_INVALID,
    BundleVerificationResult,
    DatasetStorageStatus,
    compute_independent_fingerprint,
    create_evaluation_bundle,
    verify_audit_bundle,
)
from src.external_dataset_adapter import ExternalDatasetAdapter
from src.audit_invariants import (
    audit_bundle_privacy,
    run_publication_release_audit,
    validate_audit_mathematical_invariants,
)
from src.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_EXECUTION_BLOCKED,
    EXIT_POLICY_FAILED,
    EXIT_SUCCESS,
    EXIT_VERIFICATION_FAILED,
    main,
)


@pytest.fixture
def clean_audit_fixture(tmp_path):
    """Generates a valid evaluated suite and audit result."""
    yt = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    yp = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 5)
    s = np.array(["GroupA"] * 25 + ["GroupB"] * 25)

    policy = FairnessPolicy(
        name="Operational_Policy_v1",
        min_balanced_accuracy=0.50,
        max_equalized_odds_diff=0.20,
        max_demographic_parity_diff=0.20,
        policy_version="1.0.0",
    )

    estimand = {
        "population": "Operational test population",
        "decision_unit": "ROW_LEVEL",
        "observation_unit": "ROW_LEVEL",
        "weighting_rule": "UNIFORM",
        "target_quantity": "PROBABILITY_OF_POSITIVE_OUTCOME",
        "fairness_quantity": "EQUALIZED_ODDS_AND_DEMOGRAPHIC_PARITY",
        "estimand_status": "SPECIFIED",
    }

    suite = evaluate_fairness_defensively(
        y_true=yt,
        y_pred=yp,
        sensitive=s,
        policy=policy,
        statistical_estimand=estimand,
        evaluation_unit_level=EvaluationUnitLevel.ROW_LEVEL,
        evaluation_unit_key="row_id",
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
        decision_gate_required=True,
    )

    audit_res = build_audit_result_from_suite(
        suite=suite,
        purpose="Release Candidate Verification Audit",
        dataset_name="synthetic_fixture",
        sensitive_attribute_name="sensitive_group",
        sensitive_lineage=SensitiveAttributeLineage.OBSERVED,
    )

    bundle_dir = tmp_path / "valid_bundle"
    create_evaluation_bundle(
        output_dir=bundle_dir,
        audit_result=audit_res,
        dataset_storage_status=DatasetStorageStatus.CUSTOMER_HELD,
    )
    return audit_res, bundle_dir


# ---------------------------------------------------------------------------
# Test 1: Clean Bundle Verification
# ---------------------------------------------------------------------------
def test_bundle_verification_valid_bundle(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    result = verify_audit_bundle(bundle_dir)

    assert result.is_valid is True
    assert len(result.failure_codes) == 0
    assert result.checks_performed["required_files"] is True
    assert result.checks_performed["raw_data_cleanliness"] is True
    assert result.checks_performed["schema_conformance"] is True
    assert result.checks_performed["semantic_invariants"] is True
    assert result.checks_performed["cryptographic_fingerprint"] is True
    assert result.checks_performed["manifest_alignment"] is True
    assert "THREAT MODEL NOTICE" in result.threat_model_disclaimer


# ---------------------------------------------------------------------------
# Test 2: Tampered Metric Invalidates Fingerprint
# ---------------------------------------------------------------------------
def test_bundle_verification_tampered_metric_fails_fingerprint(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    audit_file = bundle_dir / "audit_result.json"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tamper with accuracy without updating fingerprint
    data["global_performance"]["accuracy"] = 0.9999
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    result = verify_audit_bundle(bundle_dir)
    assert result.is_valid is False
    assert BUNDLE_FINGERPRINT_MISMATCH in result.failure_codes


# ---------------------------------------------------------------------------
# Test 3: Updated Fingerprint Fails Manifest & Cross-Artifact Checks
# ---------------------------------------------------------------------------
def test_bundle_verification_updated_fingerprint_fails_manifest(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    audit_file = bundle_dir / "audit_result.json"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tamper with metric AND recompute fingerprint in audit_result.json, leaving manifest stale
    data["global_performance"]["accuracy"] = 0.9999
    new_fp = compute_canonical_audit_fingerprint(data)
    data["audit_fingerprint"] = new_fp
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    result = verify_audit_bundle(bundle_dir)
    assert result.is_valid is False
    # Must catch manifest mismatch and cross-artifact discrepancy
    assert (
        BUNDLE_MANIFEST_MISMATCH in result.failure_codes
        or BUNDLE_CROSS_ARTIFACT_MISMATCH in result.failure_codes
    )


# ---------------------------------------------------------------------------
# Test 4: Forged Consistency Violating Semantic Invariant Fails
# ---------------------------------------------------------------------------
def test_bundle_verification_forged_consistency_violating_semantic_invariant(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    audit_file = bundle_dir / "audit_result.json"
    man_file = bundle_dir / "evaluation_manifest.json"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Forge impossible TPR > 1.0
    for g in data.get("subgroup_evaluations", {}).values():
        g["TPR"] = 1.45

    new_fp = compute_canonical_audit_fingerprint(data)
    data["audit_fingerprint"] = new_fp
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Update manifest to agree with forged fingerprint
    with open(man_file, "r", encoding="utf-8") as f:
        man = json.load(f)
    man["fingerprint"] = new_fp
    with open(man_file, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)

    result = verify_audit_bundle(bundle_dir)
    assert result.is_valid is False
    assert BUNDLE_INVARIANTS_INVALID in result.failure_codes


# ---------------------------------------------------------------------------
# Test 5: Smuggled Raw CSV Fails Verification
# ---------------------------------------------------------------------------
def test_bundle_verification_smuggled_raw_csv_fails(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    csv_file = bundle_dir / "results" / "raw_customer_dump.csv"
    csv_file.write_text("user_id,revenue,zip_code\n101,450.0,90210\n102,120.0,10001\n", encoding="utf-8")

    result = verify_audit_bundle(bundle_dir)
    assert result.is_valid is False
    assert BUNDLE_RAW_DATA_DETECTED in result.failure_codes


# ---------------------------------------------------------------------------
# Test 6: Smuggled Raw Parquet Fails Verification
# ---------------------------------------------------------------------------
def test_bundle_verification_smuggled_raw_parquet_fails(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    parquet_file = bundle_dir / "provenance" / "transactions.parquet"
    parquet_file.write_bytes(b"PAR1_MOCK_PARQUET_DATA")

    result = verify_audit_bundle(bundle_dir)
    assert result.is_valid is False
    assert BUNDLE_RAW_DATA_DETECTED in result.failure_codes


# ---------------------------------------------------------------------------
# Test 7: Missing Required File Fails Verification
# ---------------------------------------------------------------------------
def test_bundle_verification_missing_file_fails(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    fc_file = bundle_dir / "configuration" / "feature_contract.json"
    if fc_file.is_file():
        fc_file.unlink()

    result = verify_audit_bundle(bundle_dir)
    assert result.is_valid is False
    assert BUNDLE_MISSING_FILE in result.failure_codes
    assert "configuration/feature_contract.json" in result.details["missing_files"]


# ---------------------------------------------------------------------------
# Test 8: Invalid Draft-07 Schema Fails Verification
# ---------------------------------------------------------------------------
def test_bundle_verification_invalid_schema_fails(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    audit_file = bundle_dir / "audit_result.json"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Invalidate schema_version semver
    data["schema_version"] = "INVALID_NOT_SEMVER"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    result = verify_audit_bundle(bundle_dir)
    assert result.is_valid is False
    assert BUNDLE_SCHEMA_INVALID in result.failure_codes


# ---------------------------------------------------------------------------
# Test 9: Status Taxonomy Precedence - Blocked Execution
# ---------------------------------------------------------------------------
def test_taxonomy_precedence_blocked_execution():
    tax = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.BLOCKED,
        details={"block_reason": "Missing ground truth labels"},
    )
    assert tax.audit_execution_status == AuditExecutionStatus.BLOCKED.value
    assert tax.policy_status == PolicyStatus.NOT_EVALUABLE.value
    assert tax.precedence_rule_applied == "PRECEDENCE_EXECUTION_BLOCKED"


# ---------------------------------------------------------------------------
# Test 10: Status Taxonomy Precedence - Invalid Model (Leakage)
# ---------------------------------------------------------------------------
def test_taxonomy_precedence_invalid_model_blocks_confirmatory():
    tax = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        leakage_findings_count=3,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS.value,
    )
    assert tax.model_status == ModelStatus.INVALID.value
    assert tax.policy_status == PolicyStatus.NOT_EVALUABLE.value
    assert tax.precedence_rule_applied == "PRECEDENCE_MODEL_INVALID"


# ---------------------------------------------------------------------------
# Test 11: Status Taxonomy Precedence - Degenerate Model
# ---------------------------------------------------------------------------
def test_taxonomy_precedence_degenerate_model_fails_and_insufficient():
    tax = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        is_degenerate=True,
    )
    assert tax.model_status == ModelStatus.DEGENERATE.value
    assert tax.policy_status == PolicyStatus.FAIL.value
    assert tax.evidence_status == EvidenceStatus.INSUFFICIENT.value
    assert tax.precedence_rule_applied == "PRECEDENCE_MODEL_DEGENERATE"


# ---------------------------------------------------------------------------
# Test 12: Status Taxonomy Precedence - Insufficient Evidence
# ---------------------------------------------------------------------------
def test_taxonomy_precedence_insufficient_evidence_confirmatory():
    tax = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        evidence_status=EvidenceStatus.INSUFFICIENT,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS.value,
    )
    assert tax.evidence_status == EvidenceStatus.INSUFFICIENT.value
    assert tax.policy_status == PolicyStatus.NOT_EVALUABLE.value
    assert tax.precedence_rule_applied == "PRECEDENCE_EVIDENCE_INSUFFICIENT"


# ---------------------------------------------------------------------------
# Test 13: Decoupled Human Review Boundary
# ---------------------------------------------------------------------------
def test_human_review_boundary_decoupling():
    tax_failing = AuditTaxonomyState(
        audit_execution_status=AuditExecutionStatus.COMPLETE.value,
        evidence_status=EvidenceStatus.SUFFICIENT.value,
        model_status=ModelStatus.ACCEPTABLE.value,
        policy_status=PolicyStatus.FAIL.value,
    )
    hrb = determine_human_review_boundary(
        taxonomy_state=tax_failing,
        sensitive_lineage="DERIVED_PROXY",
        estimand_specified=True,
    )
    assert hrb.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value in hrb.reason_codes
    assert HumanReviewReasonCode.REVIEW_DERIVED_SENSITIVE_ATTRIBUTE.value in hrb.reason_codes


# ---------------------------------------------------------------------------
# Test 14: Policy Decision Gate Canonical & Deprecated Certification Gate
# ---------------------------------------------------------------------------
def test_policy_decision_gate_canonical_and_deprecated_alias(clean_audit_fixture):
    audit_res, _ = clean_audit_fixture

    policy = FairnessPolicy(
        name="Gate_Test_Policy",
        policy_version="1.0.0",
    )

    class MockSuite:
        statistical_estimand = {"estimand_status": "SPECIFIED"}
        subgroup_metrics = {}
        uncertainty_intervals = {}
        policy_result = type("MockPR", (), {"evidence_trace": ["rule_1_satisfied"]})()

    suite = MockSuite()

    # Canonical decision gate call
    ok, errs, codes = verify_policy_decision_gate(
        suite=suite,
        policy=policy,
        evaluation_unit_valid=True,
        split_valid=True,
        leakage_findings_count=0,
        invariants_valid=True,
    )
    assert ok is True
    assert len(errs) == 0

    # Deprecated alias call emits DeprecationWarning
    with pytest.deprecated_call():
        alias_ok, alias_errs, alias_codes = verify_policy_certification_gate(
            suite=suite,
            policy=policy,
            evaluation_unit_valid=True,
            split_valid=True,
            leakage_findings_count=0,
            invariants_valid=True,
        )
    assert alias_ok == ok


# ---------------------------------------------------------------------------
# Test 15: CLI Deterministic Exit Codes
# ---------------------------------------------------------------------------
def test_cli_deterministic_exit_codes(tmp_path):
    # 1. Config error on unknown command
    exit_code = main(["unknown_cmd"])
    assert exit_code == EXIT_CONFIG_ERROR

    # 2. Ingest synthetic success -> 0
    exit_code = main(["ingest", "--synthetic", "--samples", "50"])
    assert exit_code == EXIT_SUCCESS

    # 3. Missing file for external ingest -> 4 (config error)
    exit_code = main(["ingest", "--file", str(tmp_path / "nonexistent.csv")])
    assert exit_code == EXIT_CONFIG_ERROR

    # 4. Verification of invalid bundle directory -> 4 (missing directory)
    exit_code = main(["verify", str(tmp_path / "nonexistent_dir")])
    assert exit_code == EXIT_CONFIG_ERROR


# ---------------------------------------------------------------------------
# Test 16: Publication Release Audit
# ---------------------------------------------------------------------------
def test_publication_release_audit():
    audit_out = run_publication_release_audit()
    # Ensure no raw operational customer datasets, no secrets, no local user paths, and valid license
    assert audit_out["release_audit_passed"] is True, f"Release audit failed: {audit_out['findings']}"
    assert audit_out["total_issues_found"] == 0
    assert audit_out["status_recommendation"] == (
        "Release-candidate engineering verification complete. Repository publication checks passed. "
        "External real-data validation has not been executed and remains pending authorized operational-data staging. "
        "The system must not be interpreted as providing customer-specific, regulatory, or production suitability conclusions "
        "without additional human and empirical validation."
    )


# ---------------------------------------------------------------------------
# Test 17: Human-Review Reason Codes & Aliases Exhaustive Test
# ---------------------------------------------------------------------------
def test_human_review_reason_codes_exhaustive_and_aliases():
    # 1. Test every individual documented reason condition
    # a. Derived sensitive attribute
    h1 = determine_human_review_boundary(sensitive_lineage="DERIVED_PROXY")
    assert h1.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_DERIVED_SENSITIVE_ATTRIBUTE.value in h1.reason_codes

    # b. Regulatory mapping
    h2 = determine_human_review_boundary(policy_type="REGULATORY_MAPPING")
    assert h2.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_REGULATORY_MAPPING.value in h2.reason_codes

    # c. Low support
    mock_suite = type("MockSuite", (), {
        "subgroup_metrics": {"Group_Minority": {"support_status": EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value}},
        "statistical_estimand": {"estimand_status": "SPECIFIED"},
    })()
    h3 = determine_human_review_boundary(suite=mock_suite)
    assert h3.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_LOW_SUPPORT.value in h3.reason_codes

    # d. Ambiguous estimand
    h4 = determine_human_review_boundary(estimand_specified=False)
    assert h4.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_AMBIGUOUS_ESTIMAND.value in h4.reason_codes

    # e. Feature timing
    h5 = determine_human_review_boundary(has_timing_features=True)
    assert h5.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_FEATURE_TIMING.value in h5.reason_codes

    # f. Unit mismatch
    h6 = determine_human_review_boundary(unit_mismatch=True)
    assert h6.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_UNIT_MISMATCH.value in h6.reason_codes

    # g. Material policy violation
    tax_fail = AuditTaxonomyState(policy_status=PolicyStatus.FAIL.value)
    h7 = determine_human_review_boundary(taxonomy_state=tax_fail)
    assert h7.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value in h7.reason_codes

    # h. Inconclusive evidence
    tax_no_eval = AuditTaxonomyState(policy_status=PolicyStatus.NOT_EVALUABLE.value, evidence_status=EvidenceStatus.INSUFFICIENT.value)
    h8 = determine_human_review_boundary(taxonomy_state=tax_no_eval)
    assert h8.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_INCONCLUSIVE_EVIDENCE.value in h8.reason_codes

    # i. Policy ambiguity
    h9 = determine_human_review_boundary(policy_ambiguity=True)
    assert h9.human_review_required is True
    assert HumanReviewReasonCode.REVIEW_POLICY_AMBIGUITY.value in h9.reason_codes

    # 2. Test alias mappings
    aliases = [
        ("MARGINAL_DISPARITY", HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value),
        ("SPARSE_SUBGROUP", HumanReviewReasonCode.REVIEW_LOW_SUPPORT.value),
        ("SHIFT_DETECTED", HumanReviewReasonCode.REVIEW_INCONCLUSIVE_EVIDENCE.value),
        ("POLICY_AMBIGUITY", HumanReviewReasonCode.REVIEW_POLICY_AMBIGUITY.value),
        ("HIGH_STAKES_DECISION", HumanReviewReasonCode.REVIEW_REGULATORY_MAPPING.value),
        ("UNCALIBRATED_PROBABILITIES", HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value),
    ]
    for raw_alias, canonical in aliases:
        assert canonicalize_human_review_reason_code(raw_alias) == canonical
        h_alias = determine_human_review_boundary(additional_reason_codes=[raw_alias])
        assert h_alias.human_review_required is True
        assert canonical in h_alias.reason_codes


# ---------------------------------------------------------------------------
# Test 18: Taxonomy Precedence Invariants Exhaustive Enforcement
# ---------------------------------------------------------------------------
def test_taxonomy_precedence_invariants_exhaustive():
    # Invariant 1: BLOCKED cannot become POLICY_STATUS=PASS even if requested
    tax_blocked = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.BLOCKED,
        policy_status=PolicyStatus.PASS,
        evidence_status=EvidenceStatus.SUFFICIENT,
    )
    assert tax_blocked.audit_execution_status == AuditExecutionStatus.BLOCKED.value
    assert tax_blocked.policy_status == PolicyStatus.NOT_EVALUABLE.value
    assert tax_blocked.evidence_status == EvidenceStatus.INSUFFICIENT.value
    assert tax_blocked.policy_status != PolicyStatus.PASS.value

    # Invariant 2: INVALID model cannot produce a confirmatory policy conclusion
    tax_invalid = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        model_status=ModelStatus.INVALID,
        policy_status=PolicyStatus.PASS,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS.value,
    )
    assert tax_invalid.model_status == ModelStatus.INVALID.value
    assert tax_invalid.policy_status == PolicyStatus.NOT_EVALUABLE.value
    assert tax_invalid.evidence_status == EvidenceStatus.INSUFFICIENT.value

    # Invariant 3: INSUFFICIENT evidence cannot produce an evidence-sufficient conclusion
    tax_insuff = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        evidence_status=EvidenceStatus.INSUFFICIENT,
        policy_status=PolicyStatus.PASS,
        analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS.value,
    )
    assert tax_insuff.evidence_status == EvidenceStatus.INSUFFICIENT.value
    assert tax_insuff.policy_status == PolicyStatus.NOT_EVALUABLE.value

    # Invariant 4: DEGENERATE remains distinct from INVALID
    tax_degen = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        model_status=ModelStatus.DEGENERATE,
    )
    assert tax_degen.model_status == ModelStatus.DEGENERATE.value
    assert tax_degen.policy_status == PolicyStatus.FAIL.value  # Degenerate models strictly fail
    assert tax_degen.model_status != tax_invalid.model_status
    assert tax_degen.policy_status != tax_invalid.policy_status

    # Invariant 5: NOT_EVALUABLE remains distinct from FAIL
    assert PolicyStatus.NOT_EVALUABLE.value != PolicyStatus.FAIL.value

    # Invariant 6: Exploratory execution remains distinguishable from confirmatory
    tax_exploratory = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        analysis_type="EXPLORATORY",
        policy_status=PolicyStatus.PASS,
    )
    assert tax_exploratory.details.get("is_exploratory") is True
    assert tax_exploratory.details.get("analysis_type") == "EXPLORATORY"


# ---------------------------------------------------------------------------
# Test 19: Machine-Readable EXTERNAL_VALIDATION_PENDING Status
# ---------------------------------------------------------------------------
def test_external_validation_pending_machine_readable_status():
    adapter = ExternalDatasetAdapter(dataset_name="delhivery_logistics")
    assert adapter.status == "EXTERNAL_VALIDATION_PENDING"
    st = adapter.get_external_validation_status()
    assert st["status"] == "EXTERNAL_VALIDATION_PENDING"
    assert st["operational_data_staged"] is False
    assert st["execution_classification"] == "EXTERNAL_REAL_DATA_VALIDATION"
    assert "pending" in st["notes"].lower()


# ---------------------------------------------------------------------------
# Test 20: Evaluation-Unit Evidence Gating Strictness
# ---------------------------------------------------------------------------
def test_evaluation_unit_evidence_gating_strictness():
    # 1. Empty dataframe must fail and not emit VERIFIED_FROM_DATA
    diag_empty = validate_evaluation_unit_configuration(
        df=pd.DataFrame(),
        evaluation_unit_key="trip_uuid",
        raise_on_error=False,
    )
    assert diag_empty.get("semantic_validation_status") == "PENDING_RAW_DATA_VERIFICATION"

    # 2. Non-dataframe input must not emit VERIFIED_FROM_DATA
    diag_nondf = validate_evaluation_unit_configuration(
        df=None,
        evaluation_unit_key="trip_uuid",
        raise_on_error=False,
    )
    assert diag_nondf.get("semantic_validation_status") == "PENDING_RAW_DATA_VERIFICATION"

    # 3. Missing column must not emit VERIFIED_FROM_DATA
    df_missing = pd.DataFrame({"other_col": [1, 2, 3]})
    diag_missing = validate_evaluation_unit_configuration(
        df=df_missing,
        evaluation_unit_key="trip_uuid",
        raise_on_error=False,
    )
    assert diag_missing.get("semantic_validation_status") == "PENDING_RAW_DATA_VERIFICATION"

    # 4. Valid inspected non-empty dataframe emits VERIFIED_FROM_DATA
    df_valid = pd.DataFrame({"trip_uuid": ["T1", "T2", "T3"]})
    diag_valid = validate_evaluation_unit_configuration(
        df=df_valid,
        evaluation_unit_key="trip_uuid",
        raise_on_error=True,
    )
    assert diag_valid.get("semantic_validation_status") == "VERIFIED_FROM_DATA"
    assert diag_valid.get("unit_cardinality") == 3


# ---------------------------------------------------------------------------
# Test 21: Adversarial Multi-Artifact Bundle Tampering Test
# ---------------------------------------------------------------------------
def test_bundle_verification_adversarial_multi_artifact_tamper(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    audit_file = bundle_dir / "audit_result.json"
    res_file = bundle_dir / "results" / "fairness_metrics.json"
    man_file = bundle_dir / "evaluation_manifest.json"

    # Adversary alters both fairness_metrics.json AND audit_result.json to forge a better metric
    with open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    audit_data["fairness_metrics"]["demographic_parity_difference"] = 0.0123

    # Update results/fairness_metrics.json to match forged audit_result.json
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(audit_data["fairness_metrics"], f, indent=2)

    # Recompute fingerprint in audit_result.json
    forged_fp = compute_canonical_audit_fingerprint(audit_data)
    audit_data["audit_fingerprint"] = forged_fp
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # Verification must catch manifest mismatch because evaluation_manifest.json still has old fingerprint
    res = verify_audit_bundle(bundle_dir)
    assert res.is_valid is False
    assert BUNDLE_MANIFEST_MISMATCH in res.failure_codes


# ---------------------------------------------------------------------------
# Test 22: Bundle Privacy Violation Detection
# ---------------------------------------------------------------------------
def test_bundle_verification_privacy_violation_caught(clean_audit_fixture):
    audit_res, bundle_dir = clean_audit_fixture
    # Smuggle an AWS access key into the bundle's warnings file
    warn_file = bundle_dir / "warnings" / "audit_warnings.json"
    warn_data = {"warnings": ["Connection reset with key AKIA1111222233334444"]}
    with open(warn_file, "w", encoding="utf-8") as f:
        json.dump(warn_data, f, indent=2)

    res = verify_audit_bundle(bundle_dir)
    assert res.is_valid is False
    assert BUNDLE_PRIVACY_VIOLATION in res.failure_codes


# ---------------------------------------------------------------------------
# Test 23: Complete CLI Exit Codes & JSON Output Matrix
# ---------------------------------------------------------------------------
def test_cli_deterministic_exit_codes_and_json_output(tmp_path, capsys):
    # 1. Exit 0: Ingest synthetic with --json
    code_ingest = main(["ingest", "--synthetic", "--samples", "50", "--output", str(tmp_path / "syn.csv"), "--json"])
    assert code_ingest == EXIT_SUCCESS
    captured = capsys.readouterr()
    json_out = json.loads(captured.out)
    assert json_out["status"] == "SUCCESS"
    assert json_out["records"] == 50

    # 2. Exit 0: Audit synthetic with --json
    audit_out_path = tmp_path / "cli_audit_result.json"
    code_audit = main([
        "audit",
        "--synthetic",
        "--samples", "100",
        "--output", str(audit_out_path),
        "--json",
    ])
    assert code_audit in (EXIT_SUCCESS, EXIT_POLICY_FAILED)
    captured = capsys.readouterr()
    audit_json = json.loads(captured.out)
    assert "audit_execution_status" in audit_json
    assert audit_json["audit_execution_status"] == "COMPLETE"

    # 3. Exit 0: Bundle with --json
    bundle_out_path = tmp_path / "cli_bundle"
    code_bundle = main([
        "bundle",
        "--audit-result", str(audit_out_path),
        "--output-dir", str(bundle_out_path),
        "--json",
    ])
    assert code_bundle == EXIT_SUCCESS
    captured = capsys.readouterr()
    bundle_json = json.loads(captured.out)
    assert bundle_json["status"] == "SUCCESS"

    # 4. Exit 0: Verify with --json
    code_verify = main(["verify", str(bundle_out_path), "--json"])
    assert code_verify == EXIT_SUCCESS
    captured = capsys.readouterr()
    verify_json = json.loads(captured.out)
    assert verify_json["is_valid"] is True
    assert "checks_performed" in verify_json

    # 5. Exit 1: Policy Failure
    code_fail = main([
        "audit",
        "--synthetic",
        "--samples", "100",
        "--min-bal-acc", "0.99",
        "--max-eo-diff", "0.0001",
    ])
    assert code_fail == EXIT_POLICY_FAILED

    # 6. Exit 2: Execution Blocked (bad data columns)
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("irrelevant_col1,irrelevant_col2\n1,2\n", encoding="utf-8")
    code_blocked = main(["audit", "--file", str(bad_csv), "--target-col", "missing_target"])
    assert code_blocked == EXIT_EXECUTION_BLOCKED

    # 7. Exit 3: Verification Failed (tampered bundle)
    res_file = bundle_out_path / "results" / "fairness_metrics.json"
    if res_file.is_file():
        res_file.write_text('{"tampered": true}', encoding="utf-8")
    code_tampered = main(["verify", str(bundle_out_path)])
    assert code_tampered == EXIT_VERIFICATION_FAILED

    # 8. Exit 4: Configuration Error (missing file)
    code_cfg = main(["audit", "--file", str(tmp_path / "nonexistent.csv")])
    assert code_cfg == EXIT_CONFIG_ERROR


# ---------------------------------------------------------------------------
# Test 24: Reason-Code Alias Normalization & Full Reachability
# ---------------------------------------------------------------------------
def test_human_review_reason_code_normalization_and_reachability():
    """
    Proves that:
    1. Every alias in HUMAN_REVIEW_REASON_CODE_ALIASES maps to exactly one valid canonical HumanReviewReasonCode.
    2. Every canonical HumanReviewReasonCode is reachable from actual review-boundary logic.
    """
    canonical_codes = {c.value for c in HumanReviewReasonCode}

    # 1. Alias normalization verification
    for alias, canonical in HUMAN_REVIEW_REASON_CODE_ALIASES.items():
        assert isinstance(alias, str) and len(alias) > 0
        assert canonical in canonical_codes, f"Alias '{alias}' maps to invalid code '{canonical}'"
        norm = canonicalize_human_review_reason_code(alias)
        assert norm == canonical

    # 2. Canonical reachability verification
    # Prove that every single canonical code can be triggered through domain inputs
    reached = set()

    # REVIEW_MATERIAL_POLICY_VIOLATION
    r1 = determine_human_review_boundary(taxonomy_state=AuditTaxonomyState(policy_status=PolicyStatus.FAIL.value))
    reached.update(r1.reason_codes)

    # REVIEW_INCONCLUSIVE_EVIDENCE
    r2 = determine_human_review_boundary(taxonomy_state=AuditTaxonomyState(policy_status=PolicyStatus.NOT_EVALUABLE.value))
    reached.update(r2.reason_codes)

    # REVIEW_DERIVED_SENSITIVE_ATTRIBUTE
    r3 = determine_human_review_boundary(sensitive_lineage="DERIVED_PROXY")
    reached.update(r3.reason_codes)

    # REVIEW_REGULATORY_MAPPING
    r4 = determine_human_review_boundary(policy_type="REGULATORY_MAPPING")
    reached.update(r4.reason_codes)

    # REVIEW_LOW_SUPPORT
    mock_suite = type("MockSuite", (), {
        "subgroup_metrics": {"grp": {"support_status": EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value}},
        "statistical_estimand": {"estimand_status": "SPECIFIED"},
    })()
    r5 = determine_human_review_boundary(suite=mock_suite)
    reached.update(r5.reason_codes)

    # REVIEW_AMBIGUOUS_ESTIMAND
    r6 = determine_human_review_boundary(estimand_specified=False)
    reached.update(r6.reason_codes)

    # REVIEW_FEATURE_TIMING
    r7 = determine_human_review_boundary(has_timing_features=True)
    reached.update(r7.reason_codes)

    # REVIEW_UNIT_MISMATCH
    r8 = determine_human_review_boundary(unit_mismatch=True)
    reached.update(r8.reason_codes)

    # REVIEW_POLICY_AMBIGUITY
    r9 = determine_human_review_boundary(policy_ambiguity=True)
    reached.update(r9.reason_codes)

    # Assert that all canonical codes are reached (zero unreachable codes)
    missing = canonical_codes - reached
    assert len(missing) == 0, f"Unreachable canonical reason codes: {missing}"
    assert reached == canonical_codes


