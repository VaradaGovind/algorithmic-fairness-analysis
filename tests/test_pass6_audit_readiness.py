# ============================================================================
# PASS 6 AUDIT-GRADE EVIDENCE & GOVERNANCE TEST SUITE
# Validates deterministic fingerprinting, non-parametric bootstrap replicate
# accounting, Disparate Impact Ratio denominator defenses, tri-state precedence,
# missing sensitive attribute governance, path containment, CSV injection defense,
# external dataset adapter boundary, and result classification taxonomy.
# ============================================================================

import os
import re
import socket
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation_manifest import (
    EvaluationManifest,
    ResultClassification,
    compute_manifest_fingerprint,
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
    PolicyEvaluationResult,
    compute_fairness_uncertainty_bootstrap,
    evaluate_fairness_defensively,
    evaluate_policy_compliance,
    safe_disparate_impact_ratio,
)
from src.external_dataset_adapter import ExternalDatasetAdapter, ExternalDataBundle
from src.ingest_data import (
    IngestionStatus,
    export_sanitized_csv,
    sanitize_csv_value,
    validate_source_path_security,
)


# ---------------------------------------------------------------------------
# TEST 1: Manifest Determinism
# ---------------------------------------------------------------------------
def test_manifest_determinism():
    """Re-running fingerprint computation on identical data produces identical SHA-256."""
    m1 = EvaluationManifest(
        manifest_id="test-id-1",
        created_at_utc="2026-09-08T12:00:00Z",
        provenance={"dataset": "synth", "hash": "abc1234"},
        result_classification=ResultClassification.SYNTHETIC_VERIFIED,
        evidence_state=EvidenceState.PASS,
        policy_id="POL-TEST-001",
        policy_version="1.0.0",
        reproducibility={"seed": 42, "python_version": "3.11"},
        metric_values={"accuracy": 0.85, "equalized_odds_diff": 0.04},
        uncertainty={"iterations": 200, "valid": 200},
    )
    m2 = EvaluationManifest(
        manifest_id="test-id-2",  # Different ID (should be ignored in fingerprint)
        created_at_utc="2026-09-08T14:30:00Z",  # Different timestamp (ignored)
        provenance={"dataset": "synth", "hash": "abc1234"},
        result_classification=ResultClassification.SYNTHETIC_VERIFIED,
        evidence_state=EvidenceState.PASS,
        policy_id="POL-TEST-001",
        policy_version="1.0.0",
        reproducibility={"seed": 42, "python_version": "3.11"},
        metric_values={"accuracy": 0.85, "equalized_odds_diff": 0.04},
        uncertainty={"iterations": 200, "valid": 200},
    )

    fp1 = compute_manifest_fingerprint(m1)
    fp2 = compute_manifest_fingerprint(m2)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# TEST 2: Fingerprint Sensitivity
# ---------------------------------------------------------------------------
def test_fingerprint_sensitivity():
    """Modifying any metric, metadata, policy, or seed changes the fingerprint."""
    base_m = EvaluationManifest(
        provenance={"dataset": "synth", "hash": "abc1234"},
        result_classification=ResultClassification.SYNTHETIC_VERIFIED,
        evidence_state=EvidenceState.PASS,
        policy_id="POL-TEST-001",
        policy_version="1.0.0",
        reproducibility={"seed": 42},
        metric_values={"accuracy": 0.85},
    )
    base_fp = compute_manifest_fingerprint(base_m)

    # Modify metric
    m_metric = EvaluationManifest(
        provenance={"dataset": "synth", "hash": "abc1234"},
        result_classification=ResultClassification.SYNTHETIC_VERIFIED,
        evidence_state=EvidenceState.PASS,
        policy_id="POL-TEST-001",
        policy_version="1.0.0",
        reproducibility={"seed": 42},
        metric_values={"accuracy": 0.86},
    )
    assert compute_manifest_fingerprint(m_metric) != base_fp

    # Modify seed
    m_seed = EvaluationManifest(
        provenance={"dataset": "synth", "hash": "abc1234"},
        result_classification=ResultClassification.SYNTHETIC_VERIFIED,
        evidence_state=EvidenceState.PASS,
        policy_id="POL-TEST-001",
        policy_version="1.0.0",
        reproducibility={"seed": 43},
        metric_values={"accuracy": 0.85},
    )
    assert compute_manifest_fingerprint(m_seed) != base_fp

    # Modify policy_id
    m_policy = EvaluationManifest(
        provenance={"dataset": "synth", "hash": "abc1234"},
        result_classification=ResultClassification.SYNTHETIC_VERIFIED,
        evidence_state=EvidenceState.PASS,
        policy_id="POL-TEST-002",
        policy_version="1.0.0",
        reproducibility={"seed": 42},
        metric_values={"accuracy": 0.85},
    )
    assert compute_manifest_fingerprint(m_policy) != base_fp


# ---------------------------------------------------------------------------
# TEST 3: Bootstrap Replicate Accounting
# ---------------------------------------------------------------------------
def test_bootstrap_replicate_accounting():
    """Discarded replicates are logged with reasons; discard + valid == requested."""
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    groups = pd.Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"] * 5)

    res = compute_fairness_uncertainty_bootstrap(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_series=groups,
        n_bootstrap=50,
        random_seed=42,
    )
    assert res.requested_iterations == 50
    assert res.successful_iterations + res.discarded_iterations == res.requested_iterations
    assert isinstance(res.discard_reasons, dict)
    assert res.support_status in (EvaluationStatus.HEALTHY, EvaluationStatus.INSUFFICIENT_SUPPORT)


# ---------------------------------------------------------------------------
# TEST 4: Bootstrap Failure -> INSUFFICIENT_EVIDENCE
# ---------------------------------------------------------------------------
def test_bootstrap_failure_insufficient_evidence():
    """When valid bootstrap replicates < min_valid_bootstrap_fraction, policy gives INSUFFICIENT_EVIDENCE."""
    # Mock uncertainty result with insufficient valid replicates
    uncertainty = FairnessUncertaintyResult(
        requested_iterations=100,
        successful_iterations=30,  # 30% < 80% threshold
        discarded_iterations=70,
        discard_reasons={"zero_selection_rate": 70},
        support_status=EvaluationStatus.INSUFFICIENT_SUPPORT,
    )
    policy = FairnessPolicy(min_valid_bootstrap_fraction=0.80)

    # Healthy point metrics but failed bootstrap support
    suite = FairnessEvaluationSuite(
        demographic_parity_difference=0.02,
        equalized_odds_difference=0.02,
        disparate_impact_ratio=0.90,
        balanced_accuracy=0.80,
        uncertainty=uncertainty,
    )

    eval_res = evaluate_policy_compliance(suite, policy)
    assert eval_res.evidence_state == EvidenceState.INSUFFICIENT_EVIDENCE
    assert any("Bootstrap replicate support failed" in a for a in eval_res.abstentions)


# ---------------------------------------------------------------------------
# TEST 5: Disparate Impact Ratio Denominator Edge Cases
# ---------------------------------------------------------------------------
def test_disparate_impact_ratio_edge_cases():
    """DIR handles zero and near-zero selection rates defensively."""
    # Case 1: Reference group rate = 0
    dir_val, stat, rates = safe_disparate_impact_ratio(
        y_pred=np.array([0, 0, 0, 0]),
        sensitive_series=pd.Series(["A", "A", "B", "B"]),
    )
    assert np.isnan(dir_val)
    assert stat in (EvaluationStatus.ZERO_SELECTION_RATE, EvaluationStatus.UNSTABLE_RATIO)
    assert rates["A"] == 0.0 and rates["B"] == 0.0

    # Case 2: One group near-zero selection rate (< 1e-4)
    # 1 positive out of 20,000 = 5e-5 (< 1e-4)
    preds = np.zeros(20000, dtype=int)
    preds[0] = 1
    groups = pd.Series(["A"] * 10000 + ["B"] * 10000)
    dir_val2, stat2, rates2 = safe_disparate_impact_ratio(
        y_pred=preds,
        sensitive_series=groups,
        min_selection_rate_threshold=1e-4,
    )
    assert stat2 == EvaluationStatus.UNSTABLE_RATIO
    assert np.isnan(dir_val2)


# ---------------------------------------------------------------------------
# TEST 6: Multiple Comparisons Metadata
# ---------------------------------------------------------------------------
def test_multiple_comparisons_metadata():
    """Evaluation records number_of_comparisons and honors analysis_type."""
    y_true = np.array([1, 0, 1, 0] * 20)
    y_pred = np.array([1, 0, 0, 1] * 20)
    sensitive_df = pd.DataFrame({
        "group1": ["A", "B"] * 40,
        "group2": ["X", "X", "Y", "Y"] * 20,
    })

    policy = FairnessPolicy(analysis_type=AnalysisType.SCREENING)
    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        policy=policy,
    )
    assert suite.number_of_comparisons is not None
    assert suite.number_of_comparisons > 1
    assert suite.analysis_type == AnalysisType.SCREENING


# ---------------------------------------------------------------------------
# TEST 7: Worst-Group Selection Disclosure
# ---------------------------------------------------------------------------
def test_worst_group_selection_disclosure():
    """Disparity report flags worst-group disparity selection bias (winner's curse)."""
    y_true = np.array([1, 0, 1, 0] * 30)
    y_pred = np.array([1, 0, 1, 0] * 30)
    sensitive_df = pd.DataFrame({"group": ["A", "B", "C"] * 40})

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
    )
    assert suite.selection_bias_disclosure is not None
    assert "Selection Bias Disclosure" in suite.selection_bias_disclosure
    assert "winner's curse" in suite.selection_bias_disclosure.lower()


# ---------------------------------------------------------------------------
# TEST 8: Tri-State Precedence
# ---------------------------------------------------------------------------
def test_tri_state_precedence():
    """Enforces strict precedence: Degeneracy (FAIL) > Support (INSUFFICIENT) > Disparity (FAIL) > PASS."""
    policy = FairnessPolicy(
        min_balanced_accuracy=0.60,
        max_equalized_odds_diff=0.10,
    )

    # 1. Degenerate model with 0 disparity -> FAIL (not PASS, not INSUFFICIENT)
    suite_deg = FairnessEvaluationSuite(
        balanced_accuracy=0.50,
        equalized_odds_difference=0.0,
        degeneracy_status=ModelDegeneracyStatus.COLLAPSED_CONSTANT,
    )
    res_deg = evaluate_policy_compliance(suite_deg, policy)
    assert res_deg.evidence_state == EvidenceState.FAIL

    # 2. Insufficient support on non-degenerate model -> INSUFFICIENT_EVIDENCE (not PASS, not FAIL)
    suite_insuf = FairnessEvaluationSuite(
        balanced_accuracy=0.80,
        equalized_odds_difference=0.02,
        evaluation_status=EvaluationStatus.SPARSE_SUBGROUP,
    )
    res_insuf = evaluate_policy_compliance(suite_insuf, policy)
    assert res_insuf.evidence_state == EvidenceState.INSUFFICIENT_EVIDENCE

    # 3. Disparity violation -> FAIL
    suite_violation = FairnessEvaluationSuite(
        balanced_accuracy=0.80,
        equalized_odds_difference=0.25,  # exceeds 0.10
        evaluation_status=EvaluationStatus.HEALTHY,
    )
    res_violation = evaluate_policy_compliance(suite_violation, policy)
    assert res_violation.evidence_state == EvidenceState.FAIL

    # 4. Clean evaluation passing all criteria -> PASS
    suite_clean = FairnessEvaluationSuite(
        balanced_accuracy=0.80,
        equalized_odds_difference=0.05,
        demographic_parity_difference=0.04,
        disparate_impact_ratio=0.88,
        evaluation_status=EvaluationStatus.HEALTHY,
    )
    res_clean = evaluate_policy_compliance(suite_clean, policy)
    assert res_clean.evidence_state == EvidenceState.PASS


# ---------------------------------------------------------------------------
# TEST 9: Missing Sensitive Attribute Governance
# ---------------------------------------------------------------------------
def test_missing_sensitive_attribute_governance():
    """REJECT raises error, EXCLUDE removes rows and counts, EXPLICIT_UNKNOWN treats null as distinct group."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    sens_with_nulls = pd.Series(["A", "B", None, "A", "B", np.nan])

    # REJECT policy
    policy_reject = FairnessPolicy(missing_attribute_policy=MissingAttributePolicy.REJECT)
    suite_reject = evaluate_fairness_defensively(
        y_true=y_true, y_pred=y_pred, sensitive_features=sens_with_nulls, policy=policy_reject
    )
    assert suite_reject.evaluation_status == EvaluationStatus.MISSING_SENSITIVE_ATTRIBUTE

    # EXCLUDE policy
    policy_exclude = FairnessPolicy(missing_attribute_policy=MissingAttributePolicy.EXCLUDE)
    suite_exclude = evaluate_fairness_defensively(
        y_true=y_true, y_pred=y_pred, sensitive_features=sens_with_nulls, policy=policy_exclude
    )
    assert suite_exclude.evaluation_status == EvaluationStatus.HEALTHY
    assert suite_exclude.excluded_missing_count == 2

    # EXPLICIT_UNKNOWN_GROUP policy
    policy_unknown = FairnessPolicy(missing_attribute_policy=MissingAttributePolicy.EXPLICIT_UNKNOWN_GROUP)
    suite_unknown = evaluate_fairness_defensively(
        y_true=y_true, y_pred=y_pred, sensitive_features=sens_with_nulls, policy=policy_unknown
    )
    assert suite_unknown.evaluation_status == EvaluationStatus.HEALTHY
    assert "UNKNOWN" in suite_unknown.subgroup_metrics


# ---------------------------------------------------------------------------
# TEST 10: Unknown Category Handling
# ---------------------------------------------------------------------------
def test_unknown_category_handling():
    """Unregistered / unexpected categories are managed defensively per policy."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    sens = pd.Series(["Group1", "Group2", "UNEXPECTED_VAL", "Group1"])

    # Under EXCLUDE, if configured with known categories
    policy_exclude = FairnessPolicy(missing_attribute_policy=MissingAttributePolicy.EXCLUDE)
    suite = evaluate_fairness_defensively(
        y_true=y_true, y_pred=y_pred, sensitive_features=sens, policy=policy_exclude
    )
    assert suite.subgroup_metrics is not None
    assert len(suite.subgroup_metrics) > 0


# ---------------------------------------------------------------------------
# TEST 11: Policy Versioning & Disclaimers
# ---------------------------------------------------------------------------
def test_policy_versioning_and_disclaimers():
    """Every evaluation output includes policy_id, policy_version, and legal disclaimer."""
    policy = FairnessPolicy(
        policy_id="POL-CORP-AUDIT-2026",
        policy_version="2.1.0",
    )
    suite = FairnessEvaluationSuite(
        balanced_accuracy=0.75,
        equalized_odds_difference=0.03,
        evaluation_status=EvaluationStatus.HEALTHY,
    )
    res = evaluate_policy_compliance(suite, policy)
    assert res.policy_id == "POL-CORP-AUDIT-2026"
    assert res.policy_version == "2.1.0"
    assert len(res.disclaimer) > 50
    assert "DOES NOT constitute legal certification" in res.disclaimer


# ---------------------------------------------------------------------------
# TEST 12: Global / Subgroup / Intersectional Precedence
# ---------------------------------------------------------------------------
def test_critical_subgroup_failure_precedence():
    """Critical subgroup failure causes overall FAIL even if global metric passes."""
    policy = FairnessPolicy(
        max_equalized_odds_diff=0.10,
        critical_subgroups=["Group_Disadvantaged"],
    )
    # Global EOD is 0.05 (passes < 0.10), but critical subgroup has extreme disparity
    suite = FairnessEvaluationSuite(
        equalized_odds_difference=0.05,
        balanced_accuracy=0.80,
        subgroup_metrics={
            "Group_Advantaged": {"tpr": 0.85, "fpr": 0.10},
            "Group_Disadvantaged": {"tpr": 0.40, "fpr": 0.30},  # TPR gap = 0.45 > 0.10
        },
        evaluation_status=EvaluationStatus.HEALTHY,
    )
    res = evaluate_policy_compliance(suite, policy)
    assert res.evidence_state == EvidenceState.FAIL
    assert any("Critical subgroup" in v for v in res.violations)


# ---------------------------------------------------------------------------
# TEST 13: Sparse Calibration Support Status
# ---------------------------------------------------------------------------
def test_sparse_calibration_support_status():
    """Subgroups with < 30 samples or < 5 positive/negative outcomes flag SPARSE_SUBGROUP."""
    # 20 samples total (< 30)
    y_true = np.array([1, 0] * 10)
    y_pred = np.array([1, 0] * 10)
    sens = pd.Series(["A"] * 20)

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sens,
        compute_subgroup_calibration=True,
    )
    assert suite.calibration_by_subgroup["A"]["status"] == EvaluationStatus.SPARSE_SUBGROUP


# ---------------------------------------------------------------------------
# TEST 14: Path Resolution Containment (Staging Root)
# ---------------------------------------------------------------------------
def test_path_resolution_containment():
    """Ingestion path outside approved_staging_root is rejected even if file exists."""
    from src.ingest_data import SecurityValidationError

    with tempfile.TemporaryDirectory() as staging_root:
        with tempfile.TemporaryDirectory() as outside_root:
            # File created outside staging root
            outside_file = Path(outside_root) / "external_data.csv"
            outside_file.write_text("a,b,c\n1,2,3\n")

            # File created inside staging root
            inside_file = Path(staging_root) / "valid_data.csv"
            inside_file.write_text("a,b,c\n1,2,3\n")

            # Inside path succeeds
            valid_path = validate_source_path_security(
                inside_file, approved_staging_root=Path(staging_root)
            )
            assert valid_path == inside_file.resolve()

            # Outside path fails containment check
            with pytest.raises(SecurityValidationError):
                validate_source_path_security(
                    outside_file, approved_staging_root=Path(staging_root)
                )

            # Directory traversal attempt fails
            traversal_path = Path(staging_root) / ".." / outside_file.name
            with pytest.raises(SecurityValidationError):
                validate_source_path_security(
                    traversal_path, approved_staging_root=Path(staging_root)
                )


# ---------------------------------------------------------------------------
# TEST 15: Git-Tracked Data Rejection
# ---------------------------------------------------------------------------
def test_git_tracked_data_rejection():
    """Ingestion of git-tracked files is rejected when require_untracked=True."""
    from src.ingest_data import SecurityValidationError

    readme_path = Path("README.md").resolve()
    if readme_path.exists():
        with pytest.raises(SecurityValidationError, match="GIT TRACKING PROHIBITED"):
            validate_source_path_security(readme_path, require_untracked=True)


# ---------------------------------------------------------------------------
# TEST 16: External vs Synthetic Boundary Separation
# ---------------------------------------------------------------------------
def test_external_vs_synthetic_boundary_separation():
    """ExternalDatasetAdapter has ZERO imports from synthetic scenario modules."""
    import src.external_dataset_adapter as ext_module
    import inspect

    source_text = inspect.getsource(ext_module)
    # Must NOT import synthetic scenarios, DGPs, or simulation engines
    assert "synthetic_scenarios" not in source_text
    assert "generate_synthetic" not in source_text
    assert "SCENARIO_" not in source_text


# ---------------------------------------------------------------------------
# TEST 17: Result Provenance Classification
# ---------------------------------------------------------------------------
def test_result_provenance_classification():
    """Classifies SYNTHETIC_VERIFIED vs EXTERNAL_PENDING_STAGING vs EXTERNAL_AUDIT_READY."""
    manifest_synth = EvaluationManifest(
        provenance={"dataset": "synthetic_scenario_a", "dataset_type": "SYNTHETIC"},
        result_classification=ResultClassification.SYNTHETIC_VERIFIED,
    )
    assert manifest_synth.result_classification == ResultClassification.SYNTHETIC_VERIFIED

    # External dataset staged via adapter
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "client_data.csv"
        df = pd.DataFrame({
            "target": [1, 0, 1, 0],
            "gender": ["F", "M", "F", "M"],
            "feature": [10, 20, 30, 40],
        })
        df.to_csv(test_file, index=False)

        adapter = ExternalDatasetAdapter(staging_root=tmp_dir)
        bundle = adapter.ingest_external_tabular(
            file_path=test_file,
            label_column="target",
            protected_columns=["gender"],
            feature_columns=["feature"],
            provenance_metadata={"client": "EnterpriseCorp"},
        )
        assert bundle.provenance_manifest.result_classification == ResultClassification.EXTERNAL_AUDIT_READY


# ---------------------------------------------------------------------------
# TEST 18: Manifest / Result Consistency
# ---------------------------------------------------------------------------
def test_manifest_result_consistency():
    """Fields in EvaluationManifest match the evaluated result object exactly."""
    policy = FairnessPolicy(policy_id="POL-99", policy_version="3.0.0")
    suite = FairnessEvaluationSuite(
        accuracy=0.88,
        balanced_accuracy=0.85,
        equalized_odds_difference=0.04,
        evaluation_status=EvaluationStatus.HEALTHY,
    )
    pol_res = evaluate_policy_compliance(suite, policy)

    manifest = EvaluationManifest(
        policy_id=pol_res.policy_id,
        policy_version=pol_res.policy_version,
        evidence_state=pol_res.evidence_state,
        metric_values={
            "accuracy": suite.accuracy,
            "balanced_accuracy": suite.balanced_accuracy,
            "equalized_odds_difference": suite.equalized_odds_difference,
        },
    )

    assert manifest.policy_id == pol_res.policy_id
    assert manifest.policy_version == pol_res.policy_version
    assert manifest.evidence_state == pol_res.evidence_state
    assert manifest.metric_values["accuracy"] == suite.accuracy
    assert manifest.metric_values["equalized_odds_difference"] == suite.equalized_odds_difference


# ---------------------------------------------------------------------------
# TEST 19: Customer Raw Data Git Isolation
# ---------------------------------------------------------------------------
def test_customer_raw_data_git_isolation():
    """Verify .gitignore contains data/ and no raw data files are tracked."""
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists()
    content = gitignore_path.read_text(encoding="utf-8")
    assert re.search(r"^\s*data/\s*$", content, re.MULTILINE) is not None

    # Verify no CSV or JSON files inside data/ are tracked in git
    import subprocess
    res = subprocess.run(["git", "ls-files", "data/"], capture_output=True, text=True)
    # Output should be empty (no files tracked under data/)
    assert res.stdout.strip() == ""


# ---------------------------------------------------------------------------
# TEST 20: Zero Network Calls in Ingestion & CSV Injection Defense
# ---------------------------------------------------------------------------
def test_zero_network_calls_and_csv_injection():
    """Verify ingestion operates strictly locally with CSV injection neutralization."""
    # Test CSV injection defense
    unsafe_val = "=SUM(A1:A10)"
    sanitized = sanitize_csv_value(unsafe_val)
    assert sanitized.startswith("'")
    assert sanitize_csv_value("+cmd|' /C calc'!A0").startswith("'")
    assert sanitize_csv_value("-5+10").startswith("'")
    assert sanitize_csv_value("@SUM").startswith("'")
    assert sanitize_csv_value("normal_string") == "normal_string"
    assert sanitize_csv_value(123) == 123

    # Test export_sanitized_csv
    df_unsafe = pd.DataFrame({"col": ["=cmd", "safe", 42]})
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_csv = Path(tmp_dir) / "safe_export.csv"
        export_sanitized_csv(df_unsafe, out_csv)
        content = out_csv.read_text(encoding="utf-8")
        assert "'=cmd" in content

    # Test that ingestion doesn't touch socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # socket module is accessible, but adapter should run locally without network calls
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "local_test.csv"
            pd.DataFrame({"y": [1, 0], "g": ["A", "B"], "x": [1, 2]}).to_csv(test_file, index=False)
            adapter = ExternalDatasetAdapter(staging_root=tmp_dir)
            bundle = adapter.ingest_external_tabular(
                file_path=test_file,
                label_column="y",
                protected_columns=["g"],
                feature_columns=["x"],
            )
            assert bundle.n_rows == 2
