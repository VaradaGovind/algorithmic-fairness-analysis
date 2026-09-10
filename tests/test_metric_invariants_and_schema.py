# ============================================================================
# Metric Invariants, Output Schema, and Hygiene Test Suite
# Tests count identities, metric equations, schema validation, JSON roundtrip,
# content hygiene scanning, and manifest determinism.
# ============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.metric_invariants import (
    ERR_INVALID_COUNT_IDENTITY,
    ERR_INVALID_METRIC_IDENTITY,
    scan_content_hygiene,
    validate_metric_mathematical_invariants,
)
from src.audit_result_schema import (
    AuditResult,
    SensitiveAttributeLineage,
    build_audit_result_from_suite,
    validate_audit_result_dict,
)
from src.evaluation_manifest import (
    EvaluationManifest,
    ResultClassification,
    compute_manifest_fingerprint,
    create_evaluation_manifest,
)
from src.fairness_evaluator import (
    EvaluationStatus,
    EvaluationUnitLevel,
    EvidenceState,
    evaluate_fairness_defensively,
)


# ---------------------------------------------------------------------------
# Test 1: Mathematical Invariants - Confusion Counts & Rates
# ---------------------------------------------------------------------------

def test_metric_invariants_confusion_counts():
    """Validates that correct confusion matrix counts and rates pass invariant checks."""
    valid_audit = {
        "dataset": {"total_records": 100, "entity_count": 50},
        "subgroup_evaluations": {
            "GroupA": {
                "sample_size": 50,
                "positive_count": 30,
                "negative_count": 20,
                "TP": 25,
                "TN": 15,
                "FP": 5,
                "FN": 5,
                "accuracy": 40 / 50,
                "TPR": 25 / 30,
                "TNR": 15 / 20,
                "FPR": 5 / 20,
                "FNR": 5 / 30,
                "PPV": 25 / 30,
                "balanced_accuracy": (25 / 30 + 15 / 20) / 2,
                "selection_rate": 30 / 50,
            },
            "GroupB": {
                "sample_size": 50,
                "positive_count": 20,
                "negative_count": 30,
                "TP": 15,
                "TN": 20,
                "FP": 10,
                "FN": 5,
                "accuracy": 35 / 50,
                "TPR": 15 / 20,
                "TNR": 20 / 30,
                "FPR": 10 / 30,
                "FNR": 5 / 20,
                "PPV": 15 / 25,
                "balanced_accuracy": (15 / 20 + 20 / 30) / 2,
                "selection_rate": 25 / 50,
            },
        },
        "global_performance": {
            "demographic_parity_difference": (30 / 50) - (25 / 50),
            "equal_opportunity_difference": (25 / 30) - (15 / 20),
            "equalized_odds_difference": max(abs((25 / 30) - (15 / 20)), abs((5 / 20) - (10 / 30))),
            "disparate_impact_ratio": (25 / 50) / (30 / 50),
        },
    }

    is_valid, errors = validate_metric_mathematical_invariants(valid_audit)
    assert is_valid is True, f"Invariant check failed: {errors}"
    assert len(errors) == 0


# ---------------------------------------------------------------------------
# Test 2: Invariant Violation Detection
# ---------------------------------------------------------------------------

def test_metric_invariants_violation_detection():
    """Verifies that forged or mathematically impossible metrics are rejected."""
    invalid_counts = {
        "dataset": {"total_records": 50, "entity_count": 50},
        "subgroup_evaluations": {
            "GroupA": {
                "sample_size": 50,
                "positive_count": 25,
                "negative_count": 25,
                "TP": 20,
                "TN": 20,
                "FP": 10,
                "FN": 10,  # Sum = 60 != 50
            }
        },
    }
    is_valid, errors = validate_metric_mathematical_invariants(invalid_counts)
    assert is_valid is False
    assert any(ERR_INVALID_COUNT_IDENTITY in e for e in errors)


# ---------------------------------------------------------------------------
# Test 3: Disparity Identity Checks
# ---------------------------------------------------------------------------

def test_metric_invariants_disparity_identities():
    """Checks that global disparity metrics match the subgroup extrema."""
    tampered_dpd = {
        "dataset": {"total_records": 100, "entity_count": 50},
        "subgroup_evaluations": {
            "G1": {"sample_size": 50, "selection_rate": 0.80},
            "G2": {"sample_size": 50, "selection_rate": 0.40},
        },
        "global_performance": {
            "demographic_parity_difference": 0.10,  # Should be 0.80 - 0.40 = 0.40!
        },
    }
    is_valid, errors = validate_metric_mathematical_invariants(tampered_dpd)
    assert is_valid is False
    assert any(ERR_INVALID_METRIC_IDENTITY in e for e in errors)


# ---------------------------------------------------------------------------
# Test 4: Manifest Determinism & Fingerprint Sensitivity
# ---------------------------------------------------------------------------

def test_manifest_determinism():
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
        manifest_id="test-id-2",  # Different ID (ignored in content fingerprint)
        created_at_utc="2026-09-08T14:30:00Z",  # Different timestamp
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
    assert len(fp1) == 64


def test_fingerprint_sensitivity():
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


# ---------------------------------------------------------------------------
# Test 5: Audit Result Schema Validation & JSON Roundtrip
# ---------------------------------------------------------------------------

def test_audit_result_json_canonical_roundtrip(tmp_path):
    yt = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    yp = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    s = np.array(["G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2"])
    suite = evaluate_fairness_defensively(yt, yp, sensitive=s)

    audit_res = build_audit_result_from_suite(suite)
    fp_orig = audit_res.audit_fingerprint

    save_file = tmp_path / "test_audit_result.json"
    audit_res.save_json(save_file)
    assert save_file.exists()

    loaded = AuditResult.load_json(save_file)
    assert loaded.audit_id == audit_res.audit_id
    assert loaded.audit_fingerprint == fp_orig


# ---------------------------------------------------------------------------
# Test 6: Content Hygiene Scanner
# ---------------------------------------------------------------------------

def test_content_hygiene_scanner():
    clean_text = "This is clean documentation with no secrets or emails."
    res_clean = scan_content_hygiene(clean_text)
    assert res_clean["clean"] is True
    assert res_clean["total_leaks_found"] == 0

    dirty_text = "Contact user@example.com or check C:\\Users\\test\\secret."
    res_dirty = scan_content_hygiene(dirty_text, sanitize=True)
    assert res_dirty["clean"] is False
    assert "[EMAIL_REDACTED]" in res_dirty["sanitized_content"]
    assert "[PATH_REDACTED]" in res_dirty["sanitized_content"]
