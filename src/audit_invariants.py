# ============================================================================
# Audit Invariants Compatibility Module
# Re-exports invariant validation and hygiene checking from src.metric_invariants.
# ============================================================================

from __future__ import annotations

from src.metric_invariants import (
    ERR_INVALID_BOOTSTRAP_CONFIGURATION,
    ERR_INVALID_COUNT_IDENTITY,
    ERR_INVALID_GROUP_MAPPING,
    ERR_INVALID_METRIC_IDENTITY,
    ERR_INVALID_UNIT_ALIGNMENT,
    ERR_MISSING_ESTIMAND,
    audit_artifact_privacy,
    run_publication_release_audit,
    run_repository_hygiene_check,
    scan_content_hygiene,
    validate_audit_mathematical_invariants,
    validate_metric_mathematical_invariants,
)

__all__ = [
    "ERR_INVALID_BOOTSTRAP_CONFIGURATION",
    "ERR_INVALID_COUNT_IDENTITY",
    "ERR_INVALID_GROUP_MAPPING",
    "ERR_INVALID_METRIC_IDENTITY",
    "ERR_INVALID_UNIT_ALIGNMENT",
    "ERR_MISSING_ESTIMAND",
    "audit_artifact_privacy",
    "run_publication_release_audit",
    "run_repository_hygiene_check",
    "scan_content_hygiene",
    "validate_audit_mathematical_invariants",
    "validate_metric_mathematical_invariants",
]
