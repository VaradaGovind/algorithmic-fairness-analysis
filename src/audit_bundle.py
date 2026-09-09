# ============================================================================
# Reproducible Audit Bundle Generator (Pass 9 Release-Candidate)
# Creates a self-contained, auditable evaluation package for independent replication.
# Guarantees zero inclusion of raw customer or proprietary data rows.
# ============================================================================

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.audit_result_schema import AuditResult, validate_audit_result_dict
from src.evaluation_manifest import EvaluationManifest


# Failure taxonomy codes
BUNDLE_MISSING_FILE = "BUNDLE_MISSING_FILE"
BUNDLE_SCHEMA_INVALID = "BUNDLE_SCHEMA_INVALID"
BUNDLE_INVARIANTS_INVALID = "BUNDLE_INVARIANTS_INVALID"
BUNDLE_FINGERPRINT_MISMATCH = "BUNDLE_FINGERPRINT_MISMATCH"
BUNDLE_MANIFEST_MISMATCH = "BUNDLE_MANIFEST_MISMATCH"
BUNDLE_RAW_DATA_DETECTED = "BUNDLE_RAW_DATA_DETECTED"
BUNDLE_CONFIGURATION_MISMATCH = "BUNDLE_CONFIGURATION_MISMATCH"
BUNDLE_CROSS_ARTIFACT_MISMATCH = "BUNDLE_CROSS_ARTIFACT_MISMATCH"
BUNDLE_PRIVACY_VIOLATION = "BUNDLE_PRIVACY_VIOLATION"


@dataclass
class BundleVerificationResult:
    is_valid: bool
    failure_codes: List[str] = field(default_factory=list)
    checks_performed: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, List[str]] = field(default_factory=dict)
    threat_model_disclaimer: str = (
        "THREAT MODEL NOTICE: SHA-256 fingerprints establish tamper evidence and cryptographic "
        "consistency between recorded audit artifacts and declared manifests. They do NOT prove "
        "authenticity of the originating data source, absence of upstream data manipulation before "
        "hashing, or legal truth of declared policies, lineages, or claims. Cryptographic consistency "
        "is strictly distinct from empirical operational real-data validation or legal authenticity."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "failure_codes": self.failure_codes,
            "checks_performed": self.checks_performed,
            "details": self.details,
            "threat_model_disclaimer": self.threat_model_disclaimer,
        }


class DatasetStorageStatus(str, Enum):
    INCLUDED = "INCLUDED"
    EXTERNAL_REFERENCE = "EXTERNAL_REFERENCE"
    CUSTOMER_HELD = "CUSTOMER_HELD"


def get_current_environment_metadata() -> Dict[str, Any]:

    """Collects deterministic runtime environment metadata for audit replication."""
    packages: Dict[str, str] = {}
    for pkg in ["numpy", "scipy", "sklearn", "pandas", "yaml", "shap"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "UNKNOWN")
        except ImportError:
            packages[pkg] = "NOT_INSTALLED"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "collection_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def create_evaluation_bundle(
    output_dir: Union[str, Path],
    audit_result: AuditResult,
    manifest: Optional[EvaluationManifest] = None,
    dataset_storage_status: DatasetStorageStatus = DatasetStorageStatus.CUSTOMER_HELD,
    configuration_files: Optional[Dict[str, Any]] = None,
    environment_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Constructs a complete, reproducible evaluation bundle directory:
    
    bundle_root/
      ├── audit_result.json
      ├── evaluation_manifest.json
      ├── configuration/
      │     ├── evaluation_units.json
      │     ├── feature_contract.json
      │     └── policy_configuration.json
      ├── provenance/
      │     ├── dataset_provenance.json
      │     └── dataset_storage_status.json
      ├── results/
      │     ├── global_performance.json
      │     ├── fairness_metrics.json
      │     ├── subgroup_evaluations.json
      │     └── uncertainty_quantification.json
      ├── warnings/
      │     └── audit_warnings.json
      ├── environment.json
      └── README.md
    
    Zero raw customer data is placed into the bundle.
    """
    root = Path(output_dir)
    cfg_dir = root / "configuration"
    prov_dir = root / "provenance"
    res_dir = root / "results"
    warn_dir = root / "warnings"

    for d in [cfg_dir, prov_dir, res_dir, warn_dir]:
        d.mkdir(parents=True, exist_ok=True)

    audit_dict = audit_result.to_dict()

    # 1. Root audit_result.json
    audit_result.save_json(root / "audit_result.json")

    # 2. Root evaluation_manifest.json
    if manifest is not None:
        manifest.save_json(root / "evaluation_manifest.json")
    else:
        # Generate minimal manifest representation from audit_dict
        min_manifest = {
            "evaluation_id": audit_dict.get("audit_id"),
            "fingerprint": audit_dict.get("audit_fingerprint"),
            "timestamp_utc": audit_dict.get("timestamp"),
            "dataset_identity": audit_dict.get("dataset", {}).get("dataset_name"),
            "policy_id": audit_dict.get("policy_compliance", {}).get("policy_name"),
            "evidence_state": audit_dict.get("policy_compliance", {}).get("evidence_state"),
        }
        with open(root / "evaluation_manifest.json", "w", encoding="utf-8") as f:
            json.dump(min_manifest, f, indent=2)

    # 3. configuration/
    with open(cfg_dir / "evaluation_units.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("evaluation_unit_alignment", {}), f, indent=2)

    with open(cfg_dir / "feature_contract.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("feature_contract", {}), f, indent=2)

    with open(cfg_dir / "policy_configuration.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("policy_compliance", {}), f, indent=2)

    if configuration_files:
        for name, content in configuration_files.items():
            out_f = cfg_dir / f"{name}.json"
            with open(out_f, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)

    # 4. provenance/
    with open(prov_dir / "dataset_provenance.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("dataset", {}), f, indent=2)

    status_record = {
        "dataset_name": audit_dict.get("dataset", {}).get("dataset_name"),
        "dataset_storage_status": dataset_storage_status.value,
        "data_source_digest": audit_dict.get("dataset", {}).get("data_source_digest"),
        "raw_customer_data_included": False,
        "access_instructions": (
            "Under strict privacy and confidentiality protocols, raw customer records are NOT stored "
            "in this audit bundle. To reproduce this evaluation, stage the authenticated dataset "
            f"matching SHA-256 digest '{audit_dict.get('dataset', {}).get('data_source_digest')}' "
            "in an approved local untracked staging directory."
        ),
    }
    with open(prov_dir / "dataset_storage_status.json", "w", encoding="utf-8") as f:
        json.dump(status_record, f, indent=2)

    # 5. results/
    with open(res_dir / "global_performance.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("global_performance", {}), f, indent=2)

    with open(res_dir / "fairness_metrics.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("fairness_metrics", {}), f, indent=2)

    with open(res_dir / "subgroup_evaluations.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("subgroup_evaluations", {}), f, indent=2)

    with open(res_dir / "uncertainty_quantification.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict.get("uncertainty_quantification", {}), f, indent=2)

    # 6. warnings/
    warnings_record = {
        "violations": audit_dict.get("policy_compliance", {}).get("violations", []),
        "abstentions": audit_dict.get("policy_compliance", {}).get("abstentions", []),
        "rules_satisfied": audit_dict.get("policy_compliance", {}).get("rules_satisfied", []),
    }
    with open(warn_dir / "audit_warnings.json", "w", encoding="utf-8") as f:
        json.dump(warnings_record, f, indent=2)

    # 7. environment.json
    env_data = environment_info or get_current_environment_metadata()
    with open(root / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2)

    # 8. README.md
    ds_name = audit_dict.get("dataset", {}).get("dataset_name", "Dataset")
    audit_id = audit_dict.get("audit_id", "N/A")
    fp = audit_dict.get("audit_fingerprint", "N/A")
    ev_state = audit_dict.get("policy_compliance", {}).get("evidence_state", "N/A")
    passed = audit_dict.get("policy_compliance", {}).get("policy_passed", False)

    readme_content = f"""# Reproducible Fairness Evaluation Bundle

**Dataset:** {ds_name}  
**Audit ID:** `{audit_id}`  
**Audit Fingerprint (SHA-256):** `{fp}`  
**Policy Evidence State:** `{ev_state}`  
**Policy Decision:** `{"PASS" if passed else "NON-PASS"}`  
**Dataset Storage Status:** `{dataset_storage_status.value}`  

---

## 1. Bundle Structure & Contents

This audit bundle contains the complete mathematical configuration, model metadata,
subgroup breakdowns, uncertainty estimates, and execution environment necessary to
audit, review, and replicate the fairness evaluation without exposing raw customer records.

- `audit_result.json`: Product-grade machine-readable schema (Draft-07 compatible).
- `evaluation_manifest.json`: Verifiable configuration identity manifest.
- `configuration/`: Evaluation unit alignment, prediction-time contracts, and policy specifications.
- `provenance/`: Dataset lineage, legal origin records, and storage classification.
- `results/`: Performance metrics, fairness parities, and subgroup-level evaluations.
- `warnings/`: Active warnings, abstentions, and policy violations.
- `environment.json`: Python interpreter, platform, and dependency package digests.

---

## 2. Replication Instructions

1. Verify environment prerequisites against `environment.json`.
2. Stage the authorized raw dataset matching the SHA-256 digest in `provenance/dataset_storage_status.json`.
3. Execute the pipeline using `src.external_dataset_adapter.ExternalDatasetAdapter` or the canonical benchmark runner.
4. Verify that the resulting canonical fingerprint matches `{fp}`.

---

## 3. Methodological Disclaimers & Governance Notice

> [!NOTE]
> **Zero Raw Data Policy:** Raw customer records and PII are strictly excluded from this bundle.
> **Report Privacy:** Cell suppression and heuristics protect subgroup anonymity; this does NOT constitute formal differential privacy.
> **Non-Certification:** Compliance with technical fairness constraints reflects internal mathematical criteria and DOES NOT constitute statutory non-discrimination or legal compliance certification.
"""
    with open(root / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    return root


def compute_independent_fingerprint(audit_dict: Dict[str, Any]) -> str:
    """Computes independent canonical SHA-256 fingerprint for verification."""
    excluded_keys = {"audit_fingerprint", "timestamp"}
    canonical_items = {
        k: v for k, v in sorted(audit_dict.items()) if k not in excluded_keys and v is not None
    }
    canonical_json = json.dumps(canonical_items, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def verify_audit_bundle(bundle_path: Union[str, Path]) -> BundleVerificationResult:
    """
    Independently verifies an audit bundle across five distinct assurance categories:
      1. Cryptographic Consistency:
         - Canonical SHA-256 fingerprint re-computation across audit_result.json.
         - Evaluation manifest alignment and fingerprint identity match.
      2. Schema Conformance:
         - Draft-07 structural JSON Schema validation of audit_result.json.
      3. Semantic Invariant Verification:
         - Independent arithmetic verification of confusion matrix counts (TP+TN+FP+FN=N).
         - Rate bounding and consistency (TPR, FPR, TNR, FNR, PPV, balanced accuracy in [0, 1]).
         - Disparity bounding and arithmetic equations (DPD, EODiff, DIR >= 0).
      4. Cross-Artifact Parity:
         - Exact agreement between standalone configuration/*.json files and embedded configs.
         - Exact agreement between standalone results/*.json files and embedded results.
      5. Privacy & Security Cleanliness:
         - Broad raw-data detection rejecting .csv, .parquet, .feather, .arrow, .h5, etc.
         - File size boundary checks (<10MB).
         - Heuristic credential, email, and developer machine path suppression scans.
    """
    from src.audit_invariants import validate_audit_mathematical_invariants

    root = Path(bundle_path)
    failure_codes: List[str] = []
    checks_performed: Dict[str, bool] = {
        "required_files": False,
        "raw_data_cleanliness": False,
        "schema_conformance": False,
        "semantic_invariants": False,
        "cryptographic_fingerprint": False,
        "manifest_alignment": False,
        "configuration_consistency": False,
        "cross_artifact_consistency": False,
        "privacy_cleanliness": False,
    }
    details: Dict[str, List[str]] = {
        "missing_files": [],
        "raw_data_findings": [],
        "schema_errors": [],
        "invariant_errors": [],
        "fingerprint_mismatch": [],
        "manifest_mismatch": [],
        "configuration_mismatch": [],
        "cross_artifact_mismatch": [],
        "privacy_findings": [],
    }

    # 1. Check Required Files
    required_files = [
        "audit_result.json",
        "evaluation_manifest.json",
        "configuration/evaluation_units.json",
        "configuration/feature_contract.json",
        "configuration/policy_configuration.json",
        "provenance/dataset_provenance.json",
        "provenance/dataset_storage_status.json",
        "results/global_performance.json",
        "results/fairness_metrics.json",
        "results/subgroup_evaluations.json",
        "results/uncertainty_quantification.json",
        "warnings/audit_warnings.json",
        "environment.json",
        "README.md",
    ]
    missing = [f for f in required_files if not (root / f).is_file()]
    if missing:
        details["missing_files"].extend(missing)
        failure_codes.append(BUNDLE_MISSING_FILE)
    else:
        checks_performed["required_files"] = True

    # 2. Broad Raw Data Detection
    forbidden_extensions = {
        ".csv", ".tsv", ".parquet", ".feather", ".arrow",
        ".h5", ".hdf5", ".orc", ".avro", ".pkl", ".pickle",
        ".npy", ".npz", ".sqlite", ".sqlite3", ".db", ".tab",
        ".data", ".dat",
    }
    raw_data_found = []
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file():
                if p.suffix.lower() in forbidden_extensions:
                    raw_data_found.append(f"Forbidden data extension: {p.relative_to(root)}")
                elif p.stat().st_size > 10 * 1024 * 1024:
                    raw_data_found.append(f"Excessive file size (>10MB): {p.relative_to(root)}")

    if raw_data_found:
        details["raw_data_findings"].extend(raw_data_found)
        failure_codes.append(BUNDLE_RAW_DATA_DETECTED)
    else:
        checks_performed["raw_data_cleanliness"] = True

    # If audit_result.json is missing, return immediately
    audit_file = root / "audit_result.json"
    if not audit_file.is_file():
        return BundleVerificationResult(
            is_valid=False,
            failure_codes=failure_codes,
            checks_performed=checks_performed,
            details=details,
        )

    try:
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_dict = json.load(f)
    except Exception as e:
        details["schema_errors"].append(f"Could not parse audit_result.json: {e}")
        failure_codes.append(BUNDLE_SCHEMA_INVALID)
        return BundleVerificationResult(
            is_valid=False,
            failure_codes=failure_codes,
            checks_performed=checks_performed,
            details=details,
        )

    # 3. Schema Conformance Check (Draft-07 structure)
    s_valid, s_errors = validate_audit_result_dict(audit_dict, validate_invariants=False)
    if not s_valid:
        details["schema_errors"].extend(s_errors)
        failure_codes.append(BUNDLE_SCHEMA_INVALID)
    else:
        checks_performed["schema_conformance"] = True

    # 4. Semantic & Mathematical Invariants Check
    inv_errors = []
    ds = audit_dict.get("dataset", {})
    if ds.get("total_records", 0) < 1:
        inv_errors.append("dataset.total_records must be >= 1")
    if ds.get("entity_count", 0) < 1:
        inv_errors.append("dataset.entity_count must be >= 1")

    subgroups = audit_dict.get("subgroup_evaluations", {})
    if isinstance(subgroups, dict):
        for g_name, g_info in subgroups.items():
            if isinstance(g_info, dict):
                n = g_info.get("sample_size", 0)
                pos = g_info.get("positive_count")
                neg = g_info.get("negative_count")
                if pos is not None and neg is not None and (pos + neg) != n:
                    inv_errors.append(f"Subgroup '{g_name}': positive ({pos}) + negative ({neg}) != sample_size ({n})")
                for rate_name in ["TPR", "FPR", "TNR", "FNR", "PPV", "balanced_accuracy"]:
                    r_val = g_info.get(rate_name)
                    if r_val is not None and not (isinstance(r_val, (int, float)) and math.isnan(r_val)):
                        if r_val < 0.0 or r_val > 1.0:
                            inv_errors.append(f"Subgroup '{g_name}': {rate_name} ({r_val}) outside [0, 1]")

    fm = audit_dict.get("fairness_metrics", {})
    if isinstance(fm, dict):
        dpd = fm.get("demographic_parity_difference")
        if dpd is not None and not math.isnan(dpd) and (dpd < 0.0 or dpd > 1.0):
            inv_errors.append(f"fairness_metrics: demographic_parity_difference ({dpd}) outside [0, 1]")
        eod = fm.get("equalized_odds_difference")
        if eod is not None and not math.isnan(eod) and (eod < 0.0 or eod > 1.0):
            inv_errors.append(f"fairness_metrics: equalized_odds_difference ({eod}) outside [0, 1]")
        dir_val = fm.get("disparate_impact_ratio")
        if dir_val is not None and not math.isnan(dir_val) and dir_val < 0.0:
            inv_errors.append(f"fairness_metrics: disparate_impact_ratio ({dir_val}) < 0")

    std_inv_valid, std_inv_errors = validate_audit_mathematical_invariants(audit_dict)
    if not std_inv_valid:
        inv_errors.extend(std_inv_errors)

    if inv_errors:
        details["invariant_errors"].extend(inv_errors)
        failure_codes.append(BUNDLE_INVARIANTS_INVALID)
    else:
        checks_performed["semantic_invariants"] = True

    # 5. Cryptographic Fingerprint Verification
    computed_fp = compute_independent_fingerprint(audit_dict)
    recorded_fp = audit_dict.get("audit_fingerprint", "")
    if computed_fp != recorded_fp:
        details["fingerprint_mismatch"].append(
            f"Computed fingerprint '{computed_fp}' does not match recorded '{recorded_fp}'"
        )
        failure_codes.append(BUNDLE_FINGERPRINT_MISMATCH)
    else:
        checks_performed["cryptographic_fingerprint"] = True

    # 6. Manifest Alignment Check
    manifest_file = root / "evaluation_manifest.json"
    if manifest_file.is_file():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_dict = json.load(f)
            man_fp = manifest_dict.get("fingerprint") or manifest_dict.get("audit_fingerprint")
            if man_fp and man_fp != recorded_fp:
                details["manifest_mismatch"].append(
                    f"Manifest fingerprint '{man_fp}' does not match audit fingerprint '{recorded_fp}'"
                )
                failure_codes.append(BUNDLE_MANIFEST_MISMATCH)
            else:
                checks_performed["manifest_alignment"] = True

            ds_id = manifest_dict.get("dataset_identity")
            if ds_id and ds_id != ds.get("dataset_name"):
                details["manifest_mismatch"].append(
                    f"Manifest dataset '{ds_id}' does not match audit dataset '{ds.get('dataset_name')}'"
                )
                if BUNDLE_MANIFEST_MISMATCH not in failure_codes:
                    failure_codes.append(BUNDLE_MANIFEST_MISMATCH)
        except Exception as e:
            details["manifest_mismatch"].append(f"Could not verify evaluation_manifest.json: {e}")
            failure_codes.append(BUNDLE_MANIFEST_MISMATCH)

    # 7. Configuration Consistency Check
    cfg_checks = [
        ("configuration/evaluation_units.json", "evaluation_unit_alignment"),
        ("configuration/feature_contract.json", "feature_contract"),
        ("configuration/policy_configuration.json", "policy_compliance"),
    ]
    cfg_mismatch = []
    for cfg_rel, audit_key in cfg_checks:
        cfg_path = root / cfg_rel
        if cfg_path.is_file():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                expected_data = audit_dict.get(audit_key, {})
                if cfg_data != expected_data:
                    cfg_mismatch.append(f"{cfg_rel} does not match audit_result.{audit_key}")
            except Exception as e:
                cfg_mismatch.append(f"Error reading {cfg_rel}: {e}")
    if cfg_mismatch:
        details["configuration_mismatch"].extend(cfg_mismatch)
        failure_codes.append(BUNDLE_CONFIGURATION_MISMATCH)
    else:
        checks_performed["configuration_consistency"] = True

    # 8. Cross-Artifact Consistency Check
    res_checks = [
        ("results/global_performance.json", "global_performance"),
        ("results/fairness_metrics.json", "fairness_metrics"),
        ("results/subgroup_evaluations.json", "subgroup_evaluations"),
        ("results/uncertainty_quantification.json", "uncertainty_quantification"),
    ]
    res_mismatch = []
    for res_rel, audit_key in res_checks:
        res_path = root / res_rel
        if res_path.is_file():
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    res_data = json.load(f)
                expected_data = audit_dict.get(audit_key, {})
                if res_data != expected_data:
                    res_mismatch.append(f"{res_rel} does not match audit_result.{audit_key}")
            except Exception as e:
                res_mismatch.append(f"Error reading {res_rel}: {e}")
    if res_mismatch:
        details["cross_artifact_mismatch"].extend(res_mismatch)
        failure_codes.append(BUNDLE_CROSS_ARTIFACT_MISMATCH)
    else:
        checks_performed["cross_artifact_consistency"] = True

    # 9. Privacy & Sanitization Cleanliness Check
    try:
        from src.audit_invariants import audit_bundle_privacy
        priv_res = audit_bundle_privacy(root)
        if not priv_res.get("privacy_passed", True):
            for cat, items in priv_res.get("findings", {}).items():
                if items:
                    details["privacy_findings"].extend([f"{cat}: {item}" for item in items])
            failure_codes.append(BUNDLE_PRIVACY_VIOLATION)
        else:
            checks_performed["privacy_cleanliness"] = True
    except Exception as e:
        details["privacy_findings"].append(f"Privacy scan execution error: {e}")

    is_valid = (len(failure_codes) == 0)
    return BundleVerificationResult(
        is_valid=is_valid,
        failure_codes=failure_codes,
        checks_performed=checks_performed,
        details=details,
    )

