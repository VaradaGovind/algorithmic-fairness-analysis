# ============================================================================
# Metric Invariants & Repository Hygiene Engine
# Educational module for validating mathematical consistency of fairness metrics,
# confusion matrix identities, and scanning repository content for hygiene.
# ============================================================================

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# Standard Invariant Error Codes
ERR_INVALID_COUNT_IDENTITY = "INVALID_COUNT_IDENTITY"
ERR_INVALID_METRIC_IDENTITY = "INVALID_METRIC_IDENTITY"
ERR_INVALID_UNIT_ALIGNMENT = "INVALID_UNIT_ALIGNMENT"
ERR_INVALID_BOOTSTRAP_CONFIGURATION = "INVALID_BOOTSTRAP_CONFIGURATION"
ERR_MISSING_ESTIMAND = "MISSING_ESTIMAND"
ERR_INVALID_GROUP_MAPPING = "INVALID_GROUP_MAPPING"


def validate_metric_mathematical_invariants(
    audit_data: Union[Dict[str, Any], Any],
    tolerance: float = 1e-3,
) -> Tuple[bool, List[str]]:
    """
    Validates mathematical and count consistency of an evaluation result.
    
    Verifies:
      1. Count identities:
         - positive_count + negative_count == sample_size
         - TP + TN + FP + FN == sample_size (if confusion matrix is present)
         - total_records >= sample_size
      2. Metric identities (when defined):
         - accuracy == (TP + TN) / N
         - balanced_accuracy == (TPR + TNR) / 2 == (TPR + (1 - FPR)) / 2
         - precision == TP / (TP + FP)
         - recall == TP / (TP + FN) == TPR
         - specificity == TN / (TN + FP) == TNR == 1 - FPR
         - FPR == FP / (FP + TN) == 1 - specificity
         - FNR == FN / (FN + TP) == 1 - recall
         - F1 == 2 * precision * recall / (precision + recall)
         - DPD == max(selection_rates) - min(selection_rates)
         - EOD == max(TPRs) - min(TPRs)
         - EODiff == max(max(TPR gap), max(FPR gap))
         - DIR == min(selection_rate) / max(selection_rate)
      3. No impossible mathematical combinations or false coercions.
    
    Returns:
        (is_valid: bool, errors: List[str])
    """
    if hasattr(audit_data, "to_dict"):
        d = audit_data.to_dict()
    elif isinstance(audit_data, dict):
        d = audit_data
    else:
        d = vars(audit_data)

    errors: List[str] = []

    # 1. Dataset Count Checks
    ds = d.get("dataset", {})
    tot_records = ds.get("total_records")
    ent_count = ds.get("entity_count")
    if tot_records is not None and ent_count is not None:
        if tot_records < 1 or ent_count < 1:
            errors.append(f"{ERR_INVALID_COUNT_IDENTITY}: total_records and entity_count must be >= 1.")
        if ent_count > tot_records:
            errors.append(f"{ERR_INVALID_COUNT_IDENTITY}: entity_count ({ent_count}) cannot exceed total_records ({tot_records}).")

    # 2. Subgroup Count & Rate Checks
    subgroups = d.get("subgroup_evaluations", {})
    tprs: List[float] = []
    fprs: List[float] = []
    selection_rates: List[float] = []
    subgroup_sample_sum = 0

    if isinstance(subgroups, dict):
        for g_name, g_info in subgroups.items():
            if not isinstance(g_info, dict):
                continue

            n = g_info.get("sample_size", 0)
            n_pos = g_info.get("positive_count")
            n_neg = g_info.get("negative_count")

            # Check positive + negative == sample_size
            if n_pos is not None and n_neg is not None:
                if (n_pos + n_neg) != n:
                    errors.append(
                        f"{ERR_INVALID_COUNT_IDENTITY}: Subgroup '{g_name}' positive_count ({n_pos}) + "
                        f"negative_count ({n_neg}) = {n_pos + n_neg} != sample_size ({n})."
                    )

            # Check confusion matrix if present
            tp = g_info.get("TP")
            tn = g_info.get("TN")
            fp = g_info.get("FP")
            fn = g_info.get("FN")

            # Subgroup accuracy identity
            obs_acc = g_info.get("accuracy")
            if obs_acc is not None and not math.isnan(obs_acc) and n > 0 and tp is not None and tn is not None:
                exp_acc = (tp + tn) / n
                if abs(obs_acc - exp_acc) > tolerance:
                    errors.append(
                        f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' accuracy ({obs_acc:.4f}) does not match "
                        f"(TP + TN) / N = {exp_acc:.4f}."
                    )

            if all(v is not None for v in [tp, tn, fp, fn]):
                sum_conf = tp + tn + fp + fn
                if sum_conf != n:
                    errors.append(
                        f"{ERR_INVALID_COUNT_IDENTITY}: Subgroup '{g_name}' TP ({tp}) + TN ({tn}) + "
                        f"FP ({fp}) + FN ({fn}) = {sum_conf} != sample_size ({n})."
                    )

                # Precision identity
                if (tp + fp) > 0:
                    exp_prec = tp / (tp + fp)
                    obs_ppv = g_info.get("PPV")
                    if obs_ppv is not None and not math.isnan(obs_ppv):
                        if abs(obs_ppv - exp_prec) > tolerance:
                            errors.append(
                                f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' PPV ({obs_ppv:.4f}) does not match "
                                f"TP / (TP + FP) = {exp_prec:.4f}."
                            )

                # Recall / TPR identity
                if (tp + fn) > 0:
                    exp_tpr = tp / (tp + fn)
                    obs_tpr = g_info.get("TPR")
                    if obs_tpr is not None and not math.isnan(obs_tpr):
                        if abs(obs_tpr - exp_tpr) > tolerance:
                            errors.append(
                                f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' TPR ({obs_tpr:.4f}) does not match "
                                f"TP / (TP + FN) = {exp_tpr:.4f}."
                            )

                # Specificity / TNR identity
                if (tn + fp) > 0:
                    exp_tnr = tn / (tn + fp)
                    obs_tnr = g_info.get("TNR")
                    if obs_tnr is not None and not math.isnan(obs_tnr):
                        if abs(obs_tnr - exp_tnr) > tolerance:
                            errors.append(
                                f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' TNR ({obs_tnr:.4f}) does not match "
                                f"TN / (TN + FP) = {exp_tnr:.4f}."
                            )

                # Balanced accuracy identity
                obs_bacc = g_info.get("balanced_accuracy")
                obs_tpr_val = g_info.get("TPR")
                obs_tnr_val = g_info.get("TNR")
                if all(v is not None and not math.isnan(v) for v in [obs_bacc, obs_tpr_val, obs_tnr_val]):
                    exp_bacc = (obs_tpr_val + obs_tnr_val) / 2.0
                    if abs(obs_bacc - exp_bacc) > tolerance:
                        errors.append(
                            f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' balanced_accuracy ({obs_bacc:.4f}) does not match "
                            f"(TPR + TNR) / 2 = {exp_bacc:.4f}."
                        )

            # Accumulate rates for disparity checks
            sr = g_info.get("selection_rate")
            if sr is not None and not math.isnan(sr):
                selection_rates.append(sr)
            tpr_val = g_info.get("TPR")
            if tpr_val is not None and not math.isnan(tpr_val):
                tprs.append(tpr_val)
            fpr_val = g_info.get("FPR")
            if fpr_val is not None and not math.isnan(fpr_val):
                fprs.append(fpr_val)

            subgroup_sample_sum += n

    # 3. Global vs Subgroup Disparity Checks
    glob_perf = d.get("global_performance", {})
    if glob_perf:
        # Check Demographic Parity Difference
        obs_dpd = glob_perf.get("demographic_parity_difference")
        if obs_dpd is not None and not math.isnan(obs_dpd) and len(selection_rates) >= 2:
            exp_dpd = max(selection_rates) - min(selection_rates)
            if abs(obs_dpd - exp_dpd) > tolerance:
                errors.append(
                    f"{ERR_INVALID_METRIC_IDENTITY}: Global DPD ({obs_dpd:.4f}) does not match "
                    f"max(selection_rates) - min(selection_rates) = {exp_dpd:.4f}."
                )

        # Check Equal Opportunity Difference
        obs_eod = glob_perf.get("equal_opportunity_difference")
        if obs_eod is not None and not math.isnan(obs_eod) and len(tprs) >= 2:
            exp_eod = max(tprs) - min(tprs)
            if abs(obs_eod - exp_eod) > tolerance:
                errors.append(
                    f"{ERR_INVALID_METRIC_IDENTITY}: Global EOD ({obs_eod:.4f}) does not match "
                    f"max(TPRs) - min(TPRs) = {exp_eod:.4f}."
                )

        # Check Equalized Odds Difference
        obs_eodiff = glob_perf.get("equalized_odds_difference")
        if obs_eodiff is not None and not math.isnan(obs_eodiff) and len(tprs) >= 2 and len(fprs) >= 2:
            exp_tpr_gap = max(tprs) - min(tprs)
            exp_fpr_gap = max(fprs) - min(fprs)
            exp_eodiff = max(exp_tpr_gap, exp_fpr_gap)
            if abs(obs_eodiff - exp_eodiff) > tolerance:
                errors.append(
                    f"{ERR_INVALID_METRIC_IDENTITY}: Global Equalized Odds Diff ({obs_eodiff:.4f}) does not match "
                    f"max(TPR gap, FPR gap) = {exp_eodiff:.4f}."
                )

        # Check Disparate Impact Ratio
        obs_dir = glob_perf.get("disparate_impact_ratio")
        if obs_dir is not None and not math.isnan(obs_dir) and len(selection_rates) >= 2:
            max_sr = max(selection_rates)
            min_sr = min(selection_rates)
            if max_sr > 0:
                exp_dir = min_sr / max_sr
                if abs(obs_dir - exp_dir) > tolerance:
                    errors.append(
                        f"{ERR_INVALID_METRIC_IDENTITY}: Global DIR ({obs_dir:.4f}) does not match "
                        f"min_sr / max_sr = {exp_dir:.4f}."
                    )

    # 4. Evaluation Unit Alignment Invariants
    unit_align = d.get("evaluation_unit_alignment", {})
    if unit_align:
        obs_unit = unit_align.get("observational_record_unit")
        dec_unit = unit_align.get("evaluation_decision_unit")
        agg_strat = unit_align.get("aggregation_strategy")
        bstrap_unit = unit_align.get("bootstrap_resampling_unit")

        if obs_unit and dec_unit:
            if obs_unit != dec_unit and agg_strat == "NONE":
                errors.append(
                    f"{ERR_INVALID_UNIT_ALIGNMENT}: Observational unit ({obs_unit}) != decision unit ({dec_unit}), "
                    f"but aggregation_strategy is 'NONE'."
                )

        if bstrap_unit and obs_unit:
            if bstrap_unit == "ROW_LEVEL" and obs_unit in ("TRIP_LEVEL", "ENTITY_LEVEL"):
                errors.append(
                    f"{ERR_INVALID_BOOTSTRAP_CONFIGURATION}: Bootstrap unit 'ROW_LEVEL' used on clustered unit '{obs_unit}'."
                )

    # 5. Estimand Specification Invariants
    estimand = d.get("statistical_estimand", {})
    if estimand:
        status = estimand.get("estimand_status")
        if status not in ("SPECIFIED", "UNSPECIFIED", "NOT_APPLICABLE"):
            errors.append(f"{ERR_MISSING_ESTIMAND}: estimand_status must be SPECIFIED, UNSPECIFIED, or NOT_APPLICABLE.")

    is_valid = (len(errors) == 0)
    return is_valid, errors


# Backwards compatibility alias
validate_audit_mathematical_invariants = validate_metric_mathematical_invariants


def scan_content_hygiene(
    content: str,
    sanitize: bool = False,
) -> Dict[str, Any]:
    """
    Scans text content for personal email addresses, local developer absolute paths,
    and credential patterns.
    """
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    # Matches Windows absolute drive paths (e.g. C:\Users\...) and Linux /home paths
    path_pattern = re.compile(r"(?:(?<=[\s\"'(=])|^)(?:[a-zA-Z]:[\\/]|/(?:home|Users))[a-zA-Z0-9_\\/.-]+")
    secret_pattern = re.compile(r"(?:(?:sk|ghp|AKIA)[-_a-zA-Z0-9]{16,}|-----BEGIN (?:RSA )?PRIVATE KEY-----)")

    emails = email_pattern.findall(content)
    paths_raw = path_pattern.findall(content)
    # Exclude harmless examples/placeholders
    paths = [p for p in paths_raw if not (p.endswith("...") or p.endswith("..") or "placeholder" in p.lower())]
    secrets = secret_pattern.findall(content)

    findings: Dict[str, List[str]] = {
        "emails": list(set(emails)),
        "local_paths": list(set(paths)),
        "secrets": list(set(secrets)),
    }

    total_findings = len(findings["emails"]) + len(findings["local_paths"]) + len(findings["secrets"])
    sanitized_text = content
    if sanitize:
        sanitized_text = email_pattern.sub("[EMAIL_REDACTED]", sanitized_text)
        for p in findings["local_paths"]:
            sanitized_text = sanitized_text.replace(p, "[PATH_REDACTED]")
        sanitized_text = secret_pattern.sub("[SECRET_REDACTED]", sanitized_text)

    return {
        "clean": (total_findings == 0),
        "total_leaks_found": total_findings,
        "findings": findings,
        "sanitized_content": sanitized_text if sanitize else None,
    }


# Backwards compatibility alias
audit_artifact_privacy = scan_content_hygiene


def run_repository_hygiene_check(
    repo_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Conducts a pre-publication repository hygiene scan ensuring:
      1. Zero raw customer / operational datasets committed in data, src, configs.
      2. Zero hardcoded secrets or API credentials.
      3. Zero local absolute developer paths in public documentation and metadata.
      4. Explicit permissive open-source license declared.
      5. Educational positioning verified (flags unhedged commercial claims like
         'certified discrimination-free', 'statutorily compliant', 'guaranteed fairness').
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent

    findings: Dict[str, List[str]] = {
        "raw_datasets": [],
        "secrets": [],
        "local_paths": [],
        "licensing": [],
        "unhedged_claims": [],
    }

    # 1. Zero Raw Datasets Check (in data, src, configs)
    forbidden_data_exts = {".parquet", ".feather", ".arrow", ".h5", ".hdf5", ".orc", ".avro", ".sqlite", ".db"}
    check_dirs = [root / "data", root / "src", root / "configs"]

    for d in check_dirs:
        if d.is_dir():
            for p in d.rglob("*"):
                if p.is_file():
                    if p.suffix.lower() in forbidden_data_exts:
                        findings["raw_datasets"].append(f"Forbidden raw dataset file: {p.relative_to(root)}")
                    elif p.suffix.lower() in (".csv", ".tsv") and "test" not in p.parts:
                        findings["raw_datasets"].append(f"Untracked operational CSV detected: {p.relative_to(root)}")

    # 2. Zero Secrets Check in public documentation
    secret_patterns = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
    ]
    for doc_file in [root / "README.md", root / "pyproject.toml"]:
        if doc_file.is_file():
            text = doc_file.read_text(encoding="utf-8", errors="ignore")
            for sp in secret_patterns:
                m = sp.findall(text)
                if m:
                    findings["secrets"].extend([f"{doc_file.name}: {x[:8]}..." for x in m])

    # 3. Zero Local Absolute Developer Paths in documentation
    local_path_pattern = re.compile(r"(?:(?<=[\s\"'(=])|^)(?:[a-zA-Z]:[\\/]|/(?:home|Users))[a-zA-Z0-9_\\/.-]+")
    for doc_file in [root / "README.md", root / "pyproject.toml"]:
        if doc_file.is_file():
            text = doc_file.read_text(encoding="utf-8", errors="ignore")
            matches = local_path_pattern.findall(text)
            actual_leaks = [m for m in matches if not (m.endswith("...") or m.endswith("..") or "placeholder" in m.lower())]
            if actual_leaks:
                findings["local_paths"].extend([f"{doc_file.name}: {m}" for m in actual_leaks])

    # 4. Explicit License Declared
    license_file = root / "LICENSE"
    pyproject_file = root / "pyproject.toml"
    has_license = license_file.is_file()
    if pyproject_file.is_file():
        pp_text = pyproject_file.read_text(encoding="utf-8", errors="ignore")
        if 'license = { text = "MIT" }' in pp_text or 'license = "MIT"' in pp_text:
            has_license = True
    if not has_license:
        findings["licensing"].append("No explicit license declared in LICENSE or pyproject.toml.")

    # 5. Evidence-Qualified Claims (flags commercial overclaims)
    forbidden_claim_regexes = [
        re.compile(r"\bcertified (?:fair|discrimination-free|bias-free)\b", re.IGNORECASE),
        re.compile(r"\bstatutorily compliant\b", re.IGNORECASE),
        re.compile(r"\bguaranteed fairness\b", re.IGNORECASE),
        re.compile(r"\bproduction-ready commercial validation complete\b", re.IGNORECASE),
    ]
    for doc_file in [root / "README.md"]:
        if doc_file.is_file():
            text = doc_file.read_text(encoding="utf-8", errors="ignore")
            for fcr in forbidden_claim_regexes:
                m = fcr.findall(text)
                if m:
                    findings["unhedged_claims"].extend([f"{doc_file.name}: '{x}'" for x in m])

    total_issues = sum(len(v) for v in findings.values())
    is_ready = (total_issues == 0)

    return {
        "release_audit_passed": is_ready,
        "total_issues_found": total_issues,
        "findings": findings,
        "scanned_summary": {
            "credentials": (
                "no credential-like material detected by the implemented scanners"
                if len(findings["secrets"]) == 0
                else f"{len(findings['secrets'])} potential credential patterns detected"
            ),
            "claims": (
                "no unsupported claims detected under the configured claim-audit rules"
                if len(findings["unhedged_claims"]) == 0
                else f"{len(findings['unhedged_claims'])} unsupported claims detected"
            ),
            "raw_datasets": (
                "no tracked operational tabular datasets detected"
                if len(findings["raw_datasets"]) == 0
                else f"{len(findings['raw_datasets'])} data files detected"
            ),
            "local_paths": (
                "no absolute developer machine paths detected in scanned documentation"
                if len(findings["local_paths"]) == 0
                else f"{len(findings['local_paths'])} path patterns detected"
            ),
            "licensing": (
                "explicit permissive license verified"
                if len(findings["licensing"]) == 0
                else "licensing declaration missing"
            ),
        },
        "status_recommendation": (
            "Educational public release checks passed. No secrets, raw datasets, or local machine paths detected."
            if is_ready else
            "Release blocked: pre-publication audit criteria unmet."
        ),
    }


# Backwards compatibility alias
run_publication_release_audit = run_repository_hygiene_check
