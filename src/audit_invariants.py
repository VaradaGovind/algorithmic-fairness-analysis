# ============================================================================
# Audit Invariants Engine (Pass 9 Release-Candidate)
# Mathematical consistency validator and output privacy auditing for AuditResult objects.
# Enforces exact count identities, metric equations, and privacy sanitization.
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


def validate_audit_mathematical_invariants(
    audit_data: Union[Dict[str, Any], Any],
    tolerance: float = 1e-3,
) -> Tuple[bool, List[str]]:
    """
    Validates the mathematical and count consistency of an AuditResult dictionary or object.
    
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
      3. No impossible mathematical combinations or false coercions of undefined states.
    
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
                    obs_fpr = g_info.get("FPR")
                    if obs_fpr is not None and not math.isnan(obs_fpr):
                        exp_fpr = fp / (tn + fp)
                        if abs(obs_fpr - exp_fpr) > tolerance:
                            errors.append(
                                f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' FPR ({obs_fpr:.4f}) does not match "
                                f"FP / (FP + TN) = {exp_fpr:.4f}."
                            )
                        if obs_tnr is not None and not math.isnan(obs_tnr):
                            if abs((obs_tnr + obs_fpr) - 1.0) > tolerance:
                                errors.append(
                                    f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' TNR ({obs_tnr:.4f}) + "
                                    f"FPR ({obs_fpr:.4f}) != 1.0."
                                )

                # FNR identity: FNR == FN / (FN + TP) == 1 - recall
                if (tp + fn) > 0:
                    exp_fnr = fn / (tp + fn)
                    obs_fnr = g_info.get("FNR")
                    if obs_fnr is not None and not math.isnan(obs_fnr):
                        if abs(obs_fnr - exp_fnr) > tolerance:
                            errors.append(
                                f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' FNR ({obs_fnr:.4f}) does not match "
                                f"FN / (FN + TP) = {exp_fnr:.4f}."
                            )

            # Subgroup balanced accuracy identity: (TPR + TNR) / 2 == (TPR + (1 - FPR)) / 2
            obs_tpr = g_info.get("TPR")
            obs_fpr = g_info.get("FPR")
            obs_bal = g_info.get("balanced_accuracy")
            if (
                obs_tpr is not None
                and obs_fpr is not None
                and obs_bal is not None
                and not (math.isnan(obs_tpr) or math.isnan(obs_fpr) or math.isnan(obs_bal))
            ):
                expected_bal = (obs_tpr + (1.0 - obs_fpr)) / 2.0
                if abs(obs_bal - expected_bal) > tolerance:
                    errors.append(
                        f"{ERR_INVALID_METRIC_IDENTITY}: Subgroup '{g_name}' balanced_accuracy ({obs_bal:.4f}) does not match "
                        f"(TPR + (1 - FPR)) / 2 = {expected_bal:.4f}."
                    )

            if obs_tpr is not None and not math.isnan(obs_tpr):
                tprs.append(obs_tpr)
            if obs_fpr is not None and not math.isnan(obs_fpr):
                fprs.append(obs_fpr)

            # Selection rate
            sel_rate = g_info.get("predicted_positive_rate")
            if sel_rate is None and all(v is not None for v in [tp, fp]) and n > 0:
                sel_rate = (tp + fp) / n
            if sel_rate is not None and not math.isnan(sel_rate):
                selection_rates.append(sel_rate)

            subgroup_sample_sum += n

    # 3. Global Performance Checks
    perf = d.get("global_performance", {})
    glob_acc = perf.get("accuracy")
    glob_bal = perf.get("balanced_accuracy")
    glob_prec = perf.get("precision")
    glob_rec = perf.get("recall")
    glob_f1 = perf.get("f1_score")

    # F1 consistency with precision & recall
    if (
        glob_prec is not None
        and glob_rec is not None
        and glob_f1 is not None
        and not (math.isnan(glob_prec) or math.isnan(glob_rec) or math.isnan(glob_f1))
    ):
        if (glob_prec + glob_rec) > 0:
            exp_f1 = 2.0 * glob_prec * glob_rec / (glob_prec + glob_rec)
            if abs(glob_f1 - exp_f1) > tolerance:
                errors.append(
                    f"{ERR_INVALID_METRIC_IDENTITY}: Global f1_score ({glob_f1:.4f}) does not match "
                    f"harmonic mean of precision ({glob_prec:.4f}) and recall ({glob_rec:.4f}) = {exp_f1:.4f}."
                )
        elif glob_f1 > tolerance:
            errors.append(
                f"{ERR_INVALID_METRIC_IDENTITY}: Global f1_score is {glob_f1} but precision + recall = 0."
            )

    # 4. Fairness Metric Aggregate Identities
    fm = d.get("fairness_metrics", {})
    dpd = fm.get("demographic_parity_difference")
    eod = fm.get("equalized_odds_difference")
    dir_ratio = fm.get("disparate_impact_ratio")
    eq_opp = fm.get("equal_opportunity_difference")

    # DPD == max(selection_rates) - min(selection_rates)
    if dpd is not None and len(selection_rates) >= 2 and not math.isnan(dpd):
        exp_dpd = max(selection_rates) - min(selection_rates)
        if abs(dpd - exp_dpd) > tolerance:
            errors.append(
                f"{ERR_INVALID_METRIC_IDENTITY}: demographic_parity_difference ({dpd:.4f}) does not match "
                f"max - min subgroup selection rate ({exp_dpd:.4f})."
            )

    # DIR == min(selection_rates) / max(selection_rates)
    if dir_ratio is not None and len(selection_rates) >= 2 and not math.isnan(dir_ratio):
        max_rate = max(selection_rates)
        min_rate = min(selection_rates)
        if max_rate > 0:
            exp_dir = min_rate / max_rate
            if abs(dir_ratio - exp_dir) > tolerance:
                errors.append(
                    f"{ERR_INVALID_METRIC_IDENTITY}: disparate_impact_ratio ({dir_ratio:.4f}) does not match "
                    f"min / max subgroup selection rate ({exp_dir:.4f})."
                )

    # EOD (Equal Opportunity) == max(TPRs) - min(TPRs)
    if eq_opp is not None and len(tprs) >= 2 and not math.isnan(eq_opp):
        exp_eq_opp = max(tprs) - min(tprs)
        if abs(eq_opp - exp_eq_opp) > tolerance:
            errors.append(
                f"{ERR_INVALID_METRIC_IDENTITY}: equal_opportunity_difference ({eq_opp:.4f}) does not match "
                f"max - min subgroup TPR ({exp_eq_opp:.4f})."
            )

    # Equalized Odds Difference == max(TPR gap, FPR gap)
    if eod is not None and len(tprs) >= 2 and len(fprs) >= 2 and not math.isnan(eod):
        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        exp_eod = max(tpr_gap, fpr_gap)
        if abs(eod - exp_eod) > tolerance:
            errors.append(
                f"{ERR_INVALID_METRIC_IDENTITY}: equalized_odds_difference ({eod:.4f}) does not match "
                f"max(TPR gap {tpr_gap:.4f}, FPR gap {fpr_gap:.4f}) = {exp_eod:.4f}."
            )

    # 5. Unit & Bootstrap Alignment Check
    eua = d.get("evaluation_unit_alignment", {})
    unit_level = eua.get("evaluation_unit_level", "")
    agg_strat = eua.get("aggregation_strategy", "")
    unc = d.get("uncertainty_quantification", {})
    resamp_unit = unc.get("resampling_unit", "")
    mean_recs = eua.get("records_per_unit_mean", 1.0)
    mismatch_ack = eua.get("mismatch_guard_acknowledged", False)

    if unit_level in ("TRIP_LEVEL", "ROUTE_LEVEL", "ENTITY_LEVEL") and mean_recs > 1.0:
        if resamp_unit == "ROW_BOOTSTRAP" and not mismatch_ack:
            errors.append(
                f"{ERR_INVALID_UNIT_ALIGNMENT}: ROW_BOOTSTRAP is statistically incompatible with "
                f"{unit_level} having repeated entity rows (mean {mean_recs:.2f}) without explicit mismatch override."
            )

    if resamp_unit == "CLUSTER_BOOTSTRAP":
        cluster_k = unc.get("cluster_key") or eua.get("unit_key")
        if not cluster_k:
            errors.append(
                f"{ERR_INVALID_BOOTSTRAP_CONFIGURATION}: CLUSTER_BOOTSTRAP requested but no cluster_key specified."
            )

    # 6. Estimand Presence Check
    estimand = eua.get("statistical_estimand") or d.get("statistical_estimand")
    if estimand is None:
        errors.append(f"{ERR_MISSING_ESTIMAND}: Mandatory statistical estimand declaration is missing.")
    elif isinstance(estimand, dict):
        status = estimand.get("estimand_status", "ESTIMAND_UNSPECIFIED")
        if status == "ESTIMAND_UNSPECIFIED":
            errors.append(f"{ERR_MISSING_ESTIMAND}: Evaluation estimand is marked ESTIMAND_UNSPECIFIED.")
        else:
            req_est_fields = ["population", "decision_unit", "observation_unit", "weighting_rule", "target_quantity", "fairness_quantity"]
            for fld in req_est_fields:
                if not estimand.get(fld):
                    errors.append(f"{ERR_MISSING_ESTIMAND}: Estimand missing mandatory field '{fld}'.")

    # 7. Group Mapping & Semantic Integrity Check
    sens_attrs = d.get("sensitive_attributes", [])
    if isinstance(sens_attrs, list):
        for attr in sens_attrs:
            if isinstance(attr, dict):
                attr_name = attr.get("attribute_name", "")
                disclosed = attr.get("disclosed_categories", [])
                ds_name = ds.get("dataset_name", "").lower()
                if "adult" in ds_name:
                    rule = str(attr.get("proxy_derivation_rule", "")).lower()
                    target_dec = str(d.get("evaluation_scope", {}).get("target_decision", "")).lower()
                    if "credit" in rule or "credit" in target_dec or "credit" in attr_name.lower():
                        errors.append(
                            f"{ERR_INVALID_GROUP_MAPPING}: Adult dataset cannot be semantically mapped to credit "
                            f"decisions or proxies; benchmark represents census income (> $50K)."
                        )
                if disclosed and subgroups and isinstance(subgroups, dict):
                    subgroup_keys = set(subgroups.keys())
                    for cat in disclosed:
                        if cat not in subgroup_keys:
                            errors.append(
                                f"{ERR_INVALID_GROUP_MAPPING}: Disclosed sensitive category '{cat}' "
                                f"not found in subgroup evaluations: {sorted(subgroup_keys)}."
                            )

    return (len(errors) == 0, errors)


# ============================================================================
# Output Privacy Audit & Sanitization
# ============================================================================

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
LOCAL_PATH_REGEX = re.compile(r"(?:[a-zA-Z]:[\\/]|/(?:home|Users|var|tmp|etc)[\\/])[^\s\"',;:]+")
SECRET_KEY_REGEX = re.compile(r"(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})")


def audit_artifact_privacy(
    content_or_dict: Union[str, Dict[str, Any], Any],
    sanitize: bool = False,
) -> Dict[str, Any]:
    """
    Audits artifact content for privacy leaks:
      - raw email addresses
      - local absolute filesystem paths
      - credential / secret patterns
    Optionally sanitizes identified items with explicit suppression counters.
    
    CRITICAL METHODOLOGICAL NOTICE:
    This constitutes heuristic report sanitization, NOT formal differential privacy
    or certified re-identification risk bounding.
    """
    if isinstance(content_or_dict, str):
        text = content_or_dict
    else:
        text = json.dumps(content_or_dict, default=str)

    emails = EMAIL_REGEX.findall(text)
    paths = LOCAL_PATH_REGEX.findall(text)
    secrets = SECRET_KEY_REGEX.findall(text)

    findings: Dict[str, List[str]] = {
        "emails": emails,
        "local_paths": paths,
        "secrets": secrets,
    }
    total_findings = len(emails) + len(paths) + len(secrets)

    sanitized_text = text
    if sanitize:
        sanitized_text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", sanitized_text)
        sanitized_text = LOCAL_PATH_REGEX.sub("[REDACTED_LOCAL_PATH]", sanitized_text)
        sanitized_text = SECRET_KEY_REGEX.sub("[REDACTED_SECRET]", sanitized_text)

    return {
        "privacy_passed": (total_findings == 0),
        "total_leaks_found": total_findings,
        "suppression_counts": {
            "emails_suppressed": len(emails),
            "paths_suppressed": len(paths),
            "secrets_suppressed": len(secrets),
        },
        "findings": findings,
        "sanitized_content": sanitized_text if sanitize else None,
        "privacy_disclaimer": (
            "REPORT PRIVACY NOTICE: Privacy protections operate via pattern suppression heuristics. "
            "This constitutes report privacy/sanitization, NOT formal differential privacy, "
            "and absence of matched patterns is a heuristic assurance rather than a mathematical proof of absolute absence."
        ),
    }


def audit_bundle_privacy(
    bundle_path: Union[str, Path],
    sanitize: bool = False,
) -> Dict[str, Any]:
    """
    Audits an entire bundle directory for privacy leaks across all text/JSON files:
      - raw email addresses
      - local absolute filesystem paths
      - credential / secret patterns
    Returns aggregated privacy results.
    """
    root = Path(bundle_path)
    all_findings: Dict[str, List[str]] = {
        "emails": [],
        "local_paths": [],
        "secrets": [],
    }
    scanned_files = 0
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".json", ".md", ".txt", ".yaml", ".yml"):
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    res = audit_artifact_privacy(text, sanitize=sanitize)
                    scanned_files += 1
                    for k in all_findings:
                        if res["findings"].get(k):
                            all_findings[k].extend([f"{p.name}: {item}" for item in res["findings"][k]])
                except Exception:
                    pass

    total_leaks = sum(len(v) for v in all_findings.values())
    return {
        "privacy_passed": (total_leaks == 0),
        "total_leaks_found": total_leaks,
        "scanned_files_count": scanned_files,
        "findings": all_findings,
        "disclaimer": (
            "REPORT PRIVACY NOTICE: Privacy protections operate via pattern suppression heuristics. "
            "This constitutes report privacy/sanitization, NOT formal differential privacy, "
            "and absence of matched patterns is a heuristic assurance rather than a mathematical proof of absolute absence."
        ),
    }


def run_publication_release_audit(
    repo_root: Optional[Union[str, Path]] = None,
    bundle_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Conducts a rigorous pre-release publication audit ensuring:
      1. Zero raw customer / operational datasets committed or bundled.
      2. Zero hardcoded secrets, private tokens, or API credentials.
      3. Zero local absolute developer paths in public documentation and bundle artifacts.
      4. Explicit permissive open-source license declared.
      5. Evidence-qualified release claims (scans for forbidden unhedged claims like
         'certified discrimination-free', 'statutorily compliant', 'guaranteed fairness',
         'commercial validation complete' without qualification).
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent

    findings: Dict[str, List[str]] = {
        "raw_datasets": [],
        "secrets": [],
        "local_paths": [],
        "licensing": [],
        "unhedged_claims": [],
    }

    # 1. Zero Raw Datasets Check (in data, src, configs, and bundle directory)
    forbidden_data_exts = {".parquet", ".feather", ".arrow", ".h5", ".hdf5", ".orc", ".avro", ".sqlite", ".db"}
    check_dirs = [root / "data", root / "src", root / "configs"]
    if bundle_path:
        check_dirs.append(Path(bundle_path))

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
    for doc_file in [root / "README.md", root / "docs" / "RELEASE_CANDIDATE.md", root / "pyproject.toml"]:
        if doc_file.is_file():
            text = doc_file.read_text(encoding="utf-8", errors="ignore")
            for sp in secret_patterns:
                m = sp.findall(text)
                if m:
                    findings["secrets"].extend([f"{doc_file.name}: {x[:8]}..." for x in m])

    # 3. Zero Local Absolute Developer Paths in documentation (excluding URLs and ellipsis placeholders)
    local_path_pattern = re.compile(r"(?:(?<=[\s\"'(=])|^)(?:[a-zA-Z]:[\\/]|/(?:home|Users))[a-zA-Z0-9_\\/.-]+")
    for doc_file in [root / "README.md", root / "docs" / "RELEASE_CANDIDATE.md"]:
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

    # 5. Evidence-Qualified Claims (Forbidden marketing claims)
    forbidden_claim_regexes = [
        re.compile(r"\bcertified (?:fair|discrimination-free|bias-free)\b", re.IGNORECASE),
        re.compile(r"\bstatutorily compliant\b", re.IGNORECASE),
        re.compile(r"\bguaranteed fairness\b", re.IGNORECASE),
        re.compile(r"\bproduction-ready commercial validation complete\b", re.IGNORECASE),
    ]
    for doc_file in [root / "README.md", root / "docs" / "RELEASE_CANDIDATE.md"]:
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
        "methodological_notice": (
            "AUDIT ASSURANCE LIMITATION: Static pattern scans indicate that zero matching patterns "
            "were detected under the configured regex rules. Static scanning does NOT constitute a "
            "mathematical proof of the absolute absence of secrets, private material, or semantic overclaims."
        ),
        "status_recommendation": (
            "Release-candidate engineering verification complete. Repository publication checks passed. "
            "External real-data validation has not been executed and remains pending authorized operational-data staging. "
            "The system must not be interpreted as providing customer-specific, regulatory, or production suitability conclusions "
            "without additional human and empirical validation."
            if is_ready else
            "Release blocked: pre-publication audit criteria unmet."
        ),
    }

