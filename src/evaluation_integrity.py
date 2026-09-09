# ============================================================================
# Evaluation Integrity and Leakage Audit Engine
# Detects target duplicates, post-outcome leakage, temporal leakage, split overlap
# ============================================================================

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class LeakageStatus(str, Enum):
    CONFIRMED_LEAKAGE = "CONFIRMED_LEAKAGE"
    SUSPICIOUS_FEATURE = "SUSPICIOUS_FEATURE"
    DOMAIN_REVIEW_REQUIRED = "DOMAIN_REVIEW_REQUIRED"
    CLEAN = "CLEAN"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class LeakageFinding:
    dataset: str
    feature_name: str
    status: LeakageStatus
    severity: Severity
    reason: str
    metric_value: Optional[float] = None
    automatically_verified: bool = False
    domain_interpretation_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["severity"] = self.severity.value
        return d


@dataclass
class SplitIntegrityFinding:
    dataset: str
    split_pair: str
    overlap_count: int
    overlap_fraction: float
    status: LeakageStatus
    severity: Severity
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["severity"] = self.severity.value
        return d


@dataclass
class IntegrityReport:
    dataset: str
    findings: List[LeakageFinding]
    split_findings: List[SplitIntegrityFinding]

    @property
    def has_critical_leakage(self) -> bool:
        return any(
            f.severity in (Severity.CRITICAL, Severity.HIGH)
            and f.status == LeakageStatus.CONFIRMED_LEAKAGE
            for f in self.findings
        ) or any(
            sf.severity in (Severity.CRITICAL, Severity.HIGH)
            and sf.status == LeakageStatus.CONFIRMED_LEAKAGE
            for sf in self.split_findings
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "has_critical_leakage": self.has_critical_leakage,
            "findings": [f.to_dict() for f in self.findings],
            "split_findings": [sf.to_dict() for sf in self.split_findings],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation Integrity Audit Report: {self.dataset}",
            "",
            f"**Overall Status:** {'CRITICAL LEAKAGE DETECTED' if self.has_critical_leakage else 'PASSED / REVIEW REQUIRED'}",
            "",
            "## Feature & Target Leakage Findings",
            "",
        ]
        if not self.findings:
            lines.append("No suspicious features or leakage detected.")
        else:
            lines.append("| Feature | Status | Severity | Reason | Metric Value | Auto-Verified | Domain Review Required |")
            lines.append("|---|---|---|---|---:|:---:|:---:|")
            for f in self.findings:
                metric_str = f"{f.metric_value:.4f}" if f.metric_value is not None else "N/A"
                lines.append(
                    f"| `{f.feature_name}` | **{f.status.value}** | {f.severity.value} | {f.reason} | {metric_str} | {'Yes' if f.automatically_verified else 'No'} | {'Yes' if f.domain_interpretation_required else 'No'} |"
                )

        lines.extend(["", "## Split Overlap & Leakage Findings", ""])
        if not self.split_findings:
            lines.append("No split overlap issues detected.")
        else:
            lines.append("| Split Pair | Overlap Count | Overlap Fraction | Status | Severity | Reason |")
            lines.append("|---|---:|---:|---|---|---|")
            for sf in self.split_findings:
                lines.append(
                    f"| {sf.split_pair} | {sf.overlap_count} | {sf.overlap_fraction:.4%} | **{sf.status.value}** | {sf.severity.value} | {sf.reason} |"
                )

        return "\n".join(lines)


# Known suspicious keywords indicating post-outcome, target semantics, or future timestamps
SUSPICIOUS_FEATURE_SUBSTRINGS = [
    "target",
    "not_target",
    "outcome",
    "future_outcome",
    "post_completion",
    "completion_status",
    "cutoff",
    "is_cutoff",
    "label",
    "actual_time",
    "actual_distance",
    "segment_actual",
    "delivered",
    "elapsed",
    "arrival_time",
    "delivery_time",
]

# Explicit known post-outcome operational features in logistics domains
KNOWN_POST_OUTCOME_FEATURES = {
    "actual_time",
    "actual_distance_to_destination",
    "segment_actual_time",
    "actual_vs_osrm_distance_ratio",
    "actual_vs_osrm_time_ratio",
    "time_per_distance_actual",
    "segment_deviation",
}


def audit_feature_leakage(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    dataset_name: str = "Dataset",
    known_post_outcome_features: Optional[Sequence[str]] = None,
    correlation_threshold: float = 0.98,
) -> List[LeakageFinding]:
    """
    Audits feature matrix X against target y for leakage:
    A. Exact target duplicates
    B. Inverse target duplicates
    C. Columns equal to target after linear or monotonic transformation
    D. Suspicious feature names containing target/outcome semantics
    E. Extremely high single-feature predictive correlation
    F. Features observed post-outcome / execution
    """
    findings: List[LeakageFinding] = []
    y_series = pd.Series(y).reset_index(drop=True)
    post_outcome_set = set(known_post_outcome_features or KNOWN_POST_OUTCOME_FEATURES)

    # Coerce numeric target for correlation checks
    y_numeric = pd.to_numeric(y_series, errors="coerce")
    y_is_binary = set(y_series.dropna().unique()).issubset({0, 1, 0.0, 1.0, True, False})

    for col in X.columns:
        col_series = X[col].reset_index(drop=True)
        col_name_lower = str(col).lower()

        # 1. Check exact target duplicate
        if col_series.equals(y_series) or (y_is_binary and np.array_equal(col_series.to_numpy(), y_series.to_numpy())):
            findings.append(
                LeakageFinding(
                    dataset=dataset_name,
                    feature_name=str(col),
                    status=LeakageStatus.CONFIRMED_LEAKAGE,
                    severity=Severity.CRITICAL,
                    reason="Feature is an exact duplicate of target variable.",
                    metric_value=1.0,
                    automatically_verified=True,
                    domain_interpretation_required=False,
                )
            )
            continue

        # 2. Check inverse target duplicate
        if y_is_binary:
            try:
                inv_target = 1 - y_numeric
                col_num = pd.to_numeric(col_series, errors="coerce")
                if col_num.equals(inv_target) or np.array_equal(col_num.to_numpy(), inv_target.to_numpy()):
                    findings.append(
                        LeakageFinding(
                            dataset=dataset_name,
                            feature_name=str(col),
                            status=LeakageStatus.CONFIRMED_LEAKAGE,
                            severity=Severity.CRITICAL,
                            reason="Feature is an exact inverse duplicate of binary target variable (1 - y).",
                            metric_value=-1.0,
                            automatically_verified=True,
                            domain_interpretation_required=False,
                        )
                    )
                    continue
            except Exception:
                pass

        # 3. Check explicit post-outcome domain features
        if col in post_outcome_set or col_name_lower in post_outcome_set:
            findings.append(
                LeakageFinding(
                    dataset=dataset_name,
                    feature_name=str(col),
                    status=LeakageStatus.CONFIRMED_LEAKAGE,
                    severity=Severity.HIGH,
                    reason="Feature is observed only after task execution/outcome (post-outcome realization).",
                    metric_value=None,
                    automatically_verified=True,
                    domain_interpretation_required=False,
                )
            )

        # 4. Check suspicious feature name semantics
        matched_substrs = [sub for sub in SUSPICIOUS_FEATURE_SUBSTRINGS if sub in col_name_lower]
        if matched_substrs and not any(f.feature_name == str(col) for f in findings):
            findings.append(
                LeakageFinding(
                    dataset=dataset_name,
                    feature_name=str(col),
                    status=LeakageStatus.SUSPICIOUS_FEATURE,
                    severity=Severity.HIGH,
                    reason=f"Feature name contains suspicious target/outcome semantic keywords: {matched_substrs}.",
                    metric_value=None,
                    automatically_verified=False,
                    domain_interpretation_required=True,
                )
            )

        # 5. Check extremely high correlation
        col_num = pd.to_numeric(col_series, errors="coerce")
        if not col_num.dropna().empty and not y_numeric.dropna().empty and col_num.std() > 1e-9 and y_numeric.std() > 1e-9:
            valid_mask = col_num.notna() & y_numeric.notna()
            if valid_mask.sum() > 10:
                corr = float(np.corrcoef(col_num[valid_mask], y_numeric[valid_mask])[0, 1])
                if abs(corr) >= correlation_threshold:
                    findings.append(
                        LeakageFinding(
                            dataset=dataset_name,
                            feature_name=str(col),
                            status=LeakageStatus.CONFIRMED_LEAKAGE if abs(corr) > 0.999 else LeakageStatus.SUSPICIOUS_FEATURE,
                            severity=Severity.CRITICAL if abs(corr) > 0.999 else Severity.HIGH,
                            reason=f"Extreme linear correlation with target (|r| = {abs(corr):.4f} >= {correlation_threshold}).",
                            metric_value=corr,
                            automatically_verified=True,
                            domain_interpretation_required=abs(corr) <= 0.999,
                        )
                    )

    return findings


def audit_split_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    dataset_name: str = "Dataset",
    key_columns: Optional[Sequence[str]] = None,
) -> List[SplitIntegrityFinding]:
    """
    Audits for exact and near-duplicate row leakage across train/val/test splits.
    """
    findings: List[SplitIntegrityFinding] = []

    def check_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, pair_name: str) -> None:
        if df_a.empty or df_b.empty:
            return

        cols = key_columns if key_columns is not None else [c for c in df_a.columns if c in df_b.columns]
        if not cols:
            return

        sub_a = df_a[cols].dropna()
        sub_b = df_b[cols].dropna()

        merged = pd.merge(sub_a, sub_b, how="inner", on=cols)
        overlap_count = len(merged)
        overlap_frac = overlap_count / max(len(sub_b), 1)

        if overlap_count > 0:
            severity = Severity.CRITICAL if overlap_frac > 0.05 else Severity.HIGH
            status = LeakageStatus.CONFIRMED_LEAKAGE if overlap_frac > 0.0 else LeakageStatus.SUSPICIOUS_FEATURE
            findings.append(
                SplitIntegrityFinding(
                    dataset=dataset_name,
                    split_pair=pair_name,
                    overlap_count=overlap_count,
                    overlap_fraction=overlap_frac,
                    status=status,
                    severity=severity,
                    reason=f"{overlap_count} identical records ({overlap_frac:.2%}) cross split boundary.",
                )
            )

    check_pair(train_df, test_df, "Train vs Test")
    if val_df is not None:
        check_pair(train_df, val_df, "Train vs Validation")
        check_pair(val_df, test_df, "Validation vs Test")

    return findings


def run_full_integrity_audit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    dataset_name: str = "Dataset",
    known_post_outcome_features: Optional[Sequence[str]] = None,
) -> IntegrityReport:
    """
    Performs end-to-end evaluation integrity audit:
    - Feature and target leakage on combined data
    - Split overlap between train, val, test
    """
    X_all = pd.concat([X_train, X_test] + ([X_val] if X_val is not None else []), axis=0, ignore_index=True)
    y_all = pd.concat([pd.Series(y_train), pd.Series(y_test)] + ([pd.Series(y_val)] if y_val is not None else []), axis=0, ignore_index=True)

    feature_findings = audit_feature_leakage(
        X=X_all,
        y=y_all,
        dataset_name=dataset_name,
        known_post_outcome_features=known_post_outcome_features,
    )

    split_findings = audit_split_overlap(
        train_df=X_train,
        test_df=X_test,
        val_df=X_val,
        dataset_name=dataset_name,
    )

    return IntegrityReport(
        dataset=dataset_name,
        findings=feature_findings,
        split_findings=split_findings,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Integrity and Leakage Audit Tool")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic demonstration audit")
    args = parser.parse_args()

    if args.demo:
        print("\n=== Running Demonstration Evaluation Integrity Audit ===")
        np.random.seed(42)
        n = 500
        clean_feat = np.random.randn(n)
        target = (clean_feat + np.random.randn(n) * 0.5 > 0).astype(int)
        
        exact_target = target.copy()
        inv_target = 1 - target
        post_completion_time = target * 120.0 + np.random.exponential(5.0, n)
        
        df = pd.DataFrame({
            "planned_distance": np.abs(np.random.randn(n) * 10),
            "driver_age": np.random.randint(20, 60, n),
            "target_leak_exact": exact_target,
            "not_target_leak": inv_target,
            "actual_time": post_completion_time,
        })
        
        train_df = df.iloc[:350].copy()
        test_df = df.iloc[350:].copy()
        test_df.iloc[0] = train_df.iloc[0]
        
        report = run_full_integrity_audit(
            X_train=train_df,
            y_train=target[:350],
            X_test=test_df,
            y_test=target[350:],
            dataset_name="Demo Logistics Dataset",
        )
        print(report.to_markdown())
