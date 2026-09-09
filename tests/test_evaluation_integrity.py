# ============================================================================
# Unit Tests: Evaluation Integrity & Leakage Detection
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from src.evaluation_integrity import (
    LeakageStatus,
    Severity,
    audit_feature_leakage,
    audit_split_overlap,
    run_full_integrity_audit,
)


def test_exact_target_duplicate_detected():
    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    X = pd.DataFrame({
        "clean_feature": np.random.randn(8),
        "target_clone": y.copy(),
    })
    findings = audit_feature_leakage(X, y)
    target_findings = [f for f in findings if f.feature_name == "target_clone"]
    assert len(target_findings) > 0
    assert target_findings[0].status == LeakageStatus.CONFIRMED_LEAKAGE
    assert target_findings[0].severity == Severity.CRITICAL


def test_inverse_target_duplicate_detected():
    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    X = pd.DataFrame({
        "clean_feature": np.random.randn(8),
        "not_target": 1 - y,
    })
    findings = audit_feature_leakage(X, y)
    inv_findings = [f for f in findings if f.feature_name == "not_target"]
    assert len(inv_findings) > 0
    assert inv_findings[0].status == LeakageStatus.CONFIRMED_LEAKAGE
    assert inv_findings[0].severity == Severity.CRITICAL


def test_post_outcome_domain_features_detected():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    X = pd.DataFrame({
        "osrm_distance": [10.0, 20.0, 15.0, 30.0, 25.0, 12.0],
        "actual_time": [45.0, 95.0, 60.0, 130.0, 110.0, 50.0],
        "segment_deviation": [2.0, 5.0, 1.0, 8.0, 4.0, 2.0],
    })
    findings = audit_feature_leakage(X, y)
    post_features = [f.feature_name for f in findings if f.status == LeakageStatus.CONFIRMED_LEAKAGE]
    assert "actual_time" in post_features
    assert "segment_deviation" in post_features


def test_suspicious_semantic_keywords_detected():
    y = pd.Series([0, 1, 0, 1])
    X = pd.DataFrame({
        "pre_dispatch_speed": [20, 25, 30, 22],
        "future_outcome_flag": [0, 1, 0, 1],
        "post_completion_status": [1, 2, 1, 2],
    })
    findings = audit_feature_leakage(X, y)
    suspicious = [f.feature_name for f in findings if f.status in (LeakageStatus.SUSPICIOUS_FEATURE, LeakageStatus.CONFIRMED_LEAKAGE)]
    assert "future_outcome_flag" in suspicious
    assert "post_completion_status" in suspicious


def test_extreme_correlation_detected():
    np.random.seed(42)
    y = pd.Series(np.random.choice([0, 1], size=100))
    # Create feature with 99.9% correlation
    noisy_target = y.astype(float) + np.random.normal(0, 0.005, size=100)
    X = pd.DataFrame({
        "clean_var": np.random.randn(100),
        "proxy_target": noisy_target,
    })
    findings = audit_feature_leakage(X, y, correlation_threshold=0.98)
    corr_findings = [f for f in findings if f.feature_name == "proxy_target"]
    assert len(corr_findings) > 0
    assert corr_findings[0].severity in (Severity.CRITICAL, Severity.HIGH)


def test_split_overlap_leakage_detected():
    train_df = pd.DataFrame({"feat_a": [1, 2, 3, 4], "feat_b": [10, 20, 30, 40]})
    test_df = pd.DataFrame({"feat_a": [3, 5, 6], "feat_b": [30, 50, 60]})
    # Row (3, 30) is present in both train and test
    overlap_findings = audit_split_overlap(train_df, test_df)
    assert len(overlap_findings) == 1
    assert overlap_findings[0].overlap_count == 1
    assert overlap_findings[0].status == LeakageStatus.CONFIRMED_LEAKAGE
