# ============================================================================
# Unit Tests: Sensitive Attribute Audit & Negative Controls
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from src.sensitive_audit import (
    DATASET_PROFILES,
    AttributeOrigin,
    audit_feature_contamination,
    run_negative_controls,
)


def test_dataset_profiles_provenance():
    assert "Delhivery Logistics" in DATASET_PROFILES
    assert DATASET_PROFILES["Delhivery Logistics"].origin == AttributeOrigin.SYNTHETIC
    assert DATASET_PROFILES["Delhivery Logistics"].inputs_in_model_features is True
    assert DATASET_PROFILES["Delhivery Logistics"].inputs_post_outcome is True
    assert DATASET_PROFILES["Delhivery Logistics"].represents_real_demographic is False

    assert "Adult Income Benchmark" in DATASET_PROFILES
    assert DATASET_PROFILES["Adult Income Benchmark"].origin == AttributeOrigin.REAL_OBSERVED

    assert "Amazon Last-Mile Routes" in DATASET_PROFILES
    assert DATASET_PROFILES["Amazon Last-Mile Routes"].origin == AttributeOrigin.SYNTHETIC


def test_feature_contamination_audit():
    np.random.seed(42)
    n = 200
    route_distance = np.random.uniform(5, 50, n)
    actual_time = route_distance * 2.5 + np.random.normal(0, 5, n)
    
    # Synthesize group using actual_time and route_distance (analogous to Delhivery/Amazon)
    signal = route_distance * 0.5 + actual_time * 0.5
    synthetic_group = np.where(signal > np.median(signal), "Tier_A", "Tier_B")
    
    features = pd.DataFrame({
        "route_distance": route_distance,
        "actual_time": actual_time,
        "clean_unrelated": np.random.randn(n),
    })
    
    contamination = audit_feature_contamination(pd.Series(synthetic_group), features)
    assert contamination["is_heavily_contaminated"] is True
    assert "route_distance" in contamination["correlated_features"]
    assert "actual_time" in contamination["correlated_features"]


def test_negative_controls_execution():
    np.random.seed(42)
    n = 300
    y_true = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    y_pred = y_true.copy()
    # Introduce small prediction error
    flip_idx = np.random.choice(n, size=30, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    
    sensitive = np.random.choice(["Group_A", "Group_B"], size=n, p=[0.6, 0.4])
    features = pd.DataFrame({"score": np.random.randn(n)})
    
    def simple_dp_gap(yt, yp, s):
        rates = [np.mean(yp[s == g] == 1) for g in np.unique(s)]
        return max(rates) - min(rates)
    
    controls = run_negative_controls(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        metric_fn=simple_dp_gap,
        feature_matrix=features,
        n_permutations=50,
    )
    
    control_names = [c.control_name for c in controls]
    assert "PERMUTED_SENSITIVE_ATTRIBUTE_CONTROL" in control_names
    assert "RANDOM_GROUP_CONTROL" in control_names
    assert "FEATURE_DERIVED_GROUP_CONTROL" in control_names
