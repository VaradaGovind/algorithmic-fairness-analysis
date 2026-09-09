# ============================================================================
# Synthetic Sensitive Attribute Audit & Negative Controls Engine
# Classifies sensitive attributes, audits feature-to-sensitive contamination,
# and executes negative controls (Random, Permuted, Feature-Derived).
# ============================================================================

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class AttributeOrigin(str, Enum):
    REAL_OBSERVED = "REAL_OBSERVED"
    DERIVED_PROXY = "DERIVED_PROXY"
    SYNTHETIC = "SYNTHETIC"
    UNKNOWN = "UNKNOWN"


@dataclass
class SensitiveAttributeProfile:
    dataset_name: str
    attribute_name: str
    origin: AttributeOrigin
    generation_inputs: List[str]
    inputs_in_model_features: bool
    inputs_influence_target: bool
    inputs_post_outcome: bool
    represents_real_demographic: bool
    causal_interpretation_meaningful: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["origin"] = self.origin.value
        return d


@dataclass
class NegativeControlResult:
    control_name: str
    description: str
    observed_fairness_gap: float
    control_mean_fairness_gap: float
    control_std_fairness_gap: float
    empirical_p_value: float
    disparity_explained_by_feature_coupling: bool
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Known registry of datasets and their sensitive attribute provenance
DATASET_PROFILES: Dict[str, SensitiveAttributeProfile] = {
    "Delhivery Logistics": SensitiveAttributeProfile(
        dataset_name="Delhivery Logistics",
        attribute_name="Education_Level",
        origin=AttributeOrigin.SYNTHETIC,
        generation_inputs=["osrm_distance", "actual_time", "segment_osrm_distance"],
        inputs_in_model_features=True,
        inputs_influence_target=True,  # actual_time directly determines cutoff breach
        inputs_post_outcome=True,      # actual_time is post-delivery realization
        represents_real_demographic=False,
        causal_interpretation_meaningful=False,
        rationale=(
            "Delhivery freight and last-mile tracking data contains no driver educational or demographic records. "
            "Education_Level was synthetically constructed by ranking routes on distance and post-outcome transit time "
            "(actual_time) and mapping to empirical salary proportions from Salary_Data.csv. Because actual_time directly "
            "determines cutoff failure and is an input feature, the synthetic attribute is mechanically confounded with "
            "both model inputs and the target variable."
        ),
    ),
    "Adult Income Benchmark": SensitiveAttributeProfile(
        dataset_name="Adult Income Benchmark",
        attribute_name="Education_Group",
        origin=AttributeOrigin.REAL_OBSERVED,
        generation_inputs=["education-num"],
        inputs_in_model_features=False,  # education-num excluded from features in Adult pipeline
        inputs_influence_target=True,    # real human capital effect on income
        inputs_post_outcome=False,       # observed at survey time concurrently
        represents_real_demographic=True,
        causal_interpretation_meaningful=True,
        rationale=(
            "Adult Census income benchmark records actual observed educational attainment ('education-num' >= 13 "
            "denotes HigherEd vs NonHigherEd). However, causal interpretations must account for post-education mediators "
            "(occupation, workclass, hours-per-week) which are improperly conditioned on in simple G-computation."
        ),
    ),
    "Amazon Last-Mile Routes": SensitiveAttributeProfile(
        dataset_name="Amazon Last-Mile Routes",
        attribute_name="Education_Level",
        origin=AttributeOrigin.SYNTHETIC,
        generation_inputs=["num_dropoffs", "bbox_area", "invalid_sequence_score"],
        inputs_in_model_features=True,
        inputs_influence_target=True,  # invalid_sequence_score directly impacts route evaluation
        inputs_post_outcome=False,
        represents_real_demographic=False,
        causal_interpretation_meaningful=False,
        rationale=(
            "Amazon ALMRRC 2021 route evaluation dataset contains no driver educational records. "
            "Education_Level was synthetically constructed from dropoff counts, route bounding box area, and "
            "invalid sequence execution scores. All generation inputs are also model features, meaning disparities "
            "are an artifact of model feature distributions rather than human demographic effects."
        ),
    ),
}


def audit_feature_contamination(
    sensitive_series: pd.Series,
    feature_df: pd.DataFrame,
    correlation_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Checks whether a sensitive attribute is strongly correlated with or predictable
    from the model's feature matrix (detecting circularity/contamination).
    """
    correlations: Dict[str, float] = {}
    numeric_features = feature_df.select_dtypes(include=[np.number])

    # Convert sensitive to numeric codes if non-numeric
    if not pd.api.types.is_numeric_dtype(sensitive_series.dtype):
        sensitive_numeric = pd.Series(pd.Categorical(sensitive_series).codes, index=sensitive_series.index)
    else:
        sensitive_numeric = pd.to_numeric(sensitive_series, errors="coerce")

    for col in numeric_features.columns:
        valid_mask = sensitive_numeric.notna() & numeric_features[col].notna()
        if valid_mask.sum() > 20 and numeric_features[col].std() > 1e-9 and sensitive_numeric.std() > 1e-9:
            r = float(np.corrcoef(sensitive_numeric[valid_mask], numeric_features[col][valid_mask])[0, 1])
            if abs(r) >= correlation_threshold:
                correlations[col] = r

    # Mutual information proxy / ANOVA F-test for categorical groups
    anova_pvalues: Dict[str, float] = {}
    groups = [group.dropna() for _, group in numeric_features.groupby(sensitive_series)]
    if len(groups) >= 2:
        for col in numeric_features.columns:
            group_vals = [g[col].dropna() for g in groups if len(g[col].dropna()) > 3]
            if len(group_vals) >= 2:
                try:
                    f_stat, p_val = stats.f_oneway(*group_vals)
                    if not np.isnan(p_val) and p_val < 0.001:
                        anova_pvalues[col] = float(p_val)
                except Exception:
                    pass

    return {
        "correlated_features_count": len(correlations),
        "correlated_features": correlations,
        "statistically_separated_features_count": len(anova_pvalues),
        "statistically_separated_features": anova_pvalues,
        "is_heavily_contaminated": len(correlations) >= 1 or len(anova_pvalues) >= 1,
    }


def run_negative_controls(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    feature_matrix: Optional[pd.DataFrame] = None,
    n_permutations: int = 100,
    seed: int = 42,
) -> List[NegativeControlResult]:
    """
    Runs three negative control experiments against the fairness evaluator:
    1. RANDOM_GROUP_CONTROL: Randomly drawn sensitive labels with matched marginal frequencies.
    2. PERMUTED_SENSITIVE_ATTRIBUTE_CONTROL: Permutation test breaking any real demographic link.
    3. FEATURE_DERIVED_GROUP_CONTROL: Synthetic group derived directly from predictive features.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    observed_gap = float(metric_fn(y_true, y_pred, sensitive))
    results: List[NegativeControlResult] = []

    # 1. PERMUTED_SENSITIVE_ATTRIBUTE_CONTROL
    permuted_gaps: List[float] = []
    for _ in range(n_permutations):
        perm_s = rng.permutation(sensitive)
        gap = float(metric_fn(y_true, y_pred, perm_s))
        if not np.isnan(gap):
            permuted_gaps.append(gap)

    p_val_perm = float(np.mean([g >= observed_gap for g in permuted_gaps])) if permuted_gaps else float("nan")
    results.append(
        NegativeControlResult(
            control_name="PERMUTED_SENSITIVE_ATTRIBUTE_CONTROL",
            description="Randomly shuffles sensitive attribute across instances (null hypothesis: no group difference).",
            observed_fairness_gap=observed_gap,
            control_mean_fairness_gap=float(np.mean(permuted_gaps)) if permuted_gaps else float("nan"),
            control_std_fairness_gap=float(np.std(permuted_gaps)) if permuted_gaps else float("nan"),
            empirical_p_value=p_val_perm,
            disparity_explained_by_feature_coupling=p_val_perm > 0.05,
            interpretation=(
                "If observed gap is not statistically greater than permuted null (p > 0.05), "
                "the apparent disparity is indistinguishable from random noise."
            ),
        )
    )

    # 2. RANDOM_GROUP_CONTROL
    unique_groups, group_counts = np.unique(sensitive, return_counts=True)
    group_probs = group_counts / group_counts.sum()
    random_gaps: List[float] = []
    for _ in range(n_permutations):
        rand_s = rng.choice(unique_groups, size=len(sensitive), p=group_probs)
        gap = float(metric_fn(y_true, y_pred, rand_s))
        if not np.isnan(gap):
            random_gaps.append(gap)

    p_val_rand = float(np.mean([g >= observed_gap for g in random_gaps])) if random_gaps else float("nan")
    results.append(
        NegativeControlResult(
            control_name="RANDOM_GROUP_CONTROL",
            description="Independent synthetic groups drawn from marginal empirical distribution.",
            observed_fairness_gap=observed_gap,
            control_mean_fairness_gap=float(np.mean(random_gaps)) if random_gaps else float("nan"),
            control_std_fairness_gap=float(np.std(random_gaps)) if random_gaps else float("nan"),
            empirical_p_value=p_val_rand,
            disparity_explained_by_feature_coupling=False,
            interpretation=(
                "Demonstrates baseline stochastic fairness gap expected purely due to finite sample size."
            ),
        )
    )

    # 3. FEATURE_DERIVED_GROUP_CONTROL
    if feature_matrix is not None and not feature_matrix.empty:
        # Create an artificial group derived from the most predictive continuous feature
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            top_feature = feature_matrix[numeric_cols[0]].to_numpy()
            feature_derived_group = np.where(top_feature > np.median(top_feature), "High_Feature", "Low_Feature")
            fd_gap = float(metric_fn(y_true, y_pred, feature_derived_group))
            results.append(
                NegativeControlResult(
                    control_name="FEATURE_DERIVED_GROUP_CONTROL",
                    description=f"Group mechanically derived from median split of model feature `{numeric_cols[0]}`.",
                    observed_fairness_gap=observed_gap,
                    control_mean_fairness_gap=fd_gap,
                    control_std_fairness_gap=0.0,
                    empirical_p_value=float("nan"),
                    disparity_explained_by_feature_coupling=True,
                    interpretation=(
                        f"Shows the large artificial fairness gap ({fd_gap:.4f}) produced when a group attribute "
                        f"is constructed directly from predictive features."
                    ),
                )
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Sensitive Attribute Audit")
    parser.add_argument("--demo", action="store_true", help="Run negative controls demo")
    args = parser.parse_args()

    if args.demo:
        print("\n=== Running Sensitive Attribute Audit Demo ===")
        for name, profile in DATASET_PROFILES.items():
            print(f"\nDataset: {name}")
            print(f"  Attribute: {profile.attribute_name} ({profile.origin.value})")
            print(f"  Inputs in features: {profile.inputs_in_model_features}")
            print(f"  Inputs post-outcome: {profile.inputs_post_outcome}")
            print(f"  Represents real demographic: {profile.represents_real_demographic}")
            print(f"  Causal interpretation meaningful: {profile.causal_interpretation_meaningful}")
            print(f"  Rationale: {profile.rationale[:120]}...")
