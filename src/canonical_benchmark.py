# ============================================================================
# Canonical Benchmark Execution Engine
# Enforces prediction-time feature contracts, group-aware splitting,
# model-selection firewall, majority-class baseline (B0), fair baseline hierarchy
# (B0-B4), prediction collapse detection, and multi-seed variance.
# ============================================================================

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from fairlearn.postprocessing import ThresholdOptimizer

try:
    from src import core as base
    from src.dataset_provenance import (
        DatasetProvenance,
        DatasetType,
        SensitiveOrigin,
        PROVENANCE_REGISTRY,
    )
    from src.evaluation_integrity import audit_feature_leakage
    from src.evaluation_manifest import (
        EvaluationManifest,
        ResultClassification,
        compute_manifest_fingerprint,
    )
    from src.fairness_evaluator import (
        EvaluationStatus,
        FairnessEvaluationSuite,
        FairnessPolicy,
        FairnessSuccessStatus,
        FairnessUncertaintyResult,
        IntersectionalGroupSpec,
        ModelDegeneracyStatus,
        PolicyDecision,
        PolicyEvaluationResult,
        build_intersectional_groups,
        classify_fairness_success,
        compute_fairness_uncertainty_bootstrap,
        detect_prediction_collapse,
        evaluate_fairness_defensively,
        evaluate_policy_compliance,
        evaluate_policy_sensitivity,
    )
    from src.firewall import (
        LockedTestData,
        LockedPolicy,
        ModelSelectionFirewallViolation,
        PolicyLockError,
        firewall_protected,
    )
    from src.ingest_data import (
        DatasetManifest,
        IngestionStatus,
        export_sanitized_csv,
        ingest_and_validate_dataset,
    )
    from src.sensitive_audit import (
        AttributeOrigin,
        DATASET_PROFILES,
        run_negative_controls,
    )
    from src.split_utils import (
        GroupLeakageError,
        SplitRegime,
        SplitResult,
        group_aware_3way_split,
        stratified_3way_split,
        temporal_3way_split,
    )
except ImportError:
    import core as base
    from dataset_provenance import (
        DatasetProvenance,
        DatasetType,
        SensitiveOrigin,
        PROVENANCE_REGISTRY,
    )
    from evaluation_integrity import audit_feature_leakage
    from evaluation_manifest import (
        EvaluationManifest,
        ResultClassification,
        compute_manifest_fingerprint,
    )
    from fairness_evaluator import (
        EvaluationStatus,
        FairnessEvaluationSuite,
        FairnessPolicy,
        FairnessSuccessStatus,
        FairnessUncertaintyResult,
        IntersectionalGroupSpec,
        ModelDegeneracyStatus,
        PolicyDecision,
        PolicyEvaluationResult,
        build_intersectional_groups,
        classify_fairness_success,
        compute_fairness_uncertainty_bootstrap,
        detect_prediction_collapse,
        evaluate_fairness_defensively,
        evaluate_policy_compliance,
        evaluate_policy_sensitivity,
    )
    from firewall import (
        LockedTestData,
        LockedPolicy,
        ModelSelectionFirewallViolation,
        PolicyLockError,
        firewall_protected,
    )
    from ingest_data import (
        DatasetManifest,
        IngestionStatus,
        export_sanitized_csv,
        ingest_and_validate_dataset,
    )
    from sensitive_audit import (
        AttributeOrigin,
        DATASET_PROFILES,
        run_negative_controls,
    )
    from split_utils import (
        GroupLeakageError,
        SplitRegime,
        SplitResult,
        group_aware_3way_split,
        stratified_3way_split,
        temporal_3way_split,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


# ============================================================================
# Trivial Lower-Bound Predictor (B0 Negative Control)
# ============================================================================

class MajorityClassClassifier:
    """
    Trivial baseline predictor that constantly predicts the empirical majority class
    observed in training data. Serves as the negative control establishing minimum
    baseline utility.
    """
    def __init__(self):
        self.majority_class_: int = 0
        self.class_probabilities_: np.ndarray = np.array([0.5, 0.5])

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> MajorityClassClassifier:
        vals, counts = np.unique(y, return_counts=True)
        self.majority_class_ = int(vals[np.argmax(counts)])
        prob_1 = float(np.mean(y == 1))
        self.class_probabilities_ = np.array([1.0 - prob_1, prob_1])
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        n = len(X)
        return np.full(n, fill_value=self.majority_class_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.tile(self.class_probabilities_, (n, 1))


@dataclass
class CanonicalCandidateResult:
    dataset_name: str
    baseline_id: str
    model_name: str
    regime: str
    seed: int
    suite: FairnessEvaluationSuite
    manifest_fingerprint: Optional[str] = None
    result_classification: str = ResultClassification.SYNTHETIC_DGP_VALIDATION.value
    evidence_state: str = "INSUFFICIENT_EVIDENCE"
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    number_of_comparisons: int = 1
    analysis_type: str = "SCREENING_ANALYSIS"

    def to_flat_dict(self) -> Dict[str, Any]:
        row = {
            "dataset": self.dataset_name,
            "baseline_id": self.baseline_id,
            "model": self.model_name,
            "regime": self.regime,
            "seed": self.seed,
            "result_classification": self.result_classification,
            "evidence_state": self.evidence_state or (
                self.suite.evidence_state.value
                if hasattr(self.suite.evidence_state, "value")
                else str(self.suite.evidence_state)
            ),
            "manifest_fingerprint": self.manifest_fingerprint or self.suite.manifest_fingerprint,
            "policy_id": self.policy_id or (self.suite.policy_result.policy_id if self.suite.policy_result else "UNKNOWN"),
            "policy_version": self.policy_version or (self.suite.policy_result.policy_version if self.suite.policy_result else "UNKNOWN"),
            "number_of_comparisons": self.number_of_comparisons or self.suite.number_of_comparisons,
            "analysis_type": self.analysis_type or self.suite.analysis_type,
            "accuracy": self.suite.metrics["Accuracy"].value,
            "balanced_accuracy": self.suite.metrics["Balanced Accuracy"].value,
            "precision": self.suite.metrics["Precision"].value,
            "recall": self.suite.metrics["Recall"].value,
            "f1_score": self.suite.metrics["F1-Score"].value,
            "fairness_gap": self.suite.fairness_gap,
            "demographic_parity_diff": self.suite.metrics["Demographic Parity Diff"].value,
            "equal_opportunity_diff": self.suite.metrics["Equal Opportunity Diff"].value,
            "equalized_odds_diff": self.suite.metrics["Equalized Odds Diff"].value,
            "predictive_parity_diff": self.suite.metrics["Predictive Parity Diff"].value,
            "disparate_impact_ratio": self.suite.metrics["Disparate Impact Ratio"].value,
            "utility_score": self.suite.user_utility_score,
            "accuracy_tradeoff_proxy": self.suite.accuracy_tradeoff_proxy,
            "TP": self.suite.metrics["TP"].value if "TP" in self.suite.metrics else np.nan,
            "TN": self.suite.metrics["TN"].value if "TN" in self.suite.metrics else np.nan,
            "FP": self.suite.metrics["FP"].value if "FP" in self.suite.metrics else np.nan,
            "FN": self.suite.metrics["FN"].value if "FN" in self.suite.metrics else np.nan,
            "TPR": self.suite.metrics["TPR"].value if "TPR" in self.suite.metrics else np.nan,
            "TNR": self.suite.metrics["TNR"].value if "TNR" in self.suite.metrics else np.nan,
            "FPR": self.suite.metrics["FPR"].value if "FPR" in self.suite.metrics else np.nan,
            "FNR": self.suite.metrics["FNR"].value if "FNR" in self.suite.metrics else np.nan,
            "PPV": self.suite.metrics["PPV"].value if "PPV" in self.suite.metrics else np.nan,
            "positive_prediction_rate": self.suite.metrics["Positive Prediction Rate"].value if "Positive Prediction Rate" in self.suite.metrics else np.nan,
            "actual_positive_prevalence": self.suite.metrics["Actual Positive Prevalence"].value if "Actual Positive Prevalence" in self.suite.metrics else np.nan,
            "degeneracy_status": self.suite.degeneracy_status.value,
            "degeneracy_message": self.suite.degeneracy_message,
            "fairness_success_status": self.suite.fairness_success_status.value,
            "policy_decision": self.suite.policy_result.decision.value if self.suite.policy_result else "POLICY_UNDEFINED",
            "is_policy_compliant": (self.suite.policy_result.decision == PolicyDecision.POLICY_COMPLIANT) if self.suite.policy_result else False,
            "brier_score": self.suite.metrics["Brier Score"].value if "Brier Score" in self.suite.metrics else np.nan,
            "expected_calibration_error": self.suite.metrics["Expected Calibration Error"].value if "Expected Calibration Error" in self.suite.metrics else np.nan,
        }

        # Bootstrap uncertainty intervals
        if self.suite.uncertainty_intervals:
            eo_unc = self.suite.uncertainty_intervals.get("Equalized Odds Diff")
            if eo_unc:
                row["eo_ci_low_bootstrap"] = eo_unc.get("ci_low", np.nan)
                row["eo_ci_high_bootstrap"] = eo_unc.get("ci_high", np.nan)
            dp_unc = self.suite.uncertainty_intervals.get("Demographic Parity Diff")
            if dp_unc:
                row["dp_ci_low_bootstrap"] = dp_unc.get("ci_low", np.nan)
                row["dp_ci_high_bootstrap"] = dp_unc.get("ci_high", np.nan)
            di_unc = self.suite.uncertainty_intervals.get("Disparate Impact Ratio")
            if di_unc:
                row["di_ci_low_bootstrap"] = di_unc.get("ci_low", np.nan)
                row["di_ci_high_bootstrap"] = di_unc.get("ci_high", np.nan)

        # Per-subgroup rate extraction
        if self.suite.subgroup_metrics:
            for g_name, g_vals in self.suite.subgroup_metrics.items():
                safe_g = g_name.replace(" ", "_").replace("-", "_")
                for k, v in g_vals.items():
                    row[f"subgroup_{safe_g}_{k}"] = v

        return row


# ============================================================================
# Clean Pre-Dispatch Data Loaders (Strict Feature Contract Enforcement)
# ============================================================================

def build_clean_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    steps = []
    if numeric_cols:
        steps.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
    if categorical_cols:
        steps.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(steps, remainder="drop")


def load_clean_delhivery(data_dir: Path = DATA_DIR) -> Dict[str, Any]:
    """
    Loads Delhivery Logistics enforcing PRE_DISPATCH_VALID feature contract.
    EXCLUDES: actual_time, segment_actual_time, actual_distance, and all derived ratios.
    EXTRACTS: trip_uuid as grouping key (Trip Segment Isolation).
    """
    delhivery_path = data_dir / "delhivery" / "delhivery_data.csv"
    if not delhivery_path.exists():
        raise FileNotFoundError(f"Delhivery raw dataset not found at: {delhivery_path}")

    df = pd.read_csv(delhivery_path)
    cutoff_flag = df["is_cutoff"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df["Task_Success"] = (~cutoff_flag).astype(int)

    # Strictly pre-dispatch valid features
    clean_features = [
        "osrm_distance",
        "osrm_time",
        "segment_osrm_distance",
        "segment_osrm_time",
    ]
    df = df.dropna(subset=clean_features + ["Task_Success"]).copy()
    df["time_per_distance_osrm"] = df["osrm_time"] / (df["osrm_distance"] + 1e-6)
    clean_features.append("time_per_distance_osrm")

    # Synthetic proxy derived strictly without post-outcome variables
    base_prior = base.get_salary_education_prior()
    proxy_signal = df["osrm_distance"].rank(pct=True) * 0.60 + df["segment_osrm_distance"].rank(pct=True) * 0.40
    df["Education_Level"] = base.assign_synthetic_education(proxy_signal, base_prior, seed=42)

    group_col = df["trip_uuid"] if "trip_uuid" in df.columns else pd.Series(df.index, index=df.index)

    return {
        "name": "Delhivery Logistics (Clean Pre-Dispatch)",
        "X": df[clean_features].copy(),
        "y": df["Task_Success"].copy(),
        "sensitive": df["Education_Level"].copy(),
        "groups": group_col.copy(),
        "group_semantics": "TRIP_SEGMENT_ISOLATION",
        "sensitive_origin": AttributeOrigin.SYNTHETIC,
    }


def load_clean_amazon(data_dir: Path = DATA_DIR) -> Dict[str, Any]:
    """
    Loads Amazon Last-Mile Routes enforcing PRE_DISPATCH_VALID feature contract.
    EXCLUDES: invalid_sequence_score (ambiguous post-execution timing).
    EXTRACTS: station_code as grouping key (Unseen Station Domain Shift).
    """
    route_path = data_dir / "amazon_data" / "almrrc2021-data-training" / "model_build_inputs" / "route_data.json"
    if not route_path.exists():
        raise FileNotFoundError(f"Amazon route dataset not found at: {route_path}")

    with route_path.open("r", encoding="utf-8") as fp:
        route_data = json.load(fp)

    records: List[Dict[str, Any]] = []
    for route_id, payload in route_data.items():
        if not isinstance(payload, dict):
            continue
        route_score = str(payload.get("route_score", "")).strip()
        if not route_score:
            continue
        features = base._extract_route_features(payload)
        if features is None:
            continue
        row = {
            "route_id": route_id,
            "station_code": str(payload.get("station_code", "UNKNOWN")),
            "departure_hour": base._parse_departure_hour(payload.get("departure_time_utc")),
            "executor_capacity_cm3": payload.get("executor_capacity_cm3"),
            "Task_Success": int(route_score.lower() == "high"),
        }
        row.update(features)
        records.append(row)

    df = pd.DataFrame(records)
    clean_numeric = [
        "departure_hour", "executor_capacity_cm3", "num_stops_total", "num_dropoffs",
        "lat_mean", "lng_mean", "lat_std", "lng_std", "bbox_area", "radial_dispersion",
    ]
    for col in clean_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Task_Success", "executor_capacity_cm3", "num_dropoffs"]).copy()
    df["departure_hour"] = df["departure_hour"].fillna(df["departure_hour"].median())

    df["stops_per_dropoff_ratio"] = df["num_stops_total"] / (df["num_dropoffs"] + 1e-6)
    df["spatial_density"] = df["num_dropoffs"] / (df["bbox_area"] + 1e-6)
    clean_numeric.extend(["stops_per_dropoff_ratio", "spatial_density"])

    # Synthetic proxy derived strictly without invalid_sequence_score
    base_prior = base.get_salary_education_prior()
    proxy_signal = df["num_dropoffs"].rank(pct=True) * 0.65 + df["bbox_area"].rank(pct=True) * 0.35
    df["Education_Level"] = base.assign_synthetic_education(proxy_signal, base_prior, seed=43)

    return {
        "name": "Amazon Last-Mile Routes (Clean Pre-Dispatch)",
        "X": df[clean_numeric + ["station_code"]].copy(),
        "y": df["Task_Success"].copy(),
        "sensitive": df["Education_Level"].copy(),
        "groups": df["station_code"].copy(),
        "group_semantics": "UNSEEN_STATION_HOLDOUT",
        "sensitive_origin": AttributeOrigin.SYNTHETIC,
    }


# ============================================================================
# Controlled Synthetic Benchmark Generator
# ============================================================================

def generate_controlled_synthetic_benchmark(
    n_samples: int = 3000,
    seed: int = 42,
    scenario: str = "SCENARIO_C_EXPLICIT_GROUP_EFFECT",
    signal_regime: str = "MODERATE_SIGNAL",
) -> Dict[str, Any]:
    """
    Generates controlled, reproducible synthetic logistics data adhering to PRE_DISPATCH_VALID rules.
    Features have mathematically defined relationships with outcome Y to guarantee non-zero mutual
    information, preventing trivial majority-class collapse while enabling rigorous fairness testing.
    """
    rng = np.random.default_rng(seed)

    # Base operational variables
    planned_distance = rng.uniform(5.0, 50.0, n_samples)
    planned_time = planned_distance * rng.uniform(1.8, 2.4, n_samples)
    stops_count = rng.integers(5, 40, n_samples).astype(float)
    vehicle_capacity = rng.uniform(100.0, 500.0, n_samples)
    station_code = rng.choice(["HUB_NORTH", "HUB_SOUTH", "HUB_EAST", "HUB_WEST", "HUB_CENTRAL"], size=n_samples)

    # Group assignment and feature correlation depending on scenario
    if scenario == "SCENARIO_B_PREDICTIVE_FEATURE_GROUP_CORRELATION":
        is_high = rng.binomial(1, 0.45, n_samples)
        planned_distance += is_high * 15.0
        stops_count += is_high * 10.0
        group = np.where(is_high == 1, "High_Complexity", "Standard_Complexity")
    elif scenario in ("SCENARIO_D_SEVERE_FAIRNESS_TRADEOFF", "SCENARIO_D_IMPOSSIBLE_FAIRNESS"):
        is_high = rng.binomial(1, 0.50, n_samples)
        planned_distance += is_high * 22.0
        stops_count += is_high * 18.0
        group = np.where(is_high == 1, "High_Complexity", "Standard_Complexity")
    else:
        is_high = rng.binomial(1, 0.50, n_samples)
        group = np.where(is_high == 1, "High_Complexity", "Standard_Complexity")

    # Standardize features for linear combination
    x1 = (planned_distance - 27.5) / 13.0
    x2 = (planned_time - 57.5) / 28.0
    x3 = (stops_count - 22.5) / 10.0
    x4 = (vehicle_capacity - 300.0) / 115.0

    # Scenario E: Constant Predictor / Zero Signal (Negative Control)
    if scenario == "SCENARIO_E_CONSTANT_PREDICTOR" or signal_regime == "ZERO_SIGNAL":
        z = rng.normal(0.0, 1.0, n_samples)
        threshold = np.quantile(z, 0.106)  # 89.4% positive class, 0 mutual info with X
        success = (z >= threshold).astype(int)
    else:
        # Signal coefficients
        if signal_regime == "LOW_SIGNAL":
            b1, b2, b3, b4, sigma = -0.25, -0.30, -0.20, 0.15, 1.5
        elif signal_regime == "HIGH_SIGNAL":
            b1, b2, b3, b4, sigma = -1.50, -1.60, -1.20, 0.90, 0.6
        else:  # MODERATE_SIGNAL
            b1, b2, b3, b4, sigma = -0.75, -0.80, -0.65, 0.45, 1.0

        # Group direct effect (gamma)
        if scenario == "SCENARIO_C_EXPLICIT_GROUP_EFFECT":
            gamma = -1.25
        elif scenario in ("SCENARIO_D_SEVERE_FAIRNESS_TRADEOFF", "SCENARIO_D_IMPOSSIBLE_FAIRNESS"):
            gamma = -1.80
        else:
            gamma = 0.0

        z = 0.1 + b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 + gamma * is_high + rng.normal(0.0, sigma, n_samples)
        prob = 1.0 / (1.0 + np.exp(-z))
        success = (prob >= 0.50).astype(int)

    # Trip UUID grouping (prevent multi-segment trip leakage)
    n_trips = max(n_samples // 4, 10)
    trip_ids = [f"TRIP_{i:04d}" for i in range(n_trips)]
    trip_assignments = rng.choice(trip_ids, size=n_samples)

    # Simulated timestamps over 60 days
    base_time = pd.Timestamp("2026-01-01")
    days = rng.uniform(0, 60, n_samples)
    timestamps = [base_time + pd.Timedelta(days=d) for d in days]

    df = pd.DataFrame({
        "planned_distance": planned_distance,
        "planned_time": planned_time,
        "stops_count": stops_count,
        "vehicle_capacity": vehicle_capacity,
        "station_code": station_code,
        "trip_uuid": trip_assignments,
        "trip_creation_time": timestamps,
    })

    return {
        "name": f"Synthetic Logistics Benchmark ({scenario} - {signal_regime})",
        "X": df[["planned_distance", "planned_time", "stops_count", "vehicle_capacity", "station_code"]].copy(),
        "y": pd.Series(success, name="Task_Success"),
        "sensitive": pd.Series(group, name="Complexity_Group"),
        "groups": pd.Series(trip_assignments, name="trip_uuid"),
        "timestamps": pd.Series(timestamps, name="trip_creation_time"),
        "sensitive_origin": AttributeOrigin.SYNTHETIC,
        "scenario": scenario,
        "signal_regime": signal_regime,
    }


def generate_clean_synthetic_benchmark(n_samples: int = 3000, seed: int = 42) -> Dict[str, Any]:
    """Backward-compatible default fixture executing MODERATE_SIGNAL with EXPLICIT_GROUP_EFFECT."""
    return generate_controlled_synthetic_benchmark(
        n_samples=n_samples,
        seed=seed,
        scenario="SCENARIO_C_EXPLICIT_GROUP_EFFECT",
        signal_regime="MODERATE_SIGNAL",
    )


# ============================================================================
# Canonical Baseline Hierarchy (B0 to B4)
# ============================================================================

def fit_baseline_hierarchy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    s_val: np.ndarray,
    seed: int = 42,
) -> Dict[str, LockedPolicy]:
    """
    Fits and validation-tunes the canonical baseline hierarchy B0 through B4:
    B0: Trivial Majority-Class Baseline (Training mode constant predictor - negative control)
    B1: Clean Unmitigated Linear Baseline (Standard logistic regression on clean features)
    B2: Fairness-Reweighted Baseline (Sample reweighting via inverse propensity weighting)
    B3: Validation-Tuned ThresholdOptimizer (Post-processing tuned strictly on validation set)
    B4: Proposed Tuned XGBoost (Cost-sensitive gradient boosting candidate)
    """
    policies: Dict[str, LockedPolicy] = {}

    # B0: Majority-Class Baseline (Negative Control)
    maj = MajorityClassClassifier()
    maj.fit(X_train, y_train)
    policies["B0_Majority_Class_Baseline"] = LockedPolicy(
        model_name="B0_Majority_Class_Baseline",
        estimator=maj,
        is_frozen=True,
    )

    # B1: Clean Unmitigated Linear Baseline
    lr_b1 = LogisticRegression(max_iter=1500, random_state=seed)
    lr_b1.fit(X_train, y_train)
    policies["B1_Clean_Unmitigated_Linear"] = LockedPolicy(
        model_name="B1_Clean_Unmitigated_Linear",
        estimator=lr_b1,
        is_frozen=True,
    )

    # B2: Fairness-Reweighted Baseline
    try:
        from src.main import reweighing_weights
    except ImportError:
        from main import reweighing_weights
    weights = reweighing_weights(y_train, s_train)
    lr_b2 = LogisticRegression(max_iter=1500, random_state=seed)
    lr_b2.fit(X_train, y_train, sample_weight=weights)
    policies["B2_Fairness_Reweighted"] = LockedPolicy(
        model_name="B2_Fairness_Reweighted",
        estimator=lr_b2,
        is_frozen=True,
    )

    # B3: Validation-Tuned ThresholdOptimizer (Enforcing Equalized Odds)
    # Mitigator is fitted STRICTLY on validation data, protecting locked test data
    try:
        mitigator = ThresholdOptimizer(estimator=lr_b1, constraints="equalized_odds", predict_method="predict_proba")
        mitigator.fit(X_val, y_val, sensitive_features=s_val)
        policies["B3_Validation_Threshold_Mitigated"] = LockedPolicy(
            model_name="B3_Validation_Threshold_Mitigated",
            estimator=lr_b1,
            threshold_optimizer=mitigator,
            is_frozen=True,
        )
    except Exception as exc:
        warnings.warn(f"ThresholdOptimizer fitting on validation failed: {exc}")
        policies["B3_Validation_Threshold_Mitigated"] = policies["B1_Clean_Unmitigated_Linear"]

    # B4: Proposed Tuned XGBoost Architecture
    pos_count = float(np.sum(y_train == 1))
    neg_count = float(np.sum(y_train == 0))
    scale_pos_weight = max(neg_count / max(pos_count, 1.0), 1.0)

    xgb_b4 = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.08,
        n_estimators=120,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_b4.fit(X_train, y_train, sample_weight=weights)
    policies["B4_Proposed_Tuned_XGB"] = LockedPolicy(
        model_name="B4_Proposed_Tuned_XGB",
        estimator=xgb_b4,
        is_frozen=True,
    )

    return policies


# ============================================================================
# Canonical Evaluation Runner
# ============================================================================

def run_canonical_evaluation(
    dataset_bundle: Dict[str, Any],
    regime: SplitRegime = SplitRegime.IID_HOLDOUT,
    seed: int = 42,
    alpha: float = 0.25,
    policy: Optional[FairnessPolicy] = None,
) -> List[CanonicalCandidateResult]:
    """
    Executes a single-seed canonical evaluation:
    1. Splits dataset according to requested regime (IID, Group, or Temporal).
    2. Enforces model-selection firewall: locks test data.
    3. Fits B0-B4 baselines and tunes thresholds on validation set.
    4. Evaluates frozen policies strictly ONCE on locked test set.
    """
    X_raw = dataset_bundle["X"]
    y_raw = dataset_bundle["y"]
    s_raw = dataset_bundle["sensitive"]
    d_name = dataset_bundle["name"]

    # Step 1: Split according to regime
    if regime == SplitRegime.GROUP_HOLDOUT and "groups" in dataset_bundle:
        split_res = group_aware_3way_split(X_raw, y_raw, s_raw, dataset_bundle["groups"], seed=seed)
    elif regime == SplitRegime.TEMPORAL_HOLDOUT and "timestamps" in dataset_bundle:
        split_res = temporal_3way_split(X_raw, y_raw, s_raw, dataset_bundle["timestamps"])
    else:
        split_res = stratified_3way_split(X_raw, y_raw, s_raw, seed=seed)

    # Step 2: Fit preprocessor strictly on Train, transform Val and Test
    preprocessor = build_clean_preprocessor(split_res.X_train)
    X_train_proc = preprocessor.fit_transform(split_res.X_train)
    X_val_proc = preprocessor.transform(split_res.X_val)
    X_test_proc = preprocessor.transform(split_res.X_test)

    # Step 3: Wrap test set in Firewall container
    locked_test = LockedTestData(
        X=pd.DataFrame(X_test_proc),
        y=split_res.y_test,
        sensitive=split_res.s_test,
        dataset_name=d_name,
    )

    # Step 4: Fit B0-B4 hierarchy using Train and Validation only
    policies = fit_baseline_hierarchy(
        X_train=X_train_proc,
        y_train=split_res.y_train.to_numpy(),
        s_train=split_res.s_train.to_numpy(),
        X_val=X_val_proc,
        y_val=split_res.y_val.to_numpy(),
        s_val=split_res.s_val.to_numpy(),
        seed=seed,
    )

    # Baseline B1 accuracy for tradeoff proxy computation
    b1_key = "B1_Clean_Unmitigated_Linear" if "B1_Clean_Unmitigated_Linear" in policies else "B1_Clean_Feature_Baseline"
    b1_pred = policies[b1_key].predict(X_test_proc, split_res.s_test.to_numpy())
    b1_acc = float(np.mean(b1_pred == split_res.y_test.to_numpy()))

    # Baseline B0 validation balanced accuracy for majority baseline comparison
    b0_policy = policies.get("B0_Majority_Class_Baseline")
    b0_bal_acc = None
    if b0_policy is not None:
        b0_val_pred = b0_policy.predict(X_val_proc)
        from sklearn.metrics import balanced_accuracy_score
        b0_bal_acc = float(balanced_accuracy_score(split_res.y_val.to_numpy(), b0_val_pred))

    # Step 5: Evaluate frozen policies strictly ONCE on locked test set
    active_policy = policy if policy is not None else FairnessPolicy(name="Standard_Operational_Policy")
    classification = (
        ResultClassification.SYNTHETIC_DGP_VALIDATION
        if "Synthetic" in d_name
        else ResultClassification.FRAMEWORK_VALIDATION
    )

    results: List[CanonicalCandidateResult] = []
    for b_id, policy_model in policies.items():
        y_pred = policy_model.predict(X_test_proc, sensitive_features=split_res.s_test.to_numpy())
        y_prob = policy_model.predict_proba(X_test_proc)
        suite = evaluate_fairness_defensively(
            y_true=split_res.y_test.to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive=split_res.s_test.to_numpy(),
            dataset_name=d_name,
            model_name=b_id,
            baseline_accuracy=b1_acc,
            alpha_utility=alpha,
            policy=active_policy,
            majority_baseline_bal_acc=b0_bal_acc,
        )

        manifest = EvaluationManifest.create(
            dataset_identity=d_name,
            dataset_sha256=dataset_bundle.get("source_sha256"),
            target_definition="Task_Success",
            sensitive_attributes=[dataset_bundle["sensitive"].name or "Complexity_Group"],
            feature_contract_version="PRE_DISPATCH_VALID_v1",
            selected_features=list(X_raw.columns),
            excluded_features=["Task_Success"],
            split_strategy=regime.value,
            grouping_strategy=dataset_bundle["groups"].name if "groups" in dataset_bundle else None,
            temporal_strategy=dataset_bundle["timestamps"].name if "timestamps" in dataset_bundle else None,
            random_seeds=[seed],
            model_configuration={"baseline_id": b_id, "model_name": policy_model.model_name},
            fairness_policy_configuration=active_policy.to_dict(),
            missing_attribute_policy=(
                active_policy.missing_attribute_policy.value
                if hasattr(active_policy.missing_attribute_policy, "value")
                else str(active_policy.missing_attribute_policy)
            ),
            result_classification=classification,
            evidence_state=(
                suite.evidence_state.value
                if hasattr(suite.evidence_state, "value")
                else str(suite.evidence_state)
            ),
            number_of_comparisons=suite.number_of_comparisons,
            analysis_type=suite.analysis_type,
            worst_group_selection_disclosure=suite.worst_group_selection_disclosure,
            uncertainty_configuration={"n_bootstraps": 200, "ci_level": 0.95},
            bootstrap_configuration=suite.uncertainty_intervals or {},
            warnings=suite.warnings,
            abstentions=suite.abstentions,
            evaluation_unit_level=(
                suite.evaluation_unit_level.value
                if hasattr(suite.evaluation_unit_level, "value")
                else str(suite.evaluation_unit_level)
            ),
            evaluation_unit_key=suite.evaluation_unit_key,
            aggregation_method=(
                suite.aggregation_strategy.value
                if hasattr(suite.aggregation_strategy, "value")
                else str(suite.aggregation_strategy)
            ),
            privacy_governance_summary=suite.privacy_suppression_metadata,
        )
        suite.manifest_fingerprint = manifest.fingerprint

        results.append(CanonicalCandidateResult(
            dataset_name=d_name,
            baseline_id=b_id,
            model_name=policy_model.model_name,
            regime=regime.value,
            seed=seed,
            suite=suite,
            manifest_fingerprint=manifest.fingerprint,
            result_classification=classification.value,
            evidence_state=manifest.evidence_state,
            policy_id=active_policy.policy_id,
            policy_version=active_policy.policy_version,
            number_of_comparisons=manifest.number_of_comparisons,
            analysis_type=manifest.analysis_type,
        ))

    return results


def run_canonical_multi_seed_suite(
    dataset_bundle: Dict[str, Any],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    regimes: Sequence[SplitRegime] = (SplitRegime.IID_HOLDOUT, SplitRegime.GROUP_HOLDOUT),
    alpha: float = 0.25,
    policy: Optional[FairnessPolicy] = None,
) -> pd.DataFrame:
    """
    Executes full multi-seed canonical benchmark across requested seeds and split regimes.
    """
    flat_rows: List[Dict[str, Any]] = []
    for regime in regimes:
        if regime == SplitRegime.GROUP_HOLDOUT and "groups" not in dataset_bundle:
            continue
        if regime == SplitRegime.TEMPORAL_HOLDOUT and "timestamps" not in dataset_bundle:
            continue

        print(f"  Evaluating Regime: {regime.value}...")
        for seed in seeds:
            eval_results = run_canonical_evaluation(dataset_bundle, regime=regime, seed=seed, alpha=alpha, policy=policy)
            for res in eval_results:
                flat_rows.append(res.to_flat_dict())

    df = pd.DataFrame(flat_rows)
    return df


def execute_locked_real_data_benchmark(
    manifest: DatasetManifest,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    alpha: float = 0.25,
    policy: Optional[FairnessPolicy] = None,
    intersectional_spec: Optional[IntersectionalGroupSpec] = None,
) -> pd.DataFrame:
    """
    Executes the formal locked real-data evaluation protocol:
    Train (60%) -> Validation (20%) -> Freeze Policy -> Locked Test (20%).
    Refuses execution if manifest indicates confirmed critical leakage.
    """
    if manifest.has_critical_leakage:
        raise RuntimeError(
            f"EXECUTION REFUSED: Dataset '{manifest.dataset_name}' failed evaluation integrity audit. "
            f"Critical post-outcome or target leakage detected ({manifest.leakage_findings_count} findings)."
        )

    data_path = Path(manifest.source_path)
    if not data_path.is_file():
        print(f"[NOTICE] REAL-DATA CANONICAL RUN NOT EXECUTED: source data unavailable at {data_path}")
        return pd.DataFrame()

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

    if intersectional_spec is not None:
        sensitive_series, _ = build_intersectional_groups(df, intersectional_spec)
    else:
        s_col = manifest.sensitive_columns[0]
        sensitive_series = df[s_col]

    # Pre-dispatch feature filtering: exclude target, sensitive, grouping, temporal
    exclude_cols = set(manifest.sensitive_columns) | {manifest.target_column}
    if manifest.grouping_column:
        exclude_cols.add(manifest.grouping_column)
    if manifest.temporal_column:
        exclude_cols.add(manifest.temporal_column)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    bundle: Dict[str, Any] = {
        "name": manifest.dataset_name,
        "X": df[feature_cols],
        "y": df[manifest.target_column],
        "sensitive": sensitive_series,
    }
    if manifest.grouping_column and manifest.grouping_column in df.columns:
        bundle["groups"] = df[manifest.grouping_column]
    if manifest.temporal_column and manifest.temporal_column in df.columns:
        bundle["timestamps"] = df[manifest.temporal_column]

    regimes = [SplitRegime.IID_HOLDOUT]
    if "groups" in bundle:
        regimes.append(SplitRegime.GROUP_HOLDOUT)
    if "timestamps" in bundle:
        regimes.append(SplitRegime.TEMPORAL_HOLDOUT)

    return run_canonical_multi_seed_suite(
        bundle,
        seeds=seeds,
        regimes=regimes,
        alpha=alpha,
        policy=policy,
    )


def aggregate_canonical_statistics(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates multi-seed metrics into mean, median, std, min, max, and 95% Seed-Variation Intervals.
    """
    group_cols = ["dataset", "baseline_id", "regime"]
    metric_cols = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1_score",
        "fairness_gap", "demographic_parity_diff", "equal_opportunity_diff",
        "equalized_odds_diff", "predictive_parity_diff", "disparate_impact_ratio",
        "utility_score", "accuracy_tradeoff_proxy",
        "brier_score", "expected_calibration_error",
        "TP", "TN", "FP", "FN", "TPR", "TNR", "FPR", "FNR", "PPV",
        "positive_prediction_rate", "actual_positive_prevalence",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for (d, b, r), grp in df_results.groupby(group_cols):
        row: Dict[str, Any] = {
            "dataset": d,
            "baseline_id": b,
            "regime": r,
            "seeds_count": len(grp),
        }

        # Dominant status codes and policy compliance across seeds
        if "degeneracy_status" in grp:
            row["degeneracy_status"] = grp["degeneracy_status"].mode().iloc[0] if len(grp["degeneracy_status"]) > 0 else "NORMAL"
        if "fairness_success_status" in grp:
            row["fairness_success_status"] = grp["fairness_success_status"].mode().iloc[0] if len(grp["fairness_success_status"]) > 0 else "UNDEFINED"
        if "policy_decision" in grp:
            row["policy_decision"] = grp["policy_decision"].mode().iloc[0] if len(grp["policy_decision"]) > 0 else "POLICY_UNDEFINED"
        if "is_policy_compliant" in grp:
            row["policy_compliant_rate"] = float(grp["is_policy_compliant"].mean())
        if "result_classification" in grp:
            row["result_classification"] = grp["result_classification"].mode().iloc[0] if len(grp["result_classification"]) > 0 else "SYNTHETIC_DGP_VALIDATION"
        if "evidence_state" in grp:
            row["evidence_state"] = grp["evidence_state"].mode().iloc[0] if len(grp["evidence_state"]) > 0 else "INSUFFICIENT_EVIDENCE"
        if "policy_id" in grp:
            row["policy_id"] = grp["policy_id"].mode().iloc[0] if len(grp["policy_id"]) > 0 else "UNKNOWN"
        if "policy_version" in grp:
            row["policy_version"] = grp["policy_version"].mode().iloc[0] if len(grp["policy_version"]) > 0 else "UNKNOWN"

        for m in metric_cols:
            if m in grp.columns:
                vals = grp[m].dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    m_mean = float(np.mean(vals))
                    m_std = float(np.std(vals))
                    row[f"{m}_mean"] = m_mean
                    row[f"{m}_median"] = float(np.median(vals))
                    row[f"{m}_std"] = m_std
                    row[f"{m}_min"] = float(np.min(vals))
                    row[f"{m}_max"] = float(np.max(vals))
                    if len(vals) > 1 and m_std > 1e-9:
                        ci = 1.96 * (m_std / np.sqrt(len(vals)))
                        row[f"{m}_seed_var_95_low"] = m_mean - ci
                        row[f"{m}_seed_var_95_high"] = m_mean + ci
                        row[f"{m}_ci95_low"] = m_mean - ci
                        row[f"{m}_ci95_high"] = m_mean + ci
                    else:
                        row[f"{m}_seed_var_95_low"] = m_mean
                        row[f"{m}_seed_var_95_high"] = m_mean
                        row[f"{m}_ci95_low"] = m_mean
                        row[f"{m}_ci95_high"] = m_mean
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compute_paired_seed_differences(
    df_results: pd.DataFrame,
    baseline_a: str = "B1_Clean_Unmitigated_Linear",
    baseline_b: str = "B3_Validation_Threshold_Mitigated",
    regime: str = "GROUP_HOLDOUT",
) -> pd.DataFrame:
    """
    Computes per-seed paired differences between two baselines evaluated on identical splits.
    delta = metric(baseline_b) - metric(baseline_a).
    Provides mean paired difference, median, standard deviation, min, and max.
    """
    subset = df_results[df_results["regime"] == regime]
    df_a = subset[subset["baseline_id"] == baseline_a].set_index("seed")
    df_b = subset[subset["baseline_id"] == baseline_b].set_index("seed")

    common_seeds = df_a.index.intersection(df_b.index)
    if len(common_seeds) == 0:
        return pd.DataFrame()

    diff_metrics = [
        "accuracy", "balanced_accuracy", "equalized_odds_diff",
        "demographic_parity_diff", "disparate_impact_ratio"
    ]

    records: List[Dict[str, Any]] = []
    for metric in diff_metrics:
        if metric in df_a.columns and metric in df_b.columns:
            diffs = df_b.loc[common_seeds, metric] - df_a.loc[common_seeds, metric]
            records.append({
                "metric": metric,
                "baseline_reference": baseline_a,
                "baseline_comparison": baseline_b,
                "regime": regime,
                "seeds_count": len(common_seeds),
                "mean_paired_delta": float(diffs.mean()),
                "std_paired_delta": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                "median_paired_delta": float(diffs.median()),
                "min_paired_delta": float(diffs.min()),
                "max_paired_delta": float(diffs.max()),
            })

    return pd.DataFrame(records)


def calculate_population_dgp_truth(
    scenario: str = "SCENARIO_C_EXPLICIT_GROUP_EFFECT",
    signal_regime: str = "MODERATE_SIGNAL",
    n_mc: int = 200000,
    seed: int = 1000,
) -> Dict[str, float]:
    """
    Computes population-level ground truth quantities by large-scale Monte Carlo
    integration (N = 200,000) over the exact data-generating process equations.
    """
    rng = np.random.default_rng(seed)

    planned_distance = rng.uniform(5.0, 50.0, n_mc)
    planned_time = planned_distance * rng.uniform(1.8, 2.4, n_mc)
    stops_count = rng.integers(5, 40, n_mc).astype(float)
    vehicle_capacity = rng.uniform(100.0, 500.0, n_mc)

    if scenario == "SCENARIO_B_PREDICTIVE_FEATURE_GROUP_CORRELATION":
        is_high = rng.binomial(1, 0.45, n_mc)
        planned_distance += is_high * 15.0
        stops_count += is_high * 10.0
    elif scenario in ("SCENARIO_D_SEVERE_FAIRNESS_TRADEOFF", "SCENARIO_D_IMPOSSIBLE_FAIRNESS"):
        is_high = rng.binomial(1, 0.50, n_mc)
        planned_distance += is_high * 22.0
        stops_count += is_high * 18.0
    else:
        is_high = rng.binomial(1, 0.50, n_mc)

    x1 = (planned_distance - 27.5) / 13.0
    x2 = (planned_time - 57.5) / 28.0
    x3 = (stops_count - 22.5) / 10.0
    x4 = (vehicle_capacity - 300.0) / 115.0

    if scenario == "SCENARIO_E_CONSTANT_PREDICTOR" or signal_regime == "ZERO_SIGNAL":
        z = rng.normal(0.0, 1.0, n_mc)
        threshold = np.quantile(z, 0.106)
        y = (z >= threshold).astype(int)
    else:
        if signal_regime == "LOW_SIGNAL":
            b1, b2, b3, b4, sigma = -0.25, -0.30, -0.20, 0.15, 1.5
        elif signal_regime == "HIGH_SIGNAL":
            b1, b2, b3, b4, sigma = -1.50, -1.60, -1.20, 0.90, 0.6
        else:  # MODERATE_SIGNAL
            b1, b2, b3, b4, sigma = -0.75, -0.80, -0.65, 0.45, 1.0

        if scenario == "SCENARIO_C_EXPLICIT_GROUP_EFFECT":
            gamma = -1.25
        elif scenario in ("SCENARIO_D_SEVERE_FAIRNESS_TRADEOFF", "SCENARIO_D_IMPOSSIBLE_FAIRNESS"):
            gamma = -1.80
        else:
            gamma = 0.0

        z = 0.1 + b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 + gamma * is_high + rng.normal(0.0, sigma, n_mc)
        prob = 1.0 / (1.0 + np.exp(-z))
        y = (prob >= 0.50).astype(int)

    mask_g0 = (is_high == 0)
    mask_g1 = (is_high == 1)

    p_y1_g0 = float(np.mean(y[mask_g0]))
    p_y1_g1 = float(np.mean(y[mask_g1]))
    base_rate_diff = float(p_y1_g0 - p_y1_g1)
    base_rate_ratio = float(p_y1_g1 / max(p_y1_g0, 1e-12))
    overall_prevalence = float(np.mean(y))

    return {
        "scenario": scenario,
        "signal_regime": signal_regime,
        "population_overall_prevalence": overall_prevalence,
        "population_p_y1_given_g0": p_y1_g0,
        "population_p_y1_given_g1": p_y1_g1,
        "population_base_rate_gap": base_rate_diff,
        "population_base_rate_ratio": base_rate_ratio,
    }


def compare_sample_to_population_truth(
    sample_subgroup_metrics: Dict[str, Dict[str, Any]],
    pop_truth: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compares finite-sample group rates to theoretical population targets and reports estimation bias.
    """
    sample_p_g0 = sample_subgroup_metrics.get("Standard_Complexity", {}).get("positive_prevalence", np.nan)
    sample_p_g1 = sample_subgroup_metrics.get("High_Complexity", {}).get("positive_prevalence", np.nan)
    sample_gap = float(sample_p_g0 - sample_p_g1) if not (np.isnan(sample_p_g0) or np.isnan(sample_p_g1)) else np.nan

    return {
        "pop_p_g0": pop_truth["population_p_y1_given_g0"],
        "sample_p_g0": sample_p_g0,
        "error_p_g0": float(sample_p_g0 - pop_truth["population_p_y1_given_g0"]) if not np.isnan(sample_p_g0) else np.nan,
        "pop_p_g1": pop_truth["population_p_y1_given_g1"],
        "sample_p_g1": sample_p_g1,
        "error_p_g1": float(sample_p_g1 - pop_truth["population_p_y1_given_g1"]) if not np.isnan(sample_p_g1) else np.nan,
        "pop_base_rate_gap": pop_truth["population_base_rate_gap"],
        "sample_base_rate_gap": sample_gap,
        "error_gap": float(sample_gap - pop_truth["population_base_rate_gap"]) if not np.isnan(sample_gap) else np.nan,
    }


# ============================================================================
# Main CLI
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Canonical Fairness Benchmark Runner (Educational Multi-Seed Benchmark)")
    parser.add_argument("--demo", action="store_true", help="Execute canonical benchmark on controlled synthetic demo dataset")
    parser.add_argument("--scenario", type=str, default="SCENARIO_C_EXPLICIT_GROUP_EFFECT", help="Synthetic fairness scenario")
    parser.add_argument("--signal", type=str, default="MODERATE_SIGNAL", help="Signal regime (LOW_SIGNAL, MODERATE_SIGNAL, HIGH_SIGNAL, ZERO_SIGNAL)")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated deterministic seeds")
    parser.add_argument("--alpha", type=float, default=0.25, help="User-defined utility scalarization parameter")
    parser.add_argument("--report", action="store_true", default=True, help="Generate canonical benchmark summary CSV and markdown report")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("\n" + "=" * 78)
    print("CANONICAL FAIRNESS BENCHMARK EXECUTION")
    print(f"Seeds: {seeds} | Demo Mode: {args.demo} | Scenario: {args.scenario} | Signal: {args.signal}")
    print("=" * 78)

    delhivery_csv = DATA_DIR / "delhivery" / "delhivery_data.csv"
    bundles_to_run: List[Dict[str, Any]] = []

    if delhivery_csv.exists() and not args.demo:
        print("\n[INFO] Staged raw datasets detected. Loading clean pre-dispatch features...")
        try:
            bundles_to_run.append(load_clean_delhivery())
        except Exception as exc:
            print(f"  [WARN] Failed loading clean Delhivery: {exc}")
        try:
            bundles_to_run.append(load_clean_amazon())
        except Exception as exc:
            print(f"  [WARN] Failed loading clean Amazon: {exc}")
    else:
        print("\n[NOTICE] REAL-DATA CANONICAL RUN NOT EXECUTED: source data unavailable or demo mode requested.")
        print(f"[INFO] Running on controlled synthetic benchmark ({args.scenario} - {args.signal})...")
        bundles_to_run.append(generate_controlled_synthetic_benchmark(
            n_samples=3000,
            seed=seeds[0],
            scenario=args.scenario,
            signal_regime=args.signal,
        ))

    all_results_df_list: List[pd.DataFrame] = []

    for b in bundles_to_run:
        print(f"\n--- Benchmarking: {b['name']} ---")
        df_res = run_canonical_multi_seed_suite(
            b,
            seeds=seeds,
            regimes=[SplitRegime.IID_HOLDOUT, SplitRegime.GROUP_HOLDOUT],
            alpha=args.alpha,
        )
        all_results_df_list.append(df_res)

    if all_results_df_list:
        combined_raw = pd.concat(all_results_df_list, axis=0, ignore_index=True)
        summary_stats = aggregate_canonical_statistics(combined_raw)

        raw_csv_path = DOCS_DIR / "canonical_benchmark_raw.csv"
        summary_csv_path = DOCS_DIR / "canonical_benchmark_summary.csv"
        export_sanitized_csv(combined_raw, raw_csv_path)
        export_sanitized_csv(summary_stats, summary_csv_path)

        print("\n" + "=" * 78)
        print("CANONICAL BENCHMARK EXECUTION COMPLETE")
        print(f"Raw metrics saved to:     {raw_csv_path}")
        print(f"Summary stats saved to:   {summary_csv_path}")
        print("=" * 78 + "\n")

        display_cols = [
            "baseline_id", "regime", "accuracy_mean", "balanced_accuracy_mean",
            "equalized_odds_diff_mean", "degeneracy_status", "fairness_success_status"
        ]
        available_display = [c for c in display_cols if c in summary_stats.columns]
        print(summary_stats[available_display].to_string(index=False))


if __name__ == "__main__":
    main()
