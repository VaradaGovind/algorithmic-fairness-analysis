# ============================================================================
# Educational Inequality in Algorithmic Project Assignment
# Fairness Analysis Core Module
# ensemble evaluation, Pareto frontier analysis, and all-dataset inclusion
# ============================================================================

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from scipy.stats import norm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PLOTS_DIR = PROJECT_ROOT / "plots"
DOCS_DIR = PROJECT_ROOT / "docs"
RANDOM_STATE = 42

SYNTHETIC_LEVELS = ["High School", "Bachelor's", "Master's"]
METRIC_ORDER = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "Demographic Parity Diff",
    "Equal Opportunity Diff",
    "Equalized Odds Diff",
    "Disparate Impact Ratio",
]


@dataclass
class DatasetBundle:
    name: str
    features: pd.DataFrame
    target: pd.Series
    sensitive: pd.Series
    sensitive_attribute: str
    sensitive_source: str
    notes: str = ""


@dataclass
class CandidateEvaluation:
    name: str
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    selected_features: List[str]
    notes: str = ""


@dataclass
class DatasetResult:
    bundle: DatasetBundle
    candidates: List[CandidateEvaluation]
    pareto_frontier: List[CandidateEvaluation]
    best_candidate: CandidateEvaluation
    y_train: np.ndarray
    y_test: np.ndarray
    sensitive_test: np.ndarray
    train_sensitive: np.ndarray
    best_model: Optional[object]
    best_x_test: Optional[np.ndarray]
    best_feature_names: List[str]
    bayes_history: pd.DataFrame


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _can_stratify(y: np.ndarray) -> bool:
    unique, counts = np.unique(y, return_counts=True)
    return unique.size >= 2 and int(np.min(counts)) >= 2


def _compute_disparate_impact(y_pred: np.ndarray, sensitive_features: np.ndarray) -> float:
    groups = np.unique(sensitive_features)
    if groups.size < 2:
        return float("nan")

    rates: List[float] = []
    for group in groups:
        mask = sensitive_features == group
        if np.sum(mask) > 0:
            rates.append(float(np.mean(y_pred[mask])))

    if not rates:
        return float("nan")

    rates_arr = np.asarray(rates, dtype=float)
    max_rate = float(np.max(rates_arr))
    positive_rates = rates_arr[rates_arr > 0]
    if max_rate <= 0:
        return float("nan")
    if positive_rates.size == 0:
        return 0.0
    return float(np.min(positive_rates) / max_rate)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    sensitive_features: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    metrics["ROC-AUC"] = _safe_roc_auc(y_true, y_prob) if y_prob is not None else float("nan")

    if sensitive_features is None:
        metrics["Demographic Parity Diff"] = float("nan")
        metrics["Equal Opportunity Diff"] = float("nan")
        metrics["Equalized Odds Diff"] = float("nan")
        metrics["Disparate Impact Ratio"] = float("nan")
        return metrics

    groups = np.unique(sensitive_features)
    if groups.size < 2:
        metrics["Demographic Parity Diff"] = float("nan")
        metrics["Equal Opportunity Diff"] = float("nan")
        metrics["Equalized Odds Diff"] = float("nan")
        metrics["Disparate Impact Ratio"] = float("nan")
        return metrics

    try:
        metrics["Demographic Parity Diff"] = float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        )
    except ValueError:
        metrics["Demographic Parity Diff"] = float("nan")

    try:
        metrics["Equal Opportunity Diff"] = float(
            equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        )
    except ValueError:
        metrics["Equal Opportunity Diff"] = float("nan")

    try:
        metrics["Equalized Odds Diff"] = float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
        )
    except ValueError:
        metrics["Equalized Odds Diff"] = float("nan")

    metrics["Disparate Impact Ratio"] = _compute_disparate_impact(y_pred, sensitive_features)
    return metrics


def fairness_gap(metrics: Dict[str, float]) -> float:
    terms = [abs(metrics.get("Demographic Parity Diff", float("nan"))), abs(metrics.get("Equal Opportunity Diff", float("nan"))), abs(metrics.get("Equalized Odds Diff", float("nan")))]
    values = [term for term in terms if not pd.isna(term)]
    return float(np.mean(values)) if values else float("nan")


def candidate_score(metrics: Dict[str, float]) -> float:
    gap = fairness_gap(metrics)
    if pd.isna(gap):
        gap = 0.5
    return float(metrics.get("Balanced Accuracy", 0.0) - 0.25 * gap)


def _fmt(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def _fmt_delta(delta: float) -> str:
    return "N/A" if pd.isna(delta) else f"{delta:+.4f}"


def _get_operating_point(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return fpr, tpr


def print_metrics_table(dataset_name: str, model_name: str, metrics: Dict[str, float]) -> None:
    print(f"\n{'=' * 62}")
    print(f"{dataset_name.upper()} | {model_name.upper()} METRICS")
    print(f"{'=' * 62}")
    print(f"{'Metric Name':<32} | {'Value':>12}")
    print("-" * 62)
    for metric in METRIC_ORDER:
        print(f"{metric:<32} | {_fmt(metrics.get(metric, float('nan'))):>12}")
    print(f"{'Fairness Gap':<32} | {_fmt(fairness_gap(metrics)):>12}")
    print(f"{'Composite Score':<32} | {_fmt(candidate_score(metrics)):>12}")
    print("=" * 62)


def plot_combined_confusion(y_true: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, output_path: Path, left_name: str, right_name: str) -> None:
    sns.set_theme(style="white", palette="pastel")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_true, y_left, labels=[0, 1]), annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title(left_name)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    sns.heatmap(confusion_matrix(y_true, y_right, labels=[0, 1]), annot=True, fmt="d", cmap="YlGn", ax=axes[1], cbar=False)
    axes[1].set_title(right_name)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pareto_frontier(result: DatasetResult, output_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "Model": candidate.name,
                "Accuracy": candidate.metrics["Accuracy"],
                "Fairness Gap": fairness_gap(candidate.metrics),
                "Pareto": candidate in result.pareto_frontier,
            }
            for candidate in result.candidates
        ]
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="Fairness Gap", y="Accuracy", hue="Model", style="Pareto", s=120)
    for _, row in df.iterrows():
        plt.text(row["Fairness Gap"] + 0.001, row["Accuracy"] + 0.001, row["Model"], fontsize=8)
    plt.title(f"Pareto Frontier - {result.bundle.name}")
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_salary_education_prior() -> Dict[str, float]:
    salary_path = DATA_DIR / "salary" / "Salary_Data.csv"
    if not salary_path.exists():
        return {level: 1 / len(SYNTHETIC_LEVELS) for level in SYNTHETIC_LEVELS}

    salary_df = pd.read_csv(salary_path)
    if "Education Level" not in salary_df.columns:
        return {level: 1 / len(SYNTHETIC_LEVELS) for level in SYNTHETIC_LEVELS}

    mapping = {"High School": "High School", "Bachelor's": "Bachelor's", "Master's": "Master's", "PhD": "Master's"}
    mapped = salary_df["Education Level"].map(mapping).dropna()
    proportions = mapped.value_counts(normalize=True)
    prior = {level: float(proportions.get(level, 0.0)) for level in SYNTHETIC_LEVELS}
    total = sum(prior.values())
    if total <= 0:
        return {level: 1 / len(SYNTHETIC_LEVELS) for level in SYNTHETIC_LEVELS}
    return {level: value / total for level, value in prior.items()}


def assign_synthetic_education(signal: pd.Series, base_prior: Dict[str, float], seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    signal = pd.Series(signal).astype(float).replace([np.inf, -np.inf], np.nan)
    signal = signal.fillna(signal.median())
    rank = signal.rank(pct=True).to_numpy()

    base = np.asarray([base_prior[level] for level in SYNTHETIC_LEVELS], dtype=float)
    base = base / base.sum()
    weights = np.tile(base, (signal.shape[0], 1))
    complexity_shift = np.clip(rank - 0.5, -0.5, 0.5)

    weights[:, 0] *= 1.0 + np.maximum(complexity_shift, 0.0) * 1.8
    weights[:, 1] *= 1.0 - np.abs(complexity_shift) * 0.35
    weights[:, 2] *= 1.0 + np.maximum(-complexity_shift, 0.0) * 1.4
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum(axis=1, keepdims=True)

    sampled = [rng.choice(SYNTHETIC_LEVELS, p=row) for row in weights]
    return pd.Series(sampled, index=signal.index, name="Education_Level")


def load_delhivery_dataset(base_prior: Dict[str, float]) -> DatasetBundle:
    df = pd.read_csv(DATA_DIR / "delhivery" / "delhivery_data.csv")
    cutoff_flag = df["is_cutoff"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df["Task_Success"] = (~cutoff_flag).astype(int)

    feature_cols = [
        "actual_distance_to_destination",
        "actual_time",
        "osrm_distance",
        "osrm_time",
        "segment_actual_time",
        "segment_osrm_time",
        "segment_osrm_distance",
    ]
    df = df.dropna(subset=feature_cols + ["Task_Success"]).copy()

    df["actual_vs_osrm_distance_ratio"] = df["actual_distance_to_destination"] / (df["osrm_distance"] + 1e-6)
    df["actual_vs_osrm_time_ratio"] = df["actual_time"] / (df["osrm_time"] + 1e-6)
    df["time_per_distance_actual"] = df["actual_time"] / (df["actual_distance_to_destination"] + 1e-6)
    df["time_per_distance_osrm"] = df["osrm_time"] / (df["osrm_distance"] + 1e-6)
    df["segment_deviation"] = np.abs(df["segment_actual_time"] - df["segment_osrm_time"])
    feature_cols.extend(["actual_vs_osrm_distance_ratio", "actual_vs_osrm_time_ratio", "time_per_distance_actual", "time_per_distance_osrm", "segment_deviation"])

    proxy_signal = df["osrm_distance"].rank(pct=True) * 0.45 + df["actual_time"].rank(pct=True) * 0.35 + df["segment_osrm_distance"].rank(pct=True) * 0.20
    df["Education_Level"] = assign_synthetic_education(proxy_signal, base_prior, RANDOM_STATE)

    return DatasetBundle(
        name="Delhivery Logistics",
        features=df[feature_cols].copy(),
        target=df["Task_Success"].astype(int).copy(),
        sensitive=df["Education_Level"].copy(),
        sensitive_attribute="Education_Level",
        sensitive_source="Synthetic (data-driven proxy)",
        notes="Primary logistics dataset with enhanced features.",
    )


def load_adult_dataset() -> DatasetBundle:
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
        "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
    ]
    train_df = pd.read_csv(DATA_DIR / "adult" / "adult.data", names=columns, skipinitialspace=True, na_values="?")
    test_df = pd.read_csv(DATA_DIR / "adult" / "adult.test", names=columns, skiprows=1, skipinitialspace=True, na_values="?")
    adult_df = pd.concat([train_df, test_df], ignore_index=True)
    for col in adult_df.select_dtypes(include=["object", "string"]).columns:
        adult_df[col] = adult_df[col].str.strip()
    adult_df["income"] = adult_df["income"].str.replace(".", "", regex=False)
    adult_df = adult_df.dropna().copy()
    adult_df["Task_Success"] = (adult_df["income"] == ">50K").astype(int)
    adult_df["Education_Group"] = np.where(adult_df["education-num"] >= 13, "HigherEd", "NonHigherEd")

    adult_df["capital_total"] = adult_df["capital-gain"] + adult_df["capital-loss"]
    adult_df["capital_ratio"] = (adult_df["capital-gain"] - adult_df["capital-loss"]) / (adult_df["capital_total"] + 1)
    adult_df["work_intensity"] = adult_df["hours-per-week"] / 40.0
    adult_df["age_group"] = pd.cut(adult_df["age"], bins=[0, 25, 45, 65, 100], labels=["Young", "Prime", "Late", "Senior"])

    feature_cols = [
        "age", "workclass", "fnlwgt", "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "capital_total", "capital_ratio", "work_intensity",
    ]

    return DatasetBundle(
        name="Adult Income Benchmark",
        features=adult_df[feature_cols].copy(),
        target=adult_df["Task_Success"].astype(int).copy(),
        sensitive=adult_df["Education_Group"].copy(),
        sensitive_attribute="Education_Group",
        sensitive_source="Real attribute (observed education)",
        notes="Real-world fairness benchmark with enhanced features.",
    )


def _extract_route_features(route_payload: Dict[str, object]) -> Optional[Dict[str, float]]:
    stops = route_payload.get("stops", {})
    if not isinstance(stops, dict):
        return None

    points: List[Tuple[float, float]] = []
    dropoff_count = 0
    for stop_data in stops.values():
        if not isinstance(stop_data, dict):
            continue
        lat = stop_data.get("lat")
        lng = stop_data.get("lng")
        stop_type = stop_data.get("type")
        if lat is None or lng is None:
            continue
        try:
            lat_f = float(lat)
            lng_f = float(lng)
            if np.isnan(lat_f) or np.isnan(lng_f):
                continue
        except (TypeError, ValueError):
            continue
        points.append((lat_f, lng_f))
        if stop_type == "Dropoff":
            dropoff_count += 1

    if len(points) < 2:
        return None

    coords = np.asarray(points, dtype=float)
    centroid = coords.mean(axis=0)
    radial_dispersion = float(np.mean(np.sqrt(np.sum((coords - centroid) ** 2, axis=1))))
    return {
        "num_stops_total": float(len(points)),
        "num_dropoffs": float(dropoff_count),
        "lat_mean": float(np.mean(coords[:, 0])),
        "lng_mean": float(np.mean(coords[:, 1])),
        "lat_std": float(np.std(coords[:, 0])),
        "lng_std": float(np.std(coords[:, 1])),
        "bbox_area": float((np.max(coords[:, 0]) - np.min(coords[:, 0])) * (np.max(coords[:, 1]) - np.min(coords[:, 1]))),
        "radial_dispersion": radial_dispersion,
    }


def _parse_departure_hour(raw_departure: object) -> float:
    if raw_departure is None:
        return float("nan")
    try:
        return float(int(str(raw_departure).split(":")[0]))
    except (ValueError, TypeError, IndexError):
        return float("nan")


def load_amazon_dataset(base_prior: Dict[str, float]) -> DatasetBundle:
    route_path = DATA_DIR / "amazon_data" / "almrrc2021-data-training" / "model_build_inputs" / "route_data.json"
    invalid_score_path = DATA_DIR / "amazon_data" / "almrrc2021-data-training" / "model_build_inputs" / "invalid_sequence_scores.json"
    with route_path.open("r", encoding="utf-8") as fp:
        route_data = json.load(fp)
    with invalid_score_path.open("r", encoding="utf-8") as fp:
        invalid_scores = json.load(fp)

    records: List[Dict[str, object]] = []
    for route_id, payload in route_data.items():
        if not isinstance(payload, dict):
            continue
        route_score = str(payload.get("route_score", "")).strip()
        if not route_score:
            continue
        route_features = _extract_route_features(payload)
        if route_features is None:
            continue

        row: Dict[str, object] = {
            "route_id": route_id,
            "station_code": str(payload.get("station_code", "UNKNOWN")),
            "departure_hour": _parse_departure_hour(payload.get("departure_time_utc")),
            "executor_capacity_cm3": payload.get("executor_capacity_cm3"),
            "route_score_label": route_score,
            "Task_Success": int(route_score.lower() == "high"),
            "invalid_sequence_score": invalid_scores.get(route_id, np.nan),
        }
        row.update(route_features)
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Amazon route dataset produced an empty table after parsing.")

    numeric_cols = [
        "departure_hour", "executor_capacity_cm3", "invalid_sequence_score", "num_stops_total", "num_dropoffs",
        "lat_mean", "lng_mean", "lat_std", "lng_std", "bbox_area", "radial_dispersion",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Task_Success", "executor_capacity_cm3", "num_dropoffs"]).copy()
    df["invalid_sequence_score"] = df["invalid_sequence_score"].fillna(df["invalid_sequence_score"].median())
    df["departure_hour"] = df["departure_hour"].fillna(df["departure_hour"].median())

    df["stops_per_dropoff_ratio"] = df["num_stops_total"] / (df["num_dropoffs"] + 1e-6)
    df["route_efficiency"] = df["executor_capacity_cm3"] / (df["invalid_sequence_score"] + 1e-6)
    df["spatial_density"] = df["num_dropoffs"] / (df["bbox_area"] + 1e-6)
    numeric_cols.extend(["stops_per_dropoff_ratio", "route_efficiency", "spatial_density"])

    proxy_signal = df["num_dropoffs"].rank(pct=True) * 0.50 + df["bbox_area"].rank(pct=True) * 0.25 + df["invalid_sequence_score"].rank(pct=True) * 0.25
    df["Education_Level"] = assign_synthetic_education(proxy_signal, base_prior, RANDOM_STATE + 1)

    return DatasetBundle(
        name="Amazon Last-Mile Routes",
        features=df[numeric_cols + ["station_code"]].copy(),
        target=df["Task_Success"].astype(int).copy(),
        sensitive=df["Education_Level"].copy(),
        sensitive_attribute="Education_Level",
        sensitive_source="Synthetic (data-driven proxy)",
        notes="Amazon routes included in the full dataset evaluation.",
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def sample_stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
    max_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) <= max_rows:
        return X, y, sensitive

    _, X_subset, _, y_subset, _, sensitive_subset = train_test_split(
        X,
        y,
        sensitive,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=y if _can_stratify(y) else None,
    )
    return X_subset, y_subset, sensitive_subset


def xgb_params_to_vector(params: Dict[str, float]) -> np.ndarray:
    return np.asarray([
        params["max_depth"], params["learning_rate"], params["n_estimators"], params["subsample"], params["colsample_bytree"],
        params["min_child_weight"], params["gamma"], params["reg_alpha"], params["reg_lambda"], params["max_delta_step"],
    ], dtype=float)


def vector_to_xgb_params(vector: Sequence[float]) -> Dict[str, float]:
    return {
        "max_depth": int(round(vector[0])),
        "learning_rate": float(vector[1]),
        "n_estimators": int(round(vector[2])),
        "subsample": float(vector[3]),
        "colsample_bytree": float(vector[4]),
        "min_child_weight": int(round(vector[5])),
        "gamma": float(vector[6]),
        "reg_alpha": float(vector[7]),
        "reg_lambda": float(vector[8]),
        "max_delta_step": int(round(vector[9])),
    }


def sample_random_xgb_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "max_depth": int(rng.integers(3, 12)),
        "learning_rate": float(rng.uniform(0.02, 0.18)),
        "n_estimators": int(rng.integers(100, 1000)),
        "subsample": float(rng.uniform(0.65, 1.0)),
        "colsample_bytree": float(rng.uniform(0.65, 1.0)),
        "min_child_weight": int(rng.integers(1, 8)),
        "gamma": float(rng.uniform(0.0, 4.0)),
        "reg_alpha": float(rng.uniform(0.0, 0.8)),
        "reg_lambda": float(rng.uniform(0.8, 4.5)),
        "max_delta_step": int(rng.integers(0, 4)),
    }


def build_xgb_model(params: Dict[str, float], scale_pos_weight: float) -> xgb.XGBClassifier:
    model_params = dict(params)
    model_params.update(
        {
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "verbosity": 0,
        }
    )
    return xgb.XGBClassifier(**model_params)


def evaluate_params_cv(
    params: Dict[str, float],
    X: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
    scale_pos_weight: float,
    n_splits: int = 2,
) -> Tuple[float, Dict[str, float]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: List[float] = []
    metric_store: List[Dict[str, float]] = []

    for train_idx, val_idx in cv.split(X, y):
        model = build_xgb_model(params, scale_pos_weight)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx]).astype(int)
        y_prob = model.predict_proba(X[val_idx])[:, 1]
        metrics = calculate_metrics(y[val_idx], y_pred, y_prob, sensitive[val_idx])
        metric_store.append(metrics)
        score = candidate_score(metrics)
        scores.append(score)

    average_metrics = {
        key: float(np.nanmean([metrics.get(key, np.nan) for metrics in metric_store]))
        for key in METRIC_ORDER
    }
    return float(np.mean(scores)), average_metrics


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_y
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def bayesian_optimize_xgb(
    X: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
    scale_pos_weight: float,
    budget: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    bounds = np.asarray([
        [3.0, 8.0], [0.02, 0.18], [140.0, 360.0], [0.65, 1.0], [0.65, 1.0],
        [1.0, 8.0], [0.0, 4.0], [0.0, 0.8], [0.8, 4.5], [0.0, 3.0],
    ], dtype=float)

    observations: List[np.ndarray] = []
    scores: List[float] = []
    metrics_rows: List[Dict[str, float]] = []

    initial_points = min(3, budget)
    for _ in range(initial_points):
        params = sample_random_xgb_params(rng)
        score, averaged_metrics = evaluate_params_cv(params, X, y, sensitive, scale_pos_weight)
        observations.append(xgb_params_to_vector(params))
        scores.append(score)
        metrics_rows.append({"score": score, **params, **averaged_metrics})

    while len(scores) < budget:
        X_obs = np.asarray(observations, dtype=float)
        y_obs = np.asarray(scores, dtype=float)
        kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(length_scale=np.ones(X_obs.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RANDOM_STATE, n_restarts_optimizer=0)
        gp.fit(X_obs, y_obs)

        candidate_pool = np.column_stack([
            rng.uniform(low=bounds[i, 0], high=bounds[i, 1], size=48)
            for i in range(bounds.shape[0])
        ])
        mu, sigma = gp.predict(candidate_pool, return_std=True)
        ei = expected_improvement(mu, sigma, float(np.max(y_obs)))
        best_candidate = candidate_pool[int(np.argmax(ei))]
        candidate_params = vector_to_xgb_params(best_candidate)
        score, averaged_metrics = evaluate_params_cv(candidate_params, X, y, sensitive, scale_pos_weight)
        observations.append(xgb_params_to_vector(candidate_params))
        scores.append(score)
        metrics_rows.append({"score": score, **candidate_params, **averaged_metrics})

    history = pd.DataFrame(metrics_rows).sort_values("score", ascending=False).reset_index(drop=True)
    best_params = vector_to_xgb_params(history.iloc[0][["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda", "max_delta_step"]].to_numpy())
    return best_params, history


def fairness_aware_rfe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    feature_names: List[str],
    params: Dict[str, float],
    scale_pos_weight: float,
) -> Tuple[List[int], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    split_idx = rng.permutation(len(y_train))
    cutoff = int(len(split_idx) * 0.8)
    train_idx, val_idx = split_idx[:cutoff], split_idx[cutoff:]

    selected = list(range(X_train.shape[1]))
    best_selected = selected.copy()
    best_score = -np.inf
    records: List[Dict[str, float]] = []
    min_features = max(8, int(X_train.shape[1] * 0.25))
    max_rounds = min(8, X_train.shape[1] - min_features)

    for round_idx in range(max_rounds + 1):
        model = build_xgb_model(params, scale_pos_weight)
        model.fit(X_train[train_idx][:, selected], y_train[train_idx])
        val_pred = model.predict(X_train[val_idx][:, selected]).astype(int)
        val_prob = model.predict_proba(X_train[val_idx][:, selected])[:, 1]
        metrics = calculate_metrics(y_train[val_idx], val_pred, val_prob, sensitive_train[val_idx])
        score = candidate_score(metrics)
        records.append({"round": round_idx, "features": len(selected), "score": score, **metrics})

        if score > best_score:
            best_score = score
            best_selected = selected.copy()

        if len(selected) <= min_features:
            break

        importances = model.feature_importances_
        remove_count = max(1, int(len(selected) * 0.15))
        remove_idx_local = np.argsort(importances)[:remove_count]
        keep_mask = np.ones(len(selected), dtype=bool)
        keep_mask[remove_idx_local] = False
        selected = [feature for keep, feature in zip(keep_mask, selected) if keep]

    return best_selected, pd.DataFrame(records)


def build_candidate(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_test: np.ndarray,
    selected_feature_names: List[str],
    y_prob_method: str = "predict_proba",
) -> CandidateEvaluation:
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test).astype(int)
    y_prob = None
    if hasattr(estimator, y_prob_method):
        pred_output = getattr(estimator, y_prob_method)(X_test)
        if isinstance(pred_output, list):
            pred_output = pred_output[0]
        if np.asarray(pred_output).ndim == 2 and np.asarray(pred_output).shape[1] >= 2:
            y_prob = np.asarray(pred_output)[:, 1]
        else:
            y_prob = np.asarray(pred_output).ravel()
    metrics = calculate_metrics(y_test, y_pred, y_prob, sensitive_test)
    return CandidateEvaluation(name=name, metrics=metrics, y_pred=y_pred, y_prob=y_prob, selected_features=selected_feature_names)


def pareto_frontier(candidates: List[CandidateEvaluation]) -> List[CandidateEvaluation]:
    def dominates(a: CandidateEvaluation, b: CandidateEvaluation) -> bool:
        acc_a = a.metrics["Accuracy"]
        acc_b = b.metrics["Accuracy"]
        gap_a = fairness_gap(a.metrics)
        gap_b = fairness_gap(b.metrics)
        better_or_equal = (acc_a >= acc_b) and (gap_a <= gap_b)
        strictly_better = (acc_a > acc_b) or (gap_a < gap_b)
        return better_or_equal and strictly_better

    frontier: List[CandidateEvaluation] = []
    for candidate in candidates:
        if any(dominates(other, candidate) for other in candidates if other is not candidate):
            continue
        frontier.append(candidate)
    frontier.sort(key=lambda item: (fairness_gap(item.metrics), -item.metrics["Accuracy"]))
    return frontier


def generate_shap_plot(model, X_sample: np.ndarray, feature_names: List[str], output_path: Path) -> None:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=min(20, len(feature_names)))
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"  [WARN] SHAP generation failed: {exc}")


def build_results_dataframe(results: List[DatasetResult]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for result in results:
        for candidate in result.candidates:
            row = {
                "Dataset": result.bundle.name,
                "Sensitive Attribute": result.bundle.sensitive_attribute,
                "Sensitive Source": result.bundle.sensitive_source,
                "Model": candidate.name,
                "Composite Score": candidate_score(candidate.metrics),
                "Fairness Gap": fairness_gap(candidate.metrics),
                "Selected Features": len(candidate.selected_features),
            }
            for metric in METRIC_ORDER:
                row[metric] = candidate.metrics.get(metric, float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(results: List[DatasetResult], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Final Analysis Results")
    lines.append("")
    lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## What Changed")
    lines.append("- Bayesian hyperparameter optimization for XGBoost")
    lines.append("- Fairness-aware recursive feature elimination")
    lines.append("- Ensemble comparison across RF, SVM, and neural networks")
    lines.append("- Pareto frontier analysis for accuracy/fairness tradeoffs")
    lines.append("- Amazon dataset included in the full run")
    lines.append("")

    for result in results:
        lines.append(f"## {result.bundle.name}")
        lines.append("")
        lines.append(f"- Sensitive Attribute: {result.bundle.sensitive_attribute}")
        lines.append(f"- Sensitive Source: {result.bundle.sensitive_source}")
        lines.append(f"- Notes: {result.bundle.notes}")
        lines.append("")
        lines.append("### Best Candidate")
        lines.append(f"- Model: {result.best_candidate.name}")
        lines.append(f"- Composite Score: {candidate_score(result.best_candidate.metrics):.4f}")
        lines.append(f"- Fairness Gap: {fairness_gap(result.best_candidate.metrics):.4f}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for metric in METRIC_ORDER:
            lines.append(f"| {metric} | {_fmt(result.best_candidate.metrics.get(metric, float('nan')))} |")
        lines.append("")
        lines.append("### Pareto Frontier Models")
        for candidate in result.pareto_frontier:
            lines.append(f"- {candidate.name}: acc={candidate.metrics['Accuracy']:.4f}, gap={fairness_gap(candidate.metrics):.4f}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_bundle(bundle: DatasetBundle) -> DatasetResult:
    X = bundle.features.copy()
    y = bundle.target.to_numpy(dtype=int)
    sensitive = bundle.sensitive.to_numpy()

    stratify = y if _can_stratify(y) else None
    X_train_raw, X_test_raw, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )

    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_ratio = float(class_weights[1] / class_weights[0]) if class_weights[0] > 0 else 1.0

    budget = 3 if len(y_train) > 50000 else 4
    print(f"  Bayesian search budget: {budget} iterations")
    opt_max_rows = 4000 if len(y_train) > 4000 else len(y_train)
    X_opt, y_opt, s_opt = sample_stratified_subset(X_train, y_train, s_train, opt_max_rows)
    best_params, bayes_history = bayesian_optimize_xgb(X_opt, y_opt, s_opt, class_weight_ratio, budget=budget)

    model_train_X = X_opt
    model_train_y = y_opt
    model_train_s = s_opt

    tuned_xgb = build_xgb_model(best_params, class_weight_ratio)
    tuned_candidate = build_candidate(
        name="Bayesian XGB",
        estimator=clone(tuned_xgb),
        X_train=model_train_X,
        y_train=model_train_y,
        X_test=X_test,
        y_test=y_test,
        sensitive_test=s_test,
        selected_feature_names=feature_names,
    )

    selected_idx, rfe_history = fairness_aware_rfe(model_train_X, model_train_y, model_train_s, feature_names, best_params, class_weight_ratio)
    rfe_xgb = build_xgb_model(best_params, class_weight_ratio)
    rfe_candidate = build_candidate(
        name="Fairness RFE XGB",
        estimator=clone(rfe_xgb),
        X_train=model_train_X[:, selected_idx],
        y_train=model_train_y,
        X_test=X_test[:, selected_idx],
        y_test=y_test,
        sensitive_test=s_test,
        selected_feature_names=[feature_names[i] for i in selected_idx],
    )

    rf = RandomForestClassifier(n_estimators=180, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
    rf_candidate = build_candidate("Random Forest", rf, model_train_X, model_train_y, X_test, y_test, s_test, feature_names)

    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000), cv=2, method="sigmoid")
    svm_candidate = build_candidate("Calibrated Linear SVM", svm, model_train_X, model_train_y, X_test, y_test, s_test, feature_names)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, max_iter=220, random_state=RANDOM_STATE)
    mlp_candidate = build_candidate("Neural Network", mlp, model_train_X, model_train_y, X_test, y_test, s_test, feature_names)

    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
            ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000), cv=2, method="sigmoid")),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, max_iter=220, random_state=RANDOM_STATE)),
        ],
        voting="soft",
        n_jobs=None,
    )
    ensemble_candidate = build_candidate("Soft Voting Ensemble", ensemble, model_train_X, model_train_y, X_test, y_test, s_test, feature_names)

    mitigated_candidate = tuned_candidate
    try:
        mitigator = ThresholdOptimizer(estimator=clone(tuned_xgb), constraints="equalized_odds", predict_method="predict_proba")
        mitigator.fit(model_train_X, model_train_y, sensitive_features=model_train_s)
        mitigated_pred = np.asarray(mitigator.predict(X_test, sensitive_features=s_test)).astype(int)
        mitigated_candidate = CandidateEvaluation(
            name="Fairness Mitigated XGB",
            metrics=calculate_metrics(y_test, mitigated_pred, None, s_test),
            y_pred=mitigated_pred,
            y_prob=None,
            selected_features=feature_names,
            notes="ThresholdOptimizer with equalized odds.",
        )
    except Exception:
        try:
            mitigator = ExponentiatedGradient(estimator=clone(tuned_xgb), constraints=EqualizedOdds(), eps=0.01)
            mitigator.fit(model_train_X, model_train_y, sensitive_features=model_train_s)
            mitigated_pred = np.asarray(mitigator.predict(X_test)).astype(int)
            mitigated_candidate = CandidateEvaluation(
                name="Fairness Mitigated XGB",
                metrics=calculate_metrics(y_test, mitigated_pred, None, s_test),
                y_pred=mitigated_pred,
                y_prob=None,
                selected_features=feature_names,
                notes="ExponentiatedGradient fallback.",
            )
        except Exception as exc:
            print(f"  [WARN] Fairness mitigation fallback failed: {exc}")

    candidates = [
        tuned_candidate,
        rfe_candidate,
        rf_candidate,
        svm_candidate,
        mlp_candidate,
        ensemble_candidate,
        mitigated_candidate,
    ]
    frontier = pareto_frontier(candidates)
    best_candidate = max(candidates, key=lambda cand: (candidate_score(cand.metrics), cand.metrics["Accuracy"]))

    best_model = None
    best_x_test = None
    if best_candidate.name == "Bayesian XGB":
        best_model = tuned_xgb.fit(model_train_X, model_train_y)
        best_x_test = X_test
    elif best_candidate.name == "Fairness RFE XGB":
        best_model = build_xgb_model(best_params, class_weight_ratio).fit(model_train_X[:, selected_idx], model_train_y)
        best_x_test = X_test[:, selected_idx]
    elif best_candidate.name == "Fairness Mitigated XGB":
        best_model = clone(tuned_xgb).fit(model_train_X, model_train_y)
        best_x_test = X_test

    return DatasetResult(
        bundle=bundle,
        candidates=candidates,
        pareto_frontier=frontier,
        best_candidate=best_candidate,
        y_train=y_train,
        y_test=y_test,
        sensitive_test=s_test,
        train_sensitive=s_train,
        best_model=best_model,
        best_x_test=best_x_test,
        best_feature_names=[candidate.selected_features for candidate in candidates if candidate.name == best_candidate.name][0],
        bayes_history=bayes_history,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    np.random.seed(RANDOM_STATE)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 74)
    print("Fairness Analysis Core Module")
    print("=" * 74)

    prior = get_salary_education_prior()
    bundles = [load_delhivery_dataset(prior), load_adult_dataset(), load_amazon_dataset(prior)]

    results: List[DatasetResult] = []
    for bundle in bundles:
        print("\n" + "-" * 74)
        print(f"Evaluating {bundle.name}")
        print(f"Sensitive attribute: {bundle.sensitive_attribute} ({bundle.sensitive_source})")
        result = evaluate_bundle(bundle)
        results.append(result)

        print_metrics_table(bundle.name, result.best_candidate.name, result.best_candidate.metrics)
        print(f"  Pareto frontier size: {len(result.pareto_frontier)}")

        if result.best_model is not None and result.best_x_test is not None and isinstance(result.best_model, xgb.XGBClassifier):
            sample_size = min(1500, result.best_x_test.shape[0])
            sample_idx = np.random.default_rng(RANDOM_STATE).choice(result.best_x_test.shape[0], size=sample_size, replace=False)
            generate_shap_plot(result.best_model, result.best_x_test[sample_idx], result.best_feature_names, PLOTS_DIR / f"{bundle.name.lower().replace(' ', '_')}_shap_core.png")

        plot_pareto_frontier(result, PLOTS_DIR / f"{bundle.name.lower().replace(' ', '_')}_pareto_core.png")
        if result.best_candidate.y_pred is not None:
            plot_combined_confusion(result.y_test, result.best_candidate.y_pred, result.pareto_frontier[0].y_pred, PLOTS_DIR / f"{bundle.name.lower().replace(' ', '_')}_confusion_core.png", result.best_candidate.name, result.pareto_frontier[0].name)

    results_df = build_results_dataframe(results)
    results_df.to_csv(DOCS_DIR / "metrics_summary_core.csv", index=False)

    report_path = DOCS_DIR / "Iteration3_Report.md"
    write_report(results, report_path)

    frontier_rows = []
    for result in results:
        for candidate in result.pareto_frontier:
            frontier_rows.append({
                "Dataset": result.bundle.name,
                "Model": candidate.name,
                "Accuracy": candidate.metrics["Accuracy"],
                "Fairness Gap": fairness_gap(candidate.metrics),
                "Composite Score": candidate_score(candidate.metrics),
            })
    pd.DataFrame(frontier_rows).to_csv(DOCS_DIR / "pareto_frontier_v3.csv", index=False)

    print("\n" + "=" * 74)
    print("Final Analysis COMPLETE")
    print("=" * 74)
    print(f"Report: {report_path}")
    print(f"Metrics: {DOCS_DIR / 'metrics_summary_core.csv'}")
    print(f"Pareto frontier: {DOCS_DIR / 'pareto_frontier_v3.csv'}")
    print(f"Plots: {PLOTS_DIR}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()

