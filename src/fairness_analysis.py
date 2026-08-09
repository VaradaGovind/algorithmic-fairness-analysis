# ============================================================================
# Educational Inequality in Algorithmic Project Assignment
# Final Fairness Analysis Pipeline: Causal-style checks, reweighting, subgroup fairness,
# robustness analysis, and enhanced tradeoff reporting
# ============================================================================

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import fairness_core as base


PROJECT_ROOT = base.PROJECT_ROOT
DATA_DIR = base.DATA_DIR
PLOTS_DIR = PROJECT_ROOT / "plots"
DOCS_DIR = PROJECT_ROOT / "docs"
RANDOM_STATE = base.RANDOM_STATE
METRIC_ORDER = base.METRIC_ORDER


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    weight_strategy: str


@dataclass
class DatasetStudy:
    bundle: base.DatasetBundle
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    sensitive_train: np.ndarray
    sensitive_test: np.ndarray
    feature_names: List[str]
    preprocessor: ColumnTransformer
    x_train: np.ndarray
    x_test: np.ndarray
    results: List[ModelResult]
    best_result: ModelResult
    subgroup_tables: Dict[str, pd.DataFrame]
    causal_summary: Optional[pd.DataFrame]
    robustness_summary: Optional[pd.DataFrame]


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def predictive_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    groups = np.unique(sensitive)
    ppv_values: List[float] = []
    for group in groups:
        mask = sensitive == group
        if np.sum(mask) == 0:
            continue
        predicted_positive = y_pred[mask] == 1
        if np.sum(predicted_positive) == 0:
            ppv_values.append(0.0)
        else:
            ppv_values.append(float(np.mean(y_true[mask][predicted_positive] == 1)))
    if len(ppv_values) < 2:
        return float("nan")
    return float(np.max(ppv_values) - np.min(ppv_values))


def disparate_impact_ratio(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    groups = np.unique(sensitive)
    if groups.size < 2:
        return float("nan")
    rates = []
    for group in groups:
        mask = sensitive == group
        if np.sum(mask) > 0:
            rates.append(float(np.mean(y_pred[mask])))
    if not rates:
        return float("nan")
    rates_arr = np.asarray(rates, dtype=float)
    high = float(np.max(rates_arr))
    low = float(np.min(rates_arr))
    if high <= 0:
        return float("nan")
    return float(low / high)


def fairness_gap(metrics: Dict[str, float]) -> float:
    values = [
        abs(metrics.get("Demographic Parity Diff", float("nan"))),
        abs(metrics.get("Equal Opportunity Diff", float("nan"))),
        abs(metrics.get("Equalized Odds Diff", float("nan"))),
        abs(metrics.get("Predictive Parity Diff", float("nan"))),
    ]
    values = [value for value in values if not pd.isna(value)]
    return float(np.mean(values)) if values else float("nan")


def candidate_score(metrics: Dict[str, float]) -> float:
    gap = fairness_gap(metrics)
    if pd.isna(gap):
        gap = 0.5
    return float(metrics.get("Balanced Accuracy", 0.0) - 0.25 * gap)


def reweighing_weights(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    sensitive = np.asarray(sensitive)
    overall_y = pd.Series(y).value_counts(normalize=True).to_dict()
    overall_a = pd.Series(sensitive).value_counts(normalize=True).to_dict()
    joint = pd.crosstab(sensitive, y, normalize=True)

    weights = np.zeros_like(y, dtype=float)
    for idx, (label, group) in enumerate(zip(y, sensitive)):
        p_a = float(overall_a.get(group, 0.0))
        p_y = float(overall_y.get(label, 0.0))
        p_ay = float(joint.loc[group, label]) if group in joint.index and label in joint.columns else 0.0
        weights[idx] = (p_a * p_y) / max(p_ay, 1e-9)

    weights = np.clip(weights, 1e-3, None)
    weights = weights / np.mean(weights)
    return weights


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    steps = []
    if numeric_cols:
        steps.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
    if categorical_cols:
        steps.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(steps, remainder="drop")


def sample_rows(X: pd.DataFrame, y: pd.Series, sensitive: pd.Series, cap: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if len(X) <= cap:
        return X.copy(), y.copy(), sensitive.copy()
    X_sample, _, y_sample, _, s_sample, _ = train_test_split(
        X,
        y,
        sensitive,
        train_size=cap,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )
    return X_sample, y_sample, s_sample


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], sensitive: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": _safe_roc_auc(y_true, y_prob) if y_prob is not None else float("nan"),
    }
    try:
        metrics["Demographic Parity Diff"] = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive))
    except Exception:
        metrics["Demographic Parity Diff"] = float("nan")
    try:
        metrics["Equal Opportunity Diff"] = float(equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive))
    except Exception:
        metrics["Equal Opportunity Diff"] = float("nan")
    try:
        metrics["Equalized Odds Diff"] = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive))
    except Exception:
        metrics["Equalized Odds Diff"] = float("nan")
    metrics["Predictive Parity Diff"] = predictive_parity_difference(y_true, y_pred, sensitive)
    metrics["Disparate Impact Ratio"] = disparate_impact_ratio(y_pred, sensitive)
    return metrics


def fit_models(x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray) -> Dict[str, object]:
    positive = float(np.sum(y_train == 1))
    negative = float(np.sum(y_train == 0))
    scale_pos_weight = max(negative / max(positive, 1.0), 1.0)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Reweighted Logistic": LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=120, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(48, 24), alpha=0.001, max_iter=160, random_state=RANDOM_STATE),
        "Weighted XGB": base.xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.08,
            n_estimators=160,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.4,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        ),
    }

    fitted: Dict[str, object] = {}
    for name, model in models.items():
        if name == "Reweighted Logistic":
            fitted[name] = model.fit(x_train, y_train, sample_weight=sample_weight)
        elif name == "Neural Network":
            fitted[name] = model.fit(x_train, y_train)
        else:
            try:
                fitted[name] = model.fit(x_train, y_train, sample_weight=sample_weight)
            except TypeError:
                fitted[name] = model.fit(x_train, y_train)

    return fitted


def evaluate_model(name: str, model, x_test: np.ndarray, y_test: np.ndarray, sensitive_test: np.ndarray) -> ModelResult:
    y_pred = model.predict(x_test).astype(int)
    y_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] >= 2:
            y_prob = np.asarray(proba)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_test), dtype=float)
        y_prob = 1.0 / (1.0 + np.exp(-scores))
    metrics = calculate_metrics(y_test, y_pred, y_prob, sensitive_test)
    return ModelResult(name=name, metrics=metrics, y_pred=y_pred, y_prob=y_prob, weight_strategy="baseline")


def fit_threshold_optimizer(base_model, x_train, y_train, sensitive_train, x_test, sensitive_test, y_test) -> ModelResult:
    try:
        mitigator = ThresholdOptimizer(estimator=base_model, constraints="equalized_odds", predict_method="predict_proba")
        mitigator.fit(x_train, y_train, sensitive_features=sensitive_train)
        y_pred = np.asarray(mitigator.predict(x_test, sensitive_features=sensitive_test)).astype(int)
        metrics = calculate_metrics(y_test, y_pred, None, sensitive_test)
        return ModelResult("Equalized Odds ThresholdOptimizer", metrics, y_pred, None, "postprocessing")
    except Exception:
        return ModelResult("Equalized Odds ThresholdOptimizer", calculate_metrics(y_test, base_model.predict(x_test).astype(int), None, sensitive_test), base_model.predict(x_test).astype(int), None, "postprocessing-failed")


def subgroup_analysis(
    raw_test: pd.DataFrame,
    sensitive_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    def summarize(group_series: pd.Series, label: str) -> pd.DataFrame:
        frame = pd.DataFrame({"group": group_series.astype(str), "y_true": y_true, "y_pred": y_pred})
        rows: List[Dict[str, object]] = []
        for group_value, part in frame.groupby("group"):
            rows.append({
                "Subgroup": label,
                "Value": group_value,
                "Count": int(len(part)),
                "Accuracy": float(accuracy_score(part["y_true"], part["y_pred"])),
                "Positive Rate": float(np.mean(part["y_pred"])),
            })
        return pd.DataFrame(rows)

    if dataset_name == "Delhivery Logistics":
        tables["Education Tiers"] = summarize(pd.Series(sensitive_test, index=raw_test.index), "Education")
        tables["Experience Levels"] = summarize(pd.qcut(raw_test["time_per_distance_actual"].rank(method="first"), 4, labels=["Low", "Mid-Low", "Mid-High", "High"]), "Experience")
        tables["Geography Proxies"] = summarize(pd.qcut(raw_test["actual_distance_to_destination"].rank(method="first"), 4, labels=["Near", "Moderate", "Far", "Very Far"]), "Geography")
    elif dataset_name == "Adult Income Benchmark":
        country = raw_test["native-country"].astype(str)
        top = country.value_counts().nlargest(5).index
        country = country.where(country.isin(top), "Other")
        tables["Education Tiers"] = summarize(pd.Series(sensitive_test, index=raw_test.index), "Education")
        tables["Experience Levels"] = summarize(pd.cut(raw_test["age"], bins=[0, 25, 40, 55, 100], labels=["Early", "Mid", "Late", "Senior"]), "Experience")
        tables["Geography Proxies"] = summarize(country, "Geography")
    else:
        station = raw_test["station_code"].astype(str)
        top = station.value_counts().nlargest(5).index
        station = station.where(station.isin(top), "Other")
        tables["Education Tiers"] = summarize(pd.Series(sensitive_test, index=raw_test.index), "Education")
        tables["Experience Levels"] = summarize(pd.cut(raw_test["departure_hour"], bins=[-1, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"]), "Experience")
        tables["Geography Proxies"] = summarize(station, "Geography")

    return tables


def causal_counterfactual_analysis(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    sensitive_train: np.ndarray,
    sensitive_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    feature_cols = list(raw_train.columns)
    all_train = raw_train.copy()
    all_test = raw_test.copy()

    binary_train = pd.Series((pd.Series(sensitive_train).astype(str) == "HigherEd").astype(int).to_numpy(), index=all_train.index, name="Education_Binary")
    binary_test = pd.Series((pd.Series(sensitive_test).astype(str) == "HigherEd").astype(int).to_numpy(), index=all_test.index, name="Education_Binary")

    outcome_frame = pd.concat([all_train[feature_cols], binary_train], axis=1)
    preprocessor = build_preprocessor(outcome_frame)
    x_outcome = preprocessor.fit_transform(outcome_frame)

    outcome_model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)
    outcome_model.fit(x_outcome, y_train)

    test_frame_0 = pd.concat([all_test[feature_cols], pd.Series(np.zeros(len(all_test)), index=all_test.index, name="Education_Binary")], axis=1)
    test_frame_1 = pd.concat([all_test[feature_cols], pd.Series(np.ones(len(all_test)), index=all_test.index, name="Education_Binary")], axis=1)
    p0 = outcome_model.predict_proba(preprocessor.transform(test_frame_0))[:, 1]
    p1 = outcome_model.predict_proba(preprocessor.transform(test_frame_1))[:, 1]

    prop_preprocessor = build_preprocessor(all_train[feature_cols])
    prop_features_train = prop_preprocessor.fit_transform(all_train[feature_cols])
    prop_features_test = prop_preprocessor.transform(all_test[feature_cols])

    prop_model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)
    prop_model.fit(prop_features_train, binary_train)
    prop_scores = np.clip(prop_model.predict_proba(prop_features_test)[:, 1], 1e-3, 1 - 1e-3)
    ipw = np.where(binary_test == 1, 1.0 / prop_scores, 1.0 / (1.0 - prop_scores))

    treated = binary_test == 1
    control = ~treated
    treated_mean = float(np.mean(p1[treated] - p0[treated])) if np.any(treated) else float("nan")
    overall_ate = float(np.mean(p1 - p0))
    ipw_ate = float(np.average(p1 - p0, weights=ipw))

    if np.any(treated) and np.any(control):
        treated_points = prop_scores[treated].reshape(-1, 1)
        control_points = prop_scores[control].reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control_points)
        matched_idx = nn.kneighbors(treated_points, return_distance=False).ravel()
        matched_ate = float(np.mean(y_test[treated] - y_test[control][matched_idx]))
    else:
        matched_ate = float("nan")

    return pd.DataFrame([
        {"Causal Estimate": "G-computation ATE", "Value": overall_ate},
        {"Causal Estimate": "Group-specific HigherEd effect", "Value": treated_mean},
        {"Causal Estimate": "IPW ATE", "Value": ipw_ate},
        {"Causal Estimate": "Matched Difference", "Value": matched_ate},
    ])


def robustness_checks(raw_test: pd.DataFrame, y_test: np.ndarray, sensitive_test: np.ndarray, baseline: ModelResult, mitigated: ModelResult, dataset_name: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rows.append({
        "Check": "Unseen Holdout",
        "Dataset": dataset_name,
        "Baseline Accuracy": baseline.metrics["Accuracy"],
        "Mitigated Accuracy": mitigated.metrics["Accuracy"],
        "Baseline Fairness Gap": fairness_gap(baseline.metrics),
        "Mitigated Fairness Gap": fairness_gap(mitigated.metrics),
    })

    if dataset_name == "Amazon Last-Mile Routes" and "departure_hour" in raw_test.columns:
        split_point = raw_test["departure_hour"].median()
        early = raw_test["departure_hour"] <= split_point
        late = ~early
        rows.append({
            "Check": "Time Split (Early)",
            "Dataset": dataset_name,
            "Baseline Accuracy": float(accuracy_score(y_test[early], baseline.y_pred[early])) if np.any(early) else float("nan"),
            "Mitigated Accuracy": float(accuracy_score(y_test[early], mitigated.y_pred[early])) if np.any(early) else float("nan"),
            "Baseline Fairness Gap": fairness_gap(calculate_metrics(y_test[early], baseline.y_pred[early], None, sensitive_test[early])) if np.any(early) else float("nan"),
            "Mitigated Fairness Gap": fairness_gap(calculate_metrics(y_test[early], mitigated.y_pred[early], None, sensitive_test[early])) if np.any(early) else float("nan"),
        })
        rows.append({
            "Check": "Time Split (Late)",
            "Dataset": dataset_name,
            "Baseline Accuracy": float(accuracy_score(y_test[late], baseline.y_pred[late])) if np.any(late) else float("nan"),
            "Mitigated Accuracy": float(accuracy_score(y_test[late], mitigated.y_pred[late])) if np.any(late) else float("nan"),
            "Baseline Fairness Gap": fairness_gap(calculate_metrics(y_test[late], baseline.y_pred[late], None, sensitive_test[late])) if np.any(late) else float("nan"),
            "Mitigated Fairness Gap": fairness_gap(calculate_metrics(y_test[late], mitigated.y_pred[late], None, sensitive_test[late])) if np.any(late) else float("nan"),
        })

    return pd.DataFrame(rows)


def plot_pareto(results: List[ModelResult], output_path: Path, title: str) -> None:
    df = pd.DataFrame([
        {
            "Model": result.name,
            "Accuracy": result.metrics["Accuracy"],
            "Fairness Gap": fairness_gap(result.metrics),
            "Pareto": True,
        }
        for result in results
    ])
    frontier: List[int] = []
    for idx, row in df.iterrows():
        dominated = False
        for jdx, other in df.iterrows():
            if idx == jdx:
                continue
            if other["Accuracy"] >= row["Accuracy"] and other["Fairness Gap"] <= row["Fairness Gap"] and (
                other["Accuracy"] > row["Accuracy"] or other["Fairness Gap"] < row["Fairness Gap"]
            ):
                dominated = True
                break
        frontier.append(not dominated)
    df["Pareto"] = frontier

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="Fairness Gap", y="Accuracy", hue="Model", style="Pareto", s=120)
    for _, row in df.iterrows():
        plt.text(row["Fairness Gap"] + 0.001, row["Accuracy"] + 0.001, row["Model"], fontsize=8)
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_No rows returned._"]

    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "---|" * len(columns))
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(studies: List[DatasetStudy], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Final Evaluation Results and Research Appendix")
    lines.append("")
    lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## What this report adds")
    lines.append("- Reweighting-based fairness intervention")
    lines.append("- Predictive parity and subgroup fairness checks")
    lines.append("- Causal-style counterfactual analysis on education intervention")
    lines.append("- Robustness checks on unseen holdouts and time splits")
    lines.append("- Updated labor-economics interpretation and welfare-loss framing")
    lines.append("")
    lines.append("## Fairness metric definitions")
    lines.append(r"- Demographic parity difference: $|P(\hat{Y}=1|A=a) - P(\hat{Y}=1|A=b)|$")
    lines.append(r"- Equal opportunity difference: $|P(\hat{Y}=1|Y=1,A=a) - P(\hat{Y}=1|Y=1,A=b)|$")
    lines.append("- Equalized odds difference: maximum of the TPR and FPR gaps across groups")
    lines.append("- Predictive parity difference: gap in positive predictive value across groups")
    lines.append("- Disparate impact ratio: ratio of positive prediction rates, closer to 1 is better")
    lines.append("")
    lines.append("## Causal graph")
    lines.append("```mermaid")
    lines.append("graph LR")
    lines.append("    E[Education] --> T[Task Allocation]")
    lines.append("    E --> O[Observed Outcome]")
    lines.append("    X[Route / Worker Context] --> T")
    lines.append("    X --> O")
    lines.append("    T --> O")
    lines.append("    P[Platform Incentives] --> T")
    lines.append("    P --> O")
    lines.append("```")
    lines.append("")
    lines.append("## Labor economics interpretation")
    lines.append("Education can act as a signaling variable, and platform assignment systems may amplify statistical discrimination when proxies correlate with route difficulty, timing, or geography. The added causal-style and robustness checks help distinguish whether observed disparities are due to model behavior, data structure, or underlying task allocation patterns.")
    lines.append("")

    for study in studies:
        lines.append(f"## {study.bundle.name}")
        lines.append("")
        lines.append(f"- Sensitive Attribute: {study.bundle.sensitive_attribute}")
        lines.append(f"- Sensitive Source: {study.bundle.sensitive_source}")
        lines.append(f"- Notes: {study.bundle.notes}")
        lines.append("")
        lines.append("### Model comparison")
        lines.append("| Model | Accuracy | Balanced Accuracy | Fairness Gap | Predictive Parity Diff | DI Ratio |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for result in study.results:
            lines.append(
                f"| {result.name} | {result.metrics['Accuracy']:.4f} | {result.metrics['Balanced Accuracy']:.4f} | {fairness_gap(result.metrics):.4f} | {result.metrics['Predictive Parity Diff']:.4f} | {result.metrics['Disparate Impact Ratio']:.4f} |"
            )
        lines.append("")
        lines.append(f"### Best model: {study.best_result.name}")
        lines.append(f"- Composite score: {candidate_score(study.best_result.metrics):.4f}")
        lines.append(f"- Welfare loss proxy: {max(0.0, study.results[0].metrics['Accuracy'] - study.best_result.metrics['Accuracy']):.4f}")
        lines.append("")

        if study.causal_summary is not None and not study.causal_summary.empty:
            lines.append("### Counterfactual / causal-style estimates")
            lines.extend(dataframe_to_markdown(study.causal_summary))
            lines.append("")

        if study.robustness_summary is not None and not study.robustness_summary.empty:
            lines.append("### Robustness checks")
            lines.extend(dataframe_to_markdown(study.robustness_summary))
            lines.append("")

        for name, table in study.subgroup_tables.items():
            lines.append(f"### Subgroup fairness: {name}")
            lines.extend(dataframe_to_markdown(table))
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_bundle(bundle: base.DatasetBundle) -> DatasetStudy:
    raw = bundle.features.copy()
    target = bundle.target.copy()
    sensitive = bundle.sensitive.copy()

    # For this diagnostic run we remove sample caps to allow full-data HPO and training
    raw, target, sensitive = raw.copy(), target.copy(), sensitive.copy()

    stratify = target if target.nunique() > 1 else None
    raw_train, raw_test, y_train, y_test, s_train, s_test = train_test_split(
        raw,
        target.to_numpy(dtype=int),
        sensitive.to_numpy(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(raw_train)
    x_train = preprocessor.fit_transform(raw_train)
    x_test = preprocessor.transform(raw_test)
    feature_names = list(preprocessor.get_feature_names_out())

    weights = reweighing_weights(y_train, s_train)
    fitted_models = fit_models(x_train, y_train, weights)

    # Attempt Bayesian HPO to tune an XGB candidate with a larger budget
    try:
        positive = float(np.sum(y_train == 1))
        negative = float(np.sum(y_train == 0))
        scale_pos_weight = max(negative / max(positive, 1.0), 1.0)
        # Increase budget for a better search (may be slower)
        hpo_budget = 24
        best_params, history = base.bayesian_optimize_xgb(x_train, y_train, s_train, scale_pos_weight, budget=hpo_budget)
        tuned_xgb = base.build_xgb_model(best_params, scale_pos_weight)
        try:
            tuned_xgb.fit(x_train, y_train, sample_weight=weights)
        except TypeError:
            tuned_xgb.fit(x_train, y_train)
        fitted_models["Tuned XGB (HPO)"] = tuned_xgb
        print(f"[INFO] HPO tuning completed for {bundle.name}; added Tuned XGB (HPO)")
        try:
            print(f"[INFO] Best HPO params sample: { {k: best_params.get(k, None) for k in list(best_params)[:5]} }")
        except Exception:
            pass
    except Exception as exc:
        print(f"[WARN] HPO tuning failed or skipped: {exc}")

    results: List[ModelResult] = []
    for name, model in fitted_models.items():
        result = evaluate_model(name, model, x_test, y_test, s_test)
        result.weight_strategy = "reweighted" if name == "Reweighted Logistic" else "baseline"
        results.append(result)

    base_candidate = next(result for result in results if result.name == "Logistic Regression")
    mitigated_candidate = next(result for result in results if result.name == "Reweighted Logistic")

    threshold_model = fit_threshold_optimizer(
        fitted_models["Logistic Regression"],
        x_train,
        y_train,
        s_train,
        x_test,
        s_test,
        y_test,
    )
    results.append(threshold_model)

    best_result = max(results, key=lambda item: (candidate_score(item.metrics), item.metrics["Accuracy"]))

    subgroup_tables = subgroup_analysis(raw_test, s_test, y_test, best_result.y_pred, bundle.name)
    causal_summary = causal_counterfactual_analysis(raw_train, raw_test, s_train, s_test, y_train, y_test) if bundle.name == "Adult Income Benchmark" else None
    robustness_summary = robustness_checks(raw_test, y_test, s_test, base_candidate, mitigated_candidate, bundle.name)

    return DatasetStudy(
        bundle=bundle,
        raw_train=raw_train,
        raw_test=raw_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=s_train,
        sensitive_test=s_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
        x_train=x_train,
        x_test=x_test,
        results=results,
        best_result=best_result,
        subgroup_tables=subgroup_tables,
        causal_summary=causal_summary,
        robustness_summary=robustness_summary,
    )


def summarize(studies: List[DatasetStudy]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for study in studies:
        for result in study.results:
            rows.append(
                {
                    "Dataset": study.bundle.name,
                    "Model": result.name,
                    "Accuracy": result.metrics["Accuracy"],
                    "Balanced Accuracy": result.metrics["Balanced Accuracy"],
                    "Fairness Gap": fairness_gap(result.metrics),
                    "Predictive Parity Diff": result.metrics["Predictive Parity Diff"],
                    "Disparate Impact Ratio": result.metrics["Disparate Impact Ratio"],
                    "Composite Score": candidate_score(result.metrics),
                }
            )
    return pd.DataFrame(rows)


def causal_graph_markdown(output_path: Path) -> None:
    output_path.write_text(
        "\n".join(
            [
                "# Causal Graph Appendix",
                "",
                "```mermaid",
                "graph LR",
                "    Education[Education / Education Proxy] --> Allocation[Task Allocation]",
                "    Education --> Outcome[Task Success / Outcome]",
                "    WorkerContext[Experience / Geography / Route Context] --> Allocation",
                "    WorkerContext --> Outcome",
                "    Platform[Platform Rules / Incentives] --> Allocation",
                "    Platform --> Outcome",
                "    Allocation --> Outcome",
                "```",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    prior = base.get_salary_education_prior()
    bundles = [
        base.load_delhivery_dataset(prior),
        base.load_adult_dataset(),
        base.load_amazon_dataset(prior),
    ]

    studies = [evaluate_bundle(bundle) for bundle in bundles]
    summary = summarize(studies)
    summary.to_csv(DOCS_DIR / "final_metrics_summary.csv", index=False)

    for study in studies:
        plot_pareto(study.results, PLOTS_DIR / f"{study.bundle.name.lower().replace(' ', '_')}_pareto_final.png", f"Final Evaluation Pareto Frontier - {study.bundle.name}")
        if study.best_result.y_prob is not None:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(10, 6))
            plt.title(f"Accuracy vs Fairness Gap - {study.bundle.name}")
            for result in study.results:
                plt.scatter(fairness_gap(result.metrics), result.metrics["Accuracy"], s=80)
                plt.text(fairness_gap(result.metrics) + 0.001, result.metrics["Accuracy"] + 0.001, result.name, fontsize=8)
            plt.xlabel("Fairness Gap")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{study.bundle.name.lower().replace(' ', '_')}_tradeoff_final.png", dpi=300)
            plt.close()

    write_report(studies, DOCS_DIR / "TECHNICAL_APPENDIX.md")
    causal_graph_markdown(DOCS_DIR / "Causal_Graph_Appendix.md")

    print("Final Evaluation complete")
    print(f"Report: {DOCS_DIR / 'TECHNICAL_APPENDIX.md'}")
    print(f"Metrics: {DOCS_DIR / 'final_metrics_summary.csv'}")
    print(f"Causal graph appendix: {DOCS_DIR / 'Causal_Graph_Appendix.md'}")


if __name__ == "__main__":
    main()


