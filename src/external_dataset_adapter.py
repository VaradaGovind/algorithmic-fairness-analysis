# ============================================================================
# External Data Ingestion Adapter & Black-Box Interface
# Extensible data ingestion interface: supports loading external operational
# or benchmark datasets and evaluating them with clean isolation boundaries.
# ============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.dataset_provenance import (
    DatasetProvenance,
    DatasetType,
    SensitiveOrigin,
    PROVENANCE_REGISTRY,
    compute_file_sha256,
)
from src.evaluation_manifest import (
    EvaluationManifest,
    ResultClassification,
)
from src.fairness_evaluator import (
    AnalysisType,
    EvidenceState,
    FairnessEvaluationSuite,
    FairnessPolicy,
    MissingAttributePolicy,
    evaluate_fairness_defensively,
    validate_evaluation_unit_configuration,
)
from src.ingest_data import (
    export_sanitized_csv,
    validate_source_path_security,
)
from src.split_utils import (
    SplitRegime,
    SplitResult,
    group_aware_3way_split,
    stratified_3way_split,
    temporal_3way_split,
)


@dataclass
class ExternalDataBundle:
    """
    Standardized container for externally sourced operational or benchmark datasets.
    Maintains complete isolation from synthetic scenario parameters.
    """
    dataset_name: str
    X: pd.DataFrame
    y: pd.Series
    sensitive: pd.Series
    provenance: DatasetProvenance
    groups: Optional[pd.Series] = None
    timestamps: Optional[pd.Series] = None
    source_sha256: Optional[str] = None
    classification: ResultClassification = ResultClassification.EXTERNAL_REAL_DATA_VALIDATION
    is_external_real: bool = True
    protected_df: Optional[pd.DataFrame] = None
    provenance_manifest: Optional[Any] = None
    evaluation_unit_key: Optional[str] = None
    evaluation_units: Optional[pd.Series] = None
    statistical_estimand: Optional[Dict[str, Any]] = None
    unit_diagnostics: Optional[Dict[str, Any]] = None
    target_aggregation_strategy: Optional[str] = None
    prediction_aggregation_strategy: Optional[str] = None

    @property
    def n_rows(self) -> int:
        return len(self.X)

    def __post_init__(self):
        # Assert complete absence of synthetic DGP markers
        forbidden_synthetic_markers = {"scenario", "signal_regime", "dgp_coefficients", "ground_truth_disparity"}
        attrs = set(dir(self))
        intersection = forbidden_synthetic_markers.intersection(attrs)
        if intersection:
            raise ValueError(f"EXTERNAL ISOLATION BREACH: External dataset container contains synthetic DGP markers: {intersection}")


class ExternalDatasetAdapter:
    """
    Independent validation boundary adapter.
    Loads and standardizes externally sourced real datasets (e.g. Adult Income, customer delivery telemetry)
    without accessing synthetic DGP coefficients, external simulator configs, or simulation metadata.
    """

    def __init__(
        self,
        dataset_name: str = "ExternalDataset",
        provenance: Optional[DatasetProvenance] = None,
        approved_staging_root: Optional[Union[str, Path]] = None,
        staging_root: Optional[Union[str, Path]] = None,
    ):
        self.dataset_name = dataset_name
        self.provenance = provenance or PROVENANCE_REGISTRY.get(dataset_name)
        self.approved_staging_root = approved_staging_root or staging_root

    @property
    def status(self) -> str:
        """Returns machine-readable status of external validation readiness."""
        return "EXTERNAL_VALIDATION_PENDING"

    def get_external_validation_status(self) -> Dict[str, Any]:
        """
        Returns machine-readable status indicating external operational data validation state.
        Remains strictly 'EXTERNAL_VALIDATION_PENDING' until external operational data
        is staged and evaluated.
        """
        return {
            "dataset_name": self.dataset_name,
            "status": "EXTERNAL_VALIDATION_PENDING",
            "operational_data_staged": False,
            "execution_classification": "EXTERNAL_REAL_DATA_VALIDATION",
            "notes": "External operational data staging pending user configuration.",
        }

    def ingest_external_tabular(
        self,
        file_path: Union[str, Path],
        label_column: str,
        protected_columns: Sequence[str],
        feature_columns: Optional[Sequence[str]] = None,
        provenance_metadata: Optional[Dict[str, Any]] = None,
        grouping_column: Optional[str] = None,
        temporal_column: Optional[str] = None,
        max_rows: Optional[int] = None,
        evaluation_unit_key: Optional[str] = None,
        statistical_estimand: Optional[Dict[str, Any]] = None,
        target_aggregation_strategy: Optional[str] = None,
        prediction_aggregation_strategy: Optional[str] = None,
    ) -> ExternalDataBundle:
        """
        Ingests external tabular data from CSV, validates schema and staging boundaries,
        and constructs an ExternalDataBundle with evaluation unit alignment.
        """
        valid_path = validate_source_path_security(
            file_path,
            approved_staging_root=self.approved_staging_root,
        )
        file_sha256 = compute_file_sha256(valid_path)
        df = pd.read_csv(valid_path, nrows=max_rows)

        if label_column not in df.columns:
            raise KeyError(f"Label column '{label_column}' not found in columns: {list(df.columns)}")
        for p in protected_columns:
            if p not in df.columns:
                raise KeyError(f"Protected column '{p}' not found in columns: {list(df.columns)}")

        unit_diagnostics = None
        eval_units = None
        if evaluation_unit_key is not None:
            validate_evaluation_unit_configuration(df, evaluation_unit_key, dataset_name=self.dataset_name)
            eval_units = df[evaluation_unit_key].copy()
            counts = eval_units.value_counts()
            unit_diagnostics = {
                "evaluation_unit_key": evaluation_unit_key,
                "n_records": len(df),
                "n_units": int(eval_units.nunique()),
                "min_records_per_unit": int(counts.min()) if len(counts) > 0 else 0,
                "mean_records_per_unit": float(counts.mean()) if len(counts) > 0 else 0.0,
                "max_records_per_unit": int(counts.max()) if len(counts) > 0 else 0,
            }

        if feature_columns is not None:
            X = df[list(feature_columns)].copy()
        else:
            exclude = set([label_column] + list(protected_columns))
            if grouping_column:
                exclude.add(grouping_column)
            if temporal_column:
                exclude.add(temporal_column)
            if evaluation_unit_key:
                exclude.add(evaluation_unit_key)
            X = df[[c for c in df.columns if c not in exclude]].copy()

        y = df[label_column].copy()
        protected_df = df[list(protected_columns)].copy()
        sensitive_series = (
            protected_df.iloc[:, 0]
            if protected_df.shape[1] == 1
            else protected_df.astype(str).agg("_x_".join, axis=1)
        )

        prov = self.provenance
        if prov is None:
            prov = DatasetProvenance(
                dataset_name=self.dataset_name,
                dataset_type=DatasetType.REAL_DATA,
                source_reference="Customer External Tabular File",
                version_or_date="2026-09",
                license_notes="Proprietary/Customer-Held",
                target_definition=label_column,
                sensitive_attributes=list(protected_columns),
                sensitive_origin=SensitiveOrigin.REAL_OBSERVED,
                feature_timing_contract="PRE_DISPATCH_VALID",
                sha256_fingerprint=file_sha256,
                notes="External tabular data ingested via black-box adapter.",
            )

        manifest = EvaluationManifest(
            dataset_identity=self.dataset_name,
            dataset_sha256=file_sha256,
            provenance={"dataset": self.dataset_name, "sha256": file_sha256, **(provenance_metadata or {})},
            result_classification=ResultClassification.EXTERNAL_AUDIT_READY,
            sensitive_attributes=list(protected_columns),
            selected_features=list(X.columns),
        )

        return ExternalDataBundle(
            dataset_name=self.dataset_name,
            X=X,
            y=y,
            sensitive=sensitive_series,
            provenance=prov,
            groups=df[grouping_column] if grouping_column and grouping_column in df.columns else None,
            timestamps=df[temporal_column] if temporal_column and temporal_column in df.columns else None,
            source_sha256=file_sha256,
            classification=ResultClassification.EXTERNAL_AUDIT_READY,
            protected_df=protected_df,
            provenance_manifest=manifest,
            evaluation_unit_key=evaluation_unit_key,
            evaluation_units=eval_units,
            statistical_estimand=statistical_estimand,
            unit_diagnostics=unit_diagnostics,
            target_aggregation_strategy=target_aggregation_strategy,
            prediction_aggregation_strategy=prediction_aggregation_strategy,
        )

    def load_from_csv(
        self,
        filepath: Union[str, Path],
        target_column: str,
        sensitive_column: str,
        feature_columns: Optional[Sequence[str]] = None,
        grouping_column: Optional[str] = None,
        temporal_column: Optional[str] = None,
        max_rows: Optional[int] = None,
        evaluation_unit_key: Optional[str] = None,
        statistical_estimand: Optional[Dict[str, Any]] = None,
        target_aggregation_strategy: Optional[str] = None,
        prediction_aggregation_strategy: Optional[str] = None,
    ) -> ExternalDataBundle:
        """
        Safely loads an external CSV dataset, enforces path security, computes SHA-256,
        and constructs an ExternalDataBundle.
        """
        valid_path = validate_source_path_security(
            filepath,
            approved_staging_root=self.approved_staging_root,
        )
        file_sha256 = compute_file_sha256(valid_path)

        df = pd.read_csv(valid_path, nrows=max_rows)
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in external dataset columns: {list(df.columns)}")
        if sensitive_column not in df.columns:
            raise KeyError(f"Sensitive column '{sensitive_column}' not found in external dataset columns: {list(df.columns)}")

        unit_diagnostics = None
        eval_units = None
        if evaluation_unit_key is not None:
            validate_evaluation_unit_configuration(df, evaluation_unit_key, dataset_name=self.dataset_name)
            eval_units = df[evaluation_unit_key].copy()
            counts = eval_units.value_counts()
            unit_diagnostics = {
                "evaluation_unit_key": evaluation_unit_key,
                "n_records": len(df),
                "n_units": int(eval_units.nunique()),
                "min_records_per_unit": int(counts.min()) if len(counts) > 0 else 0,
                "mean_records_per_unit": float(counts.mean()) if len(counts) > 0 else 0.0,
                "max_records_per_unit": int(counts.max()) if len(counts) > 0 else 0,
            }

        # Clean features: exclude target, sensitive, grouping, temporal, eval_unit_key
        exclude_cols = {target_column, sensitive_column}
        if grouping_column:
            exclude_cols.add(grouping_column)
        if temporal_column:
            exclude_cols.add(temporal_column)
        if evaluation_unit_key:
            exclude_cols.add(evaluation_unit_key)

        if feature_columns is not None:
            for f in feature_columns:
                if f not in df.columns:
                    raise KeyError(f"Required feature column '{f}' not found in external dataset.")
            selected_features = list(feature_columns)
        else:
            selected_features = [c for c in df.columns if c not in exclude_cols]

        prov = self.provenance
        if prov is None:
            prov = DatasetProvenance(
                dataset_name=self.dataset_name,
                dataset_type=DatasetType.REAL_DATA,
                source_reference=str(valid_path),
                version_or_date=None,
                license_notes="Externally supplied data; verify commercial redistribution terms.",
                target_definition=f"Target column: {target_column}",
                sensitive_attributes=[sensitive_column],
                sensitive_origin=SensitiveOrigin.REAL_OBSERVED,
                feature_timing_contract="PRE_DISPATCH_VALID: External operational contract.",
                sha256_fingerprint=file_sha256,
            )

        return ExternalDataBundle(
            dataset_name=self.dataset_name,
            X=df[selected_features].copy(),
            y=df[target_column].copy(),
            sensitive=df[sensitive_column].copy(),
            provenance=prov,
            groups=df[grouping_column].copy() if grouping_column and grouping_column in df.columns else None,
            timestamps=df[temporal_column].copy() if temporal_column and temporal_column in df.columns else None,
            source_sha256=file_sha256,
            classification=ResultClassification.EXTERNAL_REAL_DATA_VALIDATION,
            evaluation_unit_key=evaluation_unit_key,
            evaluation_units=eval_units,
            statistical_estimand=statistical_estimand,
            unit_diagnostics=unit_diagnostics,
            target_aggregation_strategy=target_aggregation_strategy,
            prediction_aggregation_strategy=prediction_aggregation_strategy,
        )

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
        sensitive_column: str,
        feature_columns: Optional[Sequence[str]] = None,
        grouping_column: Optional[str] = None,
        temporal_column: Optional[str] = None,
        dataset_source_note: str = "Memory Fixture",
        evaluation_unit_key: Optional[str] = None,
        statistical_estimand: Optional[Dict[str, Any]] = None,
        target_aggregation_strategy: Optional[str] = None,
        prediction_aggregation_strategy: Optional[str] = None,
    ) -> ExternalDataBundle:
        """
        Loads an ExternalDataBundle directly from a DataFrame (used for external validation or testing fixtures).
        """
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' missing.")
        if sensitive_column not in df.columns:
            raise KeyError(f"Sensitive column '{sensitive_column}' missing.")

        unit_diagnostics = None
        eval_units = None
        if evaluation_unit_key is not None:
            validate_evaluation_unit_configuration(df, evaluation_unit_key, dataset_name=self.dataset_name)
            eval_units = df[evaluation_unit_key].copy()
            counts = eval_units.value_counts()
            unit_diagnostics = {
                "evaluation_unit_key": evaluation_unit_key,
                "n_records": len(df),
                "n_units": int(eval_units.nunique()),
                "min_records_per_unit": int(counts.min()) if len(counts) > 0 else 0,
                "mean_records_per_unit": float(counts.mean()) if len(counts) > 0 else 0.0,
                "max_records_per_unit": int(counts.max()) if len(counts) > 0 else 0,
            }

        exclude = {target_column, sensitive_column}
        if grouping_column:
            exclude.add(grouping_column)
        if temporal_column:
            exclude.add(temporal_column)
        if evaluation_unit_key:
            exclude.add(evaluation_unit_key)

        feat_cols = list(feature_columns) if feature_columns is not None else [c for c in df.columns if c not in exclude]

        prov = self.provenance or DatasetProvenance(
            dataset_name=self.dataset_name,
            dataset_type=DatasetType.REAL_DATA,
            source_reference=dataset_source_note,
            version_or_date=None,
            license_notes="Memory dataset bundle.",
            target_definition=target_column,
            sensitive_attributes=[sensitive_column],
            sensitive_origin=SensitiveOrigin.REAL_OBSERVED,
            feature_timing_contract="CONCURRENT: All features observed concurrently.",
        )

        return ExternalDataBundle(
            dataset_name=self.dataset_name,
            X=df[feat_cols].copy(),
            y=df[target_column].copy(),
            sensitive=df[sensitive_column].copy(),
            provenance=prov,
            groups=df[grouping_column].copy() if grouping_column and grouping_column in df.columns else None,
            timestamps=df[temporal_column].copy() if temporal_column and temporal_column in df.columns else None,
            source_sha256=None,
            classification=ResultClassification.EXTERNAL_REAL_DATA_VALIDATION,
            evaluation_unit_key=evaluation_unit_key,
            evaluation_units=eval_units,
            statistical_estimand=statistical_estimand,
            unit_diagnostics=unit_diagnostics,
            target_aggregation_strategy=target_aggregation_strategy,
            prediction_aggregation_strategy=prediction_aggregation_strategy,
        )

    def evaluate_external_black_box(
        self,
        bundle: ExternalDataBundle,
        model: Any,
        policy: Optional[FairnessPolicy] = None,
        split_regime: SplitRegime = SplitRegime.IID_HOLDOUT,
        seed: int = 42,
        missing_attribute_policy: MissingAttributePolicy = MissingAttributePolicy.REJECT,
        compute_uncertainty: bool = True,
        n_bootstraps: int = 200,
        output_manifest_path: Optional[Union[str, Path]] = None,
        evaluation_unit_key: Optional[str] = None,
        statistical_estimand: Optional[Dict[str, Any]] = None,
        target_aggregation_strategy: Optional[str] = None,
        prediction_aggregation_strategy: Optional[str] = None,
    ) -> Tuple[FairnessEvaluationSuite, EvaluationManifest]:
        """
        Executes black-box fairness evaluation on an external dataset:
        1. Partitions data (Train / Val / Test) using standard split utils without synthetic bias.
        2. Generates test set predictions using the supplied frozen model.
        3. Invokes evaluate_fairness_defensively without access to synthetic scenario configs.
        4. Generates immutable EvaluationManifest tagged EXTERNAL_REAL_DATA_VALIDATION.
        """
        # Step 1: Split according to regime
        if split_regime == SplitRegime.GROUP_HOLDOUT and bundle.groups is not None:
            split_res = group_aware_3way_split(bundle.X, bundle.y, bundle.sensitive, bundle.groups, seed=seed)
        elif split_regime == SplitRegime.TEMPORAL_HOLDOUT and bundle.timestamps is not None:
            split_res = temporal_3way_split(bundle.X, bundle.y, bundle.sensitive, bundle.timestamps)
        else:
            split_res = stratified_3way_split(bundle.X, bundle.y, bundle.sensitive, seed=seed)

        # Step 2: Fit preprocessor on train, transform test
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num_cols = split_res.X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = [c for c in split_res.X_train.columns if c not in num_cols]
        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
        
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        X_train_proc = preprocessor.fit_transform(split_res.X_train)
        X_test_proc = preprocessor.transform(split_res.X_test)

        # Step 3: Model predictions
        # Fit model on training data if not already fitted
        if not hasattr(model, "classes_"):
            if hasattr(model, "fit"):
                model.fit(X_train_proc, split_res.y_train.to_numpy())

        y_pred = model.predict(X_test_proc)
        y_prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_proc)
            if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] >= 2:
                y_prob = np.asarray(proba)[:, 1]
            else:
                y_prob = np.asarray(proba).ravel()

        active_policy = policy or FairnessPolicy(name="External_Audit_Standard_Policy")

        # Resolve unit configuration
        active_unit_key = evaluation_unit_key or bundle.evaluation_unit_key
        active_estimand = statistical_estimand or bundle.statistical_estimand
        active_target_agg = target_aggregation_strategy or bundle.target_aggregation_strategy
        active_pred_agg = prediction_aggregation_strategy or bundle.prediction_aggregation_strategy

        test_units = None
        if bundle.evaluation_units is not None:
            test_units = bundle.evaluation_units.loc[split_res.y_test.index].to_numpy()

        # Step 4: Pure black-box fairness evaluation
        suite = evaluate_fairness_defensively(
            y_true=split_res.y_test.to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive=split_res.s_test.to_numpy(),
            dataset_name=bundle.dataset_name,
            model_name=getattr(model, "__class__", type(model)).__name__,
            policy=active_policy,
            compute_uncertainty=compute_uncertainty,
            n_bootstraps=n_bootstraps,
            missing_attribute_policy=missing_attribute_policy,
            analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
            evaluation_unit_key=active_unit_key,
            evaluation_units=test_units,
            statistical_estimand=active_estimand,
            target_aggregation_strategy=active_target_agg,
            prediction_aggregation_strategy=active_pred_agg,
        )

        # Step 5: Construct immutable EvaluationManifest
        manifest = EvaluationManifest.create(
            dataset_identity=bundle.dataset_name,
            dataset_sha256=bundle.source_sha256,
            dataset_provenance=bundle.provenance.to_dict() if bundle.provenance else None,
            target_definition=bundle.provenance.target_definition if bundle.provenance else "External_Y",
            sensitive_attributes=bundle.provenance.sensitive_attributes if bundle.provenance else ["Sensitive"],
            sensitive_attribute_provenance={"origin": bundle.provenance.sensitive_origin.value} if bundle.provenance else None,
            feature_contract_version="EXTERNAL_PRE_DISPATCH_VALID_v1",
            selected_features=list(bundle.X.columns),
            excluded_features=[bundle.provenance.target_definition if bundle.provenance else "Y"],
            split_strategy=split_regime.value,
            grouping_strategy=bundle.groups.name if bundle.groups is not None else None,
            temporal_strategy=bundle.timestamps.name if bundle.timestamps is not None else None,
            random_seeds=[seed],
            model_configuration={"model_class": getattr(model, "__class__", type(model)).__name__},
            fairness_policy_configuration=active_policy.to_dict(),
            missing_attribute_policy=missing_attribute_policy.value,
            result_classification=ResultClassification.EXTERNAL_REAL_DATA_VALIDATION,
            evidence_state=suite.evidence_state.value if hasattr(suite.evidence_state, "value") else str(suite.evidence_state),
            number_of_comparisons=suite.number_of_comparisons,
            analysis_type=suite.analysis_type,
            worst_group_selection_disclosure=suite.worst_group_selection_disclosure,
            uncertainty_configuration={"n_bootstraps": n_bootstraps, "ci_level": 0.95},
            bootstrap_configuration=suite.uncertainty_intervals or {},
            warnings=suite.warnings,
            abstentions=suite.abstentions,
        )

        suite.manifest_fingerprint = manifest.fingerprint

        if output_manifest_path:
            manifest.save_json(output_manifest_path)

        return suite, manifest
