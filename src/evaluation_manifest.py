# ============================================================================
# Immutable Evaluation Manifest & Deterministic Fingerprint Engine (Pass 6)
# Cryptographically verifiable metadata contract binding evaluation results
# to code versions, feature contracts, split parameters, and administrative policies.
# ============================================================================

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


class ResultClassification(str, Enum):
    # Standard Pass 7 Evidence Taxonomy
    INTERNAL_ENGINEERING_VERIFICATION = "INTERNAL_ENGINEERING_VERIFICATION"
    SYNTHETIC_DGP_VALIDATION = "SYNTHETIC_DGP_VALIDATION"
    INDEPENDENT_ORACLE_VALIDATION = "INDEPENDENT_ORACLE_VALIDATION"
    EXTERNAL_REAL_DATA_VALIDATION = "EXTERNAL_REAL_DATA_VALIDATION"
    EXPLORATORY = "EXPLORATORY"
    LEGACY = "LEGACY"
    # Backward compatibility aliases
    SYNTHETIC_VERIFIED = "SYNTHETIC_DGP_VALIDATION"
    SYNTHETIC_EXPLORATORY = "EXPLORATORY"
    EXTERNAL_PENDING_STAGING = "EXTERNAL_PENDING_STAGING"
    EXTERNAL_AUDIT_READY = "EXTERNAL_REAL_DATA_VALIDATION"
    FRAMEWORK_VALIDATION = "INTERNAL_ENGINEERING_VERIFICATION"


def compute_manifest_fingerprint(config: Union[Dict[str, Any], Any]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint from an evaluation manifest or configuration dict.
    
    CRITICAL METHODOLOGICAL NOTICE:
    This fingerprint serves strictly as an 'evaluation configuration identity'.
    It binds reported results to their exact evaluation inputs and parameters,
    ensuring that any change to features, splits, policies, models, or data triggers a change in
    the fingerprint. It does NOT constitute proof of statistical correctness or legal certification.
    """
    if hasattr(config, "to_dict"):
        raw_dict = config.to_dict()
    elif isinstance(config, dict):
        raw_dict = config
    else:
        raw_dict = asdict(config)

    # Exclude ephemeral / non-material run-time fields
    excluded_keys = {
        "manifest_id",
        "evaluation_id",
        "created_at_utc",
        "timestamp_utc",
        "fingerprint",
        "manifest_fingerprint",
    }
    canonical_items = {
        k: v for k, v in sorted(raw_dict.items()) if k not in excluded_keys and v is not None
    }
    canonical_json = json.dumps(canonical_items, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass
class EvaluationManifest:
    """
    Immutable manifest documenting the exact parameterization, provenance,
    governance choices, and evidence status of an evaluation run.
    """
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    evaluator_version: str = "7.0.0"
    benchmark_version: str = "1.2.0"
    dataset_identity: str = "EvaluationDataset"
    target_definition: str = "target"
    sensitive_attributes: List[str] = field(default_factory=list)
    feature_contract_version: str = "1.0.0"
    selected_features: List[str] = field(default_factory=list)
    excluded_features: List[str] = field(default_factory=list)
    split_strategy: str = "stratified"
    random_seeds: List[int] = field(default_factory=lambda: [42])
    model_configuration: Dict[str, Any] = field(default_factory=dict)
    fairness_policy_configuration: Dict[str, Any] = field(default_factory=dict)
    missing_attribute_policy: str = "REJECT"
    result_classification: ResultClassification = ResultClassification.SYNTHETIC_DGP_VALIDATION
    evidence_state: str = "INSUFFICIENT_EVIDENCE"
    number_of_comparisons: int = 1
    analysis_type: str = "SCREENING_ANALYSIS"
    manifest_version: str = "1.0.0"
    dataset_sha256: Optional[str] = None
    dataset_provenance: Optional[Dict[str, Any]] = None
    sensitive_attribute_provenance: Optional[Dict[str, Any]] = None
    grouping_strategy: Optional[str] = None
    temporal_strategy: Optional[str] = None
    threshold_configuration: Optional[Dict[str, Any]] = None
    intersectional_specifications: Optional[Dict[str, Any]] = None
    uncertainty_configuration: Dict[str, Any] = field(default_factory=dict)
    bootstrap_configuration: Dict[str, Any] = field(default_factory=dict)
    resampling_unit: str = "ROW_BOOTSTRAP"
    cluster_key: Optional[str] = None
    confidence_method: str = "EMPIRICAL_PERCENTILE_BOOTSTRAP"
    worst_group_selection_disclosure: Optional[Dict[str, Any]] = None
    result_artifact_paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    abstentions: List[str] = field(default_factory=list)
    leakage_findings: List[Dict[str, Any]] = field(default_factory=list)
    manifest_id: Optional[str] = None
    created_at_utc: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    metric_values: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    evaluation_unit_level: str = "ROW_LEVEL"
    evaluation_unit_key: Optional[str] = None
    aggregation_method: str = "NONE"
    statistical_estimand: Optional[Dict[str, Any]] = None
    privacy_governance_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["result_classification"] = (
            self.result_classification.value
            if isinstance(self.result_classification, ResultClassification)
            else str(self.result_classification)
        )
        return d

    def to_canonical_json(self) -> str:
        """Returns deterministic, formatted JSON representation of the manifest."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save_json(self, output_path: Union[str, Path]) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(self.to_canonical_json())
        return out

    @classmethod
    def create(
        cls,
        dataset_identity: str,
        target_definition: str,
        sensitive_attributes: Sequence[str],
        feature_contract_version: str,
        selected_features: Sequence[str],
        excluded_features: Sequence[str],
        split_strategy: str,
        random_seeds: Sequence[int],
        model_configuration: Dict[str, Any],
        fairness_policy_configuration: Dict[str, Any],
        missing_attribute_policy: str,
        result_classification: Union[ResultClassification, str],
        evidence_state: str,
        number_of_comparisons: int,
        analysis_type: str,
        evaluator_version: str = "6.0.0",
        benchmark_version: str = "1.1.0",
        dataset_sha256: Optional[str] = None,
        dataset_provenance: Optional[Dict[str, Any]] = None,
        sensitive_attribute_provenance: Optional[Dict[str, Any]] = None,
        grouping_strategy: Optional[str] = None,
        temporal_strategy: Optional[str] = None,
        threshold_configuration: Optional[Dict[str, Any]] = None,
        intersectional_specifications: Optional[Dict[str, Any]] = None,
        uncertainty_configuration: Optional[Dict[str, Any]] = None,
        bootstrap_configuration: Optional[Dict[str, Any]] = None,
        worst_group_selection_disclosure: Optional[Dict[str, Any]] = None,
        result_artifact_paths: Optional[Sequence[str]] = None,
        warnings: Optional[Sequence[str]] = None,
        abstentions: Optional[Sequence[str]] = None,
        leakage_findings: Optional[Sequence[Dict[str, Any]]] = None,
        evaluation_id: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
        evaluation_unit_level: str = "ROW_LEVEL",
        evaluation_unit_key: Optional[str] = None,
        aggregation_method: str = "NONE",
        privacy_governance_summary: Optional[Dict[str, Any]] = None,
    ) -> EvaluationManifest:
        """Factory constructor generating deterministic fingerprint and timestamps."""
        eval_id = evaluation_id or str(uuid.uuid4())
        ts = timestamp_utc or datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        classification = (
            result_classification
            if isinstance(result_classification, ResultClassification)
            else ResultClassification(result_classification)
        )

        config_for_fingerprint = {
            "dataset_identity": dataset_identity,
            "dataset_sha256": dataset_sha256,
            "evaluator_version": evaluator_version,
            "benchmark_version": benchmark_version,
            "target_definition": target_definition,
            "sensitive_attributes": list(sensitive_attributes),
            "feature_contract_version": feature_contract_version,
            "selected_features": list(selected_features),
            "excluded_features": list(excluded_features),
            "split_strategy": split_strategy,
            "grouping_strategy": grouping_strategy,
            "temporal_strategy": temporal_strategy,
            "random_seeds": list(random_seeds),
            "model_configuration": model_configuration,
            "threshold_configuration": threshold_configuration,
            "fairness_policy_configuration": fairness_policy_configuration,
            "missing_attribute_policy": missing_attribute_policy,
            "evaluation_unit_level": evaluation_unit_level,
            "aggregation_method": aggregation_method,
        }
        fp = compute_manifest_fingerprint(config_for_fingerprint)

        return cls(
            evaluation_id=eval_id,
            fingerprint=fp,
            timestamp_utc=ts,
            evaluator_version=evaluator_version,
            benchmark_version=benchmark_version,
            dataset_identity=dataset_identity,
            dataset_sha256=dataset_sha256,
            dataset_provenance=dataset_provenance,
            target_definition=target_definition,
            sensitive_attributes=list(sensitive_attributes),
            sensitive_attribute_provenance=sensitive_attribute_provenance,
            feature_contract_version=feature_contract_version,
            selected_features=list(selected_features),
            excluded_features=list(excluded_features),
            split_strategy=split_strategy,
            grouping_strategy=grouping_strategy,
            temporal_strategy=temporal_strategy,
            random_seeds=list(random_seeds),
            model_configuration=model_configuration,
            threshold_configuration=threshold_configuration,
            fairness_policy_configuration=fairness_policy_configuration,
            intersectional_specifications=intersectional_specifications,
            uncertainty_configuration=uncertainty_configuration or {},
            bootstrap_configuration=bootstrap_configuration or {},
            missing_attribute_policy=missing_attribute_policy,
            result_classification=classification,
            evidence_state=evidence_state,
            number_of_comparisons=number_of_comparisons,
            analysis_type=analysis_type,
            worst_group_selection_disclosure=worst_group_selection_disclosure,
            result_artifact_paths=list(result_artifact_paths or []),
            warnings=list(warnings or []),
            abstentions=list(abstentions or []),
            leakage_findings=list(leakage_findings or []),
            evaluation_unit_level=evaluation_unit_level,
            evaluation_unit_key=evaluation_unit_key,
            aggregation_method=aggregation_method,
            privacy_governance_summary=privacy_governance_summary,
        )


def create_evaluation_manifest(
    run_id: Optional[str] = None,
    evaluation_id: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_identity: Optional[str] = None,
    regime: Optional[str] = None,
    split_strategy: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    seeds: Optional[Sequence[int]] = None,
    random_seeds: Optional[Sequence[int]] = None,
    result_classification: Union[ResultClassification, str] = ResultClassification.INTERNAL_ENGINEERING_VERIFICATION,
    **kwargs: Any,
) -> EvaluationManifest:
    """
    Convenience factory function for creating an EvaluationManifest with sensible defaults
    and automatic fingerprint computation.
    """
    actual_eval_id = run_id or evaluation_id or str(uuid.uuid4())
    actual_ds_identity = dataset_name or dataset_identity or "EvaluationDataset"
    actual_split = regime or split_strategy or "stratified"
    actual_seeds = list(seeds if seeds is not None else (random_seeds if random_seeds is not None else [42]))

    classification = (
        result_classification
        if isinstance(result_classification, ResultClassification)
        else ResultClassification(result_classification)
    )

    manifest_args: Dict[str, Any] = {
        "evaluation_id": actual_eval_id,
        "dataset_identity": actual_ds_identity,
        "split_strategy": actual_split,
        "random_seeds": actual_seeds,
        "result_classification": classification,
        "evaluator_version": "7.0.0",
        "benchmark_version": "1.2.0",
    }
    if model_name:
        manifest_args["model_configuration"] = {"model_name": model_name}
    if metrics:
        manifest_args["metric_values"] = metrics

    for k, v in kwargs.items():
        manifest_args[k] = v

    manifest = EvaluationManifest(**manifest_args)
    if not manifest.fingerprint:
        manifest.fingerprint = compute_manifest_fingerprint(manifest)
    return manifest

