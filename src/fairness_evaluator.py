# ============================================================================
# Defensive Fairness Evaluator & Clean Split Hierarchy (Pass 5)
# Independently recomputes fairness metrics directly from predictions,
# enforces robust edge-case handling (empty/tiny groups, zero pos/neg),
# refactors composite score and welfare loss proxies, supports multi-attribute
# intersectional grouping, sparse-group support gating, non-parametric
# bootstrap uncertainty estimation, subgroup utility auditing, and policy sensitivity.
# ============================================================================

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvidenceState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class MissingAttributePolicy(str, Enum):
    REJECT = "REJECT"
    EXCLUDE = "EXCLUDE"
    EXPLICIT_UNKNOWN_GROUP = "EXPLICIT_UNKNOWN_GROUP"
    IMPUTE_WITH_DOCUMENTED_RULE = "IMPUTE_WITH_DOCUMENTED_RULE"


class ResamplingUnit(str, Enum):
    ROW_BOOTSTRAP = "ROW_BOOTSTRAP"
    CLUSTER_BOOTSTRAP = "CLUSTER_BOOTSTRAP"
    ROW = "ROW_BOOTSTRAP"
    CLUSTER = "CLUSTER_BOOTSTRAP"


class AnalysisType(str, Enum):
    SCREENING_ANALYSIS = "SCREENING_ANALYSIS"
    CONFIRMATORY_ANALYSIS = "CONFIRMATORY_ANALYSIS"
    SCREENING = "SCREENING_ANALYSIS"
    CONFIRMATORY = "CONFIRMATORY_ANALYSIS"


class EvaluationStatus(str, Enum):
    VALID = "VALID"
    HEALTHY = "VALID"
    UNDEFINED = "UNDEFINED"
    INSUFFICIENT_GROUP_SUPPORT = "INSUFFICIENT_GROUP_SUPPORT"
    SPARSE_SUBGROUP = "INSUFFICIENT_GROUP_SUPPORT"
    UNRELIABLE_ESTIMATE = "UNRELIABLE_ESTIMATE"
    ZERO_POSITIVE_SUBGROUP = "ZERO_POSITIVE_SUBGROUP"
    ZERO_NEGATIVE_SUBGROUP = "ZERO_NEGATIVE_SUBGROUP"
    MISSING_SENSITIVE = "MISSING_SENSITIVE"
    MISSING_SENSITIVE_ATTRIBUTE = "MISSING_ATTRIBUTE_VIOLATION"
    DEGENERATE_TARGET = "DEGENERATE_TARGET"
    INSUFFICIENT_BOOTSTRAP_SUPPORT = "INSUFFICIENT_BOOTSTRAP_SUPPORT"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_BOOTSTRAP_SUPPORT"
    UNSTABLE_RATIO = "UNSTABLE_RATIO"
    ZERO_SELECTION_RATE = "UNDEFINED"
    MISSING_ATTRIBUTE_VIOLATION = "MISSING_ATTRIBUTE_VIOLATION"
    INSUFFICIENT_CALIBRATION_SUPPORT = "INSUFFICIENT_CALIBRATION_SUPPORT"


class ModelDegeneracyStatus(str, Enum):
    NORMAL = "NORMAL"
    DEGENERATE = "DEGENERATE"
    COLLAPSED_CONSTANT = "DEGENERATE"
    COLLAPSED_SINGLE_CLASS = "DEGENERATE"
    INSUFFICIENT_SIGNAL = "INSUFFICIENT_SIGNAL"


class FairnessSuccessStatus(str, Enum):
    FAIR_AND_USEFUL = "FAIR_AND_USEFUL"
    FAIR_BUT_DEGENERATE = "FAIR_BUT_DEGENERATE"
    UNFAIR_AND_USEFUL = "UNFAIR_AND_USEFUL"
    UNFAIR_AND_DEGENERATE = "UNFAIR_AND_DEGENERATE"
    UNDEFINED = "UNDEFINED"


class PolicyDecision(str, Enum):
    POLICY_COMPLIANT = "POLICY_COMPLIANT"
    POLICY_NONCOMPLIANT = "POLICY_NONCOMPLIANT"
    POLICY_DEGENERATE = "POLICY_DEGENERATE"
    POLICY_UNDEFINED = "POLICY_UNDEFINED"


class EvaluationUnitLevel(str, Enum):
    ROW_LEVEL = "ROW_LEVEL"
    TRIP_LEVEL = "TRIP_LEVEL"
    ROUTE_LEVEL = "ROUTE_LEVEL"
    WORKER_LEVEL = "WORKER_LEVEL"
    INDIVIDUAL_LEVEL = "INDIVIDUAL_LEVEL"
    ENTITY_LEVEL = "ENTITY_LEVEL"
    SEGMENT_LEVEL = "SEGMENT_LEVEL"


class AggregationStrategy(str, Enum):
    NONE = "NONE"
    MAJORITY_VOTE = "MAJORITY_VOTE"
    ANY_FAILURE = "ANY_FAILURE"
    ALL_SUCCESS = "ALL_SUCCESS"
    FIRST_RECORD = "FIRST_RECORD"
    LAST_RECORD = "LAST_RECORD"
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    ENTITY_WEIGHTED = "ENTITY_WEIGHTED"
    UNSUPPORTED_EVALUATION_UNIT = "UNSUPPORTED_EVALUATION_UNIT"


class PolicyType(str, Enum):
    ANALYTICAL = "ANALYTICAL"
    ORGANIZATIONAL = "ORGANIZATIONAL"
    REGULATORY_MAPPING = "REGULATORY_MAPPING"


class ReportingPrivacyMode(str, Enum):
    FULL_INTERNAL = "FULL_INTERNAL"
    REDACTED_EXTERNAL = "REDACTED_EXTERNAL"
    SUMMARY_ONLY = "SUMMARY_ONLY"


class AuditExecutionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    ABORTED = "ABORTED"


class EvidenceStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class ModelStatus(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    DEGENERATE = "DEGENERATE"
    INVALID = "INVALID"


class PolicyStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class HumanReviewReasonCode(str, Enum):
    REVIEW_REGULATORY_MAPPING = "REVIEW_REGULATORY_MAPPING"
    REVIEW_DERIVED_SENSITIVE_ATTRIBUTE = "REVIEW_DERIVED_SENSITIVE_ATTRIBUTE"
    REVIEW_LOW_SUPPORT = "REVIEW_LOW_SUPPORT"
    REVIEW_AMBIGUOUS_ESTIMAND = "REVIEW_AMBIGUOUS_ESTIMAND"
    REVIEW_FEATURE_TIMING = "REVIEW_FEATURE_TIMING"
    REVIEW_MATERIAL_POLICY_VIOLATION = "REVIEW_MATERIAL_POLICY_VIOLATION"
    REVIEW_INCONCLUSIVE_EVIDENCE = "REVIEW_INCONCLUSIVE_EVIDENCE"
    REVIEW_UNIT_MISMATCH = "REVIEW_UNIT_MISMATCH"
    REVIEW_POLICY_AMBIGUITY = "REVIEW_POLICY_AMBIGUITY"


HUMAN_REVIEW_REASON_CODE_ALIASES: Dict[str, str] = {
    "MARGINAL_DISPARITY": HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value,
    "SPARSE_SUBGROUP": HumanReviewReasonCode.REVIEW_LOW_SUPPORT.value,
    "SHIFT_DETECTED": HumanReviewReasonCode.REVIEW_INCONCLUSIVE_EVIDENCE.value,
    "POLICY_AMBIGUITY": HumanReviewReasonCode.REVIEW_POLICY_AMBIGUITY.value,
    "HIGH_STAKES_DECISION": HumanReviewReasonCode.REVIEW_REGULATORY_MAPPING.value,
    "UNCALIBRATED_PROBABILITIES": HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value,
}


def canonicalize_human_review_reason_code(code: Union[str, HumanReviewReasonCode]) -> str:
    """Maps reason code or legacy/variant aliases to canonical HumanReviewReasonCode string."""
    raw = code.value if isinstance(code, HumanReviewReasonCode) else str(code)
    return HUMAN_REVIEW_REASON_CODE_ALIASES.get(raw, raw)


@dataclass
class HumanReviewBoundary:
    human_review_required: bool = False
    reason_codes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "human_review_required": self.human_review_required,
            "reason_codes": self.reason_codes,
            "notes": self.notes,
        }


@dataclass
class AuditTaxonomyState:
    audit_execution_status: str = AuditExecutionStatus.COMPLETE.value
    evidence_status: str = EvidenceStatus.SUFFICIENT.value
    model_status: str = ModelStatus.ACCEPTABLE.value
    policy_status: str = PolicyStatus.PASS.value
    precedence_rule_applied: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_execution_status": self.audit_execution_status,
            "evidence_status": self.evidence_status,
            "model_status": self.model_status,
            "policy_status": self.policy_status,
            "precedence_rule_applied": self.precedence_rule_applied,
            "details": self.details,
        }


def map_to_audit_taxonomy(
    audit_execution_status: Union[str, AuditExecutionStatus] = AuditExecutionStatus.COMPLETE,
    evidence_status: Optional[Union[str, EvidenceStatus]] = None,
    model_status: Optional[Union[str, ModelStatus]] = None,
    policy_status: Optional[Union[str, PolicyStatus]] = None,
    suite: Optional[Any] = None,
    policy_result: Optional[Any] = None,
    analysis_type: str = "CONFIRMATORY_ANALYSIS",
    leakage_findings_count: int = 0,
    is_degenerate: bool = False,
    gate_passed: Optional[bool] = None,
    gate_violations: Optional[List[str]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AuditTaxonomyState:
    """
    Enforces strict precedence rules across the decoupled status taxonomy:
      1. If AUDIT_EXECUTION_STATUS is BLOCKED: POLICY_STATUS must be NOT_EVALUABLE.
      2. If MODEL_STATUS is INVALID (critical leakage, contract failure): POLICY_STATUS must be NOT_EVALUABLE in confirmatory mode.
      3. If MODEL_STATUS is DEGENERATE: POLICY_STATUS must be FAIL and EVIDENCE_STATUS must be INSUFFICIENT.
      4. If EVIDENCE_STATUS is INSUFFICIENT: POLICY_STATUS must be NOT_EVALUABLE in confirmatory mode.
      5. If Confirmatory mode requirements are unmet (estimand missing, gate failed): POLICY_STATUS must be NOT_EVALUABLE.
      6. Otherwise: PASS if policy conditions satisfied, FAIL if violated.
    """
    exec_status = (
        audit_execution_status.value
        if isinstance(audit_execution_status, AuditExecutionStatus)
        else str(audit_execution_status)
    )
    det = dict(details or {})

    # Auto-infer model status if not explicitly given
    if model_status is None:
        if leakage_findings_count > 0:
            m_status = ModelStatus.INVALID.value
        elif is_degenerate or (suite and getattr(suite, "degeneracy_status", None) == ModelDegeneracyStatus.DEGENERATE):
            m_status = ModelStatus.DEGENERATE.value
        else:
            m_status = ModelStatus.ACCEPTABLE.value
    else:
        m_status = model_status.value if isinstance(model_status, ModelStatus) else str(model_status)

    # Auto-infer evidence status if not explicitly given
    if evidence_status is None:
        if m_status == ModelStatus.DEGENERATE.value:
            ev_status = EvidenceStatus.INSUFFICIENT.value
        elif suite and getattr(suite, "subgroup_metrics", None) and any(
            g.get("support_status") == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value
            for g in suite.subgroup_metrics.values()
            if isinstance(g, dict)
        ):
            ev_status = EvidenceStatus.INSUFFICIENT.value
        elif suite and getattr(suite, "uncertainty_intervals", None):
            has_low_frac = any(
                (u.get("valid_fraction") if isinstance(u, dict) else getattr(u, "valid_fraction", 1.0)) < 0.80
                for u in suite.uncertainty_intervals.values()
                if (isinstance(u, dict) and u.get("valid_fraction") is not None)
                or (hasattr(u, "valid_fraction") and getattr(u, "valid_fraction", None) is not None)
            )
            ev_status = EvidenceStatus.INSUFFICIENT.value if has_low_frac else EvidenceStatus.SUFFICIENT.value
        else:
            ev_status = EvidenceStatus.SUFFICIENT.value
    else:
        ev_status = evidence_status.value if isinstance(evidence_status, EvidenceStatus) else str(evidence_status)

    # Precedence evaluations
    precedence = None
    final_policy_status: str

    if exec_status == AuditExecutionStatus.BLOCKED.value:
        final_policy_status = PolicyStatus.NOT_EVALUABLE.value
        ev_status = EvidenceStatus.INSUFFICIENT.value
        precedence = "PRECEDENCE_EXECUTION_BLOCKED"
    elif m_status == ModelStatus.INVALID.value:
        final_policy_status = PolicyStatus.NOT_EVALUABLE.value
        ev_status = EvidenceStatus.INSUFFICIENT.value
        precedence = "PRECEDENCE_MODEL_INVALID"
    elif m_status == ModelStatus.DEGENERATE.value:
        final_policy_status = PolicyStatus.FAIL.value
        ev_status = EvidenceStatus.INSUFFICIENT.value
        precedence = "PRECEDENCE_MODEL_DEGENERATE"
    elif ev_status == EvidenceStatus.INSUFFICIENT.value and analysis_type in (
        AnalysisType.CONFIRMATORY_ANALYSIS.value,
        "CONFIRMATORY",
        "CONFIRMATORY_AUDIT",
    ):
        final_policy_status = PolicyStatus.NOT_EVALUABLE.value
        precedence = "PRECEDENCE_EVIDENCE_INSUFFICIENT"
    elif gate_passed is False and analysis_type in (
        AnalysisType.CONFIRMATORY_ANALYSIS.value,
        "CONFIRMATORY",
        "CONFIRMATORY_AUDIT",
    ):
        final_policy_status = PolicyStatus.NOT_EVALUABLE.value
        precedence = "PRECEDENCE_CONFIRMATORY_REQUIREMENTS_UNMET"
    elif policy_status is not None:
        final_policy_status = policy_status.value if isinstance(policy_status, PolicyStatus) else str(policy_status)
    elif policy_result is not None:
        if getattr(policy_result, "policy_passed", False):
            final_policy_status = PolicyStatus.PASS.value
        else:
            final_policy_status = PolicyStatus.FAIL.value
    elif suite and getattr(suite, "policy_result", None):
        if getattr(suite.policy_result, "policy_passed", False):
            final_policy_status = PolicyStatus.PASS.value
        else:
            final_policy_status = PolicyStatus.FAIL.value
    else:
        final_policy_status = PolicyStatus.PASS.value

    det["analysis_type"] = analysis_type
    det["is_exploratory"] = analysis_type not in (
        AnalysisType.CONFIRMATORY_ANALYSIS.value,
        "CONFIRMATORY",
        "CONFIRMATORY_AUDIT",
    )

    return AuditTaxonomyState(
        audit_execution_status=exec_status,
        evidence_status=ev_status,
        model_status=m_status,
        policy_status=final_policy_status,
        precedence_rule_applied=precedence,
        details=det,
    )


def determine_human_review_boundary(
    taxonomy_state: Optional[AuditTaxonomyState] = None,
    suite: Optional[Any] = None,
    policy_result: Optional[Any] = None,
    sensitive_lineage: Optional[Union[str, Any]] = None,
    policy_type: Optional[Union[str, PolicyType]] = None,
    has_timing_features: bool = False,
    estimand_specified: bool = True,
    unit_mismatch: bool = False,
    policy_ambiguity: bool = False,
    additional_reason_codes: Optional[Sequence[Union[str, HumanReviewReasonCode]]] = None,
) -> HumanReviewBoundary:
    """
    Decouples human review from technical evidence and policy validity.
    Human review sits on top of mathematically reported facts and flags instances
    requiring qualitative, legal, or operational human judgment.
    Does NOT modify statistical outcomes or silently flip pass/fail flags.
    """
    reasons: List[str] = []
    notes: List[str] = []

    # 1. Material Policy Violation
    if taxonomy_state and taxonomy_state.policy_status == PolicyStatus.FAIL.value:
        reasons.append(HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value)
        notes.append("Automated policy conditions failed; human governance review required before deployment.")
    elif policy_result and not getattr(policy_result, "policy_passed", True):
        reasons.append(HumanReviewReasonCode.REVIEW_MATERIAL_POLICY_VIOLATION.value)
        notes.append("Policy condition violations detected.")

    # 2. Inconclusive or Insufficient Evidence
    if taxonomy_state and (
        taxonomy_state.policy_status == PolicyStatus.NOT_EVALUABLE.value
        or taxonomy_state.evidence_status == EvidenceStatus.INSUFFICIENT.value
    ):
        reasons.append(HumanReviewReasonCode.REVIEW_INCONCLUSIVE_EVIDENCE.value)
        notes.append("Evidence is insufficient or policy is not evaluable; human review required to determine remediation.")

    # 3. Derived or Proxy Sensitive Attributes
    if sensitive_lineage in ("DERIVED_PROXY", "IMPUTED"):
        reasons.append(HumanReviewReasonCode.REVIEW_DERIVED_SENSITIVE_ATTRIBUTE.value)
        notes.append("Sensitive attribute lineage is derived or imputed proxy; requires legal and demographic review.")

    # 4. Regulatory Mapping
    pol_type_val = policy_type.value if hasattr(policy_type, "value") else str(policy_type or "")
    if pol_type_val == PolicyType.REGULATORY_MAPPING.value or "REGULATORY" in pol_type_val.upper():
        reasons.append(HumanReviewReasonCode.REVIEW_REGULATORY_MAPPING.value)
        notes.append("Evaluation uses regulatory mapping thresholds; legal compliance review required.")

    # 5. Low Support / Sparse Subgroups
    if suite and getattr(suite, "subgroup_metrics", None):
        sparse_groups = [
            g for g, d in suite.subgroup_metrics.items()
            if isinstance(d, dict) and d.get("support_status") == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value
        ]
        if sparse_groups:
            reasons.append(HumanReviewReasonCode.REVIEW_LOW_SUPPORT.value)
            notes.append(f"Subgroups with insufficient sample support detected: {sparse_groups}")

    # 6. Ambiguous or Unspecified Estimand
    if not estimand_specified or (suite and getattr(suite, "statistical_estimand", None) is None):
        reasons.append(HumanReviewReasonCode.REVIEW_AMBIGUOUS_ESTIMAND.value)
        notes.append("Statistical estimand is unspecified or ambiguous.")

    # 7. Feature Timing / Temporal Discrepancies
    if has_timing_features:
        reasons.append(HumanReviewReasonCode.REVIEW_FEATURE_TIMING.value)
        notes.append("Model uses temporal or sequence features; verify prediction-time feature cutoff boundaries.")

    # 8. Evaluation Unit Mismatch
    if unit_mismatch:
        reasons.append(HumanReviewReasonCode.REVIEW_UNIT_MISMATCH.value)
        notes.append("Statistical unit divergence from observational rows requires domain verification.")

    # 9. Policy Ambiguity or Conflicting Thresholds
    if policy_ambiguity:
        reasons.append(HumanReviewReasonCode.REVIEW_POLICY_AMBIGUITY.value)
        notes.append("Policy condition definitions or statutory thresholds exhibit ambiguity or conflict.")

    # 10. Additional or Alias Review Codes
    if additional_reason_codes:
        for arc in additional_reason_codes:
            canon = canonicalize_human_review_reason_code(arc)
            if canon not in reasons:
                reasons.append(canon)
                notes.append(f"Additional governance review trigger: {arc}")

    return HumanReviewBoundary(
        human_review_required=(len(reasons) > 0),
        reason_codes=reasons,
        notes=notes,
    )


@dataclass
class AggregationStrategySpecification:
    strategy: AggregationStrategy
    mathematical_definition: str
    semantic_description: str
    applicable_target_types: List[str]
    applicable_prediction_types: List[str]
    known_limitations: str
    recommended_target_operator: str
    recommended_prediction_operator: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AGGREGATION_STRATEGY_REGISTRY: Dict[AggregationStrategy, AggregationStrategySpecification] = {
    AggregationStrategy.ANY_FAILURE: AggregationStrategySpecification(
        strategy=AggregationStrategy.ANY_FAILURE,
        mathematical_definition="y_agg = 1 if all(y_i == 1) else 0 (product of binary indicators)",
        semantic_description="Fails the entire entity if any single constituent observation fails (e.g. delivery cutoff breached).",
        applicable_target_types=["BINARY_CLASSIFICATION", "STOCHASTIC_EVENTS"],
        applicable_prediction_types=["BINARY_HARD_PREDICTION", "THRESHOLDED_DECISION"],
        known_limitations="Sensitive to entity length: longer entities with more segments have mechanically higher risk of any failure.",
        recommended_target_operator="min(y_i) / all(y_i == 1)",
        recommended_prediction_operator="all(y_hat_i == 1)",
    ),
    AggregationStrategy.ALL_SUCCESS: AggregationStrategySpecification(
        strategy=AggregationStrategy.ALL_SUCCESS,
        mathematical_definition="y_agg = 1 if any(y_i == 1) else 0 (maximum of binary indicators)",
        semantic_description="Succeeds if any constituent observation succeeds (e.g. redundant carrier or multi-attempt delivery).",
        applicable_target_types=["BINARY_CLASSIFICATION"],
        applicable_prediction_types=["BINARY_HARD_PREDICTION"],
        known_limitations="Mechanically inflates success rates for entities with many attempts or sub-segments.",
        recommended_target_operator="max(y_i) / any(y_i == 1)",
        recommended_prediction_operator="any(y_hat_i == 1)",
    ),
    AggregationStrategy.MAJORITY_VOTE: AggregationStrategySpecification(
        strategy=AggregationStrategy.MAJORITY_VOTE,
        mathematical_definition="y_agg = 1 if mean(y_i) > 0.50 else 0",
        semantic_description="Assigns entity label based on majority consensus among constituent observations.",
        applicable_target_types=["BINARY_CLASSIFICATION", "MULTI_ANNOTATOR"],
        applicable_prediction_types=["BINARY_HARD_PREDICTION", "ENSEMBLE_ENSEMBLE"],
        known_limitations="Ignores severity of individual segment failures; ties require deterministic arbitration.",
        recommended_target_operator="mean(y_i) > 0.50",
        recommended_prediction_operator="mean(y_hat_i) > 0.50",
    ),
    AggregationStrategy.FIRST_RECORD: AggregationStrategySpecification(
        strategy=AggregationStrategy.FIRST_RECORD,
        mathematical_definition="y_agg = y_1 (chronologically earliest or first recorded observation)",
        semantic_description="Selects earliest recorded observation as representative of entity state.",
        applicable_target_types=["LONGITUDINAL", "TEMPORAL_SEQUENCE"],
        applicable_prediction_types=["INTAKE_PREDICTION", "BASELINE_SCORE"],
        known_limitations="Discards subsequent intra-entity transitions, adaptations, and outcome changes.",
        recommended_target_operator="y[0]",
        recommended_prediction_operator="y_hat[0]",
    ),
    AggregationStrategy.LAST_RECORD: AggregationStrategySpecification(
        strategy=AggregationStrategy.LAST_RECORD,
        mathematical_definition="y_agg = y_T (chronologically latest or terminal observation)",
        semantic_description="Selects terminal observation as representative of entity final state.",
        applicable_target_types=["LONGITUDINAL", "TERMINAL_OUTCOME"],
        applicable_prediction_types=["FINAL_DISPATCH_SCORE"],
        known_limitations="May introduce survivorship bias if attrition occurs before final observation.",
        recommended_target_operator="y[-1]",
        recommended_prediction_operator="y_hat[-1]",
    ),
    AggregationStrategy.ENTITY_WEIGHTED: AggregationStrategySpecification(
        strategy=AggregationStrategy.ENTITY_WEIGHTED,
        mathematical_definition="w_i = (1 / n_entity_i) * (N_total / N_unique_entities)",
        semantic_description="Inverse-frequency observation weighting ensuring each unique entity contributes equal aggregate weight.",
        applicable_target_types=["ALL_TYPES", "BINARY_CLASSIFICATION", "REGRESSION"],
        applicable_prediction_types=["ALL_TYPES", "CONTINUOUS_PROBABILITY", "BINARY_HARD_PREDICTION"],
        known_limitations="Does not collapse rows into single decisions; standard errors require cluster-aware estimation.",
        recommended_target_operator="weighted_mean",
        recommended_prediction_operator="weighted_mean",
    ),
    AggregationStrategy.NONE: AggregationStrategySpecification(
        strategy=AggregationStrategy.NONE,
        mathematical_definition="y_agg = y (identity mapping)",
        semantic_description="No aggregation performed. Preserves raw row-level records under i.i.d. assumption.",
        applicable_target_types=["IID_RECORDS", "CROSS_SECTIONAL_SURVEY"],
        applicable_prediction_types=["ROW_PREDICTION"],
        known_limitations="Invalid if rows exhibit intra-entity correlation or clustering.",
        recommended_target_operator="identity",
        recommended_prediction_operator="identity",
    ),
    AggregationStrategy.UNSUPPORTED_EVALUATION_UNIT: AggregationStrategySpecification(
        strategy=AggregationStrategy.UNSUPPORTED_EVALUATION_UNIT,
        mathematical_definition="Refusal to aggregate",
        semantic_description="Explicit refusal when target semantics or data structure do not justify heuristic aggregation.",
        applicable_target_types=[],
        applicable_prediction_types=[],
        known_limitations="Blocks evaluation until user explicitly configures domain-justified aggregation.",
        recommended_target_operator="raise ValueError",
        recommended_prediction_operator="raise ValueError",
    ),
}


def _apply_aggregation_operator(values: np.ndarray, strat: AggregationStrategy) -> int:
    if strat == AggregationStrategy.ANY_FAILURE:
        return int(np.all(values == 1))
    elif strat == AggregationStrategy.ALL_SUCCESS:
        return int(np.any(values == 1))
    elif strat == AggregationStrategy.MAJORITY_VOTE:
        return int(np.mean(values) > 0.5)
    elif strat == AggregationStrategy.FIRST_RECORD:
        return int(values[0])
    elif strat == AggregationStrategy.LAST_RECORD:
        return int(values[-1])
    elif strat == AggregationStrategy.UNSUPPORTED_EVALUATION_UNIT:
        raise ValueError("UNSUPPORTED_EVALUATION_AGGREGATION: Target or prediction semantics do not support safe entity aggregation.")
    else:
        raise ValueError(f"UNSUPPORTED_EVALUATION_AGGREGATION: Unknown or unsupported aggregation strategy: {strat}")


def aggregate_to_evaluation_unit(
    y_true: Union[pd.Series, np.ndarray, Sequence[Any]],
    y_pred: Union[pd.Series, np.ndarray, Sequence[Any]],
    sensitive: Union[pd.Series, np.ndarray, Sequence[Any]],
    unit_ids: Optional[Union[pd.Series, np.ndarray, Sequence[Any]]] = None,
    y_prob: Optional[Union[pd.Series, np.ndarray, Sequence[Any]]] = None,
    cluster_series: Optional[Union[pd.Series, np.ndarray, Sequence[Any]]] = None,
    strategy: Union[AggregationStrategy, str] = AggregationStrategy.ANY_FAILURE,
    entity_series: Optional[Union[pd.Series, np.ndarray, Sequence[Any]]] = None,
    target_strategy: Optional[Union[AggregationStrategy, str]] = None,
    prediction_strategy: Optional[Union[AggregationStrategy, str]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Safely collapses repeated row-level observations of an entity into a single decision per entity.
    Explicitly distinguishes observed target aggregation from prediction aggregation.
    
    Strategies:
    - ANY_FAILURE: If any row for the entity has outcome 0, the entity outcome is 0 (delivery cutoff semantics).
    - ALL_SUCCESS: Entity outcome is 1 if any record succeeded (redundant carrier semantics).
    - MAJORITY_VOTE: Entity outcome is majority label (mean > 0.5).
    - FIRST_RECORD: Earliest record for the entity.
    - LAST_RECORD: Latest record for the entity.
    - NONE: No aggregation, returns original arrays.
    - UNSUPPORTED_EVALUATION_UNIT: Explicit refusal for invalid semantics (returns UNSUPPORTED_EVALUATION_AGGREGATION).
    """
    try:
        strat = strategy if isinstance(strategy, AggregationStrategy) else AggregationStrategy(strategy)
    except Exception as exc:
        raise ValueError(f"UNSUPPORTED_EVALUATION_AGGREGATION: Invalid aggregation strategy '{strategy}': {exc}") from exc

    actual_unit_ids = unit_ids if unit_ids is not None else entity_series
    if actual_unit_ids is None and "entity_id" in kwargs:
        actual_unit_ids = kwargs["entity_id"]
    if actual_unit_ids is None:
        raise ValueError("unit_ids or entity_series must be provided for entity aggregation.")

    try:
        t_strat = (
            target_strategy if isinstance(target_strategy, AggregationStrategy)
            else (AggregationStrategy(target_strategy) if target_strategy is not None else strat)
        )
        p_strat = (
            prediction_strategy if isinstance(prediction_strategy, AggregationStrategy)
            else (AggregationStrategy(prediction_strategy) if prediction_strategy is not None else strat)
        )
    except Exception as exc:
        raise ValueError(f"UNSUPPORTED_EVALUATION_AGGREGATION: Invalid target/prediction aggregation strategy: {exc}") from exc

    if t_strat == AggregationStrategy.NONE and p_strat == AggregationStrategy.NONE:
        return (
            np.asarray(y_true),
            np.asarray(y_pred),
            np.asarray(sensitive),
            np.asarray(y_prob) if y_prob is not None else None,
            np.asarray(cluster_series) if cluster_series is not None else np.asarray(actual_unit_ids),
        )

    if (
        t_strat == AggregationStrategy.UNSUPPORTED_EVALUATION_UNIT
        or p_strat == AggregationStrategy.UNSUPPORTED_EVALUATION_UNIT
    ):
        raise ValueError("UNSUPPORTED_EVALUATION_AGGREGATION: Target or prediction semantics do not support safe entity aggregation.")

    y_t_arr = np.asarray(y_true)
    y_p_arr = np.asarray(y_pred)
    s_arr = np.asarray(sensitive)
    u_arr = np.asarray(actual_unit_ids)
    p_arr = np.asarray(y_prob) if y_prob is not None else None
    c_arr = np.asarray(cluster_series) if cluster_series is not None else None

    # Preserve order of first appearance of each unique entity
    _, idx = np.unique(u_arr, return_index=True)
    unique_units = u_arr[np.sort(idx)]

    agg_yt = []
    agg_yp = []
    agg_s = []
    agg_prob = [] if p_arr is not None else None
    agg_cluster = [] if c_arr is not None else None

    for uid in unique_units:
        mask = (u_arr == uid)
        t_sub = y_t_arr[mask]
        p_sub = y_p_arr[mask]
        s_sub = s_arr[mask]

        # Sensitive attribute must be entity-invariant (take first/mode)
        agg_s.append(s_sub[0])

        agg_yt.append(_apply_aggregation_operator(t_sub, t_strat))
        agg_yp.append(_apply_aggregation_operator(p_sub, p_strat))

        if p_arr is not None:
            prob_sub = p_arr[mask]
            agg_prob.append(float(np.mean(prob_sub)))

        if c_arr is not None:
            c_sub = c_arr[mask]
            agg_cluster.append(c_sub[0])

    return (
        np.array(agg_yt, dtype=int),
        np.array(agg_yp, dtype=int),
        np.array(agg_s),
        np.array(agg_prob, dtype=float) if agg_prob is not None else None,
        np.array(agg_cluster) if agg_cluster is not None else unique_units,
    )



def compute_entity_weights(
    unit_ids: Union[np.ndarray, pd.Series, Sequence[Any]],
    base_weights: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> np.ndarray:
    """
    Computes inverse-frequency weights per observation such that each unique entity
    contributes equal aggregate weight (1.0) to weighted metric calculations:
        w_i = (1.0 / count(unit_id_i)) * (N_total / N_unique_entities)
    If base_weights is provided, multiplies inverse-frequency weights by base_weights.
    """
    arr = np.asarray(unit_ids)
    unique_units, counts = np.unique(arr, return_counts=True)
    count_map = dict(zip(unique_units, counts))
    raw_weights = np.array([1.0 / count_map[u] for u in arr], dtype=float)
    if base_weights is not None:
        raw_weights = raw_weights * np.asarray(base_weights, dtype=float)
    return raw_weights * (len(arr) / np.sum(raw_weights))


def validate_evaluation_unit_configuration(
    df: pd.DataFrame,
    unit_config: Optional[Union[Dict[str, Any], str]] = None,
    temporal_mode: bool = False,
    cluster_mode: bool = False,
    evaluation_unit_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Validates evaluation unit configuration against an actual dataset DataFrame.
    Verifies:
      - evaluation unit key exists, is non-null, and has non-zero cardinality;
      - reports duplicate key rate and repeated rows per entity;
      - group key exists if configured;
      - cluster key exists if cluster mode or cluster key is configured;
      - timestamp key exists if temporal mode is configured.
    
    Raises:
        ValueError if invalid and raise_on_error=True.
    Returns:
        diagnostics: Dict[str, Any]
    """
    errors: List[str] = []
    diagnostics: Dict[str, Any] = {}

    if isinstance(unit_config, str):
        unit_key = unit_config
        group_key = None
        cluster_key = None
        timestamp_key = None
    elif isinstance(unit_config, dict):
        unit_key = unit_config.get("evaluation_unit_key")
        group_key = unit_config.get("group_key")
        cluster_key = unit_config.get("cluster_key")
        timestamp_key = unit_config.get("timestamp_key")
    else:
        unit_key = evaluation_unit_key
        group_key = None
        cluster_key = None
        timestamp_key = None

    if evaluation_unit_key and not unit_key:
        unit_key = evaluation_unit_key

    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        errors.append("INVALID_EVALUATION_UNIT_CONFIGURATION: Input dataset must be a non-empty pandas DataFrame.")

    # 1. Evaluation unit key validation
    if unit_key and isinstance(df, pd.DataFrame) and len(df) > 0:
        if unit_key not in df.columns:
            errors.append(f"INVALID_EVALUATION_UNIT_CONFIGURATION: Evaluation unit key '{unit_key}' does not exist in dataset.")
        else:
            s = df[unit_key]
            null_count = int(s.isna().sum())
            if null_count > 0:
                errors.append(f"INVALID_EVALUATION_UNIT_CONFIGURATION: Evaluation unit key '{unit_key}' contains {null_count} null values.")
            cardinality = int(s.nunique())
            if cardinality == 0:
                errors.append(f"INVALID_EVALUATION_UNIT_CONFIGURATION: Evaluation unit key '{unit_key}' has zero cardinality.")
            
            counts = s.value_counts()
            recs_per_entity_mean = float(counts.mean()) if len(counts) > 0 else 0.0
            recs_per_entity_max = int(counts.max()) if len(counts) > 0 else 0
            recs_per_entity_min = int(counts.min()) if len(counts) > 0 else 0
            duplicate_key_count = int(len(s) - cardinality)
            duplicate_rate = float(duplicate_key_count / len(s)) if len(s) > 0 else 0.0

            diagnostics["evaluation_unit_key"] = unit_key
            diagnostics["unit_cardinality"] = cardinality
            diagnostics["null_count"] = null_count
            diagnostics["duplicate_key_rate"] = duplicate_rate
            diagnostics["n_records"] = len(df)
            diagnostics["n_units"] = cardinality
            diagnostics["records_per_entity"] = {
                "min": recs_per_entity_min,
                "mean": recs_per_entity_mean,
                "max": recs_per_entity_max,
            }

    # 2. Group key validation
    if group_key and group_key not in df.columns:
        errors.append(f"INVALID_EVALUATION_UNIT_CONFIGURATION: Group key '{group_key}' does not exist in dataset.")

    # 3. Cluster key validation
    if (cluster_mode or cluster_key) and cluster_key:
        if cluster_key not in df.columns:
            errors.append(f"INVALID_EVALUATION_UNIT_CONFIGURATION: Cluster key '{cluster_key}' does not exist in dataset.")
        else:
            diagnostics["cluster_cardinality"] = int(df[cluster_key].nunique())

    # 4. Timestamp key validation
    if temporal_mode and timestamp_key:
        if timestamp_key not in df.columns:
            errors.append(f"INVALID_EVALUATION_UNIT_CONFIGURATION: Timestamp key '{timestamp_key}' does not exist in dataset for temporal evaluation.")

    if errors and raise_on_error:
        raise ValueError(errors[0])

    if not errors and unit_key and isinstance(df, pd.DataFrame) and len(df) > 0 and diagnostics.get("unit_cardinality", 0) > 0:
        diagnostics["semantic_validation_status"] = "VERIFIED_FROM_DATA"
    else:
        diagnostics["semantic_validation_status"] = "PENDING_RAW_DATA_VERIFICATION"

    return diagnostics


@dataclass

class IntersectionalGroupSpec:
    """
    Specification for multi-attribute intersectional grouping.
    Preserves provenance of component attributes and values.
    """
    attributes: List[str]
    name: Optional[str] = None
    min_group_size: int = 20
    min_group_positives: int = 5
    min_group_negatives: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": self.attributes,
            "name": self.name or " x ".join(self.attributes),
            "min_group_size": self.min_group_size,
            "min_group_positives": self.min_group_positives,
            "min_group_negatives": self.min_group_negatives,
        }


def build_intersectional_groups(
    df: pd.DataFrame,
    spec: IntersectionalGroupSpec,
) -> Tuple[pd.Series, Dict[str, Dict[str, Any]]]:
    """
    Constructs composite intersectional series from multiple attribute columns.
    Returns (composite_series, attribute_provenance_map).
    """
    for attr in spec.attributes:
        if attr not in df.columns:
            raise KeyError(f"Attribute '{attr}' not found in dataframe columns: {list(df.columns)}")

    series_parts = [df[attr].astype(str) for attr in spec.attributes]
    composite = series_parts[0]
    for p in series_parts[1:]:
        composite = composite + " x " + p

    group_name = spec.name or " x ".join(spec.attributes)
    composite.name = group_name

    provenance_map: Dict[str, Dict[str, Any]] = {}
    unique_combinations = df[spec.attributes].drop_duplicates()
    for _, row in unique_combinations.iterrows():
        label = " x ".join(str(row[attr]) for attr in spec.attributes)
        provenance_map[label] = {attr: row[attr] for attr in spec.attributes}

    return composite, provenance_map


@dataclass
class FairnessPolicy:
    """
    Formal administrative policy specifying fairness, predictive utility,
    robustness, and subgroup constraints.
    """
    name: str = "Standard_Operational_Policy"
    policy_id: str = "POL-STD-2026.1"
    policy_version: str = "1.0.0"
    policy_type: PolicyType = PolicyType.ANALYTICAL
    description: str = "Standard Administrative Fairness Policy"
    owner_authority: str = "Model Risk Governance Committee"
    effective_date: str = "2026-09-08"
    disclaimer: str = (
        "LEGAL DISCLAIMER: Compliance with this technical fairness policy reflects satisfaction of internal "
        "mathematical criteria and DOES NOT constitute legal certification of statutory non-discrimination "
        "or regulatory compliance immunity under Title VII, ECOA, FHA, or EU AI Act."
    )
    max_equalized_odds_diff: Optional[float] = 0.10
    max_demographic_parity_diff: Optional[float] = 0.10
    max_predictive_parity_diff: Optional[float] = None
    min_disparate_impact_ratio: Optional[float] = 0.80
    min_balanced_accuracy: float = 0.60
    min_recall: Optional[float] = 0.10
    min_specificity: Optional[float] = 0.10
    min_precision: Optional[float] = None
    min_positive_rate: float = 0.02
    max_positive_rate: float = 0.98
    min_entropy: float = 0.10
    minimum_group_support: int = 5
    min_group_positives: int = 5
    min_group_negatives: int = 5
    min_subgroup_balanced_accuracy: Optional[float] = None
    min_subgroup_recall: Optional[float] = None
    min_subgroup_specificity: Optional[float] = None
    require_better_than_majority_baseline: bool = True
    require_adequate_subgroup_support: bool = False
    min_calibration_support: int = 30
    min_valid_bootstrap_fraction: float = 0.80
    missing_attribute_policy: MissingAttributePolicy = MissingAttributePolicy.REJECT
    critical_subgroups: Optional[List[str]] = None
    analysis_type: AnalysisType = AnalysisType.SCREENING_ANALYSIS

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["missing_attribute_policy"] = (
            self.missing_attribute_policy.value
            if isinstance(self.missing_attribute_policy, MissingAttributePolicy)
            else str(self.missing_attribute_policy)
        )
        d["policy_type"] = (
            self.policy_type.value
            if isinstance(self.policy_type, PolicyType)
            else str(self.policy_type)
        )
        return d


@dataclass
class PolicyEvaluationResult:
    """
    Outcome of evaluating a model against a specific FairnessPolicy.
    Clearly designated as POLICY_CLASSIFICATION, not objective ground truth.
    """
    policy_name: str
    decision: PolicyDecision
    is_useful: bool
    is_fair: bool
    is_degenerate: bool
    violations: List[str]
    satisfied_rules: List[str]
    observed_disparity: Optional[float] = None
    evidence_strength: str = "EVALUATED"
    evidence_state: EvidenceState = EvidenceState.INSUFFICIENT_EVIDENCE
    policy_id: str = "POL-STD-2026.1"
    policy_version: str = "1.0.0"
    policy_type: PolicyType = PolicyType.ANALYTICAL
    effective_date: str = "2026-09-08"
    description: str = ""
    abstentions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = (
        "LEGAL DISCLAIMER: Compliance with this technical fairness policy reflects satisfaction of internal "
        "mathematical criteria and DOES NOT constitute legal certification of statutory non-discrimination "
        "or regulatory compliance immunity under Title VII, ECOA, FHA, or EU AI Act."
    )
    audit_notes: str = ""
    evidence_trace: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    failed_conditions: List[str] = field(default_factory=list)
    passed_conditions: List[str] = field(default_factory=list)
    missing_evidence_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_type": (
                self.policy_type.value
                if isinstance(self.policy_type, PolicyType)
                else str(self.policy_type)
            ),
            "effective_date": self.effective_date,
            "description": self.description,
            "decision": self.decision.value,
            "evidence_state": (
                self.evidence_state.value
                if isinstance(self.evidence_state, EvidenceState)
                else str(self.evidence_state)
            ),
            "is_useful": self.is_useful,
            "is_fair": self.is_fair,
            "is_degenerate": self.is_degenerate,
            "violations": self.violations,
            "satisfied_rules": self.satisfied_rules,
            "observed_disparity": self.observed_disparity,
            "evidence_strength": self.evidence_strength,
            "abstentions": self.abstentions,
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
            "audit_notes": self.audit_notes,
            "evidence_trace": self.evidence_trace,
            "reason_codes": self.reason_codes,
            "failed_conditions": self.failed_conditions,
            "passed_conditions": self.passed_conditions,
            "missing_evidence_conditions": self.missing_evidence_conditions,
        }


@dataclass
class MetricResult:
    name: str
    value: float
    status: EvaluationStatus
    message: str = ""
    selection_rates: Dict[str, float] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.value, self.status, self.selection_rates))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class FairnessUncertaintyResult:
    """
    Quantifies within-test finite-sample statistical estimation uncertainty
    via non-parametric bootstrap confidence intervals.
    """
    metric_name: str = "metric"
    point_estimate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    ci_level: float = 0.95
    sample_size: int = 0
    subgroup_sizes: Optional[Dict[str, int]] = None
    estimation_method: str = "STRATIFIED_BOOTSTRAP"
    interpretation: str = ""
    requested_iterations: int = 0
    successful_iterations: int = 0
    discarded_iterations: int = 0
    discard_reasons: Dict[str, int] = field(default_factory=dict)
    support_status: str = "VALID"
    ratio_stability_status: str = "STABLE"
    resampling_unit: str = "ROW_BOOTSTRAP"
    cluster_key: Optional[str] = None
    confidence_method: str = "EMPIRICAL_PERCENTILE_BOOTSTRAP"
    discard_fraction: float = 0.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FairnessEvaluationSuite:
    dataset_name: str = "Dataset"
    model_name: str = "Model"
    sample_size: int = 0
    group_counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    overall_status: EvaluationStatus = EvaluationStatus.VALID
    fairness_gap: float = 0.0
    user_utility_score: float = 0.0
    accuracy_tradeoff_proxy: Optional[float] = None
    degeneracy_status: ModelDegeneracyStatus = ModelDegeneracyStatus.NORMAL
    degeneracy_message: str = "Acceptable prediction diversity."
    fairness_success_status: FairnessSuccessStatus = FairnessSuccessStatus.UNDEFINED
    confusion_matrix: Optional[Dict[str, int]] = None
    subgroup_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    policy_result: Optional[PolicyEvaluationResult] = None
    calibration_details: Optional[List[Dict[str, float]]] = None
    uncertainty_intervals: Optional[Dict[str, Dict[str, Any]]] = None
    intersectional_spec: Optional[Dict[str, Any]] = None
    policy_sensitivity: Optional[Dict[str, Dict[str, Any]]] = None
    evidence_state: EvidenceState = EvidenceState.INSUFFICIENT_EVIDENCE
    abstentions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    number_of_comparisons: int = 1
    analysis_type: Union[str, AnalysisType] = "SCREENING_ANALYSIS"
    worst_group_selection_disclosure: Optional[Dict[str, Any]] = None
    manifest_fingerprint: Optional[str] = None
    missing_attribute_handling: Optional[Dict[str, Any]] = None
    calibration_by_subgroup: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    excluded_missing_count: int = 0
    selection_bias_disclosure: Optional[str] = None
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    demographic_parity_difference: float = 0.0
    equalized_odds_difference: float = 0.0
    disparate_impact_ratio: float = 1.0
    uncertainty: Optional[Any] = None
    evaluation_status: Optional[EvaluationStatus] = None
    evaluation_unit_level: Union[EvaluationUnitLevel, str] = EvaluationUnitLevel.ROW_LEVEL
    evaluation_unit_key: Optional[str] = None
    aggregation_strategy: Union[AggregationStrategy, str] = AggregationStrategy.NONE
    target_aggregation_strategy: Optional[Union[AggregationStrategy, str]] = None
    prediction_aggregation_strategy: Optional[Union[AggregationStrategy, str]] = None
    observational_record_unit: Optional[str] = None
    cluster_dependence_unit: Optional[str] = None
    statistical_estimand: Optional[Dict[str, Any]] = None
    reporting_privacy_mode: Union[ReportingPrivacyMode, str] = ReportingPrivacyMode.FULL_INTERNAL
    privacy_suppression_metadata: Optional[Dict[str, Any]] = None
    sample_weight: Optional[Any] = None
    taxonomy_state: Optional[Dict[str, Any]] = None
    human_review_boundary: Optional[Dict[str, Any]] = None


    def __post_init__(self):
        if self.evaluation_status is not None:
            self.overall_status = self.evaluation_status
        else:
            self.evaluation_status = self.overall_status

        if self.balanced_accuracy != 0.0 and "Balanced Accuracy" not in self.metrics:
            self.metrics["Balanced Accuracy"] = MetricResult("Balanced Accuracy", self.balanced_accuracy, self.overall_status)
        if self.equalized_odds_difference != 0.0 and "Equalized Odds Diff" not in self.metrics:
            self.metrics["Equalized Odds Diff"] = MetricResult("Equalized Odds Diff", self.equalized_odds_difference, self.overall_status)
        if self.demographic_parity_difference != 0.0 and "Demographic Parity Diff" not in self.metrics:
            self.metrics["Demographic Parity Diff"] = MetricResult("Demographic Parity Diff", self.demographic_parity_difference, self.overall_status)
        if self.disparate_impact_ratio != 1.0 and "Disparate Impact Ratio" not in self.metrics:
            self.metrics["Disparate Impact Ratio"] = MetricResult("Disparate Impact Ratio", self.disparate_impact_ratio, self.overall_status)
        if self.accuracy != 0.0 and "Accuracy" not in self.metrics:
            self.metrics["Accuracy"] = MetricResult("Accuracy", self.accuracy, self.overall_status)
        if self.uncertainty is not None and not self.uncertainty_intervals:
            self.uncertainty_intervals = {"Equalized Odds Diff": self.uncertainty}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "sample_size": self.sample_size,
            "group_counts": self.group_counts,
            "overall_status": self.overall_status.value,
            "evidence_state": (
                self.evidence_state.value
                if isinstance(self.evidence_state, EvidenceState)
                else str(self.evidence_state)
            ),
            "evaluation_unit_level": (
                self.evaluation_unit_level.value
                if isinstance(self.evaluation_unit_level, EvaluationUnitLevel)
                else str(self.evaluation_unit_level)
            ),
            "evaluation_unit_key": self.evaluation_unit_key,
            "aggregation_strategy": (
                self.aggregation_strategy.value
                if isinstance(self.aggregation_strategy, AggregationStrategy)
                else str(self.aggregation_strategy)
            ),
            "reporting_privacy_mode": (
                self.reporting_privacy_mode.value
                if isinstance(self.reporting_privacy_mode, ReportingPrivacyMode)
                else str(self.reporting_privacy_mode)
            ),
            "privacy_suppression_metadata": self.privacy_suppression_metadata,
            "fairness_gap": self.fairness_gap,
            "user_utility_score": self.user_utility_score,
            "accuracy_tradeoff_proxy": self.accuracy_tradeoff_proxy,
            "degeneracy_status": self.degeneracy_status.value,
            "degeneracy_message": self.degeneracy_message,
            "fairness_success_status": self.fairness_success_status.value,
            "policy_result": self.policy_result.to_dict() if self.policy_result else None,
            "confusion_matrix": self.confusion_matrix,
            "subgroup_metrics": self.subgroup_metrics,
            "calibration_details": self.calibration_details,
            "uncertainty_intervals": self.uncertainty_intervals,
            "intersectional_spec": self.intersectional_spec,
            "policy_sensitivity": self.policy_sensitivity,
            "abstentions": self.abstentions,
            "warnings": self.warnings,
            "number_of_comparisons": self.number_of_comparisons,
            "analysis_type": self.analysis_type,
            "worst_group_selection_disclosure": self.worst_group_selection_disclosure,
            "manifest_fingerprint": self.manifest_fingerprint,
            "missing_attribute_handling": self.missing_attribute_handling,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for name, res in self.metrics.items():
            val_str = f"{res.value:.4f}" if not np.isnan(res.value) else "NaN"
            rows.append({
                "Metric": name,
                "Value": val_str,
                "Status": res.status.value,
                "Note": res.message,
            })
        return pd.DataFrame(rows)


def govern_evaluation_reporting(
    suite: FairnessEvaluationSuite,
    privacy_mode: Union[ReportingPrivacyMode, str] = ReportingPrivacyMode.FULL_INTERNAL,
    min_reporting_population: int = 10,
    max_disclosed_groups: Optional[int] = None,
    mask_labels: bool = False,
) -> FairnessEvaluationSuite:
    """
    Applies privacy-aware output governance to prevent entity re-identification
    and sparse subgroup data leakage in external reporting.
    
    Modes:
    - FULL_INTERNAL: Complete granular metrics, unredacted group labels, and exact counts.
    - REDACTED_EXTERNAL: Cell suppression of subgroups with N < min_reporting_population,
      optional group label masking, and cap on disclosed groups.
    - SUMMARY_ONLY: All granular subgroup breakdowns suppressed; only global metrics reported.
    
    CRITICAL METHODOLOGICAL NOTICE:
    This constitutes report privacy (heuristic cell suppression and label masking),
    NOT formal differential privacy.
    """
    mode = privacy_mode if isinstance(privacy_mode, ReportingPrivacyMode) else ReportingPrivacyMode(privacy_mode)
    suite.reporting_privacy_mode = mode

    if mode == ReportingPrivacyMode.FULL_INTERNAL:
        suite.privacy_suppression_metadata = {
            "privacy_mode": mode.value,
            "suppression_applied": False,
            "disclosed_groups_count": len(suite.subgroup_metrics or {}),
            "disclaimer": "Full internal audit report. Zero privacy suppression applied.",
        }
        return suite

    if mode == ReportingPrivacyMode.SUMMARY_ONLY:
        orig_count = len(suite.subgroup_metrics or {})
        suite.subgroup_metrics = {}
        suite.intersectional_spec = None
        suite.calibration_by_subgroup = {}
        suite.privacy_suppression_metadata = {
            "privacy_mode": mode.value,
            "suppression_applied": True,
            "suppressed_groups_count": orig_count,
            "disclosed_groups_count": 0,
            "suppression_reason": "Summary-only mode requested for public release; individual group cells suppressed.",
            "disclaimer": "REPORT PRIVACY NOTICE: Subgroup suppression applied via cell suppression heuristics. This is not differential privacy.",
        }
        suite.warnings.append("Privacy Governance: Subgroup breakdowns suppressed under SUMMARY_ONLY mode.")
        return suite

    # REDACTED_EXTERNAL mode
    orig_subgroups = suite.subgroup_metrics or {}
    governed_subgroups: Dict[str, Dict[str, Any]] = {}
    governed_cal: Dict[str, Dict[str, Any]] = {}
    suppressed_count = 0
    group_idx = 1

    sorted_groups = sorted(
        orig_subgroups.items(),
        key=lambda item: item[1].get("count", item[1].get("sample_size", 0)),
        reverse=True,
    )

    for g_name, g_info in sorted_groups:
        sz = g_info.get("count", g_info.get("sample_size", 0))
        if sz < min_reporting_population:
            suppressed_count += 1
            continue

        if max_disclosed_groups is not None and len(governed_subgroups) >= max_disclosed_groups:
            suppressed_count += 1
            continue

        display_name = f"REDACTED_GROUP_{group_idx:03d}" if mask_labels else g_name
        group_idx += 1
        governed_subgroups[display_name] = dict(g_info)
        if suite.calibration_by_subgroup and g_name in suite.calibration_by_subgroup:
            governed_cal[display_name] = dict(suite.calibration_by_subgroup[g_name])

    suite.subgroup_metrics = governed_subgroups
    if suite.calibration_by_subgroup:
        suite.calibration_by_subgroup = governed_cal

    suppression_applied = (suppressed_count > 0 or mask_labels)
    suite.privacy_suppression_metadata = {
        "privacy_mode": mode.value,
        "suppression_applied": suppression_applied,
        "min_reporting_population": min_reporting_population,
        "suppressed_groups_count": suppressed_count,
        "disclosed_groups_count": len(governed_subgroups),
        "labels_masked": mask_labels,
        "disclaimer": "REPORT PRIVACY NOTICE: Cell suppression and label masking applied to protect entity anonymity. This is not differential privacy.",
    }
    if suppression_applied:
        suite.warnings.append(
            f"Privacy Governance: {suppressed_count} subgroup(s) suppressed under REDACTED_EXTERNAL mode (threshold={min_reporting_population})."
        )
    return suite


def compute_group_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Returns (tn, fp, fn, tp) safely, supporting optional sample weights."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight)
    return float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0]), float(cm[1, 1])



def brier_score_loss_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier score loss: mean((y_prob - y_true)^2)."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    if len(y_t) == 0:
        return float("nan")
    return float(np.mean((y_p - y_t) ** 2))


def expected_calibration_error_safe(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Computes 10-bin Expected Calibration Error (ECE) safely.
    Returns (ece, bin_details).
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    n = len(y_t)
    if n == 0:
        return float("nan"), []

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    details = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_p >= low) & (y_p <= high)
        else:
            mask = (y_p >= low) & (y_p < high)

        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_t[mask]))
            bin_conf = float(np.mean(y_p[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / n) * gap
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            gap = 0.0

        details.append({
            "bin_low": float(low),
            "bin_high": float(high),
            "bin_size": bin_size,
            "bin_accuracy": bin_acc,
            "bin_confidence": bin_conf,
            "bin_gap": gap,
        })

    return float(ece), details


def safe_disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive: Optional[Union[pd.Series, np.ndarray]] = None,
    min_group_size: int = 1,
    min_group_positives: int = 0,
    min_group_negatives: int = 0,
    min_selection_rate_threshold: float = 1e-4,
    sample_weight: Optional[np.ndarray] = None,
    **kwargs,
) -> MetricResult:
    """
    Computes min(selection_rate) / max(selection_rate).
    Guards against zero-division, empty groups, near-zero denominators,
    and extreme ratio instability when selection rates are vanishingly small.
    Supports optional sample_weight for entity-weighted evaluations.
    """
    if sensitive is None and "sensitive_series" in kwargs:
        sensitive = kwargs["sensitive_series"]
    if sensitive is None:
        raise ValueError("Sensitive attribute must be provided.")

    s_arr = np.asarray(sensitive).ravel()
    w_arr = np.asarray(sample_weight).ravel() if sample_weight is not None else None
    groups, counts = np.unique(s_arr, return_counts=True)
    if len(groups) < 2:
        return MetricResult(
            name="Disparate Impact Ratio",
            value=float("nan"),
            status=EvaluationStatus.MISSING_SENSITIVE,
            message="Fewer than 2 distinct sensitive groups available.",
            selection_rates={},
        )

    if any(c < min_group_size for c in counts):
        return MetricResult(
            name="Disparate Impact Ratio",
            value=float("nan"),
            status=EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
            message=f"At least one sensitive group has fewer than {min_group_size} instances.",
            selection_rates={},
        )

    selection_rates: Dict[str, float] = {}
    for g in groups:
        mask = s_arr == g
        if w_arr is not None:
            gw = w_arr[mask]
            rate = float(np.average(y_pred[mask] == 1, weights=gw)) if np.sum(gw) > 0 else 0.0
        else:
            rate = float(np.mean(y_pred[mask] == 1))
        selection_rates[str(g)] = rate

    rates_list = list(selection_rates.values())
    max_rate = max(rates_list)
    min_rate = min(rates_list)

    if max_rate <= 0.0:
        return MetricResult(
            name="Disparate Impact Ratio",
            value=float("nan"),
            status=EvaluationStatus.ZERO_SELECTION_RATE,
            message=f"All groups have zero positive selections (rates: {selection_rates}; 0 / 0 undefined).",
            selection_rates=selection_rates,
        )

    if max_rate < min_selection_rate_threshold or min_rate < min_selection_rate_threshold:
        return MetricResult(
            name="Disparate Impact Ratio",
            value=float("nan"),
            status=EvaluationStatus.UNSTABLE_RATIO,
            message=(
                f"Group selection rate is near zero (< {min_selection_rate_threshold}). "
                f"Ratio is statistically unstable. Group selection rates: {selection_rates}."
            ),
            selection_rates=selection_rates,
        )

    ratio = float(min_rate / max_rate)
    if np.isinf(ratio) or np.isnan(ratio):
        return MetricResult(
            name="Disparate Impact Ratio",
            value=float("nan"),
            status=EvaluationStatus.UNSTABLE_RATIO,
            message=f"Ratio calculation resulted in non-finite value ({ratio}). Group rates: {selection_rates}.",
            selection_rates=selection_rates,
        )

    return MetricResult(
        name="Disparate Impact Ratio",
        value=ratio,
        status=EvaluationStatus.VALID,
        message=f"Valid ratio computed: {ratio:.4f} (group selection rates: {selection_rates}).",
        selection_rates=selection_rates,
    )


def safe_demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = 5,
    min_group_positives: int = 0,
    min_group_negatives: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> MetricResult:
    """
    Computes max(selection_rate) - min(selection_rate), supporting optional sample_weight.
    """
    groups, counts = np.unique(sensitive, return_counts=True)
    if len(groups) < 2:
        return MetricResult(
            name="Demographic Parity Diff",
            value=float("nan"),
            status=EvaluationStatus.MISSING_SENSITIVE,
            message="Fewer than 2 distinct sensitive groups.",
        )

    if any(c < min_group_size for c in counts):
        return MetricResult(
            name="Demographic Parity Diff",
            value=float("nan"),
            status=EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
            message=f"At least one sensitive group has fewer than {min_group_size} instances.",
        )

    w_arr = np.asarray(sample_weight).ravel() if sample_weight is not None else None
    rates = []
    for g in groups:
        mask = sensitive == g
        if w_arr is not None:
            gw = w_arr[mask]
            rate = float(np.average(y_pred[mask] == 1, weights=gw)) if np.sum(gw) > 0 else 0.0
        else:
            rate = float(np.mean(y_pred[mask] == 1))
        rates.append(rate)
    gap = float(max(rates) - min(rates))
    return MetricResult(
        name="Demographic Parity Diff",
        value=gap,
        status=EvaluationStatus.VALID,
        message="Valid absolute selection gap computed.",
    )


def safe_equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = 5,
    min_group_positives: int = 0,
    min_group_negatives: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> MetricResult:
    """
    Computes max(TPR) - min(TPR) across sensitive groups (Y=1 instances).
    """
    groups, counts = np.unique(sensitive, return_counts=True)
    if len(groups) < 2:
        return MetricResult(
            name="Equal Opportunity Diff",
            value=float("nan"),
            status=EvaluationStatus.MISSING_SENSITIVE,
            message="Fewer than 2 sensitive groups.",
        )

    if any(c < min_group_size for c in counts):
        return MetricResult(
            name="Equal Opportunity Diff",
            value=float("nan"),
            status=EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
            message=f"At least one sensitive group has fewer than {min_group_size} instances.",
        )

    w_arr = np.asarray(sample_weight).ravel() if sample_weight is not None else None
    tpr_values: List[float] = []
    for g in groups:
        mask = sensitive == g
        positives = y_true[mask] == 1
        n_pos = int(np.sum(positives))
        if n_pos == 0:
            return MetricResult(
                name="Equal Opportunity Diff",
                value=float("nan"),
                status=EvaluationStatus.ZERO_POSITIVE_SUBGROUP,
                message=f"Group '{g}' has zero positive ground-truth instances (TPR undefined).",
            )
        if min_group_positives > 0 and n_pos < min_group_positives:
            return MetricResult(
                name="Equal Opportunity Diff",
                value=float("nan"),
                status=EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
                message=f"Group '{g}' has insufficient positive support ({n_pos} < {min_group_positives}).",
            )
        if w_arr is not None:
            gw = w_arr[mask][positives]
            tpr = float(np.average(y_pred[mask][positives] == 1, weights=gw)) if np.sum(gw) > 0 else 0.0
        else:
            tpr = float(np.mean(y_pred[mask][positives] == 1))
        tpr_values.append(tpr)

    gap = float(max(tpr_values) - min(tpr_values))
    return MetricResult(
        name="Equal Opportunity Diff",
        value=gap,
        status=EvaluationStatus.VALID,
        message="Valid TPR gap computed.",
    )


def safe_equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = 5,
    min_group_positives: int = 0,
    min_group_negatives: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> MetricResult:
    """
    Computes max(TPR_gap, FPR_gap) across sensitive groups.
    """
    groups, counts = np.unique(sensitive, return_counts=True)
    if len(groups) < 2:
        return MetricResult(
            name="Equalized Odds Diff",
            value=float("nan"),
            status=EvaluationStatus.MISSING_SENSITIVE,
            message="Fewer than 2 sensitive groups.",
        )

    if any(c < min_group_size for c in counts):
        return MetricResult(
            name="Equalized Odds Diff",
            value=float("nan"),
            status=EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
            message=f"At least one sensitive group has fewer than {min_group_size} instances.",
        )

    w_arr = np.asarray(sample_weight).ravel() if sample_weight is not None else None
    tpr_values: List[float] = []
    fpr_values: List[float] = []

    for g in groups:
        mask = sensitive == g
        sub_w = w_arr[mask] if w_arr is not None else None
        tn, fp, fn, tp = compute_group_confusion(y_true[mask], y_pred[mask], sample_weight=sub_w)
        n_pos = tp + fn
        n_neg = tn + fp

        if n_pos == 0:
            return MetricResult(
                name="Equalized Odds Diff",
                value=float("nan"),
                status=EvaluationStatus.ZERO_POSITIVE_SUBGROUP,
                message=f"Group '{g}' has zero positive ground-truth instances (TPR undefined).",
            )
        if n_neg == 0:
            return MetricResult(
                name="Equalized Odds Diff",
                value=float("nan"),
                status=EvaluationStatus.ZERO_NEGATIVE_SUBGROUP,
                message=f"Group '{g}' has zero negative ground-truth instances (FPR undefined).",
            )
        if min_group_positives > 0 and n_pos < min_group_positives:
            return MetricResult(
                name="Equalized Odds Diff",
                value=float("nan"),
                status=EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
                message=f"Group '{g}' has insufficient positive support ({n_pos} < {min_group_positives}).",
            )
        if min_group_negatives > 0 and n_neg < min_group_negatives:
            return MetricResult(
                name="Equalized Odds Diff",
                value=float("nan"),
                status=EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
                message=f"Group '{g}' has insufficient negative support ({n_neg} < {min_group_negatives}).",
            )

        tpr = tp / n_pos
        fpr = fp / n_neg
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    tpr_gap = max(tpr_values) - min(tpr_values)
    fpr_gap = max(fpr_values) - min(fpr_values)
    return MetricResult(
        name="Equalized Odds Diff",
        value=float(max(tpr_gap, fpr_gap)),
        status=EvaluationStatus.VALID,
        message=f"Max of TPR gap ({tpr_gap:.4f}) and FPR gap ({fpr_gap:.4f}).",
    )


def safe_predictive_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = 5,
    min_group_positives: int = 0,
    min_group_negatives: int = 0,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
) -> MetricResult:
    """
    Computes max(PPV) - min(PPV) across groups.
    Guards against zero positive predictions in any group.
    Supports optional sample_weight for entity-weighted evaluation.
    """
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return MetricResult(
            name="Predictive Parity Diff",
            value=float("nan"),
            status=EvaluationStatus.MISSING_SENSITIVE,
            message="Fewer than 2 sensitive groups.",
        )

    ppv_values: List[float] = []
    sw = np.asarray(sample_weight).ravel() if sample_weight is not None else None
    for g in groups:
        mask = sensitive == g
        pred_pos = y_pred[mask] == 1
        n_pred_pos = int(np.sum(pred_pos))
        if n_pred_pos == 0:
            return MetricResult(
                name="Predictive Parity Diff",
                value=float("nan"),
                status=EvaluationStatus.UNDEFINED,
                message=f"Group '{g}' has zero predicted positives (PPV undefined).",
            )
        if sw is not None:
            sw_sub = sw[mask][pred_pos]
            if np.sum(sw_sub) > 0:
                ppv = float(np.average(y_true[mask][pred_pos] == 1, weights=sw_sub))
            else:
                ppv = float("nan")
        else:
            ppv = float(np.mean(y_true[mask][pred_pos] == 1))
        ppv_values.append(ppv)

    gap = float(max(ppv_values) - min(ppv_values))
    return MetricResult(
        name="Predictive Parity Diff",
        value=gap,
        status=EvaluationStatus.VALID,
        message="Valid PPV gap computed.",
    )


def user_defined_utility_score(
    balanced_accuracy: float,
    fairness_gap: float,
    alpha: float = 0.25,
) -> float:
    gap = fairness_gap if not np.isnan(fairness_gap) else 0.5
    return float(balanced_accuracy - alpha * gap)


def accuracy_tradeoff_proxy(
    baseline_accuracy: float,
    mitigated_accuracy: float,
) -> float:
    return float(max(0.0, baseline_accuracy - mitigated_accuracy))


def detect_prediction_collapse(
    y_pred: Union[pd.Series, np.ndarray],
    y_true: Optional[Union[pd.Series, np.ndarray]] = None,
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    epsilon_low: float = 0.02,
    epsilon_high: float = 0.98,
    min_entropy: float = 0.10,
) -> Tuple[ModelDegeneracyStatus, str]:
    y_p = np.asarray(y_pred).ravel()
    unique_preds, pred_counts = np.unique(y_p, return_counts=True)
    if len(unique_preds) <= 1:
        c = unique_preds[0] if len(unique_preds) == 1 else "None"
        return ModelDegeneracyStatus.DEGENERATE, f"Trivial constant predictor: model predicts strictly Class {c}."

    pos_rate = float(np.mean(y_p == 1))
    if pos_rate <= epsilon_low:
        return ModelDegeneracyStatus.DEGENERATE, f"Prediction collapse: Low positive prediction rate ({pos_rate:.4f} <= {epsilon_low})."
    if pos_rate >= epsilon_high:
        return ModelDegeneracyStatus.DEGENERATE, f"Prediction collapse: High positive prediction rate ({pos_rate:.4f} >= {epsilon_high})."

    p0 = 1.0 - pos_rate
    p1 = pos_rate
    entropy = - (p0 * math.log2(p0) + p1 * math.log2(p1)) if 0.0 < pos_rate < 1.0 else 0.0
    if entropy < min_entropy:
        return ModelDegeneracyStatus.DEGENERATE, f"Prediction collapse: Low prediction entropy ({entropy:.4f} < {min_entropy:.4f})."

    if y_true is not None:
        y_t = np.asarray(y_true).ravel()
        if len(np.unique(y_t)) >= 2:
            bal_acc = float(balanced_accuracy_score(y_t, y_p))
            if abs(bal_acc - 0.50) < 0.015:
                majority_class = int(np.argmax(np.bincount(y_t)))
                pred_majority_fraction = float(np.mean(y_p == majority_class))
                if pred_majority_fraction > 0.85:
                    return ModelDegeneracyStatus.INSUFFICIENT_SIGNAL, (
                        f"Model achieves chance-level balanced accuracy ({bal_acc:.4f}) with high majority prediction ({pred_majority_fraction:.4f})."
                    )

    return ModelDegeneracyStatus.NORMAL, "Acceptable prediction diversity."


class FairnessUncertaintyDict(dict):
    @property
    def requested_iterations(self) -> int:
        first = next(iter(self.values()), None)
        return getattr(first, "requested_iterations", 0) if first else 0

    @property
    def successful_iterations(self) -> int:
        first = next(iter(self.values()), None)
        return getattr(first, "successful_iterations", 0) if first else 0

    @property
    def discarded_iterations(self) -> int:
        first = next(iter(self.values()), None)
        return getattr(first, "discarded_iterations", 0) if first else 0

    @property
    def discard_fraction(self) -> float:
        first = next(iter(self.values()), None)
        return getattr(first, "discard_fraction", 0.0) if first else 0.0

    @property
    def discard_reasons(self) -> Dict[str, int]:
        first = next(iter(self.values()), None)
        return getattr(first, "discard_reasons", {}) if first else {}

    @property
    def support_status(self) -> str:
        first = next(iter(self.values()), None)
        return getattr(first, "support_status", "VALID") if first else "VALID"

    @property
    def resampling_unit(self) -> str:
        first = next(iter(self.values()), None)
        return getattr(first, "resampling_unit", "ROW_BOOTSTRAP") if first else "ROW_BOOTSTRAP"

    @property
    def cluster_key(self) -> Optional[str]:
        first = next(iter(self.values()), None)
        return getattr(first, "cluster_key", None) if first else None

    @property
    def confidence_method(self) -> str:
        first = next(iter(self.values()), None)
        return getattr(first, "confidence_method", "EMPIRICAL_PERCENTILE_BOOTSTRAP") if first else "EMPIRICAL_PERCENTILE_BOOTSTRAP"

    @property
    def seed(self) -> int:
        first = next(iter(self.values()), None)
        return getattr(first, "seed", 42) if first else 42


def compute_fairness_uncertainty_bootstrap(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    sensitive: Optional[Union[pd.Series, np.ndarray]] = None,
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
    seed: int = 42,
    min_group_size: int = 5,
    min_valid_bootstrap_fraction: float = 0.80,
    resampling_unit: Union[ResamplingUnit, str] = ResamplingUnit.ROW_BOOTSTRAP,
    cluster_series: Optional[Union[pd.Series, np.ndarray]] = None,
    cluster_key: Optional[str] = None,
    confidence_method: str = "EMPIRICAL_PERCENTILE_BOOTSTRAP",
    **kwargs,
) -> FairnessUncertaintyDict:
    """
    Estimates within-test finite-sample statistical uncertainty via non-parametric
    bootstrap (either stratified row-level or entity-cluster bootstrap).
    Tracks requested, successful, and discarded iterations with explicit discard reasons.
    When observations within clusters are correlated (e.g. repeated trips per driver),
    CLUSTER_BOOTSTRAP resamples entire entity clusters, preventing artificial variance shrinkage.
    If valid replicates fall below min_valid_bootstrap_fraction, returns
    INSUFFICIENT_BOOTSTRAP_SUPPORT rather than ungrounded confidence intervals.
    """
    unit_str = (
        resampling_unit.value
        if isinstance(resampling_unit, ResamplingUnit)
        else str(resampling_unit).upper()
    )
    is_cluster = unit_str in ("CLUSTER", "CLUSTER_BOOTSTRAP")

    if sensitive is None and "sensitive_series" in kwargs:
        sensitive = kwargs["sensitive_series"]
    if "n_bootstrap" in kwargs:
        n_bootstraps = kwargs["n_bootstrap"]
    if "random_seed" in kwargs:
        seed = kwargs["random_seed"]
    if cluster_series is None and "clusters" in kwargs:
        cluster_series = kwargs["clusters"]
    if cluster_series is None and "cluster_col" in kwargs:
        cluster_series = kwargs["cluster_col"]
    if sensitive is None:
        raise ValueError("Sensitive attribute must be provided.")

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    s = np.asarray(sensitive).ravel()
    n = len(y_t)
    rng = np.random.RandomState(seed)

    if is_cluster:
        if cluster_series is None:
            raise ValueError(
                "Cluster bootstrap requested (resampling_unit='CLUSTER_BOOTSTRAP') but no cluster_series was provided. "
                "Silent fallback to row-level bootstrap is strictly prohibited."
            )
        clusters_arr = np.asarray(cluster_series).ravel()
        if len(clusters_arr) != n:
            raise ValueError(f"cluster_series length ({len(clusters_arr)}) does not match sample size ({n}).")
        unique_clusters = np.unique(clusters_arr)
        n_clusters = len(unique_clusters)
        if n_clusters < 5:
            # Insufficient clusters for cluster bootstrap
            pass
        cluster_to_indices = {c: np.where(clusters_arr == c)[0] for c in unique_clusters}

    groups, counts = np.unique(s, return_counts=True)
    group_sizes = {str(g): int(c) for g, c in zip(groups, counts)}

    # Point estimates
    pt_eo = safe_equalized_odds_difference(y_t, y_p, s, min_group_size).value
    pt_dp = safe_demographic_parity_difference(y_p, s, min_group_size).value
    pt_di = safe_disparate_impact_ratio(y_p, s, min_group_size).value

    eo_boots: List[float] = []
    dp_boots: List[float] = []
    di_boots: List[float] = []
    subgroup_tpr_boots: Dict[str, List[float]] = {str(g): [] for g in groups}
    subgroup_sel_boots: Dict[str, List[float]] = {str(g): [] for g in groups}

    # Tracking discard reasons
    discard_reasons_map: Dict[str, Dict[str, int]] = {
        "Equalized Odds Diff": {},
        "Demographic Parity Diff": {},
        "Disparate Impact Ratio": {},
    }
    for g in groups:
        discard_reasons_map[f"Selection Rate ({g})"] = {}
        discard_reasons_map[f"TPR ({g})"] = {}

    # Stratified bootstrap indices by group (for row mode)
    group_indices = {g: np.where(s == g)[0] for g in groups}

    for _ in range(n_bootstraps):
        if is_cluster:
            sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
            boot_idx = np.concatenate([cluster_to_indices[c] for c in sampled_c])
        else:
            boot_idx_list = []
            for g, idxs in group_indices.items():
                if len(idxs) > 0:
                    sampled = rng.choice(idxs, size=len(idxs), replace=True)
                    boot_idx_list.append(sampled)
            boot_idx = np.concatenate(boot_idx_list)

        b_yt = y_t[boot_idx]
        b_yp = y_p[boot_idx]
        b_s = s[boot_idx]

        # Equalized Odds Difference
        b_eo = safe_equalized_odds_difference(b_yt, b_yp, b_s, min_group_size=2)
        if b_eo.status == EvaluationStatus.VALID and not np.isnan(b_eo.value):
            eo_boots.append(b_eo.value)
        else:
            reason = b_eo.status.value if b_eo.status != EvaluationStatus.VALID else "NAN_VALUE"
            discard_reasons_map["Equalized Odds Diff"][reason] = (
                discard_reasons_map["Equalized Odds Diff"].get(reason, 0) + 1
            )

        # Demographic Parity Difference
        b_dp = safe_demographic_parity_difference(b_yp, b_s, min_group_size=2)
        if b_dp.status == EvaluationStatus.VALID and not np.isnan(b_dp.value):
            dp_boots.append(b_dp.value)
        else:
            reason = b_dp.status.value if b_dp.status != EvaluationStatus.VALID else "NAN_VALUE"
            discard_reasons_map["Demographic Parity Diff"][reason] = (
                discard_reasons_map["Demographic Parity Diff"].get(reason, 0) + 1
            )

        # Disparate Impact Ratio (with explicit ratio stability controls)
        b_di = safe_disparate_impact_ratio(b_yp, b_s, min_group_size=2)
        if (
            b_di.status == EvaluationStatus.VALID
            and not np.isnan(b_di.value)
            and not np.isinf(b_di.value)
            and b_di.value <= 100.0
        ):
            di_boots.append(b_di.value)
        else:
            reason = b_di.status.value if b_di.status != EvaluationStatus.VALID else "UNSTABLE_OR_EXTREME_RATIO"
            discard_reasons_map["Disparate Impact Ratio"][reason] = (
                discard_reasons_map["Disparate Impact Ratio"].get(reason, 0) + 1
            )

        # Subgroup Conditional Rates
        for g in groups:
            mask = b_s == g
            str_g = str(g)
            if np.sum(mask) > 0:
                sel_rate = float(np.mean(b_yp[mask] == 1))
                subgroup_sel_boots[str_g].append(sel_rate)
            else:
                discard_reasons_map[f"Selection Rate ({str_g})"]["EMPTY_GROUP_IN_SAMPLE"] = (
                    discard_reasons_map[f"Selection Rate ({str_g})"].get("EMPTY_GROUP_IN_SAMPLE", 0) + 1
                )

            pos_mask = (b_s == g) & (b_yt == 1)
            if np.sum(pos_mask) > 0:
                tpr_val = float(np.mean(b_yp[pos_mask] == 1))
                subgroup_tpr_boots[str_g].append(tpr_val)
            else:
                discard_reasons_map[f"TPR ({str_g})"]["ZERO_POSITIVES_IN_SAMPLE"] = (
                    discard_reasons_map[f"TPR ({str_g})"].get("ZERO_POSITIVES_IN_SAMPLE", 0) + 1
                )

    alpha_tail = (1.0 - ci_level) / 2.0
    results: Dict[str, FairnessUncertaintyResult] = {}

    def get_ci(boots: List[float], pt: float, name: str) -> FairnessUncertaintyResult:
        successful_cnt = len(boots)
        discarded_cnt = n_bootstraps - successful_cnt
        valid_fraction = successful_cnt / max(n_bootstraps, 1)
        discard_frac = float(discarded_cnt / max(n_bootstraps, 1))
        reasons = discard_reasons_map.get(name, {})

        if is_cluster and n_clusters < 5:
            support_status = EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT.value
            low, high = float("nan"), float("nan")
            interp = f"INSUFFICIENT_BOOTSTRAP_SUPPORT: Fewer than 5 unique clusters ({n_clusters} found). Cluster bootstrap requires >= 5 clusters."
        elif valid_fraction < min_valid_bootstrap_fraction:
            support_status = EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT.value
            low, high = float("nan"), float("nan")
            interp = (
                f"INSUFFICIENT_BOOTSTRAP_SUPPORT: valid replicates ({successful_cnt}/{n_bootstraps}, "
                f"{valid_fraction:.1%}) below required {min_valid_bootstrap_fraction:.1%} threshold. "
                f"Discard reasons: {reasons}."
            )
        elif successful_cnt >= 10:
            support_status = EvaluationStatus.VALID.value
            low = float(np.percentile(boots, 100.0 * alpha_tail))
            high = float(np.percentile(boots, 100.0 * (1.0 - alpha_tail)))
            interp = f"{int(ci_level*100)}% Bootstrap CI: [{low:.4f}, {high:.4f}] ({successful_cnt}/{n_bootstraps} valid)."
        else:
            support_status = EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT.value
            low, high = float("nan"), float("nan")
            interp = f"Insufficient valid bootstrap samples ({successful_cnt} < 10)."

        est_method = "CLUSTER_BOOTSTRAP" if is_cluster else "STRATIFIED_BOOTSTRAP"

        return FairnessUncertaintyResult(
            metric_name=name,
            point_estimate=pt,
            ci_low=low,
            ci_high=high,
            ci_level=ci_level,
            sample_size=n,
            subgroup_sizes=group_sizes,
            estimation_method=est_method,
            interpretation=interp,
            requested_iterations=n_bootstraps,
            successful_iterations=successful_cnt,
            discarded_iterations=discarded_cnt,
            discard_reasons=reasons,
            support_status=support_status,
            ratio_stability_status="STABLE" if support_status == EvaluationStatus.VALID.value else "UNSTABLE_OR_INSUFFICIENT",
            resampling_unit=unit_str,
            cluster_key=cluster_key,
            confidence_method=confidence_method,
            discard_fraction=discard_frac,
            seed=seed,
        )

    results["Equalized Odds Diff"] = get_ci(eo_boots, pt_eo, "Equalized Odds Diff")
    results["Demographic Parity Diff"] = get_ci(dp_boots, pt_dp, "Demographic Parity Diff")
    results["Disparate Impact Ratio"] = get_ci(di_boots, pt_di, "Disparate Impact Ratio")

    for str_g in subgroup_sel_boots:
        pt_sel = float(np.mean(y_p[s == str_g] == 1)) if np.sum(s == str_g) > 0 else float("nan")
        results[f"Selection Rate ({str_g})"] = get_ci(
            subgroup_sel_boots[str_g], pt_sel, f"Selection Rate ({str_g})"
        )

    for str_g in subgroup_tpr_boots:
        pos_m = (s == str_g) & (y_t == 1)
        pt_tpr = float(np.mean(y_p[pos_m] == 1)) if np.sum(pos_m) > 0 else float("nan")
        results[f"TPR ({str_g})"] = get_ci(
            subgroup_tpr_boots[str_g], pt_tpr, f"TPR ({str_g})"
        )

    return FairnessUncertaintyDict(results)


def evaluate_policy_compliance(
    suite: FairnessEvaluationSuite,
    policy: FairnessPolicy,
    majority_baseline_bal_acc: Optional[float] = None,
    uncertainty_result: Optional[Dict[str, FairnessUncertaintyResult]] = None,
) -> PolicyEvaluationResult:
    """
    Evaluates policy compliance with anti-degeneracy precedence, subgroup utility,
    subgroup support gating, bootstrap validity, ratio stability, and strict tri-state
    abstention precedence (PASS / FAIL / INSUFFICIENT_EVIDENCE).
    Emits an auditable evidence trace with exact numeric observations, thresholds,
    condition outcomes, and machine-readable reason codes.
    """
    violations: List[str] = []
    satisfied_rules: List[str] = []
    abstentions: List[str] = []
    warnings_list: List[str] = []
    evidence_trace: List[Dict[str, Any]] = []
    reason_codes: List[str] = []
    failed_conditions: List[str] = []
    passed_conditions: List[str] = []
    missing_evidence_conditions: List[str] = []

    is_degenerate = (
        suite.degeneracy_status != ModelDegeneracyStatus.NORMAL
        or suite.degeneracy_status in (
            ModelDegeneracyStatus.DEGENERATE,
            ModelDegeneracyStatus.COLLAPSED_CONSTANT,
            ModelDegeneracyStatus.COLLAPSED_SINGLE_CLASS,
        )
    )

    # Precedence Step 1: Model Degeneracy
    if is_degenerate:
        msg = f"Model degeneracy detected: {suite.degeneracy_message}"
        violations.append(msg)
        failed_conditions.append(msg)
        reason_codes.append("MODEL_DEGENERATE")
        evidence_trace.append({
            "rule": "anti_degeneracy",
            "metric": "model_degeneracy_status",
            "observed": suite.degeneracy_status.value,
            "condition_status": "FAILED",
            "reason_code": "MODEL_DEGENERATE",
            "details": suite.degeneracy_message,
        })
    else:
        passed_conditions.append("Model is non-degenerate.")
        evidence_trace.append({
            "rule": "anti_degeneracy",
            "metric": "model_degeneracy_status",
            "observed": suite.degeneracy_status.value,
            "condition_status": "PASSED",
            "reason_code": "MODEL_NORMAL",
        })

    # Precedence Step 2: Evaluation Status & Missing Attribute Audit
    if suite.overall_status in (EvaluationStatus.MISSING_ATTRIBUTE_VIOLATION, EvaluationStatus.MISSING_SENSITIVE):
        msg = "Evaluation refused: sensitive attribute missing/unknown values violated policy."
        abstentions.append(msg)
        missing_evidence_conditions.append(msg)
        reason_codes.append("MISSING_SENSITIVE_ATTRIBUTE")
        evidence_trace.append({
            "rule": "missing_sensitive_attributes",
            "metric": "overall_status",
            "observed": suite.overall_status.value,
            "condition_status": "INSUFFICIENT_EVIDENCE",
            "reason_code": "MISSING_SENSITIVE_ATTRIBUTE",
        })
    elif suite.overall_status in (
        EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT,
        EvaluationStatus.UNRELIABLE_ESTIMATE,
        EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT,
        EvaluationStatus.INSUFFICIENT_CALIBRATION_SUPPORT,
        EvaluationStatus.UNDEFINED,
    ):
        msg = f"Evaluation evidence insufficient due to status: {suite.overall_status.value}"
        abstentions.append(msg)
        missing_evidence_conditions.append(msg)
        reason_codes.append("STATUS_INSUFFICIENT_EVIDENCE")
        evidence_trace.append({
            "rule": "evaluation_status_integrity",
            "metric": "overall_status",
            "observed": suite.overall_status.value,
            "condition_status": "INSUFFICIENT_EVIDENCE",
            "reason_code": "STATUS_INSUFFICIENT_EVIDENCE",
        })
    else:
        passed_conditions.append("Evaluation status and missing attribute checks passed.")
        evidence_trace.append({
            "rule": "evaluation_status_integrity",
            "metric": "overall_status",
            "observed": suite.overall_status.value,
            "condition_status": "PASSED",
            "reason_code": "STATUS_VALID",
        })

    # Precedence Step 3: Critical Subgroup Support & Disparity Gating
    if policy.critical_subgroups and suite.subgroup_metrics:
        for c_grp in policy.critical_subgroups:
            if c_grp not in suite.subgroup_metrics:
                msg = f"Required critical subgroup '{c_grp}' not observed in evaluation data."
                abstentions.append(msg)
                missing_evidence_conditions.append(msg)
                reason_codes.append("CRITICAL_SUBGROUP_UNOBSERVED")
                evidence_trace.append({
                    "rule": "critical_subgroup_support",
                    "subgroup": c_grp,
                    "condition_status": "INSUFFICIENT_EVIDENCE",
                    "reason_code": "CRITICAL_SUBGROUP_UNOBSERVED",
                })
            else:
                c_data = suite.subgroup_metrics[c_grp]
                c_status = c_data.get("support_status")
                if c_status and c_status != EvaluationStatus.VALID.value:
                    msg = f"Required critical subgroup '{c_grp}' failed sample support requirements ({c_status})."
                    abstentions.append(msg)
                    missing_evidence_conditions.append(msg)
                    reason_codes.append("CRITICAL_SUBGROUP_INSUFFICIENT_SUPPORT")
                    evidence_trace.append({
                        "rule": "critical_subgroup_support",
                        "subgroup": c_grp,
                        "support_status": c_status,
                        "condition_status": "INSUFFICIENT_EVIDENCE",
                        "reason_code": "CRITICAL_SUBGROUP_INSUFFICIENT_SUPPORT",
                    })
                # Check subgroup disparity
                tpr = c_data.get("tpr") if c_data.get("tpr") is not None else c_data.get("TPR")
                if tpr is not None and policy.max_equalized_odds_diff is not None:
                    all_tprs = [
                        (g.get("tpr") if g.get("tpr") is not None else g.get("TPR"))
                        for g in suite.subgroup_metrics.values()
                        if (g.get("tpr") is not None or g.get("TPR") is not None)
                    ]
                    if all_tprs and (max(all_tprs) - tpr) > policy.max_equalized_odds_diff:
                        disp_val = max(all_tprs) - tpr
                        msg = f"Critical subgroup '{c_grp}' TPR disparity ({disp_val:.4f}) exceeds policy maximum ({policy.max_equalized_odds_diff:.4f})"
                        violations.append(msg)
                        failed_conditions.append(msg)
                        reason_codes.append("CRITICAL_SUBGROUP_DISPARITY_VIOLATION")
                        evidence_trace.append({
                            "rule": "critical_subgroup_disparity",
                            "subgroup": c_grp,
                            "observed_disparity": disp_val,
                            "policy_threshold": policy.max_equalized_odds_diff,
                            "condition_status": "FAILED",
                            "reason_code": "CRITICAL_SUBGROUP_DISPARITY_VIOLATION",
                        })

    if policy.require_adequate_subgroup_support and suite.subgroup_metrics:
        for g_name, g_info in suite.subgroup_metrics.items():
            if g_info.get("support_status") == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value:
                msg = f"Subgroup '{g_name}' has insufficient sample support under policy requirement."
                abstentions.append(msg)
                missing_evidence_conditions.append(msg)
                reason_codes.append("SUBGROUP_INSUFFICIENT_SUPPORT")

    # Precedence Step 4: Bootstrap Uncertainty Support Gating
    unc_data = uncertainty_result
    if unc_data is None and suite.uncertainty_intervals:
        unc_data = {
            k: FairnessUncertaintyResult(**v) if isinstance(v, dict) else v
            for k, v in suite.uncertainty_intervals.items()
        }

    if unc_data:
        for m_name in ["Equalized Odds Diff", "Demographic Parity Diff", "Disparate Impact Ratio"]:
            if m_name in unc_data:
                m_unc = unc_data[m_name]
                m_stat = getattr(m_unc, "support_status", "")
                if m_stat in (
                    EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT.value,
                    EvaluationStatus.INSUFFICIENT_BOOTSTRAP_SUPPORT,
                    "INSUFFICIENT_BOOTSTRAP_SUPPORT",
                    "INSUFFICIENT_SUPPORT",
                ):
                    msg = f"Bootstrap replicate support failed for '{m_name}': {getattr(m_unc, 'interpretation', '')}"
                    abstentions.append(msg)
                    missing_evidence_conditions.append(msg)
                    reason_codes.append("INSUFFICIENT_BOOTSTRAP_SUPPORT")
                    evidence_trace.append({
                        "rule": "bootstrap_uncertainty_support",
                        "metric": m_name,
                        "support_status": m_stat,
                        "condition_status": "INSUFFICIENT_EVIDENCE",
                        "reason_code": "INSUFFICIENT_BOOTSTRAP_SUPPORT",
                        "details": getattr(m_unc, "interpretation", ""),
                    })

    # Precedence Step 5: Ratio Metric Instability
    di_metric = suite.metrics.get("Disparate Impact Ratio")
    if di_metric and di_metric.status == EvaluationStatus.UNSTABLE_RATIO:
        msg = f"Disparate Impact Ratio is statistically unstable near zero: {di_metric.message}"
        abstentions.append(msg)
        missing_evidence_conditions.append(msg)
        reason_codes.append("UNSTABLE_RATIO_DENOMINATOR")
        evidence_trace.append({
            "rule": "ratio_metric_stability",
            "metric": "Disparate Impact Ratio",
            "observed": di_metric.value,
            "condition_status": "INSUFFICIENT_EVIDENCE",
            "reason_code": "UNSTABLE_RATIO_DENOMINATOR",
            "details": di_metric.message,
        })

    # Global Utility checks
    bal_acc = suite.metrics.get("Balanced Accuracy")
    bal_acc_val = bal_acc.value if (bal_acc and bal_acc.status == EvaluationStatus.VALID) else 0.0
    if bal_acc_val < policy.min_balanced_accuracy:
        msg = f"Balanced Accuracy ({bal_acc_val:.4f}) < policy minimum ({policy.min_balanced_accuracy:.4f})"
        violations.append(msg)
        failed_conditions.append(msg)
        reason_codes.append("BALANCED_ACCURACY_BELOW_MINIMUM")
        evidence_trace.append({
            "metric": "Balanced Accuracy",
            "observed": bal_acc_val,
            "support_status": "VALID" if bal_acc else "MISSING",
            "policy_threshold": policy.min_balanced_accuracy,
            "operator": ">=",
            "condition_status": "FAILED",
            "reason_code": "BALANCED_ACCURACY_BELOW_MINIMUM",
        })
    else:
        sat = f"Balanced Accuracy ({bal_acc_val:.4f}) >= {policy.min_balanced_accuracy:.4f}"
        satisfied_rules.append(sat)
        passed_conditions.append(sat)
        evidence_trace.append({
            "metric": "Balanced Accuracy",
            "observed": bal_acc_val,
            "support_status": "VALID" if bal_acc else "MISSING",
            "policy_threshold": policy.min_balanced_accuracy,
            "operator": ">=",
            "condition_status": "PASSED",
            "reason_code": "BALANCED_ACCURACY_SATISFIED",
        })

    if policy.require_better_than_majority_baseline and majority_baseline_bal_acc is not None:
        if bal_acc_val <= majority_baseline_bal_acc:
            msg = f"Balanced Accuracy ({bal_acc_val:.4f}) did not exceed majority baseline ({majority_baseline_bal_acc:.4f})"
            violations.append(msg)
            failed_conditions.append(msg)
            reason_codes.append("BELOW_MAJORITY_BASELINE")
        else:
            sat = f"Balanced Accuracy ({bal_acc_val:.4f}) exceeds majority baseline ({majority_baseline_bal_acc:.4f})"
            satisfied_rules.append(sat)
            passed_conditions.append(sat)

    rec = suite.metrics.get("Recall")
    if rec and policy.min_recall is not None:
        if rec.status == EvaluationStatus.VALID and rec.value < policy.min_recall:
            msg = f"Recall ({rec.value:.4f}) < policy minimum ({policy.min_recall:.4f})"
            violations.append(msg)
            failed_conditions.append(msg)
            reason_codes.append("RECALL_BELOW_MINIMUM")
        elif rec.status == EvaluationStatus.VALID:
            satisfied_rules.append(f"Recall ({rec.value:.4f}) >= {policy.min_recall:.4f}")
            passed_conditions.append(f"Recall ({rec.value:.4f}) >= {policy.min_recall:.4f}")

    tnr = suite.metrics.get("TNR")
    if tnr and policy.min_specificity is not None:
        if tnr.status == EvaluationStatus.VALID and tnr.value < policy.min_specificity:
            msg = f"Specificity ({tnr.value:.4f}) < policy minimum ({policy.min_specificity:.4f})"
            violations.append(msg)
            failed_conditions.append(msg)
            reason_codes.append("SPECIFICITY_BELOW_MINIMUM")
        elif tnr.status == EvaluationStatus.VALID:
            satisfied_rules.append(f"Specificity ({tnr.value:.4f}) >= {policy.min_specificity:.4f}")
            passed_conditions.append(f"Specificity ({tnr.value:.4f}) >= {policy.min_specificity:.4f}")

    # Subgroup Utility checks
    if suite.subgroup_metrics:
        for g_name, g_info in suite.subgroup_metrics.items():
            if policy.min_subgroup_balanced_accuracy is not None:
                sub_bal = g_info.get("balanced_accuracy", float("nan"))
                if not np.isnan(sub_bal) and sub_bal < policy.min_subgroup_balanced_accuracy:
                    msg = f"Subgroup '{g_name}' Balanced Accuracy ({sub_bal:.4f}) < policy minimum ({policy.min_subgroup_balanced_accuracy:.4f})"
                    violations.append(msg)
                    failed_conditions.append(msg)
                    reason_codes.append("SUBGROUP_BALANCED_ACCURACY_BELOW_MINIMUM")

            if policy.min_subgroup_recall is not None:
                sub_tpr = g_info.get("TPR", float("nan"))
                if not np.isnan(sub_tpr) and sub_tpr < policy.min_subgroup_recall:
                    msg = f"Subgroup '{g_name}' TPR ({sub_tpr:.4f}) < policy minimum ({policy.min_subgroup_recall:.4f})"
                    violations.append(msg)
                    failed_conditions.append(msg)
                    reason_codes.append("SUBGROUP_RECALL_BELOW_MINIMUM")

            if policy.min_subgroup_specificity is not None:
                sub_tnr = g_info.get("TNR", float("nan"))
                if not np.isnan(sub_tnr) and sub_tnr < policy.min_subgroup_specificity:
                    msg = f"Subgroup '{g_name}' TNR ({sub_tnr:.4f}) < policy minimum ({policy.min_subgroup_specificity:.4f})"
                    violations.append(msg)
                    failed_conditions.append(msg)
                    reason_codes.append("SUBGROUP_SPECIFICITY_BELOW_MINIMUM")

    # Fairness disparity checks
    fairness_violations: List[str] = []
    observed_disp = None

    # Equalized Odds Difference
    eo = suite.metrics.get("Equalized Odds Diff")
    eo_unc = unc_data.get("Equalized Odds Diff") if unc_data else None
    eo_ci = [eo_unc.ci_low, eo_unc.ci_high] if eo_unc else None
    if eo and policy.max_equalized_odds_diff is not None:
        observed_disp = eo.value
        if eo.status != EvaluationStatus.VALID:
            violations.append(f"Equalized Odds Diff is undefined ({eo.status.value})")
            fairness_violations.append("Equalized Odds Diff undefined")
            reason_codes.append("EO_DIFF_UNDEFINED")
            evidence_trace.append({
                "metric": "Equalized Odds Diff",
                "observed": eo.value,
                "uncertainty": eo_ci,
                "support_status": eo.status.value,
                "policy_threshold": policy.max_equalized_odds_diff,
                "operator": "<=",
                "condition_status": "INSUFFICIENT_EVIDENCE",
                "reason_code": "EO_DIFF_UNDEFINED",
            })
        elif eo.value > policy.max_equalized_odds_diff:
            msg = f"Equalized Odds Diff ({eo.value:.4f}) > policy maximum ({policy.max_equalized_odds_diff:.4f})"
            violations.append(msg)
            fairness_violations.append(msg)
            failed_conditions.append(msg)
            reason_codes.append("EO_DIFF_THRESHOLD_EXCEEDED")
            evidence_trace.append({
                "metric": "Equalized Odds Diff",
                "observed": eo.value,
                "uncertainty": eo_ci,
                "support_status": eo.status.value,
                "policy_threshold": policy.max_equalized_odds_diff,
                "operator": "<=",
                "condition_status": "FAILED",
                "reason_code": "EO_DIFF_THRESHOLD_EXCEEDED",
            })
        else:
            sat = f"Equalized Odds Diff ({eo.value:.4f}) <= {policy.max_equalized_odds_diff:.4f}"
            satisfied_rules.append(sat)
            passed_conditions.append(sat)
            evidence_trace.append({
                "metric": "Equalized Odds Diff",
                "observed": eo.value,
                "uncertainty": eo_ci,
                "support_status": eo.status.value,
                "policy_threshold": policy.max_equalized_odds_diff,
                "operator": "<=",
                "condition_status": "PASSED",
                "reason_code": "EO_DIFF_SATISFIED",
            })

    # Demographic Parity Difference
    dp = suite.metrics.get("Demographic Parity Diff")
    dp_unc = unc_data.get("Demographic Parity Diff") if unc_data else None
    dp_ci = [dp_unc.ci_low, dp_unc.ci_high] if dp_unc else None
    if dp and policy.max_demographic_parity_diff is not None:
        if dp.status != EvaluationStatus.VALID:
            violations.append(f"Demographic Parity Diff is undefined ({dp.status.value})")
            fairness_violations.append("Demographic Parity Diff undefined")
            reason_codes.append("DP_DIFF_UNDEFINED")
            evidence_trace.append({
                "metric": "Demographic Parity Diff",
                "observed": dp.value,
                "uncertainty": dp_ci,
                "support_status": dp.status.value,
                "policy_threshold": policy.max_demographic_parity_diff,
                "operator": "<=",
                "condition_status": "INSUFFICIENT_EVIDENCE",
                "reason_code": "DP_DIFF_UNDEFINED",
            })
        elif dp.value > policy.max_demographic_parity_diff:
            msg = f"Demographic Parity Diff ({dp.value:.4f}) > policy maximum ({policy.max_demographic_parity_diff:.4f})"
            violations.append(msg)
            fairness_violations.append(msg)
            failed_conditions.append(msg)
            reason_codes.append("DP_DIFF_THRESHOLD_EXCEEDED")
            evidence_trace.append({
                "metric": "Demographic Parity Diff",
                "observed": dp.value,
                "uncertainty": dp_ci,
                "support_status": dp.status.value,
                "policy_threshold": policy.max_demographic_parity_diff,
                "operator": "<=",
                "condition_status": "FAILED",
                "reason_code": "DP_DIFF_THRESHOLD_EXCEEDED",
            })
        else:
            sat = f"Demographic Parity Diff ({dp.value:.4f}) <= {policy.max_demographic_parity_diff:.4f}"
            satisfied_rules.append(sat)
            passed_conditions.append(sat)
            evidence_trace.append({
                "metric": "Demographic Parity Diff",
                "observed": dp.value,
                "uncertainty": dp_ci,
                "support_status": dp.status.value,
                "policy_threshold": policy.max_demographic_parity_diff,
                "operator": "<=",
                "condition_status": "PASSED",
                "reason_code": "DP_DIFF_SATISFIED",
            })

    # Disparate Impact Ratio
    di = suite.metrics.get("Disparate Impact Ratio")
    di_unc = unc_data.get("Disparate Impact Ratio") if unc_data else None
    di_ci = [di_unc.ci_low, di_unc.ci_high] if di_unc else None
    if di and policy.min_disparate_impact_ratio is not None:
        if di.status == EvaluationStatus.VALID and not np.isnan(di.value):
            if di.value < policy.min_disparate_impact_ratio:
                msg = f"Disparate Impact Ratio ({di.value:.4f}) < policy minimum ({policy.min_disparate_impact_ratio:.4f})"
                violations.append(msg)
                fairness_violations.append(msg)
                failed_conditions.append(msg)
                reason_codes.append("DIR_THRESHOLD_VIOLATION")
                evidence_trace.append({
                    "metric": "Disparate Impact Ratio",
                    "observed": di.value,
                    "uncertainty": di_ci,
                    "support_status": di.status.value,
                    "policy_threshold": policy.min_disparate_impact_ratio,
                    "operator": ">=",
                    "condition_status": "FAILED",
                    "reason_code": "DIR_THRESHOLD_VIOLATION",
                })
            else:
                sat = f"Disparate Impact Ratio ({di.value:.4f}) >= {policy.min_disparate_impact_ratio:.4f}"
                satisfied_rules.append(sat)
                passed_conditions.append(sat)
                evidence_trace.append({
                    "metric": "Disparate Impact Ratio",
                    "observed": di.value,
                    "uncertainty": di_ci,
                    "support_status": di.status.value,
                    "policy_threshold": policy.min_disparate_impact_ratio,
                    "operator": ">=",
                    "condition_status": "PASSED",
                    "reason_code": "DIR_SATISFIED",
                })
        elif di.status == EvaluationStatus.UNSTABLE_RATIO:
            violations.append("Disparate Impact Ratio is unstable near zero.")
            fairness_violations.append("Disparate Impact Ratio unstable near zero.")
            reason_codes.append("DIR_UNSTABLE_RATIO")

    if not policy.policy_version or str(policy.policy_version).strip() in ("", "None"):
        msg = "Policy version must be explicitly identified for audit certification."
        abstentions.append(msg)
        missing_evidence_conditions.append(msg)
        reason_codes.append("GATE_UNIDENTIFIED_POLICY_VERSION")

    if suite.statistical_estimand:
        est_stat = suite.statistical_estimand.get("estimand_status") if isinstance(suite.statistical_estimand, dict) else getattr(suite.statistical_estimand, "estimand_status", "")
        if est_stat == "ESTIMAND_UNSPECIFIED":
            msg = "Evaluation estimand is marked ESTIMAND_UNSPECIFIED. Policy certification blocked."
            abstentions.append(msg)
            missing_evidence_conditions.append(msg)
            reason_codes.append("GATE_ESTIMAND_UNSPECIFIED")

    is_useful = (not is_degenerate) and (bal_acc_val >= policy.min_balanced_accuracy)
    is_fair = (len(fairness_violations) == 0)

    # FINAL PRECEDENCE DECISION
    if is_degenerate:
        decision = PolicyDecision.POLICY_DEGENERATE
        evidence_state = EvidenceState.FAIL
        evidence_strength = "DEGENERATE_PREDICTOR"
        if "MODEL_DEGENERATE" not in reason_codes:
            reason_codes.insert(0, "MODEL_DEGENERATE")
    elif len(abstentions) > 0:
        decision = PolicyDecision.POLICY_UNDEFINED
        evidence_state = EvidenceState.INSUFFICIENT_EVIDENCE
        evidence_strength = "INSUFFICIENT_EVIDENCE"
        if not any("INSUFFICIENT" in r or "MISSING" in r or "UNSTABLE" in r for r in reason_codes):
            reason_codes.insert(0, "INSUFFICIENT_EVIDENCE_ABSTENTION")
    elif len(violations) > 0:
        decision = PolicyDecision.POLICY_NONCOMPLIANT
        evidence_state = EvidenceState.FAIL
        evidence_strength = "POLICY_CONSTRAINTS_VIOLATED"
        if not any("THRESHOLD" in r or "VIOLATION" in r or "BELOW" in r for r in reason_codes):
            reason_codes.insert(0, "POLICY_THRESHOLD_VIOLATION")
    else:
        decision = PolicyDecision.POLICY_COMPLIANT
        evidence_state = EvidenceState.PASS
        evidence_strength = "CONFIRMED_ADEQUATE"
        reason_codes.append("ALL_POLICY_RULES_SATISFIED")

    return PolicyEvaluationResult(
        policy_name=policy.name,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_type=policy.policy_type,
        effective_date=policy.effective_date,
        description=policy.description,
        decision=decision,
        evidence_state=evidence_state,
        is_useful=is_useful,
        is_fair=is_fair,
        is_degenerate=is_degenerate,
        violations=violations,
        satisfied_rules=satisfied_rules,
        observed_disparity=observed_disp,
        evidence_strength=evidence_strength,
        abstentions=abstentions,
        warnings=warnings_list,
        disclaimer=policy.disclaimer,
        audit_notes=f"Evaluated against '{policy.name}' (ID: {policy.policy_id} v{policy.policy_version}). Decision: {decision.value} | Evidence State: {evidence_state.value}.",
        evidence_trace=evidence_trace,
        reason_codes=reason_codes,
        failed_conditions=failed_conditions,
        passed_conditions=passed_conditions,
        missing_evidence_conditions=missing_evidence_conditions,
    )


def verify_policy_decision_gate(
    suite: FairnessEvaluationSuite,
    policy: FairnessPolicy,
    statistical_estimand: Optional[Dict[str, Any]] = None,
    evaluation_unit_valid: bool = True,
    split_valid: bool = True,
    leakage_findings_count: int = 0,
    invariants_valid: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """
    Evaluates the Policy Decision Gate (pre-decision evaluability and admissibility check)
    before confirmatory policy evaluation.
    Enforces 9 mandatory prerequisites:
      1. Valid estimand (not missing, not ESTIMAND_UNSPECIFIED)
      2. Valid evaluation unit (evaluation_unit_valid is True)
      3. Valid split (split_valid is True)
      4. No critical leakage (leakage_findings_count == 0)
      5. Adequate subgroup support (no unhandled sparse groups)
      6. Valid uncertainty (bootstrap fraction >= 0.80 if uncertainty requested)
      7. Valid audit invariants (invariants_valid is True)
      8. Complete evidence trace (non-empty trace and reason codes)
      9. Policy version identified (non-empty policy_version string)

    Returns:
        (gate_passed: bool, gate_violations: List[str], reason_codes: List[str])
    """
    gate_violations: List[str] = []
    reason_codes: List[str] = []

    # 1. Valid Estimand
    est = statistical_estimand or suite.statistical_estimand
    if est is None:
        gate_violations.append("Policy decision gate: Missing evaluation estimand (ESTIMAND_UNSPECIFIED).")
        reason_codes.append("GATE_MISSING_ESTIMAND")
    elif isinstance(est, dict) and est.get("estimand_status") == "ESTIMAND_UNSPECIFIED":
        gate_violations.append("Policy decision gate: Evaluation estimand is ESTIMAND_UNSPECIFIED.")
        reason_codes.append("GATE_ESTIMAND_UNSPECIFIED")

    # 2. Valid Evaluation Unit
    if not evaluation_unit_valid:
        gate_violations.append("Policy decision gate: Evaluation unit configuration is invalid or unverified.")
        reason_codes.append("GATE_INVALID_EVALUATION_UNIT")

    # 3. Valid Split
    if not split_valid:
        gate_violations.append("Policy decision gate: Holdout split protocol is invalid or contaminated.")
        reason_codes.append("GATE_INVALID_SPLIT")

    # 4. No Critical Leakage
    if leakage_findings_count > 0:
        gate_violations.append(f"Policy decision gate: {leakage_findings_count} critical feature leakage findings detected.")
        reason_codes.append("GATE_LEAKAGE_DETECTED")

    # 5. Adequate Subgroup Support
    if suite.subgroup_metrics:
        for g_name, g_data in suite.subgroup_metrics.items():
            if g_data.get("support_status") == EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value:
                gate_violations.append(f"Policy decision gate: Subgroup '{g_name}' has insufficient sample support.")
                reason_codes.append("GATE_INSUFFICIENT_SUBGROUP_SUPPORT")

    # 6. Valid Uncertainty
    if suite.uncertainty_intervals:
        for m_name, u_res in suite.uncertainty_intervals.items():
            valid_frac = u_res.get("valid_fraction") if isinstance(u_res, dict) else getattr(u_res, "valid_fraction", 1.0)
            if valid_frac is not None and valid_frac < 0.80:
                gate_violations.append(f"Policy decision gate: Metric '{m_name}' valid bootstrap fraction ({valid_frac:.2f}) < 0.80.")
                reason_codes.append("GATE_INVALID_UNCERTAINTY")

    # 7. Valid Audit Invariants
    if not invariants_valid:
        gate_violations.append("Policy decision gate: Mathematical audit invariants validation failed.")
        reason_codes.append("GATE_INVARIANT_VIOLATION")

    # 8. Complete Evidence Trace
    if suite.policy_result and not suite.policy_result.evidence_trace:
        gate_violations.append("Policy decision gate: Missing or empty policy evidence trace.")
        reason_codes.append("GATE_INCOMPLETE_EVIDENCE_TRACE")

    # 9. Policy Version Identified
    if not policy.policy_version or str(policy.policy_version).strip() in ("", "None"):
        gate_violations.append("Policy decision gate: Policy version is unidentified.")
        reason_codes.append("GATE_UNIDENTIFIED_POLICY_VERSION")

    return (len(gate_violations) == 0, gate_violations, reason_codes)


def verify_policy_certification_gate(
    *args, **kwargs
) -> Tuple[bool, List[str], List[str]]:
    """
    DEPRECATED: verify_policy_certification_gate is deprecated and retained for backward compatibility.
    Use verify_policy_decision_gate instead.
    """
    warnings.warn(
        "verify_policy_certification_gate is deprecated; use verify_policy_decision_gate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return verify_policy_decision_gate(*args, **kwargs)


def evaluate_policy_sensitivity(
    suite: FairnessEvaluationSuite,
    policies: Optional[Dict[str, FairnessPolicy]] = None,
    majority_baseline_bal_acc: Optional[float] = None,
    uncertainty_result: Optional[Dict[str, FairnessUncertaintyResult]] = None,
) -> Dict[str, PolicyEvaluationResult]:
    """
    Evaluates a single model against multiple alternative fairness policies.
    """
    if policies is None:
        policies = {
            "Standard_Operational_Policy": FairnessPolicy(
                name="Standard_Operational_Policy",
                policy_id="POL-STD-2026.1",
                policy_version="1.0.0",
                min_balanced_accuracy=0.55,
                max_equalized_odds_diff=0.10,
                min_disparate_impact_ratio=0.80,
            ),
            "Strict_Equalized_Odds_Policy": FairnessPolicy(
                name="Strict_Equalized_Odds_Policy",
                policy_id="POL-STRICT-EO-2026.1",
                policy_version="1.0.0",
                min_balanced_accuracy=0.60,
                max_equalized_odds_diff=0.05,
                min_disparate_impact_ratio=0.85,
                min_recall=0.50,
            ),
            "Lenient_Diagnostic_Policy": FairnessPolicy(
                name="Lenient_Diagnostic_Policy",
                policy_id="POL-LENIENT-2026.1",
                policy_version="1.0.0",
                min_balanced_accuracy=0.52,
                max_equalized_odds_diff=0.15,
                min_disparate_impact_ratio=0.70,
            ),
        }

    results: Dict[str, PolicyEvaluationResult] = {}
    for name, pol in policies.items():
        res = evaluate_policy_compliance(
            suite,
            policy=pol,
            majority_baseline_bal_acc=majority_baseline_bal_acc,
            uncertainty_result=uncertainty_result,
        )
        results[name] = res
    return results


def classify_fairness_success(
    suite: FairnessEvaluationSuite,
    min_balanced_acc: float = 0.55,
    max_eo_diff: float = 0.10,
    max_dp_diff: float = 0.10,
) -> FairnessSuccessStatus:
    """Preserves backward compatibility with legacy calls."""
    if suite.overall_status != EvaluationStatus.VALID:
        return FairnessSuccessStatus.UNDEFINED

    bal_acc = suite.metrics["Balanced Accuracy"].value
    eo_res = suite.metrics.get("Equalized Odds Diff")
    dp_res = suite.metrics.get("Demographic Parity Diff")

    eo_val = eo_res.value if (eo_res and eo_res.status == EvaluationStatus.VALID) else 0.0
    dp_val = dp_res.value if (dp_res and dp_res.status == EvaluationStatus.VALID) else 0.0

    is_fair = (eo_val <= max_eo_diff) and (dp_val <= max_dp_diff)
    is_useful = (suite.degeneracy_status == ModelDegeneracyStatus.NORMAL) and (bal_acc >= min_balanced_acc)

    if suite.degeneracy_status == ModelDegeneracyStatus.DEGENERATE:
        return FairnessSuccessStatus.FAIR_BUT_DEGENERATE if is_fair else FairnessSuccessStatus.UNFAIR_AND_DEGENERATE

    if is_useful and is_fair:
        return FairnessSuccessStatus.FAIR_AND_USEFUL
    elif is_useful and not is_fair:
        return FairnessSuccessStatus.UNFAIR_AND_USEFUL
    elif not is_useful and is_fair:
        return FairnessSuccessStatus.FAIR_BUT_DEGENERATE
    else:
        return FairnessSuccessStatus.UNFAIR_AND_DEGENERATE


def evaluate_fairness_defensively(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    sensitive: Optional[Union[pd.Series, np.ndarray, pd.DataFrame]] = None,
    dataset_name: str = "Dataset",
    model_name: str = "Model",
    baseline_accuracy: Optional[float] = None,
    alpha_utility: float = 0.25,
    min_group_size: int = 1,
    min_group_positives: int = 0,
    min_group_negatives: int = 0,
    min_balanced_acc: float = 0.55,
    max_eo_diff: float = 0.10,
    policy: Optional[FairnessPolicy] = None,
    majority_baseline_bal_acc: Optional[float] = None,
    compute_uncertainty: bool = False,
    n_bootstraps: int = 200,
    intersectional_spec: Optional[IntersectionalGroupSpec] = None,
    missing_attribute_policy: Optional[MissingAttributePolicy] = None,
    analysis_type: AnalysisType = AnalysisType.SCREENING_ANALYSIS,
    min_valid_bootstrap_fraction: Optional[float] = None,
    min_calibration_support: Optional[int] = None,
    resampling_unit: Union[ResamplingUnit, str] = ResamplingUnit.ROW_BOOTSTRAP,
    cluster_series: Optional[Union[pd.Series, np.ndarray]] = None,
    cluster_key: Optional[str] = None,
    evaluation_unit_level: Union[EvaluationUnitLevel, str] = EvaluationUnitLevel.ROW_LEVEL,
    evaluation_unit_key: Optional[str] = None,
    entity_series: Optional[Union[pd.Series, np.ndarray]] = None,
    aggregation_strategy: Union[AggregationStrategy, str] = AggregationStrategy.NONE,
    reporting_privacy_mode: Union[ReportingPrivacyMode, str] = ReportingPrivacyMode.FULL_INTERNAL,
    min_reporting_population: int = 10,
    max_disclosed_groups: Optional[int] = None,
    mask_privacy_labels: bool = False,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    allow_row_bootstrap_override: bool = False,
    target_aggregation_strategy: Optional[Union[AggregationStrategy, str]] = None,
    prediction_aggregation_strategy: Optional[Union[AggregationStrategy, str]] = None,
    statistical_estimand: Optional[Dict[str, Any]] = None,
    decision_gate_required: Optional[bool] = None,
    certification_gate_required: Optional[bool] = None,
    evaluation_unit_valid: bool = True,
    split_valid: bool = True,
    leakage_findings_count: int = 0,
    invariants_valid: bool = True,
    **kwargs,
) -> FairnessEvaluationSuite:
    """
    Main defensive fairness evaluation entrypoint.
    Computes performance, calibration, full confusion matrix, per-group conditional metrics,
    prediction collapse checks, optional bootstrap uncertainty intervals, missing attribute governance,
    multiple-comparison accounting, worst-group selection disclosure, defensive policy compliance,
    evaluation unit aggregation / inverse-frequency weighting, and privacy-aware report governance.
    """
    if sensitive is None and "sensitive_features" in kwargs:
        sensitive = kwargs["sensitive_features"]
    if cluster_series is None and "clusters" in kwargs:
        cluster_series = kwargs["clusters"]
    if cluster_series is None and "cluster_col" in kwargs:
        cluster_series = kwargs["cluster_col"]
    if entity_series is None and "entity_id" in kwargs:
        entity_series = kwargs["entity_id"]
    if entity_series is None and "entity_col" in kwargs:
        entity_series = kwargs["entity_col"]
    if entity_series is None and "evaluation_unit_series" in kwargs:
        entity_series = kwargs["evaluation_unit_series"]
    if sample_weight is None and "weights" in kwargs:
        sample_weight = kwargs["weights"]

    if isinstance(evaluation_unit_level, str):
        evaluation_unit_level = EvaluationUnitLevel(evaluation_unit_level)
    if isinstance(aggregation_strategy, str):
        aggregation_strategy = AggregationStrategy(aggregation_strategy)
    if isinstance(reporting_privacy_mode, str):
        reporting_privacy_mode = ReportingPrivacyMode(reporting_privacy_mode)
    if isinstance(resampling_unit, str):
        resampling_unit = ResamplingUnit(resampling_unit)

    warnings_list: List[str] = []
    abstentions_list: List[str] = []

    y_t_raw = np.asarray(y_true)
    y_p_raw = np.asarray(y_pred)

    if isinstance(sensitive, pd.DataFrame):
        if sensitive.shape[1] == 1:
            s_raw = sensitive.iloc[:, 0].to_numpy()
        else:
            s_raw = sensitive.astype(str).agg("_x_".join, axis=1).to_numpy()
    elif isinstance(sensitive, (pd.Series, list, np.ndarray)):
        s_raw = np.asarray(sensitive).ravel()
    elif sensitive is not None:
        s_raw = np.asarray(sensitive).ravel()
    else:
        raise ValueError("Sensitive attribute must be provided.")

    if len(y_t_raw) != len(y_p_raw) or len(y_t_raw) != len(s_raw):
        raise ValueError(
            f"Dimension mismatch: y_true ({len(y_t_raw)}), y_pred ({len(y_p_raw)}), sensitive ({len(s_raw)})"
        )

    # Statistical Unit Mismatch Guard:
    # If evaluating at entity level (TRIP_LEVEL, ROUTE_LEVEL, WORKER_LEVEL, ENTITY_LEVEL) with repeated rows
    # and ROW_BOOTSTRAP is requested without aggregation or clustering, raise ValueError unless overridden.
    if entity_series is not None and aggregation_strategy == AggregationStrategy.NONE:
        entity_arr = np.asarray(entity_series)
        if len(entity_arr) > len(np.unique(entity_arr)):
            is_entity_level = evaluation_unit_level not in (
                EvaluationUnitLevel.ROW_LEVEL,
                EvaluationUnitLevel.INDIVIDUAL_LEVEL,
            )
            if is_entity_level and compute_uncertainty and resampling_unit == ResamplingUnit.ROW_BOOTSTRAP:
                if not allow_row_bootstrap_override:
                    raise ValueError(
                        f"Statistical unit mismatch: Evaluation unit is {evaluation_unit_level.value} with repeated entity rows, "
                        f"but row-level bootstrap is requested without entity clustering or aggregation. "
                        f"This violates i.i.d. sampling assumptions. "
                        f"Either aggregate rows to {evaluation_unit_level.value} using aggregation_strategy, "
                        f"use cluster bootstrap with resampling_unit=ResamplingUnit.CLUSTER_BOOTSTRAP, "
                        f"or explicitly pass allow_row_bootstrap_override=True to acknowledge the limitation."
                    )
                else:
                    msg = (
                        f"Statistical unit mismatch override: Row bootstrap used on {evaluation_unit_level.value} "
                        f"with repeated entity rows. Confidence intervals assume row-level conditional independence."
                    )
                    warnings.warn(msg)
                    warnings_list.append(msg)

    # Apply Aggregation or Entity Weighting if requested
    has_custom_agg = (target_aggregation_strategy is not None or prediction_aggregation_strategy is not None)
    if entity_series is not None and (aggregation_strategy != AggregationStrategy.NONE or has_custom_agg):
        if aggregation_strategy == AggregationStrategy.ENTITY_WEIGHTED and not has_custom_agg:
            sample_weight = compute_entity_weights(entity_series, base_weights=sample_weight)
        else:
            agg_yt, agg_yp, agg_s, agg_prob, agg_cluster = aggregate_to_evaluation_unit(
                y_true=y_t_raw,
                y_pred=y_p_raw,
                sensitive=s_raw,
                entity_series=entity_series,
                y_prob=y_prob,
                cluster_series=cluster_series,
                strategy=aggregation_strategy,
                target_strategy=target_aggregation_strategy,
                prediction_strategy=prediction_aggregation_strategy,
            )
            y_t_raw = agg_yt
            y_p_raw = agg_yp
            s_raw = agg_s
            y_prob = agg_prob
            cluster_series = agg_cluster
            if sample_weight is not None:
                warnings.warn("Custom sample_weight provided alongside record collapse. Resetting row weights for aggregated entity records.")
                sample_weight = None

    try:
        y_t_float = y_t_raw.astype(float)
        if np.isnan(y_t_float).any() or np.isinf(y_t_float).any():
            raise ValueError("y_true contains NaN or Inf values.")
    except ValueError as e:
        if "contains NaN" in str(e):
            raise

    try:
        y_p_float = y_p_raw.astype(float)
        if np.isnan(y_p_float).any() or np.isinf(y_p_float).any():
            raise ValueError("y_pred contains NaN or Inf values.")
    except ValueError as e:
        if "contains NaN" in str(e):
            raise

    y_t = y_t_raw.astype(int).ravel()
    y_p = y_p_raw.astype(int).ravel()
    y_prob_arr = np.asarray(y_prob).ravel() if y_prob is not None else None
    sw = np.asarray(sample_weight).ravel() if sample_weight is not None else None
    if sw is not None and len(sw) != len(y_t):
        raise ValueError(f"sample_weight length ({len(sw)}) does not match sample size ({len(y_t)}).")

    # Determine active governance policies
    active_policy = policy if policy is not None else FairnessPolicy(
        name="Default_Standard_Policy",
        min_balanced_accuracy=min_balanced_acc,
        max_equalized_odds_diff=max_eo_diff,
        max_demographic_parity_diff=max_eo_diff,
    )
    active_missing_policy = (
        missing_attribute_policy
        if missing_attribute_policy is not None
        else active_policy.missing_attribute_policy
    )
    cal_support_min = (
        min_calibration_support
        if min_calibration_support is not None
        else active_policy.min_calibration_support
    )
    bootstrap_valid_frac = (
        min_valid_bootstrap_fraction
        if min_valid_bootstrap_fraction is not None
        else active_policy.min_valid_bootstrap_fraction
    )

    # Missing & Unknown Sensitive Attribute Governance (Phase 9)
    s_series = pd.Series(s_raw)
    is_null = s_series.isna()
    is_unknown_str = s_series.astype(str).str.strip().str.upper().isin(
        {"", "UNKNOWN", "NULL", "NONE", "OTHER", "MALFORMED", "MISSING", "NAN"}
    )
    missing_mask = (is_null | is_unknown_str).to_numpy()
    missing_count = int(np.sum(missing_mask))
    missing_fraction = float(missing_count / max(len(s_raw), 1))

    missing_handling_info: Dict[str, Any] = {
        "policy": (
            active_missing_policy.value
            if isinstance(active_missing_policy, MissingAttributePolicy)
            else str(active_missing_policy)
        ),
        "missing_count": missing_count,
        "missing_fraction": missing_fraction,
        "action_taken": "NONE",
    }

    initial_overall_status = EvaluationStatus.VALID

    s = s_raw.copy()
    if missing_count > 0:
        if active_missing_policy == MissingAttributePolicy.REJECT:
            initial_overall_status = EvaluationStatus.MISSING_ATTRIBUTE_VIOLATION
            msg = (
                f"Missing/unknown sensitive attributes detected in {missing_count} rows ({missing_fraction:.2%}). "
                f"Policy REJECT triggered."
            )
            abstentions_list.append(msg)
            warnings_list.append(msg)
            missing_handling_info["action_taken"] = "REJECTED_EVALUATION"
            s = pd.Series(s_raw).fillna("UNKNOWN").astype(str).to_numpy()
        elif active_missing_policy == MissingAttributePolicy.EXCLUDE:
            keep_mask = ~missing_mask
            y_t = y_t[keep_mask]
            y_p = y_p[keep_mask]
            s = pd.Series(s_raw[keep_mask]).astype(str).to_numpy()
            if y_prob_arr is not None:
                y_prob_arr = y_prob_arr[keep_mask]
            if cluster_series is not None:
                cluster_series = np.asarray(cluster_series)[keep_mask]
            if sw is not None:
                sw = sw[keep_mask]
            msg = (
                f"Excluded {missing_count} rows ({missing_fraction:.2%}) with missing sensitive attributes "
                f"under EXCLUDE policy."
            )
            warnings_list.append(msg)
            missing_handling_info["action_taken"] = "EXCLUDED_FROM_METRICS"
        elif active_missing_policy == MissingAttributePolicy.EXPLICIT_UNKNOWN_GROUP:
            s_clean = pd.Series(s_raw).fillna("UNKNOWN").astype(str).to_numpy()
            s_clean[missing_mask] = "UNKNOWN"
            s = s_clean
            msg = (
                f"Mapped {missing_count} rows ({missing_fraction:.2%}) with missing sensitive attributes "
                f"to explicit group 'UNKNOWN'."
            )
            warnings_list.append(msg)
            missing_handling_info["action_taken"] = "MAPPED_TO_EXPLICIT_UNKNOWN_GROUP"
        elif active_missing_policy == MissingAttributePolicy.IMPUTE_WITH_DOCUMENTED_RULE:
            rule_id = getattr(active_policy, "imputation_rule_id", "RULE-PROXY-DEFAULT") or "RULE-PROXY-DEFAULT"
            rule_desc = getattr(active_policy, "imputation_rule_description", "Default documented proxy imputation rule") or "Default proxy rule"
            imputed_count = missing_count
            s_clean = pd.Series(s_raw).fillna("DERIVED_PROXY_IMPUTED").astype(str).to_numpy()
            s = s_clean
            msg = (
                f"Imputed {imputed_count} records using rule '{rule_id}' ({rule_desc}). "
                f"Sensitive attribute classified as DERIVED_PROXY."
            )
            warnings_list.append(msg)
            missing_handling_info["action_taken"] = "IMPUTED_WITH_DOCUMENTED_RULE"
            missing_handling_info["imputation_provenance"] = {
                "rule_id": rule_id,
                "rule_description": rule_desc,
                "original_missing_count": missing_count,
                "imputed_count": imputed_count,
                "remaining_missing_count": 0,
                "sensitive_origin": "DERIVED_PROXY",
                "disclaimer": (
                    "Imputed sensitive attributes are classified strictly as DERIVED_PROXY. "
                    "They are not equivalent to observed protected attributes, do not support causal claims, "
                    "and do not constitute legal evidence."
                ),
            }
        else:
            s = pd.Series(s_raw).fillna("UNKNOWN").astype(str).to_numpy()
    else:
        s = pd.Series(s_raw).fillna("UNKNOWN").astype(str).to_numpy()

    unique_groups, group_counts = np.unique(s, return_counts=True)
    group_dict = {str(g): int(c) for g, c in zip(unique_groups, group_counts)}

    metrics: Dict[str, MetricResult] = {}

    # Performance metrics (sample_weight aware)
    acc = float(accuracy_score(y_t, y_p, sample_weight=sw)) if len(y_t) > 0 else float("nan")
    bal_acc = float(balanced_accuracy_score(y_t, y_p, sample_weight=sw)) if len(y_t) > 0 and len(np.unique(y_t)) > 1 else float("nan")
    prec = float(precision_score(y_t, y_p, sample_weight=sw, zero_division=0)) if len(y_t) > 0 else float("nan")
    rec = float(recall_score(y_t, y_p, sample_weight=sw, zero_division=0)) if len(y_t) > 0 else float("nan")
    f1 = float(f1_score(y_t, y_p, sample_weight=sw, zero_division=0)) if len(y_t) > 0 else float("nan")

    metrics["Accuracy"] = MetricResult("Accuracy", acc, EvaluationStatus.VALID)
    metrics["Balanced Accuracy"] = MetricResult("Balanced Accuracy", bal_acc, EvaluationStatus.VALID)
    metrics["Precision"] = MetricResult("Precision", prec, EvaluationStatus.VALID)
    metrics["Recall"] = MetricResult("Recall", rec, EvaluationStatus.VALID)
    metrics["F1-Score"] = MetricResult("F1-Score", f1, EvaluationStatus.VALID)

    # Full Confusion Matrix and Class-Conditional Rates (sample_weight aware)
    tn, fp, fn, tp = compute_group_confusion(y_t, y_p, sample_weight=sw)
    cm_dict = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else float("nan")
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    if sw is not None:
        pos_pred_rate = float(np.average(y_p == 1, weights=sw)) if len(y_p) > 0 else float("nan")
        actual_pos_prev = float(np.average(y_t == 1, weights=sw)) if len(y_t) > 0 else float("nan")
    else:
        pos_pred_rate = float(np.mean(y_p == 1)) if len(y_p) > 0 else float("nan")
        actual_pos_prev = float(np.mean(y_t == 1)) if len(y_t) > 0 else float("nan")

    metrics["TP"] = MetricResult("TP", float(tp), EvaluationStatus.VALID)
    metrics["TN"] = MetricResult("TN", float(tn), EvaluationStatus.VALID)
    metrics["FP"] = MetricResult("FP", float(fp), EvaluationStatus.VALID)
    metrics["FN"] = MetricResult("FN", float(fn), EvaluationStatus.VALID)
    metrics["TPR"] = MetricResult("TPR", tpr, EvaluationStatus.VALID if not np.isnan(tpr) else EvaluationStatus.ZERO_POSITIVE_SUBGROUP)
    metrics["TNR"] = MetricResult("TNR", tnr, EvaluationStatus.VALID if not np.isnan(tnr) else EvaluationStatus.ZERO_NEGATIVE_SUBGROUP)
    metrics["FPR"] = MetricResult("FPR", fpr, EvaluationStatus.VALID if not np.isnan(fpr) else EvaluationStatus.ZERO_NEGATIVE_SUBGROUP)
    metrics["FNR"] = MetricResult("FNR", fnr, EvaluationStatus.VALID if not np.isnan(fnr) else EvaluationStatus.ZERO_POSITIVE_SUBGROUP)
    metrics["PPV"] = MetricResult("PPV", ppv, EvaluationStatus.VALID if not np.isnan(ppv) else EvaluationStatus.UNDEFINED)
    metrics["Positive Prediction Rate"] = MetricResult("Positive Prediction Rate", pos_pred_rate, EvaluationStatus.VALID)
    metrics["Actual Positive Prevalence"] = MetricResult("Actual Positive Prevalence", actual_pos_prev, EvaluationStatus.VALID)

    # Subgroup breakdown & Calibration Support Governance (Phase 12, sample_weight aware)
    subgroup_dict: Dict[str, Dict[str, Any]] = {}
    for g in unique_groups:
        mask = (s == g)
        g_size = int(np.sum(mask))
        if g_size > 0:
            g_sw = sw[mask] if sw is not None else None
            g_tn, g_fp, g_fn, g_tp = compute_group_confusion(y_t[mask], y_p[mask], sample_weight=g_sw)
            n_g_pos = g_tp + g_fn
            n_g_neg = g_tn + g_fp
            g_tpr = float(g_tp / n_g_pos) if n_g_pos > 0 else float("nan")
            g_tnr = float(g_tn / n_g_neg) if n_g_neg > 0 else float("nan")
            g_fpr = float(g_fp / n_g_neg) if n_g_neg > 0 else float("nan")
            g_fnr = float(g_fn / n_g_pos) if n_g_pos > 0 else float("nan")
            g_ppv = float(g_tp / (g_tp + g_fp)) if (g_tp + g_fp) > 0 else float("nan")
            if g_sw is not None:
                g_pred_pos = float(np.average(y_p[mask] == 1, weights=g_sw))
                g_actual_pos = float(np.average(y_t[mask] == 1, weights=g_sw))
            else:
                g_pred_pos = float(np.mean(y_p[mask] == 1))
                g_actual_pos = float(np.mean(y_t[mask] == 1))
            g_bal_acc = float((g_tpr + g_tnr) / 2.0) if not (np.isnan(g_tpr) or np.isnan(g_tnr)) else float("nan")

            has_support = True
            if g_size < min_group_size or (min_group_positives > 0 and n_g_pos < min_group_positives) or (min_group_negatives > 0 and n_g_neg < min_group_negatives):
                has_support = False

            g_entry: Dict[str, Any] = {
                "count": g_size,
                "positive_count": n_g_pos,
                "negative_count": n_g_neg,
                "positive_prevalence": g_actual_pos,
                "predicted_positive_rate": g_pred_pos,
                "TP": g_tp,
                "TN": g_tn,
                "FP": g_fp,
                "FN": g_fn,
                "TPR": g_tpr,
                "TNR": g_tnr,
                "FPR": g_fpr,
                "FNR": g_fnr,
                "PPV": g_ppv,
                "balanced_accuracy": g_bal_acc,
                "support_status": EvaluationStatus.VALID.value if has_support else EvaluationStatus.INSUFFICIENT_GROUP_SUPPORT.value,
            }

            if y_prob_arr is not None:
                g_prob = y_prob_arr[mask]
                if g_size < cal_support_min or n_g_pos < 5 or n_g_neg < 5:
                    g_entry["calibration_support_status"] = EvaluationStatus.INSUFFICIENT_CALIBRATION_SUPPORT.value
                    g_entry["brier_score"] = float("nan")
                    g_entry["expected_calibration_error"] = float("nan")
                    g_entry["calibration_note"] = (
                        f"Calibration support insufficient ({g_size} samples, {n_g_pos} pos, {n_g_neg} neg; "
                        f"minimum {cal_support_min} required). High-precision calibration suppressed."
                    )
                else:
                    g_entry["calibration_support_status"] = EvaluationStatus.VALID.value
                    g_entry["brier_score"] = brier_score_loss_safe(y_t[mask], g_prob)
                    g_ece, _ = expected_calibration_error_safe(y_t[mask], g_prob)
                    g_entry["expected_calibration_error"] = g_ece
                    g_entry["calibration_note"] = "Adequate subgroup calibration support."

            subgroup_dict[str(g)] = g_entry

    cal_details: Optional[List[Dict[str, float]]] = None
    if y_prob_arr is not None:
        if len(np.unique(y_t)) >= 2:
            roc = float(roc_auc_score(y_t, y_prob_arr))
            metrics["ROC-AUC"] = MetricResult("ROC-AUC", roc, EvaluationStatus.VALID)
        else:
            metrics["ROC-AUC"] = MetricResult("ROC-AUC", float("nan"), EvaluationStatus.DEGENERATE_TARGET, "Only 1 class present in y_true.")

        brier = brier_score_loss_safe(y_t, y_prob_arr)
        ece, cal_details = expected_calibration_error_safe(y_t, y_prob_arr)
        metrics["Brier Score"] = MetricResult("Brier Score", brier, EvaluationStatus.VALID, "Mean squared probability error.")
        metrics["Expected Calibration Error"] = MetricResult("Expected Calibration Error", ece, EvaluationStatus.VALID, "10-bin ECE.")
    else:
        metrics["ROC-AUC"] = MetricResult("ROC-AUC", float("nan"), EvaluationStatus.VALID, "Probability estimates not provided (optional).")
        metrics["Brier Score"] = MetricResult("Brier Score", float("nan"), EvaluationStatus.VALID, "Probability estimates not provided.")
        metrics["Expected Calibration Error"] = MetricResult("Expected Calibration Error", float("nan"), EvaluationStatus.VALID, "Probability estimates not provided.")

    # Fairness metrics (sample_weight aware)
    dp_res = safe_demographic_parity_difference(y_p, s, min_group_size, min_group_positives, min_group_negatives, sample_weight=sw)
    eo_res = safe_equal_opportunity_difference(y_t, y_p, s, min_group_size, min_group_positives, min_group_negatives, sample_weight=sw)
    eodds_res = safe_equalized_odds_difference(y_t, y_p, s, min_group_size, min_group_positives, min_group_negatives, sample_weight=sw)
    pp_res = safe_predictive_parity_difference(y_t, y_p, s, min_group_size, min_group_positives, min_group_negatives, sample_weight=sw)
    di_res = safe_disparate_impact_ratio(y_p, s, min_group_size, min_group_positives, min_group_negatives, sample_weight=sw)

    metrics["Demographic Parity Diff"] = dp_res
    metrics["Equal Opportunity Diff"] = eo_res
    metrics["Equalized Odds Diff"] = eodds_res
    metrics["Predictive Parity Diff"] = pp_res
    metrics["Disparate Impact Ratio"] = di_res

    gap_values = [
        r.value for r in [dp_res, eo_res, eodds_res, pp_res]
        if r.status == EvaluationStatus.VALID and not np.isnan(r.value)
    ]
    mean_gap = float(np.mean(gap_values)) if gap_values else float("nan")

    has_invalid = any(r.status not in (EvaluationStatus.VALID,) for r in [dp_res, eo_res, eodds_res, di_res])
    overall_status = initial_overall_status
    if initial_overall_status == EvaluationStatus.VALID and has_invalid:
        overall_status = EvaluationStatus.UNDEFINED

    utility = user_defined_utility_score(bal_acc, mean_gap, alpha=alpha_utility) if not np.isnan(bal_acc) else 0.0
    tradeoff = accuracy_tradeoff_proxy(baseline_accuracy, acc) if baseline_accuracy is not None else None
    deg_status, deg_msg = detect_prediction_collapse(y_p, y_t, y_prob_arr)

    # Bootstrap uncertainty estimation (Phases 4 & 5)
    unc_results_obj: Optional[Dict[str, FairnessUncertaintyResult]] = None
    uncertainty_dict = None
    if compute_uncertainty and len(unique_groups) >= 2:
        try:
            unc_results_obj = compute_fairness_uncertainty_bootstrap(
                y_t,
                y_p,
                s,
                n_bootstraps=n_bootstraps,
                min_group_size=min_group_size,
                min_valid_bootstrap_fraction=bootstrap_valid_frac,
                resampling_unit=resampling_unit,
                cluster_series=cluster_series,
                cluster_key=cluster_key,
            )
            uncertainty_dict = {k: v.to_dict() for k, v in unc_results_obj.items()}
        except Exception as e:
            warnings_list.append(f"Uncertainty bootstrap failed: {e}")
            warnings.warn(f"Uncertainty bootstrap failed: {e}")

    # Worst-Group Selection Bias Disclosure (Phase 7 & 11)
    supported_groups = [g for g, d in subgroup_dict.items() if d.get("support_status") == EvaluationStatus.VALID.value]
    unsupported_groups = [g for g, d in subgroup_dict.items() if d.get("support_status") != EvaluationStatus.VALID.value]
    ranked_groups = sorted(
        supported_groups,
        key=lambda g: subgroup_dict[g].get("balanced_accuracy", float("inf"))
    )
    top_k_flagged = [
        {
            "subgroup": g,
            "balanced_accuracy": subgroup_dict[g].get("balanced_accuracy"),
            "TPR": subgroup_dict[g].get("TPR"),
            "sample_size": subgroup_dict[g].get("count"),
        }
        for g in ranked_groups[:3]
    ]
    worst_group_disclosure = {
        "groups_searched_count": len(unique_groups),
        "total_groups_evaluated": len(unique_groups),
        "supported_groups_count": len(supported_groups),
        "unsupported_groups_count": len(unsupported_groups),
        "metrics_searched_count": 5,
        "selection_criterion": "minimum subgroup balanced accuracy among supported groups",
        "selected_post_hoc": True,
        "label": "worst observed subgroup among the evaluated supported groups",
        "top_k_flagged_groups": top_k_flagged,
        "selection_rule": "minimum subgroup balanced accuracy among supported groups",
        "selection_type": analysis_type.value if hasattr(analysis_type, "value") else str(analysis_type),
        "disclosure": (
            "Selection Bias Disclosure: Flagged worst subgroups are selected extreme-value statistics from an exploratory search "
            "and represent the 'worst observed subgroup among the evaluated supported groups'. They are subject to selection bias "
            "(winner's curse) and do not represent unbiased population estimates of the true worst subgroup."
        ),
    }

    # Multiple Comparisons Metadata (Phase 6 & 10)
    n_comparisons = len(unique_groups) * 5 + len(unique_groups) * 3

    suite = FairnessEvaluationSuite(
        dataset_name=dataset_name,
        model_name=model_name,
        sample_size=len(y_t),
        group_counts=group_dict,
        metrics=metrics,
        overall_status=overall_status,
        fairness_gap=mean_gap,
        user_utility_score=utility,
        accuracy_tradeoff_proxy=tradeoff,
        degeneracy_status=deg_status,
        degeneracy_message=deg_msg,
        fairness_success_status=FairnessSuccessStatus.UNDEFINED,
        confusion_matrix=cm_dict,
        subgroup_metrics=subgroup_dict,
        calibration_details=cal_details,
        uncertainty_intervals=uncertainty_dict,
        intersectional_spec=intersectional_spec.to_dict() if intersectional_spec else None,
        evidence_state=EvidenceState.INSUFFICIENT_EVIDENCE,
        abstentions=abstentions_list,
        warnings=warnings_list,
        number_of_comparisons=n_comparisons,
        analysis_type=analysis_type.value if hasattr(analysis_type, "value") else str(analysis_type),
        worst_group_selection_disclosure=worst_group_disclosure,
        missing_attribute_handling=missing_handling_info,
        evaluation_unit_level=evaluation_unit_level,
        evaluation_unit_key=evaluation_unit_key,
        aggregation_strategy=aggregation_strategy,
        target_aggregation_strategy=target_aggregation_strategy or aggregation_strategy,
        prediction_aggregation_strategy=prediction_aggregation_strategy or aggregation_strategy,
        statistical_estimand=statistical_estimand,
        reporting_privacy_mode=reporting_privacy_mode,
        sample_weight=sw,
    )

    # Evaluate against policy
    policy_res = evaluate_policy_compliance(
        suite,
        policy=active_policy,
        majority_baseline_bal_acc=majority_baseline_bal_acc,
        uncertainty_result=unc_results_obj,
    )

    # Policy Decision Gate (confirmatory mode or explicit requirement)
    effective_gate_req = decision_gate_required if decision_gate_required is not None else certification_gate_required
    gate_req = effective_gate_req if effective_gate_req is not None else (
        analysis_type in (AnalysisType.CONFIRMATORY_ANALYSIS, "CONFIRMATORY_ANALYSIS", AnalysisType.CONFIRMATORY)
    )
    gate_ok = True
    gate_errs: List[str] = []
    gate_codes: List[str] = []
    if gate_req:
        gate_ok, gate_errs, gate_codes = verify_policy_decision_gate(
            suite=suite,
            policy=active_policy,
            statistical_estimand=statistical_estimand,
            evaluation_unit_valid=evaluation_unit_valid,
            split_valid=split_valid,
            leakage_findings_count=leakage_findings_count,
            invariants_valid=invariants_valid,
        )
        if not gate_ok:
            policy_res.decision = PolicyDecision.POLICY_UNDEFINED
            policy_res.evidence_state = EvidenceState.INSUFFICIENT_EVIDENCE
            policy_res.policy_passed = False
            policy_res.abstentions.extend(gate_errs)
            policy_res.missing_evidence_conditions.extend(gate_errs)
            policy_res.reason_codes.extend(gate_codes)

    suite.policy_result = policy_res
    suite.evidence_state = policy_res.evidence_state
    suite.abstentions = sorted(list(set(suite.abstentions + policy_res.abstentions)))
    suite.warnings = sorted(list(set(suite.warnings + policy_res.warnings)))


    # Evaluate multi-policy sensitivity
    sens_results = evaluate_policy_sensitivity(
        suite,
        majority_baseline_bal_acc=majority_baseline_bal_acc,
        uncertainty_result=unc_results_obj,
    )
    suite.policy_sensitivity = {k: v.to_dict() for k, v in sens_results.items()}

    suite.fairness_success_status = classify_fairness_success(
        suite,
        min_balanced_acc=active_policy.min_balanced_accuracy,
        max_eo_diff=active_policy.max_equalized_odds_diff or 0.10,
    )

    suite.selection_bias_disclosure = worst_group_disclosure.get("disclosure")
    suite.excluded_missing_count = missing_count if active_missing_policy == MissingAttributePolicy.EXCLUDE else 0
    suite.accuracy = acc
    suite.balanced_accuracy = bal_acc
    suite.demographic_parity_difference = dp_res.value
    suite.equalized_odds_difference = eodds_res.value
    suite.disparate_impact_ratio = di_res.value

    cal_by_sub = {}
    for g, d in subgroup_dict.items():
        g_cnt = d.get("count", 0)
        pos_cnt = d.get("positive_count", 0)
        neg_cnt = d.get("negative_count", 0)
        if g_cnt < cal_support_min or pos_cnt < 5 or neg_cnt < 5:
            c_stat = EvaluationStatus.SPARSE_SUBGROUP
        else:
            c_stat = EvaluationStatus.VALID
        cal_by_sub[str(g)] = {
            "status": c_stat,
            "brier_score": d.get("brier_score", float("nan")),
            "expected_calibration_error": d.get("expected_calibration_error", float("nan")),
            "sample_size": g_cnt,
            "count": g_cnt,
            "positive_count": pos_cnt,
            "negative_count": neg_cnt,
        }
    suite.calibration_by_subgroup = cal_by_sub

    # Apply privacy-aware output governance
    suite = govern_evaluation_reporting(
        suite,
        privacy_mode=reporting_privacy_mode,
        min_reporting_population=min_reporting_population,
        max_disclosed_groups=max_disclosed_groups,
        mask_labels=mask_privacy_labels,
    )

    # Decoupled status taxonomy & human review boundary
    tax_state = map_to_audit_taxonomy(
        audit_execution_status=AuditExecutionStatus.COMPLETE,
        suite=suite,
        policy_result=policy_res,
        analysis_type=analysis_type.value if hasattr(analysis_type, "value") else str(analysis_type),
        leakage_findings_count=leakage_findings_count,
        is_degenerate=(suite.degeneracy_status == ModelDegeneracyStatus.DEGENERATE),
        gate_passed=gate_ok if gate_req else None,
        gate_violations=gate_errs if gate_req else None,
    )
    hr_boundary = determine_human_review_boundary(
        taxonomy_state=tax_state,
        suite=suite,
        policy_result=policy_res,
        sensitive_lineage="OBSERVED",
        policy_type=active_policy.policy_type if active_policy else None,
        estimand_specified=(statistical_estimand is not None),
        unit_mismatch=False,
    )
    suite.taxonomy_state = tax_state.to_dict()
    suite.human_review_boundary = hr_boundary.to_dict()

    return suite


# ============================================================================
# Split Hierarchy Enforcement
# ============================================================================

@dataclass
class IsolatedSplitBundle:
    """
    Enforces the split hierarchy:
    Train -> Validation / Development -> Locked Holdout Test
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    s_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    s_val: pd.Series
    X_test_locked: pd.DataFrame
    y_test_locked: pd.Series
    s_test_locked: pd.Series

    def summary(self) -> str:
        return (
            f"Split Hierarchy Summary:\n"
            f"  Train:      {len(self.y_train)} rows (for model parameter fitting)\n"
            f"  Validation: {len(self.y_val)} rows (for HPO, threshold tuning, and model selection)\n"
            f"  Locked Test:{len(self.y_test_locked)} rows (evaluated strictly once for final reporting)\n"
        )
