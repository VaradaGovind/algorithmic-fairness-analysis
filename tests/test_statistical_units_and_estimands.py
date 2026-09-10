# ============================================================================
# Statistical Units, Estimands, and Entity Aggregation Test Suite
# Tests statistical evaluation units, estimand contracts, entity aggregation strategies,
# inverse-frequency weighting, and dataset semantic mappings.
# ============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.fairness_evaluator import (
    AGGREGATION_STRATEGY_REGISTRY,
    AggregationStrategy,
    AggregationStrategySpecification,
    AnalysisType,
    EvaluationStatus,
    EvaluationUnitLevel,
    EvidenceState,
    FairnessEvaluationSuite,
    FairnessPolicy,
    PolicyDecision,
    PolicyType,
    ResamplingUnit,
    aggregate_to_evaluation_unit,
    compute_entity_weights,
    evaluate_fairness_defensively,
    validate_evaluation_unit_configuration,
)
from src.external_dataset_adapter import ExternalDataBundle, ExternalDatasetAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Test 1: Evaluation Unit Registry Loading & Validation
# ---------------------------------------------------------------------------

def test_evaluation_unit_registry_loading():
    config_path = PROJECT_ROOT / "configs" / "evaluation_units.yaml"
    assert config_path.exists(), f"Configuration file {config_path} missing."

    with open(config_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    assert "version" in registry
    assert "evaluation_units" in registry

    datasets = registry["evaluation_units"]
    required_datasets = ["delhivery_logistics", "amazon_last_mile", "adult_income", "synthetic_logistics"]
    for ds in required_datasets:
        assert ds in datasets, f"Dataset '{ds}' missing from evaluation_units.yaml"
        ds_cfg = datasets[ds]
        assert "natural_decision_unit" in ds_cfg
        assert "raw_record_unit" in ds_cfg
        assert "evaluation_unit_key" in ds_cfg
        assert "historical_unit_mismatch" in ds_cfg

    dlh = datasets["delhivery_logistics"]
    assert dlh["natural_decision_unit"] == "TRIP_LEVEL"
    assert dlh["raw_record_unit"] == "SEGMENT_LEVEL"
    assert dlh["historical_unit_mismatch"]["historical_evaluated_unit"] == "SEGMENT_LEVEL"


# ---------------------------------------------------------------------------
# Test 2: Row vs. Entity Metric Divergence
# ---------------------------------------------------------------------------

def test_row_vs_entity_metric_divergence():
    """
    Demonstrates mathematically that evaluating on raw unaggregated rows
    can dramatically misrepresent entity-level outcomes.
    """
    trip_ids_a = ["trip_A_big"] * 20 + [f"trip_A_{i}" for i in range(1, 10)]
    y_true_a = [1] * 20 + [0] * 9
    y_pred_a = [1] * 20 + [0] * 9
    group_a = ["Group_A"] * 29

    trip_ids_b = []
    for i in range(10):
        trip_ids_b.extend([f"trip_B_{i}"] * 2)
    y_true_b = [1] * 20
    y_pred_b = [1] * 20
    group_b = ["Group_B"] * 20

    all_trips = np.array(trip_ids_a + trip_ids_b)
    all_yt = np.array(y_true_a + y_true_b)
    all_yp = np.array(y_pred_a + y_pred_b)
    all_groups = np.array(group_a + group_b)

    # 1. Unaggregated row-level evaluation
    suite_row = evaluate_fairness_defensively(
        y_true=all_yt,
        y_pred=all_yp,
        sensitive=all_groups,
        entity_series=all_trips,
        evaluation_unit_level=EvaluationUnitLevel.ROW_LEVEL,
        aggregation_strategy=AggregationStrategy.NONE,
    )
    row_sub = suite_row.subgroup_metrics
    rate_a_row = row_sub["Group_A"]["predicted_positive_rate"]
    assert pytest.approx(rate_a_row, 0.01) == 20 / 29

    # 2. Trip-level aggregated evaluation (ANY_FAILURE strategy)
    suite_trip = evaluate_fairness_defensively(
        y_true=all_yt,
        y_pred=all_yp,
        sensitive=all_groups,
        entity_series=all_trips,
        evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
        aggregation_strategy=AggregationStrategy.ANY_FAILURE,
    )
    trip_sub = suite_trip.subgroup_metrics
    rate_a_trip = trip_sub["Group_A"]["predicted_positive_rate"]
    assert pytest.approx(rate_a_trip, 0.01) == 0.10

    # The two rates diverge significantly
    assert abs(rate_a_row - rate_a_trip) > 0.50
    assert suite_trip.sample_size == 20


# ---------------------------------------------------------------------------
# Test 3: High-Volume Entity Domination Guard
# ---------------------------------------------------------------------------

def test_high_volume_entity_domination_guard():
    """
    Verifies that ENTITY_WEIGHTED strategy assigns 1/N_entity weight to each row,
    preventing high-volume entities from dominating global accuracy.
    """
    entities = ["ENT_DOMINANT"] * 90 + [f"ENT_SMALL_{i}" for i in range(10)]
    y_true = np.array([1] * 90 + [0] * 10)
    y_pred = np.array([1] * 100)
    sensitive = np.array(["G1"] * 50 + ["G2"] * 50)

    suite_unweighted = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        entity_series=entities,
        aggregation_strategy=AggregationStrategy.NONE,
    )
    assert pytest.approx(suite_unweighted.accuracy, 0.01) == 0.90

    suite_weighted = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        entity_series=entities,
        aggregation_strategy=AggregationStrategy.ENTITY_WEIGHTED,
    )
    assert pytest.approx(suite_weighted.accuracy, 0.01) == 1.0 / 11.0


# ---------------------------------------------------------------------------
# Test 4: Aggregation Strategies Determinism
# ---------------------------------------------------------------------------

def test_aggregation_strategies():
    entities = np.array(["T1", "T1", "T1", "T2", "T2", "T2"])
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    sensitive = np.array(["A", "A", "A", "B", "B", "B"])

    # Strategy: ANY_FAILURE
    yt_af, yp_af, s_af, _, _ = aggregate_to_evaluation_unit(
        y_true, y_pred, sensitive, entities, strategy=AggregationStrategy.ANY_FAILURE
    )
    assert yp_af[0] == 0
    assert yp_af[1] == 0

    # Strategy: MAJORITY_VOTE
    yt_mv, yp_mv, s_mv, _, _ = aggregate_to_evaluation_unit(
        y_true, y_pred, sensitive, entities, strategy=AggregationStrategy.MAJORITY_VOTE
    )
    assert yp_mv[0] == 1
    assert yp_mv[1] == 0

    # Strategy: ALL_SUCCESS
    yt_as, yp_as, s_as, _, _ = aggregate_to_evaluation_unit(
        y_true, y_pred, sensitive, entities, strategy=AggregationStrategy.ALL_SUCCESS
    )
    assert yp_as[0] == 1
    assert yp_as[1] == 0


# ---------------------------------------------------------------------------
# Test 5: Entity Weights Normalization
# ---------------------------------------------------------------------------

def test_entity_weights_computation():
    entities = ["A", "A", "A", "B"]
    weights = compute_entity_weights(entities)
    assert len(weights) == 4
    assert pytest.approx(weights[0], 1e-5) == 2.0 / 3.0
    assert pytest.approx(weights[3], 1e-5) == 2.0
    assert pytest.approx(np.sum(weights), 1e-5) == 4.0


# ---------------------------------------------------------------------------
# Test 6: Statistical Unit Mismatch Guard
# ---------------------------------------------------------------------------

def test_statistical_unit_mismatch_guard():
    entities = np.array(["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4"])
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    with pytest.raises(ValueError) as exc_info:
        evaluate_fairness_defensively(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
            entity_series=entities,
            evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
            compute_uncertainty=True,
            resampling_unit=ResamplingUnit.ROW_BOOTSTRAP,
            allow_row_bootstrap_override=False,
        )
    assert "Statistical unit mismatch" in str(exc_info.value)

    suite = evaluate_fairness_defensively(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        entity_series=entities,
        evaluation_unit_level=EvaluationUnitLevel.TRIP_LEVEL,
        compute_uncertainty=True,
        resampling_unit=ResamplingUnit.ROW_BOOTSTRAP,
        allow_row_bootstrap_override=True,
    )
    assert any("Statistical unit mismatch override" in w for w in suite.warnings)


# ---------------------------------------------------------------------------
# Test 7: Estimand Declaration & Validation
# ---------------------------------------------------------------------------

def test_estimand_declaration_and_validation():
    valid_estimand = {
        "population": "Fleet drivers",
        "decision_unit": "TRIP_LEVEL",
        "observation_unit": "SEGMENT_LEVEL",
        "weighting_rule": "UNIFORM",
        "target_quantity": "PROBABILITY_OF_COMPLETION",
        "fairness_quantity": "EQUALIZED_ODDS_DIFF",
        "estimand_status": "SPECIFIED",
    }
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    sens = np.array(["A", "A", "B", "B"])

    suite = evaluate_fairness_defensively(
        y_true, y_pred, sensitive=sens,
        statistical_estimand=valid_estimand,
    )
    assert suite.statistical_estimand is not None
    assert suite.statistical_estimand["estimand_status"] == "SPECIFIED"


# ---------------------------------------------------------------------------
# Test 8: Evaluation Unit Configuration Validation
# ---------------------------------------------------------------------------

def test_valid_and_invalid_evaluation_unit_configuration():
    df_valid = pd.DataFrame({
        "trip_uuid": ["T1", "T1", "T2", "T2"],
        "route_id": ["R1", "R1", "R2", "R2"],
    })
    diag = validate_evaluation_unit_configuration(
        df_valid,
        evaluation_unit_key="trip_uuid",
        raise_on_error=True,
    )
    assert diag["semantic_validation_status"] == "VERIFIED_FROM_DATA"

    df_invalid = pd.DataFrame({
        "other_col": [1, 2, 3],
    })
    with pytest.raises(ValueError):
        validate_evaluation_unit_configuration(
            df_invalid,
            evaluation_unit_key="non_existent_key",
            raise_on_error=True,
        )


# ---------------------------------------------------------------------------
# Test 9: Dataset Semantics
# ---------------------------------------------------------------------------

def test_dataset_semantics_in_registry():
    cfg_path = PROJECT_ROOT / "configs" / "evaluation_units.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    units = spec["evaluation_units"]
    assert "adult_income" in units
    assert units["adult_income"]["natural_decision_unit"] == "INDIVIDUAL_LEVEL"
    assert units["adult_income"]["raw_record_unit"] == "INDIVIDUAL_LEVEL"

    assert "amazon_last_mile" in units
    assert units["amazon_last_mile"]["natural_decision_unit"] == "ROUTE_LEVEL"

    assert "delhivery_logistics" in units
    assert units["delhivery_logistics"]["natural_decision_unit"] == "TRIP_LEVEL"
