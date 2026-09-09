# ============================================================================
# Unit & Regression Tests: Secure Data Ingestion & Manifest Pipeline
# Covers schema validation, path traversal, file type security, size limits,
# duplicate detection, leakage gating, SHA-256 hashing, and Git isolation.
# ============================================================================

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.canonical_benchmark import execute_locked_real_data_benchmark
from src.dataset_provenance import (
    DatasetProvenance,
    DatasetType,
    SensitiveOrigin,
    compute_file_sha256,
    PROVENANCE_REGISTRY,
)
from src.ingest_data import (
    DatasetManifest,
    IngestionStatus,
    SchemaValidationError,
    SecurityValidationError,
    ingest_and_validate_dataset,
    validate_source_path_security,
    verify_git_isolation,
)


@pytest.fixture
def clean_csv_fixture(tmp_path):
    """Creates a temporary valid synthetic CSV for ingestion testing."""
    df = pd.DataFrame({
        "planned_distance": [10.5, 20.2, 15.0, 30.1, 25.4, 18.0, 12.3, 22.1],
        "planned_time": [25.0, 45.0, 32.0, 60.0, 50.0, 38.0, 28.0, 48.0],
        "station_code": ["ST_A", "ST_A", "ST_B", "ST_B", "ST_A", "ST_B", "ST_A", "ST_B"],
        "shift_tier": ["Morning", "Night", "Morning", "Night", "Morning", "Night", "Morning", "Night"],
        "driver_type": ["Contract", "FullTime", "Contract", "FullTime", "Contract", "FullTime", "Contract", "FullTime"],
        "Task_Success": [1, 0, 1, 1, 0, 1, 0, 1],
    })
    csv_path = tmp_path / "valid_logistics.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# 1. Dataset Manifest Creation & SHA-256 Fingerprint
def test_dataset_manifest_creation_and_sha256(clean_csv_fixture, tmp_path):
    manifest = ingest_and_validate_dataset(
        source_path=clean_csv_fixture,
        dataset_name="Test Logistics",
        target_column="Task_Success",
        sensitive_columns=["driver_type"],
        grouping_column="station_code",
        output_manifest_dir=tmp_path,
    )
    assert manifest.status in (IngestionStatus.PASSED, IngestionStatus.PASSED_WITH_WARNINGS)
    assert manifest.row_count == 8
    assert manifest.column_count == 6
    assert len(manifest.sha256_hash) == 64
    assert manifest.sha256_hash == compute_file_sha256(clean_csv_fixture)

    manifest_json = tmp_path / "test_logistics_manifest.json"
    assert manifest_json.is_file()
    with open(manifest_json, "r") as f:
        loaded = json.load(f)
    assert loaded["dataset_name"] == "Test Logistics"
    assert loaded["sha256_hash"] == manifest.sha256_hash


# 2. JSON Format Ingestion Support
def test_json_file_ingestion(tmp_path):
    data = [
        {"x1": 1.0, "x2": 2.0, "group": "A", "target": 1},
        {"x1": 2.0, "x2": 3.0, "group": "B", "target": 0},
        {"x1": 1.5, "x2": 2.5, "group": "A", "target": 1},
        {"x1": 2.5, "x2": 3.5, "group": "B", "target": 0},
    ]
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    manifest = ingest_and_validate_dataset(
        source_path=json_path,
        dataset_name="Test JSON",
        target_column="target",
        sensitive_columns=["group"],
    )
    assert manifest.status in (IngestionStatus.PASSED, IngestionStatus.PASSED_WITH_WARNINGS)
    assert manifest.row_count == 4


# 3. Path Traversal Rejection
def test_path_traversal_rejection():
    evil_path = "../../etc/passwd.csv"
    with pytest.raises(SecurityValidationError, match="Path traversal detected"):
        validate_source_path_security(evil_path)


# 4. Invalid & Dangerous File Type Rejection
@pytest.mark.parametrize("bad_ext", [".pkl", ".pickle", ".joblib", ".pt", ".py", ".exe", ".sh", ".txt"])
def test_invalid_file_type_rejection(bad_ext, tmp_path):
    bad_file = tmp_path / f"malicious{bad_ext}"
    bad_file.write_text("evil contents")
    with pytest.raises(SecurityValidationError):
        validate_source_path_security(bad_file)


# 5. Oversized Input File Limit Rejection
def test_oversized_file_rejection(clean_csv_fixture):
    # Set limit artificially low (e.g. 10 bytes)
    manifest = ingest_and_validate_dataset(
        source_path=clean_csv_fixture,
        dataset_name="Oversized Dataset",
        target_column="Task_Success",
        sensitive_columns=["driver_type"],
        max_file_size_bytes=10,
    )
    assert manifest.status == IngestionStatus.REJECTED_SIZE
    assert "exceeds limit" in manifest.validation_messages[0]


# 6. Missing Required Column Rejection
def test_missing_required_column_rejection(clean_csv_fixture):
    manifest = ingest_and_validate_dataset(
        source_path=clean_csv_fixture,
        dataset_name="Missing Col Dataset",
        target_column="NonExistentTarget",
        sensitive_columns=["NonExistentSensitive"],
    )
    assert manifest.status == IngestionStatus.REJECTED_SCHEMA
    assert any("Target column 'NonExistentTarget' missing" in m for m in manifest.validation_messages)
    assert any("Sensitive column 'NonExistentSensitive' missing" in m for m in manifest.validation_messages)


# 7. Duplicate Row Detection
def test_duplicate_row_detection(tmp_path):
    df_dup = pd.DataFrame({
        "feat": [1.0, 1.0, 1.0, 2.0],
        "target": [1, 1, 1, 0],
        "sensitive": ["A", "A", "A", "B"],
    })
    csv_path = tmp_path / "duplicates.csv"
    df_dup.to_csv(csv_path, index=False)

    manifest = ingest_and_validate_dataset(
        source_path=csv_path,
        dataset_name="Duplicate Dataset",
        target_column="target",
        sensitive_columns=["sensitive"],
    )
    assert manifest.duplicate_row_count == 2
    assert manifest.duplicate_row_fraction == 0.50
    assert any("Detected 2 duplicate rows" in m for m in manifest.validation_messages)


# 8. Impossible Value Detection
def test_impossible_value_detection(tmp_path):
    df_neg = pd.DataFrame({
        "planned_distance": [-10.5, 20.0, 15.0, 25.0],  # Negative distance
        "planned_time": [30.0, -5.0, 25.0, 40.0],       # Negative time
        "target": [1, 0, 1, 0],
        "group": ["A", "A", "B", "B"],
    })
    csv_path = tmp_path / "impossible_values.csv"
    df_neg.to_csv(csv_path, index=False)

    manifest = ingest_and_validate_dataset(
        source_path=csv_path,
        dataset_name="Impossible Values Dataset",
        target_column="target",
        sensitive_columns=["group"],
    )
    assert any("negative values" in m for m in manifest.validation_messages)


# 9. Catastrophic Leakage Screening & Benchmark Refusal
def test_catastrophic_leakage_detection_and_execution_refusal(tmp_path):
    # Create dataset with exact post-outcome leakage column
    df_leak = pd.DataFrame({
        "planned_distance": [10.0, 20.0, 30.0, 40.0] * 5,
        "actual_time": [100.0, 200.0, 150.0, 400.0] * 5,  # Banned post-outcome
        "Task_Success": [1, 0, 1, 0] * 5,
        "group": ["A", "A", "B", "B"] * 5,
    })
    csv_path = tmp_path / "leaked_dataset.csv"
    df_leak.to_csv(csv_path, index=False)

    manifest = ingest_and_validate_dataset(
        source_path=csv_path,
        dataset_name="Leaked Dataset",
        target_column="Task_Success",
        sensitive_columns=["group"],
        banned_leakage_columns=["actual_time"],
    )
    assert manifest.status == IngestionStatus.REJECTED_LEAKAGE
    assert manifest.has_critical_leakage is True

    # Confirm that canonical benchmark refuses execution
    with pytest.raises(RuntimeError, match="EXECUTION REFUSED: Dataset 'Leaked Dataset' failed evaluation integrity audit"):
        execute_locked_real_data_benchmark(manifest)


# 10. Git Isolation Verification
def test_git_isolation_verification(tmp_path):
    # External path outside project root is safe
    external_path = tmp_path / "external_data.csv"
    assert verify_git_isolation(external_path) is True
