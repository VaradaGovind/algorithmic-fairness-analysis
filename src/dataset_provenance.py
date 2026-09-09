# ============================================================================
# Dataset Provenance Contract & Registry
# Enforces structured tracking of dataset lineage, licensing, target definitions,
# sensitive attribute origins, timing contracts, and redistribution permissions.
# ============================================================================

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import yaml


class DatasetType(str, Enum):
    REAL_DATA = "REAL_DATA"
    REAL_EXTERNAL = "REAL_DATA"
    SYNTHETIC_DATA = "SYNTHETIC_DATA"


class SensitiveOrigin(str, Enum):
    REAL_OBSERVED = "REAL_OBSERVED"
    REPORTED_DEMOGRAPHIC = "REAL_OBSERVED"
    CONSTRUCTED_ATTRIBUTE = "CONSTRUCTED_ATTRIBUTE"
    DERIVED_PROXY = "DERIVED_PROXY"
    SYNTHETIC = "SYNTHETIC"


@dataclass
class DatasetProvenance:
    """
    Structured metadata record establishing dataset provenance, licensing status,
    target definition, sensitive attribute origin, and feature timing contract.
    """
    dataset_name: str
    dataset_type: DatasetType
    source_reference: str
    version_or_date: Optional[str]
    license_notes: str
    target_definition: str
    sensitive_attributes: List[str]
    sensitive_origin: SensitiveOrigin
    feature_timing_contract: str
    expected_grouping_key: Optional[str] = None
    expected_temporal_key: Optional[str] = None
    redistribution_status: str = "RESTRICTED"
    sha256_fingerprint: Optional[str] = None
    verified_licensing: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type.value,
            "source_reference": self.source_reference,
            "version_or_date": self.version_or_date,
            "license_notes": self.license_notes,
            "target_definition": self.target_definition,
            "sensitive_attributes": self.sensitive_attributes,
            "sensitive_origin": self.sensitive_origin.value,
            "feature_timing_contract": self.feature_timing_contract,
            "expected_grouping_key": self.expected_grouping_key,
            "expected_temporal_key": self.expected_temporal_key,
            "redistribution_status": self.redistribution_status,
            "sha256_fingerprint": self.sha256_fingerprint,
            "verified_licensing": self.verified_licensing,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetProvenance:
        return cls(
            dataset_name=data["dataset_name"],
            dataset_type=DatasetType(data["dataset_type"]),
            source_reference=data["source_reference"],
            version_or_date=data.get("version_or_date"),
            license_notes=data["license_notes"],
            target_definition=data["target_definition"],
            sensitive_attributes=list(data.get("sensitive_attributes", [])),
            sensitive_origin=SensitiveOrigin(data["sensitive_origin"]),
            feature_timing_contract=data["feature_timing_contract"],
            expected_grouping_key=data.get("expected_grouping_key"),
            expected_temporal_key=data.get("expected_temporal_key"),
            redistribution_status=data.get("redistribution_status", "RESTRICTED"),
            sha256_fingerprint=data.get("sha256_fingerprint"),
            verified_licensing=bool(data.get("verified_licensing", False)),
            notes=data.get("notes", ""),
        )


def compute_file_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a local file safely."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Known Dataset Provenance Registry
PROVENANCE_REGISTRY: Dict[str, DatasetProvenance] = {
    "Delhivery Logistics": DatasetProvenance(
        dataset_name="Delhivery Logistics",
        dataset_type=DatasetType.REAL_DATA,
        source_reference="Kaggle / Delhivery Logistics Dataset (Historical operational delivery records)",
        version_or_date="2022",
        license_notes="Public repository / Unverified commercial redistribution terms. Not committed to repo.",
        target_definition="Task_Success = 1 - is_cutoff (Realized delivery did not breach scheduled cutoff deadline)",
        sensitive_attributes=["Education_Level"],
        sensitive_origin=SensitiveOrigin.CONSTRUCTED_ATTRIBUTE,
        feature_timing_contract="PRE_DISPATCH_VALID: OSRM planned distances/times only. Post-delivery times (actual_time) banned.",
        expected_grouping_key="trip_uuid",
        expected_temporal_key="trip_creation_time",
        redistribution_status="LOCAL_STAGING_ONLY_NO_GIT",
        sha256_fingerprint=None,
        verified_licensing=False,
        notes="Sensitive attribute 'Education_Level' is synthetically constructed from salary priors, not real driver demographics.",
    ),
    "Amazon Last-Mile Routes": DatasetProvenance(
        dataset_name="Amazon Last-Mile Routes",
        dataset_type=DatasetType.REAL_DATA,
        source_reference="Amazon Last-Mile Routing Research Challenge (Merchante et al., 2021)",
        version_or_date="2021",
        license_notes="Research use only under Amazon challenge terms. Redistribution prohibited.",
        target_definition="Task_Success = (route_score == 'High')",
        sensitive_attributes=["Education_Level"],
        sensitive_origin=SensitiveOrigin.CONSTRUCTED_ATTRIBUTE,
        feature_timing_contract="PRE_DISPATCH_VALID: Scheduled stops, capacity, bbox area. Realized sequence scores banned.",
        expected_grouping_key="station_code",
        expected_temporal_key="departure_hour",
        redistribution_status="RESTRICTED_RESEARCH_USE",
        sha256_fingerprint=None,
        verified_licensing=False,
        notes="Education_Level constructed from route geographic complexity proxies; driver demographics absent.",
    ),
    "Adult Income Benchmark": DatasetProvenance(
        dataset_name="Adult Income Benchmark",
        dataset_type=DatasetType.REAL_DATA,
        source_reference="UCI Machine Learning Repository / US Census Bureau 1994 Current Population Survey",
        version_or_date="1994",
        license_notes="Public domain / CC BY 4.0. Suitable for academic benchmarking.",
        target_definition="Task_Success = (income > 50K)",
        sensitive_attributes=["Education_Group", "Sex", "Race"],
        sensitive_origin=SensitiveOrigin.REAL_OBSERVED,
        feature_timing_contract="SURVEY_CONCURRENT: All features observed concurrently during census interview.",
        expected_grouping_key=None,
        expected_temporal_key=None,
        redistribution_status="PUBLIC_PERMISSIVE",
        sha256_fingerprint=None,
        verified_licensing=True,
        notes="Real observed demographic attributes. Causal interpretations must adjust for post-treatment mediators.",
    ),
    "Synthetic Controlled Benchmark": DatasetProvenance(
        dataset_name="Synthetic Controlled Benchmark",
        dataset_type=DatasetType.SYNTHETIC_DATA,
        source_reference="Algorithmic Fairness Analysis Framework / Controlled Logistics DGP",
        version_or_date="Pass 4 (2026)",
        license_notes="MIT License (Open Source)",
        target_definition="Task_Success = [sigmoid(z) >= 0.50] under explicit logistic structural equations",
        sensitive_attributes=["Complexity_Tier"],
        sensitive_origin=SensitiveOrigin.SYNTHETIC,
        feature_timing_contract="PRE_DISPATCH_VALID: Clean synthetic route features generated prior to dispatch.",
        expected_grouping_key="station_cluster",
        expected_temporal_key="dispatch_hour",
        redistribution_status="UNRESTRICTED_OPEN_SOURCE",
        sha256_fingerprint=None,
        verified_licensing=True,
        notes="Controlled mathematical data-generating process used for algorithmic verification.",
    ),
}


def load_dataset_provenance_config(config_path: Union[str, Path]) -> Dict[str, DatasetProvenance]:
    """Loads dataset provenance records from YAML."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Provenance config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    result: Dict[str, DatasetProvenance] = {}
    for item in data.get("provenance_records", []):
        prov = DatasetProvenance.from_dict(item)
        result[prov.dataset_name] = prov
    return result
