# External Real-Data Validation: Status & Staging Protocol

**Document Version:** 1.0.0 (`PASS-7-SPEC`)  
**Effective Date:** 2026-09-08  
**Governance Authority:** Model Risk Governance & Algorithmic Audit Committee  
**Compliance Standard:** `POL-STD-2026.1`  
**Evaluation Status:** `EXTERNAL_PENDING_STAGING` (NOT EXECUTED)

---

## 1. Executive Status & Non-Fabrication Notice

> [!IMPORTANT]
> **FORMAL STATUS: EXTERNAL EMPIRICAL VALIDATION REMAINS PENDING**  
> In accordance with the core governance principles of the `algorithmic-fairness-analysis` platform:
> - **Zero Fabrication:** Synthetic datasets and controlled Data Generating Processes (DGPs) are never represented as real-world customer data.
> - **Zero Stored PII/Proprietary Data:** No unvetted, unredacted, or proprietary customer datasets are committed or tracked in this repository's Git history.
> - **Independent Staging Requirement:** External validation is executed exclusively via `ExternalDatasetAdapter` once the customer or auditor stages a verified, legally cleared dataset in an approved staging directory outside Git tracking.
>
> Until customer data is staged and evaluated through the locked pipeline, all benchmark results in the primary repository are classified strictly as `SYNTHETIC_DGP_VALIDATION` or `INTERNAL_ENGINEERING_VERIFICATION`.

---

## 2. Pre-Requisites for External Real-Data Execution

Prior to running the locked evaluation workflow, the staging officer must verify:

| Verification Requirement | Description | Audit Evidence Required |
| :--- | :--- | :--- |
| **Genuine Observed Attribute** | Sensitive attributes must be real, legally documented demographic or protected attributes (e.g. reported gender, race/ethnicity, age) or documented proxy constructs. | Data dictionary & collection protocol. |
| **Independent Source Lineage** | Data must originate from a real-world operational or research domain, completely independent of synthetic DGP formulas. | Vendor/licensor attribution, license contract. |
| **Sample Size Sufficiency** | Total sample size $N \ge 1,000$; each evaluated subgroup must have $N_g \ge 30$ with at least 5 positive and 5 negative outcomes. | Pre-ingestion sample size audit report. |
| **Timing & Leakage Contract** | Features must be strictly observable at prediction time. Post-outcome realizations (e.g., delivery completion timestamps) are prohibited. | Prediction-time feature contract YAML. |
| **Physical Isolation** | The raw dataset must reside outside the Git workspace root (e.g., `data/external_staging/`) and be confirmed untracked. | `git ls-files --error-unmatch` exit code non-zero. |

---

## 3. Mandatory External Evaluation Record Template

When an external evaluation is executed, the resulting audit report must populate every field of the following specification:

```yaml
external_validation_audit_record:
  dataset_metadata:
    source_name: "Customer Operational Staging"
    dataset_version: "2026.Q3"
    sha256_fingerprint: "REQUIRED_HEX_DIGEST"
    sample_size_total: 0
    license_status: "CUSTOMER_HELD_PROPRIETARY"
    classification: "EXTERNAL_REAL_DATA_VALIDATION"
  
  feature_and_target_contracts:
    target_definition: "binary_operational_outcome"
    positive_class_label: 1
    negative_class_label: 0
    sensitive_attribute: "reported_group"
    sensitive_provenance: "REAL_OBSERVED" # or "DERIVED_PROXY"
    feature_contract_version: "2.0.0"
    selected_feature_count: 0
    excluded_feature_count: 0
    excluded_leakage_features: []
  
  partitioning_and_isolation:
    split_strategy: "GROUP_HOLDOUT" # or "TEMPORAL_CHRONOLOGICAL"
    grouping_key: "entity_id"
    train_size: 0.60
    validation_size: 0.20
    locked_test_size: 0.20
    random_seed: 42
  
  uncertainty_and_resampling:
    resampling_unit: "CLUSTER_BOOTSTRAP" # or "ROW_BOOTSTRAP"
    cluster_key: "trip_uuid" # or entity identifier
    confidence_method: "EMPIRICAL_PERCENTILE_BOOTSTRAP"
    bootstrap_iterations_requested: 500
    bootstrap_iterations_successful: 0
    bootstrap_discard_fraction: 0.0
  
  governance_and_policy:
    policy_id: "POL-STD-2026.1"
    policy_version: "1.0.0"
    analysis_type: "CONFIRMATORY_ANALYSIS" # or "SCREENING_ANALYSIS"
    multiplicity_correction_method: "BONFERRONI"
    number_of_comparisons: 0
    worst_group_selection_label: "worst observed subgroup among the evaluated supported groups"
    worst_group_post_hoc_selected: true
  
  final_compliance_outcome:
    evidence_state: "PENDING_EXECUTION" # PASS | FAIL | INSUFFICIENT_EVIDENCE
    tri_state_decision: "POLICY_UNDEFINED"
    manifest_fingerprint: "PENDING_EXECUTION"
```

---

## 4. Exact Customer Staging & Execution Instructions

### Step 1: Secure Data Staging (Untracked Directory)
Create an approved, local staging directory completely detached from Git version control:
```bash
# Create staging directory outside tracked tree
mkdir -p /secure/staging/customer_fairness_data/

# Copy raw tabular CSV or JSON into staging
cp /path/to/customer_export.csv /secure/staging/customer_fairness_data/operational_eval.csv
```

Verify that Git tracking is rejected:
```bash
# Must return an error (path is not in working tree or untracked)
git ls-files --error-unmatch /secure/staging/customer_fairness_data/operational_eval.csv
```

### Step 2: Configure Evaluation Policy & Feature Contract
Define the target, sensitive attribute, and timing contract in a local configuration:
```yaml
# configs/customer_evaluation_contract.yaml
target_column: "claim_approved"
sensitive_column: "applicant_age_bracket"
grouping_column: "facility_id"
cluster_key: "facility_id"
resampling_unit: "CLUSTER_BOOTSTRAP"
missing_attribute_policy: "REJECT" # or "EXCLUDE"
analysis_type: "CONFIRMATORY_ANALYSIS"
```

### Step 3: Run the Black-Box Ingestion & Evaluation Pipeline
Execute via Python using `ExternalDatasetAdapter`:
```python
from pathlib import Path
from src.external_dataset_adapter import ExternalDatasetAdapter
from src.fairness_evaluator import (
    FairnessPolicy,
    MissingAttributePolicy,
    AnalysisType,
    ResamplingUnit,
    evaluate_fairness_defensively,
    evaluate_policy_compliance,
)
from src.evaluation_manifest import (
    create_evaluation_manifest,
    compute_manifest_fingerprint,
    ResultClassification,
)

# 1. Initialize black-box adapter
adapter = ExternalDatasetAdapter(staging_root=Path("/secure/staging/customer_fairness_data/"))

# 2. Ingest external tabular dataset
bundle = adapter.ingest_external_tabular(
    source_path="/secure/staging/customer_fairness_data/operational_eval.csv",
    target_column="claim_approved",
    sensitive_column="applicant_age_bracket",
    test_size=0.20,
    val_size=0.20,
    random_seed=42,
)

# 3. Predict on locked test partition
y_pred_test = model.predict(bundle.X_test)
y_prob_test = model.predict_proba(bundle.X_test)[:, 1] if hasattr(model, "predict_proba") else None

# 4. Evaluate defensively with cluster-aware bootstrap
policy = FairnessPolicy(
    policy_id="POL-CUSTOMER-2026.1",
    missing_attribute_policy=MissingAttributePolicy.REJECT,
    analysis_type=AnalysisType.CONFIRMATORY_ANALYSIS,
)

suite = evaluate_fairness_defensively(
    y_true=bundle.y_test,
    y_pred=y_pred_test,
    y_prob=y_prob_test,
    sensitive=bundle.sensitive_test,
    policy=policy,
    compute_uncertainty=True,
    resampling_unit=ResamplingUnit.CLUSTER_BOOTSTRAP,
    cluster_series=bundle.test_data.get("facility_id"),
    cluster_key="facility_id",
)

# 5. Generate deterministic evaluation manifest
manifest = create_evaluation_manifest(
    run_id="EXTERNAL-RUN-001",
    model_name="Customer_Production_Model_v1",
    dataset_name="Customer_Operational_Staging",
    regime="EXTERNAL_LOCKED_TEST",
    metrics={k: v.value for k, v in suite.metrics.items()},
    seeds=[42],
    result_classification=ResultClassification.EXTERNAL_REAL_DATA_VALIDATION,
)

manifest_fingerprint = compute_manifest_fingerprint(manifest)
print("External Validation Complete.")
print(f"Decision: {suite.policy_result.decision.value} | State: {suite.evidence_state.value}")
print(f"Manifest SHA-256 Fingerprint: {manifest_fingerprint}")
```
