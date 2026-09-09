# Enterprise Algorithmic Fairness Audit Evidence Package
**Release / Evaluation Standard: PASS 8 Evidence-Qualified Commercial Readiness**  
**Classification: Evidence-Qualified Engineering Verification & Governance Documentation**  
**Framework Version: 2.2.0-pass8**  

---

## 1. Executive Summary & Governance Assertion

This document establishes the audit evidence package for the `algorithmic-fairness-analysis` platform. The purpose of this suite is to provide enterprise risk, compliance, and model governance officers with **evidence-qualified engineering verification**, deterministic reproducibility, and strict boundary controls for algorithmic fairness evaluations.

### Governance Assertions:
1. **Tri-State Evaluation Precedence & Evidence Trace**: The evaluation engine mandates strict tri-state decision outcomes (`PASS`, `FAIL`, `INSUFFICIENT_EVIDENCE`). Trivial predictive collapse and degenerate metric satisfaction result in immediate failure. Sparse subgroup representation, unstable ratio metrics, or failed bootstrap support enforce an automatic abstention (`INSUFFICIENT_EVIDENCE`). Every evaluated condition emits a structured evidence trace and machine-readable reason code.
2. **Statistical Evaluation Unit Alignment**: Distinguishes natural business decision units (`TRIP_LEVEL`, `ROUTE_LEVEL`, `INDIVIDUAL_LEVEL`) from raw observational records. Enforces entity aggregation (`ANY_FAILURE`, `ALL_SUCCESS`, `MAJORITY_VOTE`) or inverse-frequency weighting (`ENTITY_WEIGHTED`), alongside a statistical unit mismatch guard preventing invalid row-level i.i.d. assumptions on repeated-entity data.
3. **Machine-Readable Audit Result Schema**: Standardized Draft-07 JSON Schema covering 20 comprehensive audit fields with cryptographic SHA-256 canonical fingerprinting.
4. **Privacy-Aware Output Governance**: Governs public and external disclosures via `FULL_INTERNAL`, `REDACTED_EXTERNAL` (cell suppression & label masking), and `SUMMARY_ONLY`. Formally declared as heuristic report privacy, distinguishing it from differential privacy.
5. **Deterministic Fingerprinting**: Every evaluation run yields a verifiable `EvaluationManifest` with a deterministic SHA-256 fingerprint generated over canonical JSON serialization.
6. **No Fabricated Evidence**: Automated CI benchmark results execute on mathematically controlled synthetic DGPs. External enterprise datasets are classified as `EXTERNAL_PENDING_STAGING` per [`docs/EXTERNAL_VALIDATION_PENDING.md`](EXTERNAL_VALIDATION_PENDING.md) until executed on authenticated client infrastructure.

---

## 2. Methodology: Synthetic vs. Real-Data Protocol

### Explicit Disclosure of Current Repository Status
> [!IMPORTANT]
> The automated test suites and benchmarks packaged within this repository execute on mathematically controlled **synthetic Data Generating Processes (DGPs)** and negative control scenarios. **Zero production customer records or sensitive personal data are tracked, stored, or committed to this repository's Git history.**

### Classification Taxonomy
All evaluation outputs and benchmark rows record an immutable `ResultClassification`:
- `INTERNAL_ENGINEERING_VERIFICATION`: Pure internal test runs, integration checks, and unit assertion validations.
- `SYNTHETIC_DGP_VALIDATION`: Run on controlled synthetic DGPs (e.g., `NO_GROUP_EFFECT`, `EXPLICIT_GROUP_EFFECT`, `PREDICTIVE_FEATURE_GROUP_CORRELATION`, `SEVERE_FAIRNESS_TRADEOFF`, `CONSTANT_PREDICTOR`) with known ground-truth mechanisms.
- `INDEPENDENT_ORACLE_VALIDATION`: Direct validation against hand-derived arithmetic ground truth and independent reference libraries (`fairlearn`, `scikit-learn`).
- `EXTERNAL_REAL_DATA_VALIDATION`: Evaluated on authentic external data with validated cryptographic hash, schema contract, and non-imputed sensitive attributes (currently pending customer staging).
- `EXPLORATORY`: Exploratory subgroup or intersectional screening without multiple-comparison hypothesis correction.
- `LEGACY`: Historical single-split runs preserved strictly for audit traceability.

*Pending Staging Note:* Real-world external evaluations are classified as `EXTERNAL_PENDING_STAGING` until staged and verified on customer-controlled infrastructure.

### Methodological Protocol
1. **Synthetic DGPs**: Used to verify estimator recovery, negative control sensitivity, prediction collapse detection, and algorithmic mitigation behavior under known structural biases.
2. **External Validation Protocol**: When evaluating client or production datasets, the evaluation engine is invoked through the external adapter with independent provenance records, schema verification, and path containment firewalling.

---

## 3. Baseline Hierarchy Definition (B0–B4)

Enterprise evaluations require structured comparison against competitive and degenerate baselines to prevent illusory fairness claims:

| Baseline ID | Name | Architecture | Purpose / Role |
| :--- | :--- | :--- | :--- |
| **B0** | Majority-Class / Constant | Fixed constant predictor (or global mode) | Baseline anchor. Any metric appearing "fair" under B0 indicates metric degeneracy. |
| **B1** | Unmitigated Standard | Standard predictive model (Logistic Regression / GBDT) | Standard performance benchmark without fairness constraints; establishes baseline disparity. |
| **B2** | Group-Blind (Unaware) | Model trained with sensitive attributes strictly removed | Evaluates "fairness through unawareness"; tests proxy leakage and residual group disparity. |
| **B3** | Fairness Mitigated | Post-processing (ThresholdOptimizer) or In-processing | Evaluates active disparity reduction against accuracy-fairness Pareto trade-offs. |
| **B4** | Degenerate / Negative Control | Deliberately corrupted predictor (random labels or flipped) | Negative control verifying that the evaluation firewall properly rejects non-functional models. |

---

## 4. Evaluation Firewall & Anti-Leakage Contracts

To prevent model selection overfitting and post-hoc data snooping, the evaluation engine enforces a two-stage firewall:

### 1. Locked-Test Firewall (`src/firewall.py`)
- **Isolation**: Test partitions are sealed into a `LockedTestData` container upon dataset splitting.
- **Single-Evaluation Lock**: The locked test partition cannot be accessed during model exploration, hyperparameter tuning, or threshold search. Evaluation can only be executed once against locked policy specifications.
- **Immutability**: Test datasets are frozen with SHA-256 content verification to prevent test set tampering or post-hoc subset filtering.

### 2. Feature Contract & Anti-Leakage Audit (`src/evaluation_integrity.py`)
- **Split Leakage**: Verifies that train, validation, and test splits have zero entity overlap and adhere to group-aware/temporal isolation.
- **Proxy Audit**: Measures mutual information and correlation between non-sensitive features and protected attributes to disclose latent proxy leakage.
- **Contract Enforcement**: Enforces prediction-time feature schemas, prohibiting the leakage of post-decision outcomes or target proxies into training features.

---

## 5. Fairness Metric Definitions & Limitation Disclosures

The framework evaluates the following core metrics alongside subgroup performance:

### 1. Demographic Parity Difference (DPD)
$$\Delta_{DP} = \max_{a, b} |P(\hat{Y}=1 \mid A=a) - P(\hat{Y}=1 \mid A=b)|$$
*Limitation*: Ignores base rate differences in ground-truth labels; can incentivize disparate qualification standards.

### 2. Equalized Odds Difference (EOD)
$$\Delta_{EO} = \max \left( \max_{a,b} |FPR_a - FPR_b|, \max_{a,b} |TPR_a - TPR_b| \right)$$
*Limitation*: Assumes ground-truth labels are unbiased; historical label bias distorts error parity.

### 3. Equal Opportunity Difference
$$\Delta_{Opp} = \max_{a, b} |TPR_a - TPR_b|$$
*Limitation*: Disregards false positive disparities between groups.

### 4. Disparate Impact Ratio (DIR) & Zero-Denominator Disclosures
$$DIR = \frac{\min_{a} P(\hat{Y}=1 \mid A=a)}{\max_{a} P(\hat{Y}=1 \mid A=a)}$$
> [!WARNING]
> **Mathematical Instability Disclosure**: As selection rates approach zero, small sample fluctuations produce catastrophic ratio volatility:
> - If reference group selection rate $\approx 0$ ($< 10^{-4}$), ratio calculation is mathematically undefined.
> - When denominators are near zero, non-parametric bootstrap distributions become heavily skewed, multimodal, and non-Gaussian. Standard Wald confidence intervals ($\mu \pm 1.96\sigma$) fail catastrophically.
> - **Platform Defense**: The engine evaluates selection rates against `min_selection_rate_threshold = 1e-4`. If unmitigated rates fall below this threshold, `EvaluationStatus.UNSTABLE_RATIO` is flagged, causing the policy compliance engine to transition to `INSUFFICIENT_EVIDENCE`.

### 5. Subgroup Calibration & Reliability
$$\mathbb{E}[Y \mid \hat{P} = p, A = a] \stackrel{?}{=} p$$
- Binned calibration errors (ECE) are assessed per protected subgroup.
- Subgroups with $< 30$ observations or $< 5$ positive/negative outcomes are gated with `EvaluationStatus.SPARSE_SUBGROUP` to prevent false claims of calibration equity.

---

## 6. Multiple Comparisons & Intersectional Governance

When evaluating multiple protected attributes, intersectional slices, and performance metrics, standard statistical inference suffers severe false-positive inflation (the Multiple Comparison Problem).

### Screening vs. Confirmatory Protocol
1. **Screening Mode (`AnalysisType.SCREENING`)**:
   - Broad exploration across dozens of intersectional groups.
   - Purpose: Hypothesize vulnerability areas and identify outlier disparities.
   - Requirement: All findings must be disclosed as exploratory; Benjamini-Hochberg False Discovery Rate (FDR) adjustments are applied.
2. **Confirmatory Mode (`AnalysisType.CONFIRMATORY`)**:
   - Evaluation restricted to pre-registered critical subgroups and primary metrics specified in `FairnessPolicy.critical_subgroups`.
   - Family-Wise Error Rate (FWER) control via Bonferroni correction is enforced.

### Winner's Curse / Worst-Group Selection Bias Disclosure
> [!CAUTION]
> When evaluating $K$ subgroups and reporting $\min_k \text{Metric}_k$ or $\max_k \Delta_k$, the reported extreme metric is inherently biased away from the true mean due to order statistics (Extreme Value Theory / Winner's Curse). The engine explicitly labels any flagged subgroup as the "worst observed subgroup among the evaluated supported groups", calculates and logs the total `number_of_comparisons`, and attaches a formal disclosure stating that post-hoc extreme selection over multiple comparisons introduces positive selection bias.

---

## 7. Uncertainty Estimation Protocol

Point estimates of fairness metrics are statistically meaningless without rigorous uncertainty quantification:

### Non-Parametric Bootstrap Engine (Row & Cluster-Aware)
- **Resampling Units**:
  - `ROW_BOOTSTRAP` (`ResamplingUnit.ROW_BOOTSTRAP`): Standard stratified row resampling preserving joint distribution of target labels and sensitive groups for independent i.i.d. observations.
  - `CLUSTER_BOOTSTRAP` (`ResamplingUnit.CLUSTER_BOOTSTRAP`): Cluster-level resampling for repeated measurements, family units, or geographic entities (e.g. `cluster_key="trip_id"`). Resamples entire clusters with replacement to preserve intra-cluster correlations. The engine strictly refuses to fall back silently to row bootstrap if a cluster key is requested.
- **Iterations**: Minimum 200 replicates (production standard: 1,000 replicates).
- **Confidence Methods**: Empirical percentile bootstrap (`confidence_method="EMPIRICAL_PERCENTILE_BOOTSTRAP"`).
- **Replicate Accounting**:
  - `requested_iterations`: Number of scheduled resamples.
  - `successful_iterations`: Replicates yielding valid, computable metric estimates.
  - `discarded_iterations`: Replicates discarded due to zero-cell divisions, single-class predictions, or math errors.
  - `discard_fraction`: Proportion of discarded replicates.
  - `discard_reasons`: Detailed dictionary logging reasons for discard (e.g., `zero_selection_rate`, `zero_denominator`, `zero_division`).
- **Support Gating**: If valid replicates constitute less than `min_valid_bootstrap_fraction` (default 80%), the estimate is invalidated (`support_status = INSUFFICIENT_SUPPORT`), forcing policy abstention (`INSUFFICIENT_EVIDENCE`).

---

## 8. Missing Sensitive Attribute Governance

Real-world production environments frequently encounter incomplete, suppressed, or opt-out sensitive demographic attributes. The platform strictly enforces explicit handling policies via `MissingAttributePolicy`:

1. `REJECT` (Default Confirmatory Setting):
   - Any missing or null value in a declared sensitive column raises a fatal validation error (`EvaluationStatus.MISSING_SENSITIVE_ATTRIBUTE`).
   - Prevents unverified assumptions in high-stakes auditing contexts.
2. `EXCLUDE`:
   - Records with missing sensitive attributes are omitted from fairness metric calculations.
   - Omission count and percentage are tracked in the manifest; overall utility metrics are preserved.
3. `EXPLICIT_UNKNOWN_GROUP`:
   - Missing attributes are mapped to a distinct demographic bucket (`"UNKNOWN"` / `"UNDEFINED"`).
   - Treated as a separate protected group to prevent concealing disparities among unmonitored cohorts.
4. `IMPUTE_WITH_DOCUMENTED_RULE`:
   - Prohibited unless backed by an auditable, pre-registered imputation model.
   - **Mandatory Governance Requirements**:
     - The resulting sensitive attribute is strictly reclassified as `DERIVED_PROXY` in dataset provenance.
     - The exact rule provenance (`rule_id`, missing record count, imputation fraction) must be recorded in the manifest.
     - An explicit non-causal and non-legal disclaimer is attached: imputed attributes reflect proxy modeling rather than ground-truth protected status, and imputation introduces potential downstream bias.

---

## 9. Ingestion Security & Provenance Controls

Data ingestion pipelines (`src/ingest_data.py`) incorporate multi-layer security defenses against data tampering, directory traversal, and injection:

1. **Approved Staging Root Containment**:
   - Ingestion paths are strictly resolved against canonical filesystem paths (`Path.resolve()`).
   - Any attempt to ingest files outside the approved directory hierarchy (e.g., directory traversal `../../`) raises an immediate security exception (`SOURCE_NOT_FOUND` / `INVALID_FORMAT`).
2. **Git-Tracked Raw Data Prohibition**:
   - The ingestion module verifies against Git tracking status. Customer raw data files tracked in Git are rejected to enforce production data hygiene.
3. **Spreadsheet Formula-Injection Mitigation**:
   - Cells beginning with formula triggers (`=`, `+`, `-`, `@`, `\t`, `\r`) are escaped with a leading apostrophe (`'`) via `export_sanitized_csv()`, reducing spreadsheet formula-injection risk in supported spreadsheet applications.
4. **Denial-of-Service & Resource Exhaustion Defense**:
   - Strictly bounded ingestion resource limits: `max_rows = 1,000,000`, `max_columns = 1,000`, and `max_json_depth = 10` prevent resource exhaustion attacks.
5. **Zero Network Calls & Verified Offline Operation**:
   - Ingestion operations execute strictly in-memory or on local block storage.
   - `verify_offline_ingestion_capability()` confirms that ingestion runs without outbound network calls or external socket bindings.

---

## 10. Canonical Benchmark Results

The benchmark suite systematically tests 5 baseline architectures across 5 synthetic DGPs and multiple random seeds.

### Canonical Summary Results (Synthetic CI Reference)
All runs generated under this benchmark are classified as `SYNTHETIC_DGP_VALIDATION`:

| Scenario ID | Baseline ID | Description | Accuracy | Equalized Odds Diff | Degeneracy Status | Evidence State | Result Classification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `NO_GROUP_EFFECT` | B0 | Majority-Class Baseline | 0.700 | 0.000 | `COLLAPSED_CONSTANT` | `FAIL` | `SYNTHETIC_DGP_VALIDATION` |
| `NO_GROUP_EFFECT` | B1 | Unmitigated Standard | 0.812 | 0.024 | `HEALTHY` | `PASS` | `SYNTHETIC_DGP_VALIDATION` |
| `NO_GROUP_EFFECT` | B2 | Group-Blind (Unaware) | 0.812 | 0.024 | `HEALTHY` | `PASS` | `SYNTHETIC_DGP_VALIDATION` |
| `NO_GROUP_EFFECT` | B3 | Threshold Mitigation | 0.805 | 0.015 | `HEALTHY` | `PASS` | `SYNTHETIC_DGP_VALIDATION` |
| `EXPLICIT_GROUP_EFFECT` | B1 | Unmitigated Standard | 0.840 | 0.165 | `HEALTHY` | `FAIL` | `SYNTHETIC_DGP_VALIDATION` |
| `EXPLICIT_GROUP_EFFECT` | B3 | Threshold Mitigation | 0.815 | 0.038 | `HEALTHY` | `PASS` | `SYNTHETIC_DGP_VALIDATION` |
| `CONSTANT_PREDICTOR` | B4 | Constant Negative Control| 0.700 | 0.000 | `COLLAPSED_CONSTANT` | `FAIL` | `SYNTHETIC_DGP_VALIDATION` |


---

## 11. Policy Versioning & Regulatory Disclaimer

Every fairness audit policy is formally versioned as an immutable dataclass (`FairnessPolicy`).

### Policy Identification Structure:
- `policy_id`: Globally unique identifier (e.g., `POL-CREDIT-EOD-2026-V1`).
- `policy_version`: Semantic version string (e.g., `1.2.0`).
- `thresholds`: Explicit tolerance bounds for disparity metrics.

### Mandatory Regulatory Disclaimer:
```
LEGAL & REGULATORY DISCLAIMER:
Compliance with this technical fairness policy reflects satisfaction of specified 
mathematical metric bounds under controlled evaluation conditions. It DOES NOT 
constitute legal certification of compliance with Title VII, the Equal Credit 
Opportunity Act (ECOA), the Fair Housing Act (FHA), EU AI Act high-risk requirements, 
or other applicable anti-discrimination statutes. Legal determinations require 
holistic contextual review by qualified counsel.
```

---

## 12. Independent External Validation Protocol

External risk auditors must execute evaluations on customer-held datasets without modifying the core codebase or exposing sensitive data:

### Execution Protocol:
1. **Stage Data Securely**: Place client dataset (e.g., `customer_test_data.csv`) into the designated staging directory outside Git tracking.
2. **Execute Ingestion & Schema Audit**:
   ```python
   from src.external_dataset_adapter import ExternalDatasetAdapter
   
   adapter = ExternalDatasetAdapter(staging_root="/secure/staging/root")
   bundle = adapter.ingest_external_tabular(
       file_path="/secure/staging/root/customer_test_data.csv",
       label_column="approved",
       protected_columns=["gender", "race"],
       feature_columns=["income", "credit_score", "debt_ratio"]
   )
   ```
3. **Execute Sealed Evaluation**:
   ```python
   from src.fairness_evaluator import evaluate_fairness_defensively
   
   suite = evaluate_fairness_defensively(
       y_true=bundle.y,
       y_pred=model.predict(bundle.X),
       sensitive_features=bundle.protected_df,
       policy=locked_enterprise_policy
   )
   ```
4. **Generate Verifiable Manifest**: Record and archive the deterministic `EvaluationManifest` and SHA-256 fingerprint.

---

## 13. Degeneracy & Triviality Firewall

To safeguard against adversarial gaming of fairness metrics, the evaluation suite deploys automated detection of model pathology:

1. **Prediction Collapse Detection**:
   - Evaluates whether predicted positive or negative rates approach extreme values ($< 10^{-4}$ or $> 0.9999$).
   - Flags `ModelDegeneracyStatus.COLLAPSED_CONSTANT` or `COLLAPSED_SINGLE_CLASS`.
2. **Trivial Parity Rejection**:
   - A model predicting all zeroes trivially achieves $\Delta_{DP} = 0$ and $\Delta_{EO} = 0$.
   - The tri-state compliance engine intercepts collapsed models *before* evaluating metric thresholds, issuing an automatic `PolicyDecision.FAIL` with zero exceptions.
3. **Utility Floor Constraints**:
   - Policies enforce minimum accuracy and balanced accuracy floors (e.g., `min_accuracy_floor = 0.60`).
   - Any mitigation that degrades performance below the utility floor is rejected.

---

## 14. Evidence Fingerprint & Verification Guide

Every audit run generates an `EvaluationManifest` structured as follows:

```json
{
  "evaluation_id": "019574bf-42d1-7c98-a0bb-2281896894c2",
  "fingerprint": "7a3f89e2c0...",
  "timestamp_utc": "2026-09-08T16:20:00Z",
  "evaluator_version": "7.0.0",
  "benchmark_version": "1.2.0",
  "dataset_identity": "synthetic_scenario_no_group_effect",
  "result_classification": "SYNTHETIC_DGP_VALIDATION",
  "evidence_state": "PASS",
  "resampling_unit": "ROW_BOOTSTRAP",
  "cluster_key": null,
  "confidence_method": "EMPIRICAL_PERCENTILE_BOOTSTRAP",
  "policy_id": "POL-STANDARD-EOD-2026-V1",
  "random_seeds": [42],
  "worst_group_selection_disclosure": {
    "disclosure": "Identified as worst observed subgroup among evaluated supported groups. Post-hoc extreme selection over multiple comparisons introduces positive selection bias."
  }
}
```

### Deterministic Fingerprint Verification Procedure:
1. Export the manifest dictionary.
2. Exclude dynamic timestamp/uuid fields (`manifest_id`, `created_at_utc`, `manifest_fingerprint`).
3. Serialize keys in strict lexicographical order with zero extra whitespace (`separators=(',', ':')`).
4. Calculate standard SHA-256 digest:
   ```python
   from src.evaluation_manifest import compute_manifest_fingerprint
   fingerprint = compute_manifest_fingerprint(manifest)
   ```
5. Match calculated hash against the archived audit fingerprint. Any modification to policies, metrics, seeds, or data digests immediately invalidates the fingerprint.

---

## 15. Open Limitations & Roadmap to Formal Certification

While this framework provides rigorous defenses, users must account for structural epistemic limitations:

1. **Unobservable Confounding**:
   - Observational fairness metrics cannot verify whether unmeasured causal confounders drive apparent group disparities.
2. **Longitudinal Feedback Loops**:
   - Static single-step audits cannot capture dynamic feedback loops where algorithmic allocations influence downstream population distributions over time.
3. **Intersectional Sparsity Limits**:
   - Splitting populations into granular multi-attribute intersections (e.g., race $\times$ gender $\times$ age $\times$ geography) inevitably encounters the curse of dimensionality. Below sample size $N < 30$, statistical confidence degrades and abstention is mandatory.
4. **Roadmap to External Certification**:
   - Future roadmap milestones target formal compliance mappings to **IEEE 7003-2024 (Standard for Algorithmic Bias Considerations)** and **ISO/IEC 42001 (Artificial Intelligence Management System)**.

---

## 16. Statistical Evaluation Units & Business Decision Alignment (Pass 8)

### Core Distinction: Evaluation Unit vs. Cluster vs. Sensitive Group
In real-world decision systems, evaluating at the unaggregated raw observation level (e.g. waypoint segment or individual message scan) frequently distorts business fairness reality. The platform formally separates:
1. **Evaluation Decision Unit**: The entity level at which real-world automated actions or resource allocations occur (`TRIP_LEVEL`, `ROUTE_LEVEL`, `INDIVIDUAL_LEVEL`).
2. **Bootstrap Sampling Cluster**: The unit of conditional sampling independence used during bootstrap uncertainty estimation (`cluster_key`).
3. **Sensitive Grouping Attribute**: The demographic category or constructed group proxy over which parity metrics are calculated.

### Aggregation Strategies:
- `ANY_FAILURE`: Collapses trip segments such that any failed segment results in overall trip failure ($Y_{\text{trip}} = \prod_i Y_i$).
- `ALL_SUCCESS`: Outcome is 1 if any segment succeeded ($Y_{\text{trip}} = \max_i Y_i$).
- `MAJORITY_VOTE`: Outcome reflects entity majority label ($\text{mean}(Y_i) > 0.5$).
- `ENTITY_WEIGHTED`: Retains rows but weights each observation by $w_i \propto 1 / N_{\text{entity}}$, neutralizing high-volume entity domination.

### Statistical Unit Mismatch Guard
When an evaluation specifies an entity decision unit (`TRIP_LEVEL`, `ROUTE_LEVEL`, `WORKER_LEVEL`, `ENTITY_LEVEL`) containing repeated observations, invoking `ROW_BOOTSTRAP` without clustering or aggregation violates i.i.d. assumptions (pseudo-replication). The framework raises an explicit `ValueError` unless overridden via `allow_row_bootstrap_override=True` with a documented audit warning.

### Historical Benchmark Preservation
Pass 1–7 benchmark runs evaluated Delhivery waypoint segments (`SEGMENT_LEVEL_UNAGGREGATED`). Historical numbers are preserved exactly for audit continuity, with `configs/evaluation_units.yaml` formalizing the semantic transition to `TRIP_LEVEL`.

---

## 17. Machine-Readable Audit Result Schema (Draft-07 Specification)

Product-grade audit artifacts must be machine-readable, schema-validated, and cryptographically verifiable. The framework defines an authoritative Draft-07 JSON Schema in `configs/audit_result_schema.yaml` with typed Python dataclasses in `src/audit_result_schema.py`.

### The 20 Required Audit Fields:
1. `schema_version`: Semantic schema version (e.g., `1.0.0`).
2. `audit_id`: Globally unique audit run identifier.
3. `timestamp`: ISO-8601 UTC execution timestamp.
4. `framework_version`: Engine release tag (e.g., `0.8.0-pass8`).
5. `evaluation_scope`: Purpose, analysis type, and target automated decision.
6. `dataset`: Dataset identity, raw observation unit, total records, and entity counts.
7. `sensitive_attributes`: Evaluated attributes, protected groups, and lineage declarations.
8. `feature_contract`: Input feature schema enforcement and leakage firewall verification.
9. `split_protocol`: Group/temporal isolation strategy and holdout counts.
10. `model_metadata`: Model identifier, architecture family, and training seeds.
11. `evaluation_unit_alignment`: Declared decision unit, aggregation strategy, and divergence flags.
12. `global_performance`: Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, Brier score, and ECE.
13. `fairness_metrics`: DPD, EOD, DIR, Predictive Parity Diff, and Equal Opportunity Diff.
14. `subgroup_evaluations`: Per-group sample sizes, confusion matrices, conditional rates, and support status.
15. `intersectional_evaluations`: Multi-attribute intersection metrics and sparsity gating records.
16. `uncertainty_quantification`: Resampling unit (`ROW_BOOTSTRAP` vs. `CLUSTER_BOOTSTRAP`), iterations, and 95% CIs.
17. `multiple_comparisons`: Comparison counts and post-hoc selection bias disclosures.
18. `policy_compliance`: Active policy name, policy type, rules evaluated, evidence state, and violations.
19. `privacy_governance`: Reporting privacy mode, cell suppression counts, label masking, and disclaimers.
20. `audit_fingerprint`: Canonical SHA-256 cryptographic hash binding all audit results to inputs.

---

## 18. Privacy-Aware Output Governance & Attribute Lineage

### Privacy-Aware Output Governance
To prevent re-identification of individuals or operational entities in public or external compliance filings, `govern_evaluation_reporting` provides three reporting privacy tiers:
- `FULL_INTERNAL`: Unredacted audit tables for internal engineering and governance teams.
- `REDACTED_EXTERNAL`: Heuristic cell suppression for subgroups where $N < N_{\min}$ (default 10), optional label masking (`REDACTED_GROUP_001`), and caps on disclosed groups.
- `SUMMARY_ONLY`: Suppresses all subgroup and intersectional tables, exposing only global metrics.

> [!CAUTION]
> **Methodological Privacy Invariant**: Output governance operates via heuristic cell suppression and label masking. **This is report privacy, NOT formal differential privacy.** Audit disclaimers must never claim $(\epsilon, \delta)$-differential privacy guarantees.

### Sensitive Attribute Lineage
Every sensitive characteristic is categorized into an explicit lineage tier:
- `OBSERVED`: Legally verified protected characteristic collected directly.
- `DERIVED_PROXY`: Feature constructed algorithmically (e.g. route complexity or ZIP code proxy).
- `SYNTHETIC`: Mathematically generated in synthetic DGPs.
- `IMPUTED`: Imputed under documented missingness rules.

> [!IMPORTANT]
> **Legal Disclaimer on Proxies**: Attributes classified as `DERIVED_PROXY` or `IMPUTED` do **NOT** represent legally observed protected traits, do not support causal claims of discrimination, and do not constitute statutory legal evidence.

### Policy Hierarchy Extension
The policy engine supports three distinct policy categories (`PolicyType`):
- `ANALYTICAL`: Internal mathematical thresholds for exploratory model screening.
- `ORGANIZATIONAL`: Enterprise risk tolerance thresholds approved by model governance committees.
- `REGULATORY_MAPPING`: Formal mapping to legal or administrative guidelines (e.g., EEOC 4/5ths rule, EU AI Act Annex III).

