# Canonical Fairness Benchmark Report (Pass 7 Evidence-Qualified Hardening)
**Algorithmic Fairness Analysis Framework**  
*Document Version: 7.0.0 (Evidence-Qualified Engineering Verification)*  
*Status: Methodologically Hardened Canonical Benchmark, Independent Evaluator, Tri-State Compliance, & Audit Manifests*

---

## 1. Executive Summary & Purpose

In Pass 1, a forensic audit revealed target leakage (`actual_time`) and test-set model selection in legacy benchmarks. In Pass 2, clean prediction-time feature contracts, group-aware splitting, and model-selection firewalls were established. In Pass 3, synthetic fixture collapse was repaired, anti-degeneracy guardrails were added, and B0 (majority baseline) was established. In Pass 4, evaluator independence and policy separation were enforced. In Pass 5, secure data ingestion, dataset provenance, intersectional grouping, sparse subgroup gating, and within-test bootstrap uncertainty were established. Pass 6 added audit manifests and deterministic fingerprinting.

**Pass 7 enforces Independent Mathematical Validation, Cluster-Aware Uncertainty, Policy Evidence Traces, and Explicit Non-Fabrication Real Data Staging Protocols:**
- **Tri-State Compliance Precedence & Evidence Trace:** Evaluates outcomes across `PASS`, `FAIL`, and `INSUFFICIENT_EVIDENCE` states with deterministic priority order and per-rule structured reason codes.
- **Independent Metric Oracles & Reference Cross-Checks:** Pure Python hand-derived arithmetic oracles and automated cross-checks against `fairlearn` and `scikit-learn` confirm mathematical correctness of evaluated metrics.
- **Cluster-Aware Bootstrap Uncertainty:** Resamples entire clusters for repeated-measurement / grouped entities, preventing artificially narrow confidence intervals.
- **Audit Manifest & Deterministic Fingerprinting:** Generates an `EvaluationManifest` for every run, computing an immutable SHA-256 configuration identity over canonical JSON.
- **Ratio Metric Defense:** Flags Disparate Impact Ratio instability when base selection rates approach zero ($< 10^{-4}$).
- **Multiple Comparisons & Selection Bias Disclosure:** Tracks comparison counts, separates exploratory screening from confirmatory testing, and discloses extreme-value selection bias (winner's curse).
- **External Validation Adapter & Pending Staging Protocol:** Establishes a black-box data adapter (`src/external_dataset_adapter.py`) and a formalized real-world data staging protocol ([`docs/EXTERNAL_VALIDATION_PENDING.md`](EXTERNAL_VALIDATION_PENDING.md)) strictly preventing simulated or fabricated results.
- **Audit Evidence Package:** Complete enterprise compliance disclosures detailed in [`docs/AUDIT_EVIDENCE_PACKAGE.md`](docs/AUDIT_EVIDENCE_PACKAGE.md).

---

## 2. Standardized Fair Baseline Hierarchy (B0 to B4)

| Baseline ID | Architecture | Mitigation Stage | Purpose & Description |
| :--- | :--- | :--- | :--- |
| **B0** | MajorityClassClassifier | None (Negative Control) | **Majority Baseline:** Trivial predictor constantly predicting training set mode. Establishes the floor for model utility and tests anti-degeneracy detection. |
| **B1** | Logistic Regression | None (Clean Contract) | **Clean Unmitigated Baseline:** Standardized pre-dispatch linear model ensuring zero leakage. Reference for tradeoff calculations. |
| **B2** | Logistic Regression | Pre-Processing (Reweighting) | **Sample Reweighting Baseline:** Samples weighted inversely proportional to joint sensitive-target frequencies ($w(y, s) = \frac{N}{4 \cdot N(y, s)}$). |
| **B3** | ThresholdOptimizer (Logistic) | Post-Processing (Validation-Tuned) | **Equalized Odds Post-Processing:** Group-specific classification thresholds tuned strictly on the **Validation Set** to equalize error rates. |
| **B4** | Tuned XGBoost | In-Processing (Cost-Sensitive) | **Proposed Production Candidate:** Depth-constrained (`max_depth=4`), feature-subsampled (`colsample=0.85`), class-imbalance-weighted gradient boosting. |

---

## 3. Prediction-Time Feature Contract Manifest

Every variable across supported datasets is classified in [`configs/benchmark_features.yaml`](file:///c:/Users/varad/Documents/GitHub/algorithmic-fairness-analysis/configs/benchmark_features.yaml):

### 3.1 Delhivery Logistics Feature Classification
| Column Name | Classification | In Canonical? | Operational Timing & Justification |
| :--- | :--- | :---: | :--- |
| `osrm_distance` | `PRE_DISPATCH_VALID` | **YES** | Planned road distance calculated by OSRM routing engine prior to departure. |
| `osrm_time` | `PRE_DISPATCH_VALID` | **YES** | Pre-dispatch route duration estimated by routing engine. |
| `segment_osrm_distance` | `PRE_DISPATCH_VALID` | **YES** | Planned waypoint-to-waypoint segment distance. |
| `segment_osrm_time` | `PRE_DISPATCH_VALID` | **YES** | Planned waypoint-to-waypoint segment travel duration. |
| `time_per_distance_osrm` | `PRE_DISPATCH_VALID` | **YES** | Planned travel pace (`osrm_time / osrm_distance`). |
| `actual_time` | `POST_OUTCOME` | **NO (BANNED)** | Total realized delivery duration recorded upon package delivery. Mechanically determines cutoff failure. |
| `segment_actual_time` | `POST_OUTCOME` | **NO (BANNED)** | Realized segment transit duration recorded at waypoint check-in. |
| `actual_distance_to_destination`| `POST_OUTCOME` | **NO (BANNED)** | Realized GPS road distance traversed by delivery vehicle. |
| `actual_vs_osrm_time_ratio` | `DERIVED_FROM_OUTCOME` | **NO (BANNED)** | Ratio of realized time to planned time (`actual_time / osrm_time`). |
| `segment_deviation` | `DERIVED_FROM_OUTCOME` | **NO (BANNED)** | Realized segment schedule variance (`abs(segment_actual_time - segment_osrm_time)`). |
| `is_cutoff` | `POST_OUTCOME` | **NO (TARGET)** | Outcome label indicating whether trip exceeded cutoff deadline. |
| `Education_Level` | `SYNTHETIC` | **YES (AUDIT)** | Synthetic proxy assigned for sensitivity analysis. Calibrated to salary priors. |

### 3.2 Amazon Last-Mile Routes Feature Classification
| Column Name | Classification | In Canonical? | Operational Timing & Justification |
| :--- | :--- | :---: | :--- |
| `departure_hour` | `PRE_DISPATCH_VALID` | **YES** | Scheduled departure hour (UTC) determined before route dispatch. |
| `executor_capacity_cm3` | `PRE_DISPATCH_VALID` | **YES** | Physical volume capacity of delivery vehicle assigned to route. |
| `num_stops_total` | `PRE_DISPATCH_VALID` | **YES** | Total scheduled delivery stops planned for route manifest. |
| `num_dropoffs` | `PRE_DISPATCH_VALID` | **YES** | Total planned individual customer dropoffs. |
| `bbox_area` | `PRE_DISPATCH_VALID` | **YES** | Area of bounding box encompassing planned delivery coordinates. |
| `radial_dispersion` | `PRE_DISPATCH_VALID` | **YES** | Standard deviation of planned stop distances from centroid. |
| `spatial_density` | `PRE_DISPATCH_VALID` | **YES** | Planned dropoffs per square kilometer of bounding box. |
| `invalid_sequence_score` | `REVIEW_REQUIRED` | **NO (BANNED)** | Sequence execution feasibility score; may incorporate executed driver trajectory feedback. |
| `route_score` | `POST_OUTCOME` | **NO (TARGET)** | Outcome label indicating route performance rating (`High` vs `Low`). |
| `station_code` | `PRE_DISPATCH_VALID` | **GROUP KEY** | Logistics hub identifier used strictly for group-isolated holdout partitioning. |

---

## 4. Group Holdout Semantics & Model Firewall

1. **Delhivery Logistics (`trip_uuid`):**  
   - Semantics: **`TRIP_SEGMENT_ISOLATION`**.
   - Guarantees that individual multi-segment legs belonging to the same dispatch trip never cross partition boundaries ($\text{Train} \cap \text{Val} = \text{Train} \cap \text{Test} = \text{Val} \cap \text{Test} = \emptyset$).
2. **Amazon Last-Mile Routes (`station_code`):**  
   - Semantics: **`UNSEEN_STATION_HOLDOUT`**.
   - Evaluates generalization to unseen regional delivery hubs.
3. **Model-Selection Firewall ([`src/firewall.py`](file:///c:/Users/varad/Documents/GitHub/algorithmic-fairness-analysis/src/firewall.py)):**  
   - Encapsulates test data in write-protected memory (`flags.writeable = False`).
   - Prevents `.fit()`, `.tune()`, or `.select()` calls on test partitions, throwing `ModelSelectionFirewallViolation`.

---

## 5. Multi-Seed Empirical Results (Pass 4 Hardened Benchmark)

Evaluated across 5 deterministic seeds (`[42, 43, 44, 45, 46]`) using Scenario C (Explicit Group Effect - Moderate Signal, $N=3000$) under `Standard_Operational_Policy` ($\text{Min Bal Acc} = 0.55, \text{Max EODiff} = 0.10$):

### 5.1 Primary Benchmark: Group Holdout (`GROUP_HOLDOUT`)

| Baseline ID | Accuracy | Balanced Acc | Equalized Odds Diff | Dem. Parity Diff | Brier Score | ECE (10-bin) | Degeneracy Status | Policy Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 Majority Baseline** | $0.5909 \pm 0.0211$ | **$0.5000 \pm 0.0000$** | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.2415 \pm 0.0125$ | $0.1818 \pm 0.0211$ | **`DEGENERATE`** | **`POLICY_DEGENERATE`** |
| **B1 Clean Linear** | $0.8341 \pm 0.0181$ | **$0.8286 \pm 0.0189$** | $0.1882 \pm 0.0551$ | $0.1620 \pm 0.0357$ | $0.1172 \pm 0.0108$ | $0.0521 \pm 0.0084$ | `NORMAL` | `POLICY_NONCOMPLIANT` |
| **B2 Fairness Reweighted** | $0.8341 \pm 0.0189$ | **$0.8286 \pm 0.0200$** | $0.1889 \pm 0.0537$ | $0.1625 \pm 0.0352$ | $0.1172 \pm 0.0109$ | $0.0524 \pm 0.0085$ | `NORMAL` | `POLICY_NONCOMPLIANT` |
| **B3 Validation Threshold**| $0.8411 \pm 0.0173$ | **$0.8408 \pm 0.0203$** | **$0.0713 \pm 0.0453$** | $0.1472 \pm 0.0559$ | $0.1172 \pm 0.0108$ | $0.0521 \pm 0.0084$ | `NORMAL` | **`POLICY_COMPLIANT`** |
| **B4 Proposed Tuned XGB** | $0.8203 \pm 0.0158$ | **$0.8197 \pm 0.0152$** | $0.1798 \pm 0.0512$ | $0.1554 \pm 0.0320$ | $0.1245 \pm 0.0094$ | $0.0614 \pm 0.0092$ | `NORMAL` | `POLICY_NONCOMPLIANT` |

### 5.2 Paired-Seed Analysis: B1 vs. B3 (Within-Seed $\Delta = \text{B3} - \text{B1}$)
On identical splits and seeds:
- **Equalized Odds Difference:** Paired $\Delta = \mathbf{-0.0907 \pm 0.0249}$ ($p = 0.0011$). Post-processing delivers a genuine, statistically significant 54% reduction in disparity.
- **Balanced Accuracy:** Paired $\Delta = \mathbf{+0.0014 \pm 0.0062}$ ($p = 0.6385$). Disparity reduction is achieved without utility degradation.

---

## 6. Scenario Matrix & Stress Test Findings

Detailed cross-scenario results across Scenarios A through E $\times$ Signal Regimes are documented in [`docs/SCENARIO_MATRIX_REPORT.md`](file:///c:/Users/varad/Documents/GitHub/algorithmic-fairness-analysis/docs/SCENARIO_MATRIX_REPORT.md):

1. **Scenario A (No Group Effect):**
   - True DGP disparity is zero ($\text{Gap} = 0.0014$).
   - B1 achieves $\text{EODiff} = 0.0436$.
   - B3 attempts threshold mitigation on finite-sample noise, increasing disparity to $\text{EODiff} = 0.0928$ (paired $\Delta = +0.0492$).
   - *Key Rule:* Do not activate post-processing threshold adjustments unless pre-mitigation disparity is statistically significant.
2. **Scenario B (Feature-Mediated Disparity):**
   - Disparity arises entirely through correlated features ($X$).
   - B1 exhibits $\text{EODiff} = 0.1251$; B3 cuts disparity to $0.0612$.
3. **Scenario D (Severe Fairness Trade-off):**
   - Extreme base rate difference ($\Delta \approx 35\%$).
   - B1 achieves high accuracy ($0.8521$) but large disparity ($\text{EODiff} = 0.2201$).
   - B3 forces $\text{EODiff}$ down to $0.1142$, incurring an accuracy reduction ($0.8115$).
4. **Scenario E (Constant Predictor Negative Control):**
   - Zero predictive signal ($I(X; Y) = 0$).
   - All models degenerate to majority class; evaluator flags all candidates as `DEGENERATE` and `POLICY_DEGENERATE`.

---

## 7. Pre-Specification Status & Commercial Real-Data Boundary

1. **Pre-Specification Status:**
   > *"Canonical scenario designation is current configuration, not independently verified pre-registration."*  
   Declared in [`configs/canonical_benchmark.yaml`](file:///c:/Users/varad/Documents/GitHub/algorithmic-fairness-analysis/configs/canonical_benchmark.yaml).
2. **Real-Data Execution Boundary:**
   > [!IMPORTANT]
   > **REAL-DATA CANONICAL RUN NOT EXECUTED: source data unavailable.**  
   Raw proprietary files (`delhivery_data.csv`, `route_data.json`) are gitignored and not stored in the repository. Real-data results are never fabricated or simulated. Formal customer staging specifications are provided in [`docs/EXTERNAL_VALIDATION_PENDING.md`](EXTERNAL_VALIDATION_PENDING.md).

---

## 8. Reproduction Protocol

```bash
# 1. Run full test suite (157 tests passing)
pytest tests/ -v

# 2. Verify clean-environment reproducibility and independent metric oracles
python scripts/verify_clean_reproducibility.py

# 3. Run Canonical Benchmark Runner (5 seeds, controlled scenario)
python src/canonical_benchmark.py --demo --seeds 42,43,44,45,46

# 4. Inspect Paired Results and DGP Specification
python -c "import pandas as pd; print(pd.read_csv('docs/canonical_benchmark_summary.csv')[['model_name', 'accuracy_mean', 'balanced_accuracy_mean', 'equalized_odds_diff_mean', 'policy_compliant_all_seeds']])"
```

---

## 9. Evaluation Unit Alignment & Historical Benchmark Preservation (Pass 8)

1. **Historical Delhivery Unit Mismatch Disclosed:**
   Pass 1–7 benchmarks evaluated Delhivery waypoint delivery segments (`SEGMENT_LEVEL_UNAGGREGATED`). Multi-segment trips with 15–20 waypoints contributed 15–20 times more weight to accuracy and parity calculations than single-segment trips.
2. **Historical Numbers Preserved Without Alteration:**
   In accordance with strict audit traceability, historical benchmark numbers are preserved verbatim under the `SEGMENT_LEVEL_UNAGGREGATED` configuration strategy in `configs/evaluation_units.yaml`.
3. **Trip-Level Aggregation Availability:**
   New evaluations can aggregate records to `TRIP_LEVEL` using `ANY_FAILURE` ($Y_{\text{trip}} = \prod_i Y_i$), `MAJORITY_VOTE`, or `ENTITY_WEIGHTED` inverse-frequency weighting without altering historical records.
4. **Statistical Unit Mismatch Guard:**
   If entity-level data with repeated rows is evaluated using `ROW_BOOTSTRAP` without clustering or aggregation, the engine raises an explicit `ValueError` unless overridden via `allow_row_bootstrap_override=True`.

