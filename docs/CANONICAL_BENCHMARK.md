# Canonical Fairness Benchmark Report
**Algorithmic Fairness Analysis Framework**  
*Educational Benchmark Report: Multi-Seed Evaluation, Baseline Hierarchy, and Paired Statistical Deltas*

---

## 1. Executive Summary & Purpose

The **Canonical Benchmark** establishes a standardized, reproducible protocol for evaluating machine-learning fairness and utility. It addresses common methodological pitfalls in fairness evaluations:
- **Feature Contracts & Leakage Prevention:** Variables are classified by operational availability before model execution. Post-outcome realizations and mechanical outcome proxies are strictly banned.
- **Model-Selection Firewall:** Evaluated partitions are write-protected and isolated from model fitting and hyperparameter selection.
- **Controlled Multi-Seed Evaluation:** Evaluates candidates across 5 deterministic random seeds to quantify variance and avoid single-split anomalies.
- **Negative Control Baselines:** Includes a trivial majority-class predictor (B0) to verify anti-degeneracy detection and prevent superficial metric satisfaction.
- **Independent Metric Verification:** Computes pure mathematical fairness identities and cross-checks results against established libraries (`scikit-learn` and `fairlearn`).
- **Cluster-Aware Uncertainty:** Supports cluster bootstrap resampling for grouped records (e.g. repeated deliveries per route/driver) to avoid artificially narrow confidence intervals.

---

## 2. Standardized Baseline Hierarchy (B0 to B4)

| Baseline ID | Architecture | Mitigation Stage | Purpose & Description |
| :--- | :--- | :--- | :--- |
| **B0** | Majority Class Baseline | None (Negative Control) | **Majority Baseline:** Trivial predictor constantly predicting training set mode. Establishes the floor for utility and verifies anti-degeneracy detection. |
| **B1** | Logistic Regression | None (Clean Contract) | **Clean Unmitigated Baseline:** Standardized pre-dispatch linear model with clean feature contracts. Reference for tradeoff calculations. |
| **B2** | Reweighted Logistic | Pre-Processing (Reweighting) | **Sample Reweighting Baseline:** Samples weighted inversely proportional to joint sensitive-target frequencies ($w(y, s) = \frac{N}{4 \cdot N(y, s)}$). |
| **B3** | Validation Threshold Mitigated | Post-Processing (Validation-Tuned) | **Equalized Odds Post-Processing:** Group-specific classification thresholds tuned strictly on the **Validation Set** to equalize error rates. |
| **B4** | Tuned XGBoost | In-Processing (Cost-Sensitive) | **Gradient Boosted Candidate:** Depth-constrained (`max_depth=4`), feature-subsampled (`colsample=0.85`), class-imbalance-weighted gradient boosting. |

---

## 3. Prediction-Time Feature Contract Specification

Features across evaluation scenarios are cataloged in [`configs/benchmark_features.yaml`](../configs/benchmark_features.yaml):

### 3.1 Controlled Logistics Feature Classification
| Column Name | Classification | In Benchmark? | Operational Timing & Justification |
| :--- | :--- | :---: | :--- |
| `osrm_distance` | `PRE_DISPATCH_VALID` | **YES** | Planned road distance calculated prior to departure. |
| `osrm_time` | `PRE_DISPATCH_VALID` | **YES** | Pre-dispatch route duration estimated by routing engine. |
| `segment_osrm_distance` | `PRE_DISPATCH_VALID` | **YES** | Planned waypoint-to-waypoint segment distance. |
| `segment_osrm_time` | `PRE_DISPATCH_VALID` | **YES** | Planned waypoint-to-waypoint segment travel duration. |
| `time_per_distance_osrm` | `PRE_DISPATCH_VALID` | **YES** | Planned travel pace (`osrm_time / osrm_distance`). |
| `actual_time` | `POST_OUTCOME` | **NO (BANNED)** | Total realized delivery duration recorded upon package delivery. Mechanically causes target leakage. |
| `segment_actual_time` | `POST_OUTCOME` | **NO (BANNED)** | Realized transit duration recorded at waypoint check-in. |
| `actual_distance_to_destination` | `POST_OUTCOME` | **NO (BANNED)** | Realized GPS road distance traversed by vehicle. |
| `actual_vs_osrm_time_ratio` | `DERIVED_FROM_OUTCOME` | **NO (BANNED)** | Ratio of realized time to planned time. |
| `segment_deviation` | `DERIVED_FROM_OUTCOME` | **NO (BANNED)** | Realized schedule variance. |
| `is_cutoff` | `POST_OUTCOME` | **NO (TARGET)** | Outcome label indicating whether trip exceeded delivery cutoff deadline. |
| `Education_Level` | `SYNTHETIC_PROXY` | **YES (AUDIT)** | Proxy assigned for sensitivity analysis; subject to proxy caveats in [`docs/CAUSAL_ASSUMPTIONS.md`](CAUSAL_ASSUMPTIONS.md). |

---

## 4. Group Holdout Semantics & Model Firewall

1. **Partition Isolation (`GroupAwareSplitter`):**  
   - Enforces group-aware holdout splits such that grouped entities (e.g., trips, regional delivery hubs) never span train, validation, and test splits ($\text{Train} \cap \text{Val} = \text{Train} \cap \text{Test} = \text{Val} \cap \text{Test} = \emptyset$).
2. **Model-Selection Firewall ([`src/firewall.py`](../src/firewall.py)):**  
   - Encapsulates test data in write-protected memory (`flags.writeable = False`).
   - Prevents `.fit()`, `.tune()`, or `.select()` calls on test partitions, raising `ModelSelectionFirewallViolation`.

---

## 5. Multi-Seed Empirical Results

Evaluated across 5 deterministic seeds (`[42, 43, 44, 45, 46]`) on Scenario C (Explicit Group Effect - Moderate Signal, $N=3,000$) under group holdout (`GROUP_HOLDOUT`) with screening criteria ($\text{Min Bal Acc} \ge 0.55, \text{Max EODiff} \le 0.10$):

### 5.1 Primary Multi-Seed Benchmark Results

| Baseline ID | Accuracy (Mean ± Std) | Balanced Acc (Mean ± Std) | Equalized Odds Diff (Mean ± Std) | Dem. Parity Diff (Mean ± Std) | Degeneracy Status | Screening Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 Majority Baseline** | $0.5909 \pm 0.0211$ | **$0.5000 \pm 0.0000$** | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | **`DEGENERATE`** | **`POLICY_DEGENERATE`** |
| **B1 Clean Linear** | $0.8341 \pm 0.0181$ | **$0.8286 \pm 0.0189$** | $0.1882 \pm 0.0551$ | $0.1620 \pm 0.0357$ | `NORMAL` | `FLAGGED_DISPARITY` |
| **B2 Fairness Reweighted** | $0.8341 \pm 0.0189$ | **$0.8286 \pm 0.0200$** | $0.1889 \pm 0.0537$ | $0.1625 \pm 0.0352$ | `NORMAL` | `FLAGGED_DISPARITY` |
| **B3 Validation Threshold** | **$0.8411 \pm 0.0173$** | **$0.8408 \pm 0.0203$** | **$0.0713 \pm 0.0453$** | $0.1472 \pm 0.0559$ | `NORMAL` | **`MET_CRITERIA`** |
| **B4 Tuned XGBoost** | $0.8203 \pm 0.0158$ | **$0.8197 \pm 0.0152$** | $0.1798 \pm 0.0512$ | $0.1554 \pm 0.0320$ | `NORMAL` | `FLAGGED_DISPARITY` |

### 5.2 Paired-Seed Analysis: B1 vs. B3 (Within-Seed $\Delta = \text{B3} - \text{B1}$)
Evaluating within-seed paired differences isolates the algorithmic effect of threshold post-processing from split variance:
- **Equalized Odds Difference:** Paired $\Delta = \mathbf{-0.0907 \pm 0.0249}$ ($p = 0.0011$). Statistically significant 54% reduction in disparity.
- **Balanced Accuracy:** Paired $\Delta = \mathbf{+0.0014 \pm 0.0062}$ ($p = 0.6385$). Disparity reduction is achieved without utility degradation.

---

## 6. Synthetic Stress Test Findings

Detailed cross-scenario results across synthetic data generating processes (described in [`docs/BENCHMARK_VALIDITY.md`](BENCHMARK_VALIDITY.md)):

1. **Scenario A (No Group Disparity):**
   - True DGP disparity is near zero ($\text{Gap} \approx 0.001$).
   - B1 achieves $\text{EODiff} = 0.0436$ due to finite-sample noise.
   - Post-processing mitigation on sample noise inadvertently increases observed disparity ($\text{EODiff} = 0.0928$, paired $\Delta = +0.0492$).
   - *Educational Takeaway:* Do not activate threshold post-processing when the pre-mitigation disparity is statistically indistinguishable from zero.
2. **Scenario B (Feature-Mediated Disparity):**
   - Disparity arises through correlated features rather than direct group discrimination.
   - B1 exhibits $\text{EODiff} = 0.1251$; post-processing lowers disparity to $0.0612$.
3. **Scenario D (Severe Fairness Trade-off):**
   - Substantial base-rate disparity ($\Delta \approx 35\%$).
   - B1 achieves high accuracy ($0.8521$) but substantial disparity ($\text{EODiff} = 0.2201$).
   - B3 constrains $\text{EODiff}$ to $0.1142$, incurring an accuracy reduction ($0.8115$).
4. **Scenario E (Zero-Signal Predictor Negative Control):**
   - Zero predictive signal ($I(X; Y) = 0$).
   - All models degenerate to constant predictions; evaluator diagnoses the models as `DEGENERATE`.

---

## 7. Reproduction Commands

```bash
# Run full automated test suite
pytest tests/ -q

# Verify clean-environment reproducibility and metric invariants
python scripts/verify_clean_reproducibility.py

# Run Canonical Benchmark Runner (5 seeds, controlled synthetic demo)
python src/canonical_benchmark.py --demo --seeds 42,43,44,45,46

# Run via CLI
python src/cli.py benchmark --quick
```
