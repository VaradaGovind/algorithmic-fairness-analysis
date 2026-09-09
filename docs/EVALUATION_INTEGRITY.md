# Evaluation Integrity & Audit Architecture

## 1. Overview & Motivation

Algorithmic fairness evaluations frequently suffer from subtle methodological failures that undermine their commercial and scientific credibility:
1. **Target and Post-Outcome Leakage:** Features that encode or occur chronologically after the prediction decision inflate accuracy scores artificially.
2. **Synthetic Sensitive Attribute Contamination:** When demographic attributes are synthetically constructed from predictive model features, observed fairness disparities become mathematical artifacts of the feature distribution rather than evidence of societal discrimination.
3. **Evaluation Split Contamination:** Performing hyperparameter optimization, threshold tuning, or model selection directly on the holdout test set introduces severe selection bias and over-optimistic performance estimates.

This document formalizes the evaluation integrity architecture implemented in the repository.

---

## 2. Leakage Detection Engine (`src/evaluation_integrity.py`)

The leakage audit scans dataset features and targets across six automated screening categories:

```mermaid
graph TD
    A[Input Features & Target] --> B[Exact Target Duplication Check]
    A --> C[Inverse Target Duplication Check]
    A --> D[Post-Outcome Execution Feature Check]
    A --> E[Semantic Keyword Heuristic Inspection]
    A --> F[Extreme Correlation Screening |r| >= 0.98]
    A --> G[Cross-Split Record Overlap Audit]
```

### Classification Taxonomy
- **`CONFIRMED_LEAKAGE`:** Definitively invalidates evaluation (e.g. identical target copies, post-outcome delivery durations).
- **`SUSPICIOUS_FEATURE`:** Features requiring domain expert review (e.g. outcome keywords or extreme statistical correlations).
- **`DOMAIN_REVIEW_REQUIRED`:** Features that may represent valid strong predictors or invalid leakage depending on dispatch timeline.
- **`CLEAN`:** Pre-dispatch operational features with no cross-split leakage.

### Confirmed Leakage Findings in Legacy Logistics Pipeline
- **Delhivery Logistics:** The target `Task_Success` was defined as trip completion within scheduled cutoff (`~is_cutoff`). The input features included `actual_time`, `segment_actual_time`, and `actual_vs_osrm_time_ratio`. Because `actual_time` is the post-trip realization whose deadline defines `is_cutoff`, using it to predict cutoff completion constitutes severe target leakage.

---

## 3. Sensitive Attribute Provenance & Negative Controls (`src/sensitive_audit.py`)

### Provenance Registry
Every sensitive attribute is classified into exactly one category:

| Dataset | Attribute Name | Classification | Inputs Used | Real Demographic? | Causal Meaning? |
|---|---|---|---|:---:|:---:|
| **Delhivery Logistics** | `Education_Level` | `SYNTHETIC` | `osrm_distance`, `actual_time`, `segment_osrm_distance` | No | No |
| **Amazon Last-Mile** | `Education_Level` | `SYNTHETIC` | `num_dropoffs`, `bbox_area`, `invalid_sequence_score` | No | No |
| **Adult Income** | `Education_Group` | `REAL_OBSERVED` | `education-num` (Census survey) | Yes | Yes (Assumption-dependent) |

### Negative Controls Suite
To verify whether observed fairness disparities reflect genuine demographic disparities or evaluation artifacts, three negative controls are executed:
1. **`RANDOM_GROUP_CONTROL`:** Replaces the sensitive attribute with independent random group draws matching empirical marginal probabilities. Establishes the expected finite-sample stochastic fairness gap.
2. **`PERMUTED_SENSITIVE_ATTRIBUTE_CONTROL`:** Randomly permutes group labels across instances ($N=100$ permutations). Computes an empirical p-value for the observed fairness gap against the null hypothesis of zero disparity.
3. **`FEATURE_DERIVED_GROUP_CONTROL`:** Deliberately constructs group labels from median splits of model features, demonstrating how feature correlation mechanically generates spurious disparities.

---

## 4. Split Hierarchy Enforcement (`src/fairness_evaluator.py`)

To eliminate test-set optimization bias, the evaluation pipeline enforces a strict 3-way split:

```text
Full Dataset
   │
   ├── Train Set (60%)           ──> Feature scaling, imputer fitting, candidate model training
   │
   ├── Validation / Dev Set (20%) ──> Hyperparameter tuning (HPO), ThresholdOptimizer fitting, Model selection
   │
   └── Locked Holdout Test (20%)  ──> Evaluated STRICTLY ONCE on selected model for unbiased reporting
```

### Prohibited Operations on Locked Test Set
- Hyperparameter optimization
- Feature selection / RFE
- Fairness threshold tuning (`ThresholdOptimizer`)
- Model selection (`max(candidates, key=...)`)

---

## 5. Defensive Fairness Evaluator

The fairness evaluator recomputes all metrics directly from raw arrays without relying on training code assumptions:
- **Zero Positive Subgroup:** Returns `ZERO_POSITIVE_SUBGROUP` with `NaN` (does not silently invent 0.0 or 1.0).
- **Zero Negative Subgroup:** Returns `ZERO_NEGATIVE_SUBGROUP` with `NaN`.
- **Tiny Group Support:** Enforces minimum group size ($N \ge 5$); returns `INSUFFICIENT_GROUP_SUPPORT` if violated.
- **Utility Refactoring:** Refactors composite score into an explicit user-defined utility function `user_defined_utility_score(balanced_acc, fairness_gap, alpha=0.25)`.
- **Tradeoff Refactoring:** Refactors welfare loss into `accuracy_tradeoff_proxy(baseline_acc, mitigated_acc) = max(0.0, baseline_acc - mitigated_acc)`.
