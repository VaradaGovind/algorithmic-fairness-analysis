# Legacy Experimental Results & Research Appendix (Historical Audit Record)

> [!WARNING]
> **AUDIT CLASSIFICATION: LEGACY_NONCOMPARABLE (METHODOLOGICALLY CONTAMINATED & UNIT-MISALIGNED)**  
> The results below are preserved strictly for provenance and audit traceability from earlier project iterations. 
> A commercial-readiness forensic audit across Passes 1–9 identified that:
> 1. **Unit Misalignment (`LEGACY_NONCOMPARABLE`):** Historical evaluations operated directly on raw observation records (rows) without defining or validating statistical estimands, decision units, or grouping definitions. Results cannot be compared to confirmatory entity-level estimands.
> 2. **Delhivery Logistics:** Input features contained post-outcome execution duration (`actual_time`), trivially leaking the cutoff outcome threshold.
> 3. **Delhivery & Amazon:** Driver `Education_Level` is **100% synthetic** and was derived directly from model input features and post-outcome trip times.
> 4. **Model Selection Bias:** Best models were selected based on performance evaluated directly on the final holdout test set (`x_test`), rather than an isolated validation set.
> 5. **Welfare Loss Misnomer:** "Welfare loss" was defined merely as accuracy loss ($\max(0, \text{Acc}_{\text{base}} - \text{Acc}_{\text{mitigated}})$) rather than economic welfare.
> 
> In Pass 9, all historical un-aggregated metrics are designated `LEGACY_NONCOMPARABLE`. For production evaluation, see [`configs/evaluation_units.yaml`](../configs/evaluation_units.yaml) and [`docs/AUDIT_EVIDENCE_PACKAGE.md`](AUDIT_EVIDENCE_PACKAGE.md).

---

## Fairness Metric Definitions
- **Demographic Parity Difference:** $|P(\hat{Y}=1 \mid A=a) - P(\hat{Y}=1 \mid A=b)|$
- **Equal Opportunity Difference:** $|P(\hat{Y}=1 \mid Y=1, A=a) - P(\hat{Y}=1 \mid Y=1, A=b)|$
- **Equalized Odds Difference:** $\max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)$
- **Predictive Parity Difference:** $|\text{PPV}_a - \text{PPV}_b|$
- **Disparate Impact Ratio:** $\frac{\min_g P(\hat{Y}=1 \mid A=g)}{\max_g P(\hat{Y}=1 \mid A=g)}$ *(Exploratory screening heuristic, not legal certification)*

---

## 1. Delhivery Logistics (Historical Single-Run Legacy Evaluation)

- **Sensitive Attribute:** `Education_Level` *(Synthetic group proxy derived from distance and actual trip duration)*
- **Sensitive Source:** Synthetic (data-driven proxy)
- **Leakage Finding:** Features include `actual_time` and `segment_actual_time` (post-outcome leakage).
- **Target:** `Task_Success` = Completed before delivery cutoff (`~is_cutoff`)

### Model Comparison
| Model | Accuracy | Balanced Accuracy | Fairness Gap | Predictive Parity Diff | DI Ratio | Audit Status |
|---|---:|---:|---:|---:|---:|---|
| **Logistic Regression** | 0.7231 | 0.7558 | 0.0764 | 0.0151 | 0.7103 | Legacy baseline |
| **Reweighted Logistic** | 0.8650 | 0.6542 | 0.0347 | 0.0236 | 0.6669 | Reweighted |
| **Random Forest** | 0.9752 | 0.9492 | 0.0205 | 0.0276 | 0.7693 | Contaminated by leakage |
| **Neural Network** | 0.9359 | 0.8673 | 0.0295 | 0.0236 | 0.7526 | Contaminated by leakage |
| **Weighted XGB** | 0.9391 | 0.9374 | 0.0272 | 0.0445 | 0.7933 | Contaminated by leakage |
| **Tuned XGB (HPO)** | 0.9759 | 0.9693 | 0.0249 | 0.0364 | 0.7875 | Contaminated by leakage & test selection |
| **Equalized Odds ThresholdOptimizer** | 0.8613 | 0.7172 | 0.0245 | 0.0677 | 0.8389 | Contaminated by leakage |

- **Nominal Best Model:** Tuned XGB (HPO)
- **Reported Composite Score:** 0.9631
- **Reported Accuracy Tradeoff Proxy:** 0.0000

### Subgroup Distribution: Education Tiers (Synthetic Proxy)
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---:|---:|---:|
| Education | Bachelor's | 6355 | 0.9751 | 0.1849 |
| Education | High School | 5097 | 0.9770 | 0.1579 |
| Education | Master's | 17522 | 0.9759 | 0.2005 |

---

## 2. Adult Income Benchmark (Historical Single-Run Legacy Evaluation)

- **Sensitive Attribute:** `Education_Group` (`HigherEd` vs `NonHigherEd`)
- **Sensitive Source:** Real attribute (observed U.S. Census survey data)
- **Target:** `Task_Success` = Income $> \$50\text{K}$

### Model Comparison
| Model | Accuracy | Balanced Accuracy | Fairness Gap | Predictive Parity Diff | DI Ratio |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.7726 | 0.8019 | 0.1453 | 0.3507 | 0.6410 |
| **Reweighted Logistic** | 0.8312 | 0.7363 | 0.2070 | 0.2853 | 0.3781 |
| **Random Forest** | 0.8454 | 0.7670 | 0.1980 | 0.2196 | 0.3551 |
| **Neural Network** | 0.8196 | 0.7455 | 0.2477 | 0.2507 | 0.3435 |
| **Weighted XGB** | 0.8057 | 0.8245 | 0.1447 | 0.3367 | 0.5987 |
| **Tuned XGB (HPO)** | 0.7941 | 0.8184 | 0.1517 | 0.3539 | 0.6344 |
| **Equalized Odds ThresholdOptimizer** | 0.8156 | 0.7256 | 0.1225 | 0.3156 | 0.5547 |

- **Nominal Best Model:** Weighted XGB
- **Reported Composite Score:** 0.7883
- **Reported Accuracy Tradeoff Proxy:** 0.0000

### Counterfactual / Causal-Style Estimates (Assumption-Dependent)
| Causal Estimate | Value | Methodological Caveat |
|---|---:|---|
| **G-computation ATE** | `+0.1313` | Parametric contrast; conditions on post-treatment mediators (`occupation`, `hours-per-week`). |
| **Group-Specific HigherEd Effect** | `+0.1343` | Conditional regression contrast. |
| **IPW ATE** | `+0.1305` | Inverse propensity weighting; subject to positivity and unmeasured ability confounding. |
| **Matched Difference** | `+0.1464` | Nearest neighbor propensity matching. |

---

## 3. Amazon Last-Mile Routes (Historical Single-Run Legacy Evaluation)

- **Sensitive Attribute:** `Education_Level` *(Synthetic group proxy derived from dropoff counts, bounding box area, and invalid sequence scores)*
- **Sensitive Source:** Synthetic (data-driven proxy)
- **Target:** `Task_Success` = Route evaluation score is "High"

### Model Comparison
| Model | Accuracy | Balanced Accuracy | Fairness Gap | Predictive Parity Diff | DI Ratio |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.6288 | 0.6211 | 0.0699 | 0.0260 | 0.8190 |
| **Reweighted Logistic** | 0.6271 | 0.6063 | 0.0793 | 0.0747 | 0.7888 |
| **Random Forest** | 0.6631 | 0.6469 | 0.1169 | 0.1040 | 0.6675 |
| **Neural Network** | 0.6419 | 0.6298 | 0.0906 | 0.0344 | 0.7392 |
| **Weighted XGB** | 0.6476 | 0.6379 | 0.0945 | 0.0607 | 0.7432 |
| **Tuned XGB (HPO)** | 0.6631 | 0.6562 | 0.1145 | 0.0887 | 0.7181 |
| **Equalized Odds ThresholdOptimizer** | 0.6198 | 0.6026 | 0.0652 | 0.0224 | 0.8663 |

- **Nominal Best Model:** Tuned XGB (HPO)
- **Reported Composite Score:** 0.6276
- **Reported Accuracy Tradeoff Proxy:** 0.0000
- **ThresholdOptimizer Disparate Impact Ratio:** 0.8663 *(Meets 80% screening heuristic, but based on synthetic proxy)*
