# Final Evaluation Results and Research Appendix

_Generated on 2026-08-09 17:26:40_

## What this report adds
- Reweighting-based fairness intervention
- Predictive parity and subgroup fairness checks
- Causal-style counterfactual analysis on education intervention
- Robustness checks on unseen holdouts and time splits
- Updated labor-economics interpretation and welfare-loss framing

## Fairness metric definitions
- Demographic parity difference: $|P(\hat{Y}=1|A=a) - P(\hat{Y}=1|A=b)|$
- Equal opportunity difference: $|P(\hat{Y}=1|Y=1,A=a) - P(\hat{Y}=1|Y=1,A=b)|$
- Equalized odds difference: maximum of the TPR and FPR gaps across groups
- Predictive parity difference: gap in positive predictive value across groups
- Disparate impact ratio: ratio of positive prediction rates, closer to 1 is better

## Causal graph
```mermaid
graph LR
    E[Education] --> T[Task Allocation]
    E --> O[Observed Outcome]
    X[Route / Worker Context] --> T
    X --> O
    T --> O
    P[Platform Incentives] --> T
    P --> O
```

## Labor economics interpretation
Education can act as a signaling variable, and platform assignment systems may amplify statistical discrimination when proxies correlate with route difficulty, timing, or geography. The added causal-style and robustness checks help distinguish whether observed disparities are due to model behavior, data structure, or underlying task allocation patterns.

## Delhivery Logistics

- Sensitive Attribute: Education_Level
- Sensitive Source: Synthetic (data-driven proxy)
- Notes: Primary logistics dataset with enhanced features.

### Model comparison
| Model | Accuracy | Balanced Accuracy | Fairness Gap | Predictive Parity Diff | DI Ratio |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7231 | 0.7558 | 0.0764 | 0.0151 | 0.7103 |
| Reweighted Logistic | 0.8650 | 0.6542 | 0.0347 | 0.0236 | 0.6669 |
| Random Forest | 0.9752 | 0.9492 | 0.0205 | 0.0276 | 0.7693 |
| Neural Network | 0.9359 | 0.8673 | 0.0295 | 0.0236 | 0.7526 |
| Weighted XGB | 0.9391 | 0.9374 | 0.0272 | 0.0445 | 0.7933 |
| Tuned XGB (HPO) | 0.9759 | 0.9693 | 0.0249 | 0.0364 | 0.7875 |
| Equalized Odds ThresholdOptimizer | 0.8613 | 0.7172 | 0.0245 | 0.0677 | 0.8389 |

### Best model: Tuned XGB (HPO)
- Composite score: 0.9631
- Welfare loss proxy: 0.0000

### Robustness checks
| Check | Dataset | Baseline Accuracy | Mitigated Accuracy | Baseline Fairness Gap | Mitigated Fairness Gap |
|---|---|---|---|---|---|
| Unseen Holdout | Delhivery Logistics | 0.7231 | 0.8650 | 0.0764 | 0.0347 |

### Subgroup fairness: Education Tiers
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Education | Bachelor's | 6355 | 0.9751 | 0.1849 |
| Education | High School | 5097 | 0.9770 | 0.1579 |
| Education | Master's | 17522 | 0.9759 | 0.2005 |

### Subgroup fairness: Experience Levels
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Experience | High | 7244 | 0.9564 | 0.3444 |
| Experience | Low | 7244 | 0.9862 | 0.0991 |
| Experience | Mid-High | 7243 | 0.9750 | 0.1932 |
| Experience | Mid-Low | 7243 | 0.9861 | 0.1218 |

### Subgroup fairness: Geography Proxies
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Geography | Far | 7243 | 0.9780 | 0.1557 |
| Geography | Moderate | 7243 | 0.9511 | 0.3554 |
| Geography | Near | 7244 | 0.9818 | 0.2057 |
| Geography | Very Far | 7244 | 0.9927 | 0.0417 |

## Adult Income Benchmark

- Sensitive Attribute: Education_Group
- Sensitive Source: Real attribute (observed education)
- Notes: Real-world fairness benchmark with enhanced features.

### Model comparison
| Model | Accuracy | Balanced Accuracy | Fairness Gap | Predictive Parity Diff | DI Ratio |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7726 | 0.8019 | 0.1453 | 0.3507 | 0.6410 |
| Reweighted Logistic | 0.8312 | 0.7363 | 0.2070 | 0.2853 | 0.3781 |
| Random Forest | 0.8454 | 0.7670 | 0.1980 | 0.2196 | 0.3551 |
| Neural Network | 0.8196 | 0.7455 | 0.2477 | 0.2507 | 0.3435 |
| Weighted XGB | 0.8057 | 0.8245 | 0.1447 | 0.3367 | 0.5987 |
| Tuned XGB (HPO) | 0.7941 | 0.8184 | 0.1517 | 0.3539 | 0.6344 |
| Equalized Odds ThresholdOptimizer | 0.8156 | 0.7256 | 0.1225 | 0.3156 | 0.5547 |

### Best model: Weighted XGB
- Composite score: 0.7883
- Welfare loss proxy: 0.0000

### Counterfactual / causal-style estimates
| Causal Estimate | Value |
|---|---|
| G-computation ATE | 0.1313 |
| Group-specific HigherEd effect | 0.1343 |
| IPW ATE | 0.1305 |
| Matched Difference | 0.1464 |

### Robustness checks
| Check | Dataset | Baseline Accuracy | Mitigated Accuracy | Baseline Fairness Gap | Mitigated Fairness Gap |
|---|---|---|---|---|---|
| Unseen Holdout | Adult Income Benchmark | 0.7726 | 0.8312 | 0.1453 | 0.2070 |

### Subgroup fairness: Education Tiers
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Education | HigherEd | 2247 | 0.8153 | 0.5349 |
| Education | NonHigherEd | 6798 | 0.8026 | 0.3202 |

### Subgroup fairness: Experience Levels
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Experience | Early | 1736 | 0.9798 | 0.0109 |
| Experience | Late | 2631 | 0.7434 | 0.5728 |
| Experience | Mid | 3600 | 0.7878 | 0.3714 |
| Experience | Senior | 1078 | 0.7375 | 0.4787 |

### Subgroup fairness: Geography Proxies
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Geography | Canada | 37 | 0.8919 | 0.4054 |
| Geography | India | 36 | 0.6389 | 0.6111 |
| Geography | Mexico | 173 | 0.9538 | 0.0520 |
| Geography | Other | 469 | 0.8060 | 0.3028 |
| Geography | Philippines | 58 | 0.8793 | 0.3276 |
| Geography | United-States | 8272 | 0.8025 | 0.3835 |

## Amazon Last-Mile Routes

- Sensitive Attribute: Education_Level
- Sensitive Source: Synthetic (data-driven proxy)
- Notes: Amazon routes included in the full dataset evaluation.

### Model comparison
| Model | Accuracy | Balanced Accuracy | Fairness Gap | Predictive Parity Diff | DI Ratio |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6288 | 0.6211 | 0.0699 | 0.0260 | 0.8190 |
| Reweighted Logistic | 0.6271 | 0.6063 | 0.0793 | 0.0747 | 0.7888 |
| Random Forest | 0.6631 | 0.6469 | 0.1169 | 0.1040 | 0.6675 |
| Neural Network | 0.6419 | 0.6298 | 0.0906 | 0.0344 | 0.7392 |
| Weighted XGB | 0.6476 | 0.6379 | 0.0945 | 0.0607 | 0.7432 |
| Tuned XGB (HPO) | 0.6631 | 0.6562 | 0.1145 | 0.0887 | 0.7181 |
| Equalized Odds ThresholdOptimizer | 0.6198 | 0.6026 | 0.0652 | 0.0224 | 0.8663 |

### Best model: Tuned XGB (HPO)
- Composite score: 0.6276
- Welfare loss proxy: 0.0000

### Robustness checks
| Check | Dataset | Baseline Accuracy | Mitigated Accuracy | Baseline Fairness Gap | Mitigated Fairness Gap |
|---|---|---|---|---|---|
| Unseen Holdout | Amazon Last-Mile Routes | 0.6288 | 0.6271 | 0.0699 | 0.0793 |
| Time Split (Early) | Amazon Last-Mile Routes | 0.6460 | 0.6476 | 0.0755 | 0.0640 |
| Time Split (Late) | Amazon Last-Mile Routes | 0.6115 | 0.6066 | 0.0758 | 0.0751 |

### Subgroup fairness: Education Tiers
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Education | Bachelor's | 276 | 0.6812 | 0.3949 |
| Education | High School | 222 | 0.6892 | 0.3288 |
| Education | Master's | 725 | 0.6483 | 0.4579 |

### Subgroup fairness: Experience Levels
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Experience | Afternoon | 1065 | 0.6554 | 0.4441 |
| Experience | Morning | 158 | 0.7152 | 0.2595 |

### Subgroup fairness: Geography Proxies
| Subgroup | Value | Count | Accuracy | Positive Rate |
|---|---|---|---|---|
| Geography | DBO3 | 122 | 0.7295 | 0.2049 |
| Geography | DLA7 | 227 | 0.6388 | 0.3524 |
| Geography | DLA9 | 141 | 0.6241 | 0.5816 |
| Geography | DSE4 | 88 | 0.6364 | 0.5795 |
| Geography | DSE5 | 106 | 0.7358 | 0.4151 |
| Geography | Other | 539 | 0.6586 | 0.4304 |
