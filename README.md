# Educational Inequality in Algorithmic Project Assignment

**Author:** Varada Govind (ID: IEC2023004)

This repository contains a fairness-focused machine learning analysis investigating assignment outcomes across logistics and benchmark datasets. The primary objective is to quantify and mitigate structural educational biases in gig economy routing algorithms using constrained optimization and structural causal models.

## Pipeline Architecture
The analysis pipeline combines:
- **Predictive Modeling:** Extreme Gradient Boosting (XGBoost), Random Forests, and Multi-Layer Perceptrons.
- **Bias Mitigation:** Inverse Propensity Weighting (IPW) and Fairlearn (`ThresholdOptimizer` enforcing Equalized Odds).
- **Causal Analysis:** Structural Causal Models (SCMs) estimating counterfactual discrimination via exact matching and G-computation.
- **Evaluation:** Pareto frontier analysis quantifying the welfare loss tradeoff between accuracy and the fairness gap.
- **Optimization:** Bayesian Hyperparameter Optimization via Gaussian Process Surrogates.

## Project Structure

```
.
├── data/
│   └── raw/                      # Input datasets (Adult Income, Delhivery, Amazon)
├── docs/
│   ├── TECHNICAL_APPENDIX.md     # Auto-generated runtime metrics and subgroup tables
│   ├── Causal_Graph_Appendix.md  # SCM graphical definitions
│   └── final_metrics_summary.csv # Machine-readable aggregated results
├── plots/                        # Generated visual artifacts (Pareto frontiers, SHAP, confusion matrices)
├── src/
│   ├── fairness_core.py          # Data ingestion, preprocessing, and core metric definitions
│   └── fairness_analysis.py      # Main executable pipeline
├── FINAL_REPORT.md               # Final academic report
└── README.md                     # Repository documentation
```

## Environment Setup

Install Python requirements:
```powershell
pip install pandas numpy scikit-learn xgboost fairlearn shap matplotlib seaborn
```

## Execution

To reproduce the analysis, execute the pipeline from the project root:
```powershell
python src/fairness_analysis.py
```
*Note: Depending on your local environment, you may need to use `python3`, `py`, or invoke your virtual environment directly (e.g., `.\venv\Scripts\python.exe`).*

## Documentation

- Please refer to **`FINAL_REPORT.md`** for the formal academic paper detailing methodology, results, and labor economics interpretations.
- Please refer to the files generated in `docs/` and `plots/` for detailed statistical outputs and visualizations.
