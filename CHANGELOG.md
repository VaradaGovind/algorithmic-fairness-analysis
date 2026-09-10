# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-09-10

Initial public educational open-source release.

### Added
- **Educational Fairness Evaluation Pipeline:** Modular fairness analysis pipeline evaluating demographic parity, equalized odds, predictive parity, and disparate impact ratio across classification models (Logistic Regression, Random Forest, MLP, XGBoost).
- **Evaluation Integrity Safeguards:** Prediction-time feature firewall (`LockedTestData`), group-aware holdout partitioning (`GroupAwareSplitter`), and leakage prevention contracts.
- **Fairness Mitigation Techniques:** Pre-processing sample reweighting and post-processing threshold adjustment strategies.
- **Causal DAG Modeling:** Structural causal model specification (`docs/CAUSAL_ASSUMPTIONS.md`) distinguishing direct, indirect, and spurious pathways under explicit identifying assumptions.
- **Intersectional Analysis & Uncertainty:** Subgroup disparity analysis with sparse-cell gating and cluster-aware bootstrap confidence intervals (`docs/INTERSECTIONAL_FAIRNESS.md`).
- **Benchmark Validity Framework:** Controlled synthetic data generating processes (DGPs), anti-degeneracy checks, calibration analysis (Brier score, ECE), and multi-seed paired delta comparisons (`docs/BENCHMARK_VALIDITY.md`).
- **Comprehensive Technical Documentation:** Mathematical metric definitions, split isolation protocols, and legacy methodological disclosures (`docs/TECHNICAL_APPENDIX.md`).
- **Automated Test Suite:** Comprehensive test suite covering fairness metrics, independent metric oracles, causal estimands, cluster uncertainty, and repository hygiene.
- **CLI Interface:** Educational command-line interface (`afa {demo,benchmark,ingest,audit}`) with human-readable and structured outputs.

### Changed
- Refactored repository from commercial audit packaging into a clean educational and research-engineering structure.
- Replaced pass-named test suites with focused, domain-organized test modules.
- Clarified that the 80% disparate impact rule is used as an educational screening heuristic, not a definitive legal determination.
