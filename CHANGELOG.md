# Changelog

All notable changes to this project are documented in this file. This project follows semantic versioning principles.

## [1.0.0-rc1] - Release Candidate (2026)

### Added
- **Auditable Fairness Evaluation Pipeline:** Decoupled 4-dimension audit taxonomy (`AUDIT_EXECUTION_STATUS`, `EVIDENCE_STATUS`, `MODEL_STATUS`, `POLICY_STATUS`) with deterministic precedence engine.
- **Evaluation-Unit and Estimand Governance:** Schema-driven alignment between observation units, decision units, and statistical estimands (`configs/evaluation_units.yaml`).
- **Prediction-Time Feature Controls:** Schema-enforced feature contracts (`configs/benchmark_features.yaml`) prohibiting post-outcome realizations and target proxies.
- **Model Firewall and Holdout Isolation:** `LockedTestData` firewall preventing test-set model selection; group-aware and temporal holdout partitioning.
- **Independent Metric and Reference Checks:** Standalone metric computation verified against independent Fairlearn/scikit-learn reference oracle suites.
- **Uncertainty and Intersectional Analysis:** Non-parametric bootstrap confidence intervals (row and cluster-aware) and multi-attribute intersectional disparity auditing.
- **Audit Bundle Generation and Verification:** Zero-raw-data reproducible evaluation packaging with Draft-07 JSON Schema and independent mathematical invariant verifier (`verify_audit_bundle`).
- **Human-Review Boundaries:** Machine-readable trigger codes (`HumanReviewReasonCode`) and canonical alias normalization.
- **Structured CLI:** Unified command-line interface (`afa {ingest,audit,bundle,verify}`) with deterministic exit codes and `--json` machine-readable output.
- **Publication and Release Audits:** Pre-publication scan verifying zero raw datasets, zero credential patterns, zero local paths, and permissive MIT licensing.

### Validation
- **Automated Tests:** 209 unit, integration, and regression tests passing.
- **Reproducibility:** 10/10 clean-environment reproducibility verification steps passing from scratch (Exit code 0).
- **Bundle Integrity:** Independent cryptographic SHA-256 fingerprinting, Draft-07 JSON schema validation, and mathematical count identity checks.
- **Publication Hygiene:** Zero secrets, zero tracked tabular data, zero developer machine paths, verified MIT license.

### Known Limitations
- **External Real-Data Validation Pending:** Authentic customer operational datasets have not yet been evaluated through the pipeline; status remains strictly `EXTERNAL_VALIDATION_PENDING`.
- **Scope Boundary:** Technical metric compliance demonstrates satisfaction of mathematical criteria under specified evaluation conditions; it does not constitute statutory certification under Title VII, ECOA, FHA, EU AI Act, or GDPR. Customer-specific deployment suitability requires staging authentic client data and human legal oversight.
