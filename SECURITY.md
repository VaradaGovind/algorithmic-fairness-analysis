# Security Policy

## Reporting Security Issues

We take the security and integrity of `algorithmic-fairness-analysis` seriously. If you discover a security vulnerability, credential leak, or sensitive data exposure risk in this repository, please report it responsibly.

### Responsible Disclosure Process
- **Do NOT open a public GitHub issue** to report potential security vulnerabilities, leaked credentials, or sensitive data exposures.
- Please use **GitHub Private Vulnerability Reporting** via the repository's **Security Advisories** tab:
  1. Navigate to the repository on GitHub.
  2. Click on the **Security** tab.
  3. Under "Reporting", click **Report a vulnerability** to open a private advisory.

This allows us to investigate and resolve the issue in private before public disclosure.

---

## What Constitutes a Security Issue?

In the context of this algorithmic fairness evaluation framework, security and integrity issues include:
1. **Secret or Credential Leaks:** Unintended inclusion of API keys, authentication tokens, or private credentials in code, configuration, or documentation.
2. **Proprietary or Personal Data Ingestion:** Committing or tracking authentic customer datasets, proprietary operational records, or Personally Identifiable Information (PII) within version control.
3. **Evaluation Integrity Bypasses:** Circumvention of the `LockedTestData` firewall, prediction-time feature contracts, or split-isolation checks that could lead to unnoticed target leakage or biased model deployment.
4. **Audit Bundle Tampering:** Flaws in `verify_audit_bundle()` that fail to detect forged metric values, altered JSON configurations, or inconsistent cryptographic fingerprints.
5. **Code Execution Vulnerabilities:** Unsafe deserialization or arbitrary code execution via file ingestion or serialization handlers.

---

## Sensitive and Operational Data Notice

- **Never submit customer datasets, production logs, or private data** in GitHub issues, pull requests, or public discussions.
- All evaluation workflows in this repository must operate on controlled synthetic data generating processes (DGPs) or approved public benchmarks.
- Authentic operational datasets must remain strictly in isolated, customer-held environments outside Git version control.

---

## Limitations of Static Scanning

- The repository implements automated scanners and pre-publication checks for pattern-based secret detection and raw data exclusion.
- **Static scanning does not guarantee the mathematical absence of all vulnerabilities or sensitive material.** Heuristic pattern scans indicate zero matched expressions under configured regular expression rules and should be supplemented with organizational security policies.
