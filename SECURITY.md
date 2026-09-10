# Security Policy

## Reporting Security Issues

If you discover a security vulnerability, credential leak, or sensitive data exposure risk in this repository, please report it responsibly.

### Responsible Disclosure Process
- **Do NOT open a public GitHub issue** to report potential security vulnerabilities or sensitive data exposures.
- Please use **GitHub Private Vulnerability Reporting** via the repository's **Security Advisories** tab:
  1. Navigate to the repository on GitHub.
  2. Click on the **Security** tab.
  3. Under "Reporting", click **Report a vulnerability** to open a private advisory.

This allows investigation and resolution of the issue before public disclosure.

---

## Scope and Guidelines

In the context of this educational algorithmic fairness framework:
1. **Secret or Credential Leaks:** Unintended inclusion of API keys, authentication tokens, or private credentials in code, configuration, or documentation.
2. **Proprietary or Personal Data:** Committing or tracking authentic proprietary records or Personally Identifiable Information (PII) within version control.
3. **Evaluation Integrity Safeguards:** Flaws in feature firewalls, split isolation, or invariant checks that could lead to silent evaluation leakage or data corruption.
4. **Code Execution Vulnerabilities:** Unsafe deserialization or arbitrary code execution via file ingestion or serialization handlers.

---

## Data Notice

- **Never submit proprietary datasets, production logs, or private data** in GitHub issues, pull requests, or public discussions.
- All evaluation workflows in this repository are designed to operate on controlled synthetic data or documented public benchmark datasets (such as the UCI Adult Census dataset).
- Any local user datasets should remain strictly outside Git version control in accordance with `.gitignore`.
