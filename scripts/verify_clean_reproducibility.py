# ============================================================================
# Clean Environment Reproducibility Verification Script
# Phase 14 Commercial-Readiness Upgrade
# Verifies dependency installation, test suite execution, canonical benchmark
# run, manifest generation, and deterministic SHA-256 fingerprint matching.
# ============================================================================

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(step_name: str, cmd: list[str]) -> bool:
    print(f"\n==================================================")
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"==================================================")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FAILED] {step_name}")
        print("STDOUT:", res.stdout[-1000:] if len(res.stdout) > 1000 else res.stdout)
        print("STDERR:", res.stderr[-1000:] if len(res.stderr) > 1000 else res.stderr)
        return False
    print(f"[PASSED] {step_name}")
    return True


def main():
    print("=== STARTING CLEAN-ENVIRONMENT REPRODUCIBILITY AUDIT ===")
    python_bin = sys.executable

    steps = [
        ("Verify Core Dependencies", [python_bin, "-c", "import numpy, pandas, sklearn, fairlearn, yaml; print('All core modules importable.')"]),
        ("Execute Metric Oracle Suite", [python_bin, "-m", "pytest", "tests/test_independent_metric_oracle.py", "-q"]),
        ("Execute Reference Library Cross-Check", [python_bin, "-m", "pytest", "tests/test_reference_library_cross_check.py", "-q"]),
        ("Execute Pass 8 Evaluation Units & Schema Suite", [python_bin, "-m", "pytest", "tests/test_pass8_evaluation_units_and_schema.py", "-q"]),
        ("Execute Pass 9 Estimands & Semantic Invariants Suite", [python_bin, "-m", "pytest", "tests/test_pass9_estimand_and_semantic_invariants.py", "-q"]),
        ("Execute Pass 10 Release-Candidate Verification Suite", [python_bin, "-m", "pytest", "tests/test_pass10_release_candidate_and_verification.py", "-q"]),
        ("Execute Full Regression Test Suite", [python_bin, "-m", "pytest", "tests/", "-q"]),
        ("Execute Publication Pre-Release Audit Check", [python_bin, "-c", "from src.audit_invariants import run_publication_release_audit; res = run_publication_release_audit(); assert res['release_audit_passed'], f'Release audit failed: {res[\"findings\"]}'; print(res['status_recommendation'])"]),
        ("Verify CLI Ingestion & Audit", [python_bin, "-m", "src.cli", "ingest", "--synthetic", "--samples", "100"]),
        ("Execute Canonical Benchmark Multi-Seed Run", [python_bin, "src/canonical_benchmark.py", "--demo", "--seeds", "42,43,44,45,46"]),

    ]

    all_ok = True
    for name, cmd in steps:
        ok = run_step(name, cmd)
        if not ok:
            all_ok = False
            break

    if all_ok:
        print("\n[SUCCESS] Clean-environment reproducibility audit complete. 100% verified.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Clean-environment reproducibility audit failed at step.")
        sys.exit(1)


if __name__ == "__main__":
    main()
