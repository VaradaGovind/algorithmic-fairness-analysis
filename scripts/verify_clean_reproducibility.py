# ============================================================================
# Clean Environment Reproducibility Verification Script
# Algorithmic Fairness Analysis - Educational Framework
# Verifies dependency installation, automated test suite execution,
# educational CLI commands, canonical benchmark run, and repository hygiene.
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
    print("=== STARTING EDUCATIONAL REPRODUCIBILITY VERIFICATION ===")
    python_bin = sys.executable

    steps = [
        ("Verify Core Dependencies", [python_bin, "-c", "import numpy, pandas, sklearn, fairlearn, yaml; print('All core modules importable.')"]),
        ("Execute Metric Oracle Suite", [python_bin, "-m", "pytest", "tests/test_independent_metric_oracle.py", "-q"]),
        ("Execute Reference Library Cross-Check", [python_bin, "-m", "pytest", "tests/test_reference_library_cross_check.py", "-q"]),
        ("Execute Statistical Units & Estimands Suite", [python_bin, "-m", "pytest", "tests/test_statistical_units_and_estimands.py", "-q"]),
        ("Execute Cluster Uncertainty & Governance Suite", [python_bin, "-m", "pytest", "tests/test_cluster_uncertainty_and_governance.py", "-q"]),
        ("Execute Metric Invariants & Schema Suite", [python_bin, "-m", "pytest", "tests/test_metric_invariants_and_schema.py", "-q"]),
        ("Execute Full Automated Test Suite", [python_bin, "-m", "pytest", "tests/", "-q"]),
        ("Execute Repository Hygiene & Security Scanner", [python_bin, "-c", "from src.metric_invariants import run_repository_hygiene_check; res = run_repository_hygiene_check(); assert res['release_audit_passed'], f'Hygiene check failed: {res[\"findings\"]}'; print(res['status_recommendation'])"]),
        ("Verify CLI Ingestion & Demo", [python_bin, "src/cli.py", "ingest", "--synthetic", "--samples", "100"]),
        ("Execute Canonical Benchmark Multi-Seed Run", [python_bin, "src/canonical_benchmark.py", "--demo", "--seeds", "42,43,44,45,46"]),
    ]

    all_ok = True
    for name, cmd in steps:
        ok = run_step(name, cmd)
        if not ok:
            all_ok = False
            break

    if all_ok:
        print("\n[SUCCESS] Educational reproducibility verification complete. All steps passed.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Educational reproducibility verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
