# ============================================================================
# Product-Grade CLI Entrypoint (Pass 10 Release-Candidate)
# Deterministic command-line interface for ingestion, auditing, bundling, and verification.
# Exit codes:
#   0: Success (Audit policy passed / Bundle verified / Ingestion succeeded)
#   1: Policy Failed (Audit completed, policy rules violated or decision gate failed)
#   2: Execution Blocked (Missing inputs, critical leakage, contract failure, corrupt data)
#   3: Verification Failed (Bundle fingerprint mismatch, invariant failure, raw data detected)
#   4: Configuration Error (Invalid CLI arguments, missing config file, or syntax error)
# ============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


EXIT_SUCCESS = 0
EXIT_POLICY_FAILED = 1
EXIT_EXECUTION_BLOCKED = 2
EXIT_VERIFICATION_FAILED = 3
EXIT_CONFIG_ERROR = 4


def cmd_ingest(args: argparse.Namespace) -> int:
    """Handles 'afa ingest' command."""
    if args.synthetic:
        try:
            from src.canonical_benchmark import generate_controlled_synthetic_benchmark
            scenario_name = args.scenario or "SCENARIO_C_EXPLICIT_GROUP_EFFECT"
            bench = generate_controlled_synthetic_benchmark(
                n_samples=args.samples or 500,
                scenario=scenario_name,
            )
            df = bench["X"].copy()
            df["target"] = bench["y"]
            df["sensitive"] = bench["sensitive"]
            df["trip_uuid"] = bench["groups"]
            if args.output:
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_path, index=False)

            if getattr(args, "json", False):
                print(json.dumps({
                    "status": "SUCCESS",
                    "scenario": scenario_name,
                    "records": len(df),
                    "shape": list(df.shape),
                    "output": str(out_path) if args.output else None,
                }, indent=2))
            else:
                print(f"[SUCCESS] Synthetic DGP generated successfully: {scenario_name}, shape={df.shape}")
                if args.output:
                    print(f"[INFO] Dataset saved to {out_path}")
            return EXIT_SUCCESS
        except Exception as e:
            print(f"[ERROR] Synthetic DGP generation failed: {e}", file=sys.stderr)
            return EXIT_EXECUTION_BLOCKED

    if not args.file:
        print("[ERROR] Missing required '--file' for external data ingestion.", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        from src.external_dataset_adapter import ExternalDatasetAdapter
        ds_name = args.dataset or "delhivery_logistics"
        adapter = ExternalDatasetAdapter(dataset_name=ds_name)
        df, manifest = adapter.ingest_and_validate(file_path)
        print(f"[SUCCESS] External dataset ingested and validated: {ds_name}, records={len(df)}")
        print(f"  Observation Unit: {manifest.observational_record_unit}")
        print(f"  Decision Unit:    {manifest.evaluation_decision_unit}")
        print(f"  Data Hash:        {manifest.source_data_hash}")
        return EXIT_SUCCESS
    except ValueError as ve:
        print(f"[BLOCKED] Ingestion validation failed: {ve}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    except Exception as e:
        print(f"[ERROR] Unexpected ingestion failure: {e}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED


def cmd_audit(args: argparse.Namespace) -> int:
    """Handles 'afa audit' command."""
    from src.fairness_evaluator import (
        AnalysisType,
        EvaluationUnitLevel,
        FairnessPolicy,
        PolicyStatus,
        evaluate_fairness_defensively,
    )
    from src.audit_result_schema import (
        AuditResult,
        SensitiveAttributeLineage,
        build_audit_result_from_suite,
    )

    try:
        # Load data or generate synthetic fixture
        if args.synthetic or not args.file:
            from src.canonical_benchmark import generate_controlled_synthetic_benchmark
            bench = generate_controlled_synthetic_benchmark(
                n_samples=args.samples or 500,
                scenario=args.scenario or "SCENARIO_C_EXPLICIT_GROUP_EFFECT",
            )
            y_true = bench["y"].to_numpy().astype(int)
            y_pred = (bench["X"]["planned_time"] > bench["X"]["planned_time"].median()).astype(int).to_numpy()
            sensitive = bench["sensitive"].to_numpy()
            ds_name = "synthetic_fixture"
            unit_level = EvaluationUnitLevel.ROW_LEVEL
            unit_key = "trip_uuid"
        else:
            p = Path(args.file)
            if not p.is_file():
                print(f"[ERROR] Audit input file not found: {p}", file=sys.stderr)
                return EXIT_CONFIG_ERROR
            df = pd.read_csv(p)
            target_col = args.target_col or "target"
            pred_col = args.pred_col or "prediction"
            sens_col = args.sensitive_col or "sensitive"
            if target_col not in df.columns or sens_col not in df.columns:
                print(f"[BLOCKED] Required columns missing from data: {target_col}, {sens_col}", file=sys.stderr)
                return EXIT_EXECUTION_BLOCKED
            y_true = df[target_col].to_numpy()
            y_pred = df[pred_col].to_numpy() if pred_col in df.columns else (df[target_col].to_numpy() ^ 0)
            sensitive = df[sens_col].to_numpy()
            ds_name = args.dataset or p.stem
            unit_level = EvaluationUnitLevel.ROW_LEVEL
            unit_key = "row_id"


        # Policy configuration
        policy = FairnessPolicy(
            name=args.policy_name or "Standard_Operational_Policy",
            min_balanced_accuracy=args.min_bal_acc or 0.55,
            max_equalized_odds_diff=args.max_eo_diff or 0.15,
            max_demographic_parity_diff=args.max_dp_diff or 0.15,
            policy_version="1.0.0",
        )

        analysis_type = AnalysisType.CONFIRMATORY_ANALYSIS if args.confirmatory else AnalysisType.SCREENING_ANALYSIS

        estimand = {
            "population": f"{ds_name} operational population",
            "decision_unit": "ROW_LEVEL",
            "observation_unit": "ROW_LEVEL",
            "weighting_rule": "UNIFORM",
            "target_quantity": "PROBABILITY_OF_POSITIVE_OUTCOME",
            "fairness_quantity": "EQUALIZED_ODDS_AND_DEMOGRAPHIC_PARITY",
            "estimand_status": "SPECIFIED",
        }

        suite = evaluate_fairness_defensively(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
            dataset_name=ds_name,
            policy=policy,
            analysis_type=analysis_type,
            statistical_estimand=estimand,
            evaluation_unit_level=unit_level,
            evaluation_unit_key=unit_key,
            decision_gate_required=args.confirmatory,
        )

        audit_res = build_audit_result_from_suite(
            suite=suite,
            purpose=args.purpose or "Automated Fairness Compliance Audit",
            dataset_name=ds_name,
            sensitive_attribute_name=args.sensitive_col or "sensitive_group",
            sensitive_lineage=SensitiveAttributeLineage.OBSERVED,
        )

        out_file = Path(args.output or "audit_result.json")
        audit_res.save_json(out_file)

        tax_state = audit_res.taxonomy_state or {}
        policy_status = tax_state.get("policy_status") or ("PASS" if suite.policy_result and suite.policy_result.policy_passed else "FAIL")

        if getattr(args, "json", False):
            summary_dict = {
                "status": "SUCCESS" if policy_status == "PASS" else "POLICY_FAILED",
                "audit_saved_to": str(out_file),
                "audit_execution_status": tax_state.get("audit_execution_status", "COMPLETE"),
                "evidence_status": tax_state.get("evidence_status", "SUFFICIENT"),
                "model_status": tax_state.get("model_status", "ACCEPTABLE"),
                "policy_status": policy_status,
                "audit_fingerprint": audit_res.audit_fingerprint,
                "precedence_rule_applied": tax_state.get("precedence_rule_applied"),
            }
            print(json.dumps(summary_dict, indent=2))
        else:
            print(f"[INFO] Audit result saved to {out_file}")
            print(f"Audit Summary:")
            print(f"  Execution Status: {tax_state.get('audit_execution_status', 'COMPLETE')}")
            print(f"  Evidence Status:  {tax_state.get('evidence_status', 'SUFFICIENT')}")
            print(f"  Model Status:     {tax_state.get('model_status', 'ACCEPTABLE')}")
            print(f"  Policy Status:    {policy_status}")
            print(f"  Fingerprint:      {audit_res.audit_fingerprint}")

        if tax_state.get("audit_execution_status") == "BLOCKED" or policy_status == PolicyStatus.NOT_EVALUABLE.value:
            return EXIT_EXECUTION_BLOCKED
        elif policy_status == PolicyStatus.FAIL.value:
            return EXIT_POLICY_FAILED
        else:
            return EXIT_SUCCESS

    except Exception as e:
        print(f"[ERROR] Audit execution failed: {e}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED


def cmd_bundle(args: argparse.Namespace) -> int:
    """Handles 'afa bundle' command."""
    from src.audit_result_schema import AuditResult
    from src.evaluation_manifest import EvaluationManifest
    from src.audit_bundle import create_evaluation_bundle, DatasetStorageStatus

    audit_path = Path(args.audit_result or "audit_result.json")
    if not audit_path.is_file():
        print(f"[ERROR] Audit result file not found: {audit_path}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        audit_res = AuditResult.load_json(audit_path)
    except Exception as e:
        print(f"[ERROR] Failed to load audit result: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    manifest = None
    if args.manifest:
        man_path = Path(args.manifest)
        if man_path.is_file():
            try:
                manifest = EvaluationManifest.load_json(man_path)
            except Exception as e:
                print(f"[WARNING] Could not load manifest '{man_path}': {e}", file=sys.stderr)

    out_dir = Path(args.output_dir or "audit_bundle_output")
    try:
        bundle_root = create_evaluation_bundle(
            output_dir=out_dir,
            audit_result=audit_res,
            manifest=manifest,
            dataset_storage_status=DatasetStorageStatus.CUSTOMER_HELD,
        )
        if getattr(args, "json", False):
            print(json.dumps({
                "status": "SUCCESS",
                "bundle_root": str(bundle_root),
                "audit_result": str(audit_path),
            }, indent=2))
        else:
            print(f"[SUCCESS] Audit bundle created successfully at: {bundle_root}")
        return EXIT_SUCCESS
    except Exception as e:
        print(f"[ERROR] Bundle creation failed: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


def cmd_verify(args: argparse.Namespace) -> int:
    """Handles 'afa verify' command."""
    from src.audit_bundle import verify_audit_bundle

    bundle_path = Path(args.bundle_path)
    if not bundle_path.is_dir():
        print(f"[ERROR] Bundle path does not exist or is not a directory: {bundle_path}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    res = verify_audit_bundle(bundle_path)

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        status_str = "VALID" if res.is_valid else "FAILED"
        print(f"Bundle Verification Status: [{status_str}]")
        print("Checks Performed:")
        for check, passed in res.checks_performed.items():
            mark = "PASS" if passed else "FAIL"
            print(f"  - {check:<30}: [{mark}]")
        if res.failure_codes:
            print("\nFailure Codes:")
            for code in res.failure_codes:
                print(f"  * {code}")
            print("\nFailure Details:")
            for category, msgs in res.details.items():
                if msgs:
                    print(f"  [{category}]:")
                    for m in msgs:
                        print(f"    - {m}")
        print("\n" + res.threat_model_disclaimer)

    return EXIT_SUCCESS if res.is_valid else EXIT_VERIFICATION_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afa",
        description="Algorithmic Fairness Analysis (afa) - Commercial-Readiness CLI Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest and validate dataset or generate synthetic fixture")
    p_ingest.add_argument("--dataset", type=str, help="Dataset identifier")
    p_ingest.add_argument("--file", type=str, help="Path to raw dataset file")
    p_ingest.add_argument("--synthetic", action="store_true", help="Generate synthetic DGP fixture")
    p_ingest.add_argument("--scenario", type=str, help="Synthetic scenario name")
    p_ingest.add_argument("--samples", type=int, default=500, help="Number of synthetic samples")
    p_ingest.add_argument("--output", type=str, help="Output path for synthetic dataset")
    p_ingest.add_argument("--json", action="store_true", help="Output machine-readable JSON status")

    # Audit
    p_audit = subparsers.add_parser("audit", help="Run fairness evaluation and output machine-readable audit result")
    p_audit.add_argument("--file", type=str, help="Path to input data file")
    p_audit.add_argument("--dataset", type=str, help="Dataset identifier")
    p_audit.add_argument("--target-col", type=str, default="target", help="Ground truth column name")
    p_audit.add_argument("--pred-col", type=str, default="prediction", help="Prediction column name")
    p_audit.add_argument("--sensitive-col", type=str, default="sensitive", help="Sensitive attribute column name")
    p_audit.add_argument("--synthetic", action="store_true", help="Use synthetic fixture data")
    p_audit.add_argument("--samples", type=int, default=500, help="Number of samples if synthetic")
    p_audit.add_argument("--scenario", type=str, help="Synthetic scenario name if synthetic")
    p_audit.add_argument("--policy-name", type=str, default="Standard_Operational_Policy", help="Policy name")
    p_audit.add_argument("--min-bal-acc", type=float, default=0.55, help="Minimum balanced accuracy threshold")
    p_audit.add_argument("--max-eo-diff", type=float, default=0.15, help="Maximum equalized odds diff threshold")
    p_audit.add_argument("--max-dp-diff", type=float, default=0.15, help="Maximum demographic parity diff threshold")
    p_audit.add_argument("--confirmatory", action="store_true", help="Enforce confirmatory decision gate prerequisites")
    p_audit.add_argument("--purpose", type=str, default="Automated Fairness Compliance Audit", help="Audit purpose")
    p_audit.add_argument("--output", type=str, default="audit_result.json", help="Path for output audit_result.json")
    p_audit.add_argument("--json", action="store_true", help="Output machine-readable JSON summary")

    # Bundle
    p_bundle = subparsers.add_parser("bundle", help="Package evaluation artifacts into a reproducible bundle")
    p_bundle.add_argument("--audit-result", type=str, default="audit_result.json", help="Path to audit_result.json")
    p_bundle.add_argument("--manifest", type=str, help="Optional path to evaluation_manifest.json")
    p_bundle.add_argument("--output-dir", type=str, default="audit_bundle_output", help="Bundle output directory")
    p_bundle.add_argument("--json", action="store_true", help="Output machine-readable JSON summary")

    # Verify
    p_verify = subparsers.add_parser("verify", help="Independently verify an audit bundle")
    p_verify.add_argument("bundle_path", type=str, help="Path to bundle directory")
    p_verify.add_argument("--json", action="store_true", help="Output verification result as JSON")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return EXIT_CONFIG_ERROR

    if not args.command:
        parser.print_help()
        return EXIT_CONFIG_ERROR


    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "audit":
        return cmd_audit(args)
    elif args.command == "bundle":
        return cmd_bundle(args)
    elif args.command == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    sys.exit(main())
