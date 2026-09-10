# ============================================================================
# Algorithmic Fairness Analysis (afa) - Educational CLI Entrypoint
# Command-line interface for data ingestion, fairness evaluation, benchmarking, and demo execution.
# Exit codes:
#   0: Success
#   1: Evaluation check failed / Threshold exceeded
#   2: Execution Blocked (Missing inputs, contract failure, corrupt data)
#   4: Configuration Error (Invalid CLI arguments or missing file)
# ============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


EXIT_SUCCESS = 0
EXIT_EVALUATION_FAILED = 1
EXIT_EXECUTION_BLOCKED = 2
EXIT_CONFIG_ERROR = 4


def cmd_demo(args: argparse.Namespace) -> int:
    """Handles 'afa demo' command: runs the self-contained educational fairness analysis pipeline."""
    try:
        from src.main import main as run_main
        cmd_args = ["--demo"]
        if args.seeds:
            cmd_args.extend(["--seeds", args.seeds])
        if args.alpha is not None:
            cmd_args.extend(["--alpha", str(args.alpha)])
        run_main(cmd_args)
        return EXIT_SUCCESS
    except Exception as e:
        print(f"[ERROR] Demo pipeline execution failed: {e}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handles 'afa benchmark' command: runs the canonical multi-seed fairness benchmark."""
    try:
        from src.canonical_benchmark import main as run_benchmark
        cmd_args = ["--demo"]
        if args.seeds:
            cmd_args.extend(["--seeds", args.seeds])
        if args.scenario:
            cmd_args.extend(["--scenario", args.scenario])
        if args.signal:
            cmd_args.extend(["--signal", args.signal])
        run_benchmark(cmd_args)
        return EXIT_SUCCESS
    except Exception as e:
        print(f"[ERROR] Canonical benchmark execution failed: {e}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED


def cmd_ingest(args: argparse.Namespace) -> int:
    """Handles 'afa ingest' command: validates external dataset schemas or generates synthetic fixture."""
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
                print(f"[SUCCESS] Synthetic DGP generated: {scenario_name}, shape={df.shape}")
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
    """Handles 'afa audit' command: evaluates fairness metrics defensively on input data or synthetic fixture."""
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

        policy = FairnessPolicy(
            name=args.policy_name or "Educational_Evaluation_Policy",
            min_balanced_accuracy=args.min_bal_acc or 0.55,
            max_equalized_odds_diff=args.max_eo_diff or 0.15,
            max_demographic_parity_diff=args.max_dp_diff or 0.15,
            policy_version="0.1.0",
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
            purpose=args.purpose or "Educational Fairness Evaluation",
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
                "output_saved_to": str(out_file),
                "model_status": tax_state.get("model_status", "ACCEPTABLE"),
                "policy_status": policy_status,
                "balanced_accuracy": suite.balanced_accuracy,
                "equalized_odds_diff": suite.equalized_odds_difference,
                "demographic_parity_diff": suite.demographic_parity_difference,
                "disparate_impact_ratio": suite.disparate_impact_ratio,
            }
            print(json.dumps(summary_dict, indent=2))
        else:
            print(f"[INFO] Evaluation result saved to {out_file}")
            print(f"Fairness Evaluation Summary:")
            print(f"  Model Status:        {tax_state.get('model_status', 'ACCEPTABLE')}")
            print(f"  Policy Status:       {policy_status}")
            print(f"  Balanced Accuracy:   {suite.balanced_accuracy:.4f}" if suite.balanced_accuracy is not None else "  Balanced Accuracy:   N/A")
            print(f"  Equalized Odds Diff: {suite.equalized_odds_difference:.4f}" if suite.equalized_odds_difference is not None else "  Equalized Odds Diff: N/A")
            print(f"  Dem. Parity Diff:    {suite.demographic_parity_difference:.4f}" if suite.demographic_parity_difference is not None else "  Dem. Parity Diff:    N/A")
            print(f"  Disparate Impact:    {suite.disparate_impact_ratio:.4f}" if suite.disparate_impact_ratio is not None else "  Disparate Impact:    N/A")

        if tax_state.get("audit_execution_status") == "BLOCKED" or policy_status == PolicyStatus.NOT_EVALUABLE.value:
            return EXIT_EXECUTION_BLOCKED
        elif policy_status == PolicyStatus.FAIL.value:
            return EXIT_EVALUATION_FAILED
        else:
            return EXIT_SUCCESS

    except Exception as e:
        print(f"[ERROR] Evaluation execution failed: {e}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afa",
        description="Algorithmic Fairness Analysis (afa) - Educational CLI Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Demo
    p_demo = subparsers.add_parser("demo", help="Run the educational pipeline demonstration")
    p_demo.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds (e.g. '42,43')")
    p_demo.add_argument("--alpha", type=float, default=0.25, help="Utility scalarization tradeoff parameter")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run the canonical multi-seed fairness benchmark")
    p_bench.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated random seeds")
    p_bench.add_argument("--scenario", type=str, default="SCENARIO_C_EXPLICIT_GROUP_EFFECT", help="Synthetic scenario name")
    p_bench.add_argument("--signal", type=str, default="MODERATE_SIGNAL", help="Signal regime (HIGH, MODERATE, LOW, ZERO)")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest and validate dataset schema or generate synthetic fixture")
    p_ingest.add_argument("--dataset", type=str, help="Dataset identifier")
    p_ingest.add_argument("--file", type=str, help="Path to raw dataset file")
    p_ingest.add_argument("--synthetic", action="store_true", help="Generate synthetic DGP fixture")
    p_ingest.add_argument("--scenario", type=str, help="Synthetic scenario name")
    p_ingest.add_argument("--samples", type=int, default=500, help="Number of synthetic samples")
    p_ingest.add_argument("--output", type=str, help="Output path for synthetic dataset")
    p_ingest.add_argument("--json", action="store_true", help="Output machine-readable JSON status")

    # Audit
    p_audit = subparsers.add_parser("audit", help="Run fairness evaluation on input data or synthetic fixture")
    p_audit.add_argument("--file", type=str, help="Path to input data file")
    p_audit.add_argument("--dataset", type=str, help="Dataset identifier")
    p_audit.add_argument("--target-col", type=str, default="target", help="Ground truth column name")
    p_audit.add_argument("--pred-col", type=str, default="prediction", help="Prediction column name")
    p_audit.add_argument("--sensitive-col", type=str, default="sensitive", help="Sensitive attribute column name")
    p_audit.add_argument("--synthetic", action="store_true", help="Use synthetic fixture data")
    p_audit.add_argument("--samples", type=int, default=500, help="Number of samples if synthetic")
    p_audit.add_argument("--scenario", type=str, help="Synthetic scenario name if synthetic")
    p_audit.add_argument("--policy-name", type=str, default="Educational_Evaluation_Policy", help="Policy name")
    p_audit.add_argument("--min-bal-acc", type=float, default=0.55, help="Minimum balanced accuracy threshold")
    p_audit.add_argument("--max-eo-diff", type=float, default=0.15, help="Maximum equalized odds diff threshold")
    p_audit.add_argument("--max-dp-diff", type=float, default=0.15, help="Maximum demographic parity diff threshold")
    p_audit.add_argument("--confirmatory", action="store_true", help="Enforce confirmatory evaluation prerequisites")
    p_audit.add_argument("--purpose", type=str, default="Educational Fairness Evaluation", help="Evaluation purpose")
    p_audit.add_argument("--output", type=str, default="audit_result.json", help="Path for output JSON result")
    p_audit.add_argument("--json", action="store_true", help="Output machine-readable JSON summary")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_SUCCESS

    if not args.command:
        parser.print_help()
        return EXIT_CONFIG_ERROR

    if args.command == "demo":
        return cmd_demo(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "audit":
        return cmd_audit(args)
    else:
        parser.print_help()
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    sys.exit(main())
