"""Evaluate local RAG answers and benchmark installed models; no training or downloads."""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.runner import load_predictions, resolve_documents, run_live, saved_demo_records
from benchmarks.scoring import load_suite, make_report, write_report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="Validate the benchmark cases and fixtures without model calls.")
    mode.add_argument("--live", action="store_true", help="Run the app's retrieval/model pipeline with local installed models.")
    mode.add_argument("--saved-demo", nargs="+", type=Path, help="Score existing report_demo.py result.json files offline.")
    mode.add_argument("--rescore", type=Path, help="Re-evaluate a benchmark results.json, preserving saved timings and manual review.")
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks" / "cases.json")
    parser.add_argument("--case", dest="cases", action="append", default=[], help="Select a case ID; repeat to select several.")
    parser.add_argument("--models", nargs="+", help="Installed local Ollama models to compare; defaults to LLM_MODEL.")
    parser.add_argument("--repeat", type=int, default=1, help="Live repetitions per model/case; seeds increment for each repetition.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300, help="Generation timeout per live case; retrieval uses its existing library timeout.")
    parser.add_argument("--max-tokens", type=int, help="Override the app's generation cap for this experiment; may cause truncation failures.")
    parser.add_argument("--context-size", type=int, help="Use the same context size across models instead of their app defaults.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--require-pass", action="store_true", help="Return exit code 1 when any automatic case checks fail (CI gate).")
    args = parser.parse_args()
    try:
        if args.repeat < 1 or args.timeout <= 0 or args.timeout == float("inf") or args.timeout != args.timeout:
            raise ValueError("Use a positive finite --timeout and --repeat of at least 1.")
        if args.max_tokens is not None and args.max_tokens < 1:
            raise ValueError("--max-tokens must be positive.")
        if args.context_size is not None and args.context_size < 1:
            raise ValueError("--context-size must be positive.")
        if args.seed < 0 or args.seed + args.repeat - 1 > 2**32 - 1:
            raise ValueError("Seeds must remain within the unsigned 32-bit range across repetitions.")
        if not args.live and (args.models or args.max_tokens is not None or args.context_size is not None or args.repeat != 1):
            raise ValueError("--models, --max-tokens, --context-size, and --repeat apply only to --live.")
        suite = load_suite(args.suite)
        known = {case["id"] for case in suite["cases"]}
        if set(args.cases) - known:
            raise ValueError("Unknown case IDs: " + ", ".join(sorted(set(args.cases) - known)))
        if args.validate:
            paths = resolve_documents(suite["cases"])
            print(f"Validated {len(suite['cases'])} cases and {len(paths)} benchmark fixtures. No model calls.")
            return
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=True)
        if any((args.output / name).exists() for name in ("results.json", "report.md", "progress.json")):
            raise ValueError("Output already contains benchmark results; choose a new --output directory.")
        if args.saved_demo:
            records = saved_demo_records(args.saved_demo, suite["cases"])
            metadata = {"mode": "saved_demo", "timing_measured": False,
                        "note": "Different historical prompt profiles are kept separate. Word limits are applied only when present in the saved request."}
        elif args.rescore:
            previous = load_predictions(args.rescore)
            records = previous["records"]
            metadata = {**previous.get("metadata", {}), "rescored_from": str(args.rescore),
                        "original_created_at": previous.get("created_at")}
        else:
            def checkpoint(records, metadata):
                # Preserve each completed attempt, even if a later case is interrupted.
                payload = make_report(deepcopy(records), metadata)
                (args.output / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

            records, metadata = asyncio.run(run_live(
                args, suite, progress=lambda message: print(message, flush=True), checkpoint=checkpoint,
            ))
        if args.cases:
            records = [record for record in records if record["case"]["id"] in args.cases]
        report = make_report(records, metadata)
        json_path, markdown_path = write_report(report, args.output)
        for row in report["summary"]:
            print(f"{row['model']} / {row['profile']}: {row['automatic_passed']}/{row['attempted']} automatic passes; "
                  f"{row['manual_reviewed']} manually reviewed.")
        print(f"JSON: {json_path}\nReport: {markdown_path}")
        if args.require_pass and any(not record["evaluation"]["passes_all_automatic_checks"] for record in report["records"]):
            parser.exit(1, "Automatic benchmark checks failed; inspect the saved report.\n")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f"Benchmark error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Benchmark interrupted. Completed live cases, if any, are saved in progress.json.\n")


if __name__ == "__main__":
    main()
