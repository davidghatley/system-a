"""Terminal entry point for Reflex Replay."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

from .core import ADVISORY_WARNING, PROVENANCE_WARNING, ReplayError, load_predictions, load_suite, replay


APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]


def _print_case(case: dict) -> None:
    record = case["record"]
    metadata = record["metadata"]
    print("=" * 78)
    print(f"CASE: {case['name']}  [{', '.join(case['tags'])}]")
    print(f"RECORD: {case['record_id']}  split=dev  source_step={metadata['source_step']}")
    print("COMPACT INPUT (exact model-facing state):")
    print(record["state"])
    print(f"OBSERVED REFERENCE: {case['observed_label']} (source-trace action, not an ideal action)")
    for result in case["predictions"]:
        identity = result["model"]
        status = "CORRECT" if result["correct"] else "WRONG"
        ambiguity = "AMBIGUOUS/LOW-MARGIN" if result["ambiguous_low_margin"] else "not low-margin"
        print(f"\nMODEL: {identity['id']}@{identity['revision']}")
        print("PROBABILITIES: " + "  ".join(f"{label}={value:.4f}" for label, value in result["probabilities"].items()))
        print(f"PREDICTION: {result['selected']}  EVALUATION: {status}  UNCERTAINTY FLAG: {ambiguity}")
        print(f"TOP-TWO MARGIN: {result['top_two_margin']:.4f}")
        print(f"ELAPSED: {result['elapsed_seconds']:.6f}s ({result['elapsed_scope']})")
        print(f"PREDICTION PROVENANCE: {result['provenance']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay stored observed next-action predictions on a frozen real-record suite.")
    parser.add_argument("--suite", type=pathlib.Path, default=APP_ROOT / "examples" / "frozen_suite.json")
    parser.add_argument("--predictions", type=pathlib.Path, default=APP_ROOT / "examples" / "stored_predictions.jsonl")
    parser.add_argument("--case", help="display one frozen case name or record ID")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        cases = load_suite(REPO_ROOT, args.suite)
        results = replay(cases, load_predictions(args.predictions))
        if args.case:
            results = [case for case in results if args.case in (case["name"], case["record_id"])]
            if not results:
                raise ReplayError(f"unknown case: {args.case}")
    except (ReplayError, OSError, ValueError) as exc:
        print(f"reflex-replay: error: {exc}", file=sys.stderr)
        return 2
    print("TASK: observed next-action prediction/evaluation")
    print(f"WARNING: {ADVISORY_WARNING}")
    print(f"PROVENANCE WARNING: {PROVENANCE_WARNING}")
    for case in results:
        _print_case(case)
    print("=" * 78)
    print(f"REPLAY ELAPSED: {time.perf_counter() - started:.6f}s (file loading, validation, rule, rendering)")
    print(f"SUMMARY: {len(results)} fixed cases; this suite is diagnostic and not an accuracy estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
