#!/usr/bin/env python3
"""Rerun conversion and independently validate contract files and hashes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert import ONTOLOGY, canonical_json, convert, sha256_file
from shared.typed_decisions import ValidationError, validate_record


def validate_dataset(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text())
    rows = 0
    failures: list[str] = []
    for split in ("train", "dev", "test"):
        path = directory / f"{split}.jsonl"
        if sha256_file(path) != report["files"][path.name]["sha256"]:
            failures.append(f"{path.name}: hash mismatch")
        split_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                split_rows += 1
                record = json.loads(line)
                if set(record) != {"state", "questions", "gold"}:
                    failures.append(f"{path.name}:{line_number}: top-level contract")
                    continue
                if record["questions"] != {"next_action": report["contract"]["question"]}:
                    failures.append(f"{path.name}:{line_number}: question mismatch")
                distribution = record.get("gold", {}).get("next_action", {}).get("probabilities", {})
                if set(distribution) != set(ONTOLOGY) or sorted(distribution.values()) != [0.0] * 5 + [1.0]:
                    failures.append(f"{path.name}:{line_number}: gold is not one-hot")
                try:
                    validate_record(record)
                except ValidationError as exc:
                    failures.append(f"{path.name}:{line_number}: shared validator: {exc}")
                if "TARGET_CANARY_9a31" in record["state"]:
                    failures.append(f"{path.name}:{line_number}: target canary leaked")
        if split_rows != report["files"][path.name]["rows"]:
            failures.append(f"{path.name}: row count mismatch")
        rows += split_rows
    if rows != report["conversion"]["retained_rows"]:
        failures.append("total row count mismatch")
    for key in ("trajectory_crossings", "task_group_crossings", "state_hash_crossings"):
        if report["split"][key] != 0:
            failures.append(f"nonzero {key}")
    if report["conversion"]["target_mutation_failures"] != 0:
        failures.append("target mutation leakage")
    return {"rows": rows, "failures": failures, "valid": not failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-dir", type=Path, required=True)
    parser.add_argument("--rerun-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    expected_report = json.loads((args.expected_dir / "report.json").read_text())
    rerun_report = convert(args.source, args.rerun_dir)
    expected_validation = validate_dataset(args.expected_dir)
    rerun_validation = validate_dataset(args.rerun_dir)
    compared = ["train.jsonl", "dev.jsonl", "test.jsonl", "samples.jsonl"]
    comparison = {
        name: {
            "expected": expected_report["files"][name]["sha256"],
            "rerun": rerun_report["files"][name]["sha256"],
            "equal": expected_report["files"][name]["sha256"] == rerun_report["files"][name]["sha256"],
        }
        for name in compared
    }
    report_equal = canonical_json(expected_report) == canonical_json(rerun_report)
    evidence = {
        "expected_validation": expected_validation,
        "rerun_validation": rerun_validation,
        "file_comparison": comparison,
        "report_equal": report_equal,
        "pass": expected_validation["valid"] and rerun_validation["valid"] and report_equal and all(item["equal"] for item in comparison.values()),
    }
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(evidence))
    if not evidence["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
