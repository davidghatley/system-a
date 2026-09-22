#!/usr/bin/env python3
"""Verify v3 semantics, leakage boundaries, grouping, and deterministic output."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from i3_convert import (  # noqa: E402
    HEAD_MAX_LEN,
    LayaQ,
    MAX_LEN,
    TOKENIZER_DEFAULT,
    AutoTokenizer,
    build_sequence,
    canon,
    classify_target,
    convert,
    file_sha,
    load_v2_splits,
    serialize_state,
    state_for,
)
from shared.typed_decisions import ValidationError, validate_record  # noqa: E402


def source_rows(path: Path) -> dict[int, dict]:
    return {line: json.loads(text) for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if text.strip()}


def validate(directory: Path, source: Path, tokenizer_path: Path, v2_dir: Path) -> dict:
    report = json.loads((directory / "report.json").read_text())
    sources = source_rows(source)
    v2 = load_v2_splits(v2_dir)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    empty, markers = build_sequence(tok, "", LayaQ, MAX_LEN, HEAD_MAX_LEN)
    room = MAX_LEN - len(empty)
    failures = []
    ids, trajectories, groups = set(), collections.defaultdict(set), collections.defaultdict(set)
    state_locations = collections.defaultdict(set)
    state_counts = collections.Counter()
    counts = collections.Counter()
    lengths = []
    unique_target_string_checks = 0
    rows = 0
    for split in ("train", "dev", "test"):
        path = directory / f"{split}.jsonl"
        if file_sha(path) != report["files"][path.name]["sha256"]:
            failures.append(f"{path.name}: hash mismatch")
        with path.open(encoding="utf-8") as handle:
            for line, text in enumerate(handle, 1):
                rows += 1
                record = json.loads(text)
                meta = record.get("metadata", {})
                try:
                    validate_record(record)
                except ValidationError as error:
                    failures.append(f"{path.name}:{line}: shared validator: {error}")
                if meta.get("id") in ids:
                    failures.append(f"{path.name}:{line}: duplicate stable id")
                ids.add(meta.get("id"))
                trajectories[meta.get("trajectory_id")].add(split)
                groups[meta.get("task_group")].add(split)
                state_locations[record["state"]].add(split)
                state_counts[record["state"]] += 1
                counts[(split, record["gold"]["next_action"]["label"])] += 1
                source_row = sources.get(meta.get("source_line"))
                if source_row is None:
                    failures.append(f"{path.name}:{line}: missing source row")
                    continue
                classification = classify_target(source_row)
                expected_label = classification.get("label")
                if not classification.get("retain") or expected_label != record["gold"]["next_action"]["label"]:
                    failures.append(f"{path.name}:{line}: whole-turn label mismatch")
                expected_state, _ = state_for(source_row["messages"][:-1], tok, room)
                if record["state"] != expected_state:
                    failures.append(f"{path.name}:{line}: state is not exact prefix rendering")
                if v2.get((meta.get("trajectory_id"), int(meta.get("source_step", -1)))) != split:
                    failures.append(f"{path.name}:{line}: v2 split assignment changed")
                if any(str(meta.get(key, "")) in record["state"] for key in ("id", "trajectory_id", "task_group")):
                    failures.append(f"{path.name}:{line}: provenance metadata leaked")
                target = source_row["messages"][-1]
                prefix_text = canon(source_row["messages"][:-1])
                target_strings = [str(target.get("content") or ""), str(target.get("reasoning_content") or "")]
                target_strings.extend(call["function"]["arguments"] for call in target.get("tool_calls", []))
                for value in target_strings:
                    if len(value) >= 24 and value not in prefix_text:
                        unique_target_string_checks += 1
                        if value in record["state"]:
                            failures.append(f"{path.name}:{line}: target-only string leaked")
                full = tok(serialize_state(record["state"]).replace(tok.mask_token, " "), add_special_tokens=False, truncation=False)["input_ids"]
                sequence, got_markers = build_sequence(tok, record["state"], LayaQ, MAX_LEN, HEAD_MAX_LEN)
                if len(full) > room or sequence != empty[:-1] + full + [tok.sep_token_id] or got_markers != markers:
                    failures.append(f"{path.name}:{line}: invalid actual Laya construction")
                lengths.append(len(sequence))
        if sum(value for (assigned, _), value in counts.items() if assigned == split) != report["split"]["rows"][split]:
            failures.append(f"{split}: report row count mismatch")
    if rows != report["conversion"]["retained_rows"]:
        failures.append("total retained count mismatch")
    if any(len(value) > 1 for value in trajectories.values()):
        failures.append("trajectory crosses split")
    if any(len(value) > 1 for value in groups.values()):
        failures.append("task group crosses split")
    cross_split_duplicate_states = sum(len(splits) > 1 for splits in state_locations.values())
    if cross_split_duplicate_states:
        failures.append("identical rendered state crosses split")
    for name in ("exclusions.jsonl", "sample_review.jsonl"):
        path = directory / name
        if file_sha(path) != report["files"][name]["sha256"]:
            failures.append(f"{name}: hash mismatch")
    exclusion_rows = sum(1 for _ in (directory / "exclusions.jsonl").open(encoding="utf-8"))
    if exclusion_rows != report["conversion"]["excluded_rows"]:
        failures.append("exclusion count mismatch")
    no_call_exclusions = 0
    with (directory / "exclusions.jsonl").open(encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            item = json.loads(text)
            source_row = sources.get(item.get("source_line"), {})
            target = (source_row.get("messages") or [{}])[-1]
            calls = target.get("tool_calls") if isinstance(target, dict) else None
            if isinstance(calls, list) and not calls:
                no_call_exclusions += 1
                if item.get("reason") != "no_call_without_positive_completion":
                    failures.append(f"exclusions.jsonl:{line}: no-call reason is not conservative policy")
    if no_call_exclusions != report["conversion"]["no_call_audit"]["counts"]["total"]:
        failures.append("no-call ledger total mismatch")
    if report["conversion"]["no_call_audit"]["all_excluded"] is not True:
        failures.append("no-call ledger does not prove all excluded")
    if report["ontology"]["absent_classes"] != ["other_tool", "respond_or_finish"]:
        failures.append("fixed ontology absent-class declaration mismatch")
    return {
        "valid": not failures,
        "failures": failures,
        "rows": rows,
        "stable_ids": len(ids),
        "trajectories": len(trajectories),
        "task_groups": len(groups),
        "duplicate_exact_states": sum(count - 1 for count in state_counts.values()),
        "cross_split_duplicate_states": cross_split_duplicate_states,
        "unique_target_strings_checked": unique_target_string_checks,
        "actual_laya_lengths": {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 2)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-dir", type=Path, required=True)
    parser.add_argument("--rerun-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DEFAULT)
    parser.add_argument("--v2-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_report = json.loads((args.expected_dir / "report.json").read_text())
    rerun_report = convert(args.source, args.rerun_dir, args.tokenizer, args.v2_dir)
    names = ("train.jsonl", "dev.jsonl", "test.jsonl", "exclusions.jsonl", "sample_review.jsonl")
    evidence = {
        "expected": validate(args.expected_dir, args.source, args.tokenizer, args.v2_dir),
        "rerun": validate(args.rerun_dir, args.source, args.tokenizer, args.v2_dir),
        "report_equal": expected_report == rerun_report,
        "file_comparison": {
            name: {
                "expected": expected_report["files"][name]["sha256"],
                "rerun": rerun_report["files"][name]["sha256"],
                "equal": expected_report["files"][name]["sha256"] == rerun_report["files"][name]["sha256"],
            }
            for name in names
        },
    }
    evidence["pass"] = (
        evidence["expected"]["valid"]
        and evidence["rerun"]["valid"]
        and evidence["report_equal"]
        and all(item["equal"] for item in evidence["file_comparison"].values())
    )
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    raise SystemExit(0 if evidence["pass"] else 1)


if __name__ == "__main__":
    main()
