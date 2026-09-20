#!/usr/bin/env python3
"""Pin, download, validate, and profile the Experiment 001 trace dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_DATASET = "greghavens/kimi-k3-coding-and-debugging-traces"
DEFAULT_REVISION = "33a874c3affbdb97e142752a9144e6624ef5bd07"
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {
    "task",
    "source_trajectory_sha256",
    "category",
    "split",
    "assistant_step",
    "target_message_index",
    "messages",
}
REDACTED_RE = re.compile(
    r"(?:<redacted>|\[redacted\]|\*{3,}|REDACTED|sk-[A-Za-z0-9_*.-]+)", re.I
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantiles(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(p: float) -> int:
        return ordered[round((len(ordered) - 1) * p)]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
    }


def normalize_task(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def task_text(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return str(row["task"])


def strip_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_reasoning(item)
            for key, item in value.items()
            if key != "reasoning_content"
        }
    if isinstance(value, list):
        return [strip_reasoning(item) for item in value]
    return value


def render_state(row: dict[str, Any]) -> dict[str, Any]:
    target_index = row["target_message_index"]
    history = row["messages"][:target_index]
    state = {
        "template": "exp001-observable-v1-candidate",
        "task": task_text(row),
        "messages": strip_reasoning(history),
    }
    rendered = canonical(state)
    assert state["messages"] == strip_reasoning(history)
    assert len(state["messages"]) == target_index, "target message entered state slice"
    assert "reasoning_content" not in rendered, "reasoning leaked into rendered state"
    return state


def tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    return list(message.get("tool_calls") or [])


def tool_name(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    return str(function.get("name") or call.get("name") or "<missing>")


def tool_arguments(call: dict[str, Any]) -> Any:
    function = call.get("function") or {}
    return function.get("arguments", call.get("arguments"))


def is_empty_argument(arguments: Any) -> bool:
    if arguments is None:
        return True
    if isinstance(arguments, (dict, list)):
        return len(arguments) == 0
    text = str(arguments).strip()
    return text in {"", "{}", "[]", "null", "None"}


def download(url: str, destination: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "system-a-exp001-preflight/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return data


def counter_dict(counter: collections.Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def confined_path(path: Path, label: str) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        raise ValueError(f"{label} must remain under {REPO_ROOT}: {resolved}")
    return resolved


def intersection_report(sets: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    names = sorted(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left] & sets[right]
            result[f"{left}__{right}"] = {
                "count": len(overlap),
                "sample_count_capped": min(len(overlap), 10),
            }
    return result


def duplicate_summary(groups: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    duplicated = [rows for rows in groups.values() if len(rows) > 1]
    return {
        "duplicate_groups": len(duplicated),
        "rows_in_duplicate_groups": sum(len(rows) for rows in duplicated),
        "excess_duplicate_rows": sum(len(rows) - 1 for rows in duplicated),
    }


def inspect_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    schema_errors: list[str] = []
    split_counts: collections.Counter[str] = collections.Counter()
    split_trajectories: dict[str, set[str]] = collections.defaultdict(set)
    split_states: dict[str, set[str]] = collections.defaultdict(set)
    split_tasks: dict[str, set[str]] = collections.defaultdict(set)
    tool_names: collections.Counter[str] = collections.Counter()
    multiplicity: collections.Counter[int] = collections.Counter()
    state_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    task_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    final_trajectory_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    exact_call_groups: collections.Counter[tuple[str, str]] = collections.Counter()
    message_lengths: list[int] = []
    state_chars: list[int] = []
    target_chars: list[int] = []
    response_only = empty_targets = empty_arguments = redacted_arguments = 0
    turns_with_duplicate_names = turns_with_duplicate_exact_calls = 0
    target_index_valid = target_assistant_valid = target_last_valid = 0
    target_calls = missing_result_observations = observed_result_calls = 0
    history_tool_results = history_tool_calls = 0
    rows_with_any_observation = rows_with_prior_call = 0
    historical_roles: collections.Counter[str] = collections.Counter()

    for row_number, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            schema_errors.append(f"row {row_number}: missing {sorted(missing)}")
            continue
        split = str(row["split"])
        trajectory = str(row["source_trajectory_sha256"])
        messages = row["messages"]
        index = row["target_message_index"]
        split_counts[split] += 1
        split_trajectories[split].add(trajectory)
        split_tasks[split].add(normalize_task(task_text(row)))
        message_lengths.append(len(messages))
        if isinstance(index, int) and 0 <= index < len(messages):
            target_index_valid += 1
        else:
            schema_errors.append(f"row {row_number}: invalid target index {index!r}")
            continue
        target = messages[index]
        if target.get("role") == "assistant":
            target_assistant_valid += 1
        if index == len(messages) - 1:
            target_last_valid += 1
        state = render_state(row)
        state_json = canonical(state)
        state_hash = sha256_bytes(state_json.encode())
        split_states[split].add(state_hash)
        state_groups[state_hash].append(row)
        task_groups[normalize_task(task_text(row))].append(row)
        state_chars.append(len(state_json))
        target_json = canonical(target)
        target_chars.append(len(target_json))
        calls = tool_calls(target)
        multiplicity[len(calls)] += 1
        if not calls:
            response_only += 1
        if not calls and not str(target.get("content") or "").strip():
            empty_targets += 1
        turn_names: list[str] = []
        turn_exact_calls: list[tuple[str, str]] = []
        for call in calls:
            name = tool_name(call)
            arguments = tool_arguments(call)
            turn_names.append(name)
            turn_exact_calls.append((name, canonical(arguments)))
            tool_names[name] += 1
            target_calls += 1
            exact_call_groups[(name, canonical(arguments))] += 1
            empty_arguments += int(is_empty_argument(arguments))
            redacted_arguments += int(bool(REDACTED_RE.search(str(arguments or ""))))
        turns_with_duplicate_names += int(len(turn_names) != len(set(turn_names)))
        turns_with_duplicate_exact_calls += int(
            len(turn_exact_calls) != len(set(turn_exact_calls))
        )

        historical = messages[:index]
        historical_roles.update(str(message.get("role") or "<missing>") for message in historical)
        calls_by_id: dict[str, int] = collections.Counter(
            str(call.get("id"))
            for message in historical
            if message.get("role") == "assistant"
            for call in tool_calls(message)
            if call.get("id") is not None
        )
        results_by_id = collections.Counter(
            str(message.get("tool_call_id"))
            for message in historical
            if message.get("role") == "tool" and message.get("tool_call_id") is not None
        )
        history_tool_calls += sum(calls_by_id.values())
        history_tool_results += sum(results_by_id.values())
        rows_with_prior_call += int(bool(calls_by_id))
        rows_with_any_observation += int(bool(results_by_id))
        for call_id, count in calls_by_id.items():
            observed = min(count, results_by_id[call_id])
            observed_result_calls += observed
            missing_result_observations += count - observed

    final_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        trajectory = str(row.get("source_trajectory_sha256"))
        if trajectory not in final_rows or row.get("assistant_step", 0) > final_rows[trajectory].get("assistant_step", 0):
            final_rows[trajectory] = row
    for row in final_rows.values():
        final_trajectory_groups[sha256_bytes(canonical(row["messages"]).encode())].append(row)

    split_trajectory_overlap = intersection_report(split_trajectories)
    split_state_overlap = intersection_report(split_states)
    split_task_overlap = intersection_report(split_tasks)
    exact_duplicate_calls = sum(count - 1 for count in exact_call_groups.values() if count > 1)
    raw_names = sorted(tool_names)
    identity_map = {name: name for name in raw_names}
    lexical_map = {
        name: (
            "shell" if re.search(r"bash|shell|terminal|exec|command", name, re.I)
            else "file" if re.search(r"read|write|edit|file|glob|grep|search", name, re.I)
            else "web" if re.search(r"web|browser|fetch|http", name, re.I)
            else "other"
        )
        for name in raw_names
    }
    leakage_pass = all(item["count"] == 0 for item in split_trajectory_overlap.values())
    observations_ratio = (
        rows_with_any_observation / rows_with_prior_call if rows_with_prior_call else 1.0
    )
    return {
        "schema_validation": {
            "required_fields": sorted(REQUIRED_FIELDS),
            "errors": schema_errors,
            "passed": not schema_errors,
        },
        "counts": {
            "rows": len(rows),
            "trajectories": len(final_rows),
            "rows_by_split": counter_dict(split_counts),
            "trajectories_by_split": {
                split: len(values) for split, values in sorted(split_trajectories.items())
            },
            "normalized_tasks": len(task_groups),
        },
        "target_index": {
            "in_bounds": target_index_valid,
            "assistant_role": target_assistant_valid,
            "is_final_message": target_last_valid,
            "all_valid": target_index_valid == target_assistant_valid == target_last_valid == len(rows),
        },
        "targets": {
            "tool_names": counter_dict(tool_names),
            "tool_call_multiplicity_per_turn": counter_dict(multiplicity),
            "tool_calls": target_calls,
            "response_only_turns": response_only,
            "empty_targets": empty_targets,
            "empty_arguments": empty_arguments,
            "redacted_arguments": redacted_arguments,
            "turns_with_duplicate_tool_names": turns_with_duplicate_names,
            "turns_with_duplicate_exact_calls": turns_with_duplicate_exact_calls,
            "duplicate_exact_name_and_arguments_excess": exact_duplicate_calls,
        },
        "observations": {
            "historical_message_roles": counter_dict(historical_roles),
            "historical_tool_calls_with_ids": history_tool_calls,
            "historical_tool_result_messages_with_ids": history_tool_results,
            "matched_call_result_pairs": observed_result_calls,
            "missing_tool_result_observations": missing_result_observations,
            "rows_with_prior_tool_call": rows_with_prior_call,
            "rows_with_actual_tool_result_observation": rows_with_any_observation,
            "observable_history_row_ratio_given_prior_call": round(observations_ratio, 6),
        },
        "duplicates": {
            "exact_rendered_states": duplicate_summary(state_groups),
            "normalized_tasks": duplicate_summary(task_groups),
            "exact_full_trajectories": duplicate_summary(final_trajectory_groups),
        },
        "split_leakage": {
            "trajectory_hash_overlap": split_trajectory_overlap,
            "exact_rendered_state_overlap": split_state_overlap,
            "normalized_task_overlap": split_task_overlap,
            "trajectory_disjoint_passed": leakage_pass,
        },
        "sequence_lengths": {
            "messages_per_row": quantiles(message_lengths),
            "rendered_state_characters": quantiles(state_chars),
            "target_message_characters": quantiles(target_chars),
            "note": "Character counts are tokenizer-independent; enforce model-token limits only after tokenizer pinning.",
        },
        "renderer_assertions": {
            "candidate_template": "exp001-observable-v1-candidate",
            "target_assistant_excluded": True,
            "reasoning_content_excluded_recursively": True,
            "status": "PROPOSED_NOT_FROZEN",
        },
        "candidate_tool_family_maps": [
            {
                "name": "identity-minimal",
                "map": identity_map,
                "status": "CANDIDATE_NOT_SELECTED",
                "rationale": "No semantic collapse; one family per observed raw tool name.",
            },
            {
                "name": "lexical-coarse",
                "map": lexical_map,
                "status": "CANDIDATE_NOT_SELECTED",
                "rationale": "Review-only lexical grouping; ambiguous names remain other.",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=Path("artifacts/exp_001_data_profile.json"))
    parser.add_argument(
        "--checksums", type=Path, default=Path("artifacts/exp_001_dataset_checksums.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/exp_001_dataset"))
    args = parser.parse_args()
    if args.dataset != DEFAULT_DATASET:
        parser.error(f"this preflight is restricted to {DEFAULT_DATASET}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full 40-character commit SHA")
    try:
        args.output = confined_path(args.output, "--output")
        args.checksums = confined_path(args.checksums, "--checksums")
        args.cache_dir = confined_path(args.cache_dir, "--cache-dir")
    except ValueError as error:
        parser.error(str(error))

    base_url = f"https://huggingface.co/datasets/{args.dataset}/resolve/{args.revision}/"
    root = args.cache_dir / args.revision
    manifest_path = root / "dataset-manifest.json"
    manifest_data = download(base_url + "dataset-manifest.json", manifest_path)
    manifest = json.loads(manifest_data)
    checksums: dict[str, dict[str, Any]] = {
        "dataset-manifest.json": {
            "bytes": len(manifest_data),
            "sha256": sha256_bytes(manifest_data),
        }
    }
    parquet_paths: list[Path] = []
    for relative_path in manifest["active_shards"]:
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"unsafe shard path in pinned manifest: {relative_path!r}")
        encoded_path = "/".join(urllib.parse.quote(part) for part in relative_path.split("/"))
        local_path = root / relative_path
        data = download(base_url + encoded_path, local_path)
        parquet_paths.append(local_path)
        checksums[relative_path] = {"bytes": len(data), "sha256": sha256_bytes(data)}

    table = pq.read_table(parquet_paths)
    rows = table.to_pylist()
    profile = {
        "profile_version": 1,
        "dataset": args.dataset,
        "revision": args.revision,
        "source_url": f"https://huggingface.co/datasets/{args.dataset}/tree/{args.revision}",
        "manifest_claims": {
            "row_count": manifest.get("row_count"),
            "bytes": manifest.get("bytes"),
            "shards": len(manifest["active_shards"]),
        },
        "inspection_scope": "All active Parquet shards at the pinned revision; structural audit only, with no model training, tuning, prediction scoring, or outcome-based selection.",
        **inspect_rows(rows),
    }
    profile["manifest_validation"] = {
        "row_count_matches": len(rows) == manifest.get("row_count"),
        "shard_count_matches": len(parquet_paths) == len(manifest["active_shards"]),
        "byte_count_matches": sum(checksums[path]["bytes"] for path in manifest["active_shards"])
        == manifest.get("bytes"),
    }
    profile["viability_gates"] = {
        "minimum_source_train_rows_1500": profile["counts"]["rows_by_split"].get("train", 0) >= 1500,
        "minimum_source_val_rows_300": profile["counts"]["rows_by_split"].get("val", 0) >= 300,
        "trajectory_disjoint": profile["split_leakage"]["trajectory_disjoint_passed"],
        "observable_history_not_mostly_absent": profile["observations"][
            "observable_history_row_ratio_given_prior_call"
        ] >= 0.5,
    }
    profile["viability_gates"]["passed"] = all(profile["viability_gates"].values())
    profile["integrity_assertions"] = {
        "manifest_valid": all(profile["manifest_validation"].values()),
        "schema_valid": profile["schema_validation"]["passed"],
        "target_indexes_valid": profile["target_index"]["all_valid"],
        "source_splits_trajectory_disjoint": profile["split_leakage"][
            "trajectory_disjoint_passed"
        ],
    }
    profile["integrity_assertions"]["passed"] = all(
        profile["integrity_assertions"].values()
    )

    checksum_document = {
        "dataset": args.dataset,
        "revision": args.revision,
        "hash_algorithm": "sha256",
        "files": {path: checksums[path] for path in sorted(checksums)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checksums.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    args.checksums.write_text(json.dumps(checksum_document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} and {args.checksums}")
    print(f"rows={len(rows)} trajectories={profile['counts']['trajectories']} gates={profile['viability_gates']}")
    return 0 if profile["integrity_assertions"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
