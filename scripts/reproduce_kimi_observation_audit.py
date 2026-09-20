#!/usr/bin/env python3
"""Independently challenge whether the pinned Kimi traces contain tool results."""

from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
REVISION = "33a874c3affbdb97e142752a9144e6624ef5bd07"
DATA_ROOT = ROOT / "data" / "exp_001_dataset" / REVISION
CHECKSUM_PATH = ROOT / "artifacts" / "exp_001_dataset_checksums.json"
OUTPUT_PATH = ROOT / "artifacts" / "reviews" / "kimi_reproduction.json"
LOG_PATH = ROOT / "artifacts" / "reviews" / "kimi_reproduction_commands.log"
OBSERVATION_KEY_RE = re.compile(
    r"(?:^|_)(?:observation|result|output|response|stdout|stderr|exit_code|return_code)(?:$|_)",
    re.I,
)
OUTPUT_LIKE_RE = re.compile(
    r"(?:^|\n)(?:[\w./~-]+[$#>] |(?:stdout|stderr|exit code|output|result):|"
    r"(?:error|warning|failed|passed|collected)\b)",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def walk(value: Any, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.append((child, item))
            found.extend(walk(item, child))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk(item, f"{path}[]"))
    return found


def sorted_counter(counter: collections.Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def main() -> int:
    manifest = json.loads((DATA_ROOT / "dataset-manifest.json").read_text())
    expected = json.loads(CHECKSUM_PATH.read_text())
    checksum_failures: list[dict[str, Any]] = []
    for relative, claim in expected["files"].items():
        path = DATA_ROOT / relative if relative != "dataset-manifest.json" else DATA_ROOT / relative
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256(path)
        if actual_bytes != claim["bytes"] or actual_sha256 != claim["sha256"]:
            checksum_failures.append(
                {
                    "path": relative,
                    "expected_bytes": claim["bytes"],
                    "actual_bytes": actual_bytes,
                    "expected_sha256": claim["sha256"],
                    "actual_sha256": actual_sha256,
                }
            )

    shard_paths = [DATA_ROOT / item for item in manifest["active_shards"]]
    schemas = collections.Counter(str(pq.read_schema(path)) for path in shard_paths)
    table = pq.read_table(shard_paths)
    rows = table.to_pylist()

    top_level_fields = collections.Counter[str]()
    nested_paths = collections.Counter[str]()
    observation_named_paths = collections.Counter[str]()
    roles = collections.Counter[str]()
    content_by_role = collections.Counter[str]()
    tool_calls_by_role = collections.Counter[str]()
    tool_call_ids_by_role = collections.Counter[str]()
    messages_with_observation_named_fields = 0
    assistant_content_after_call = 0
    assistant_output_like_after_call = 0
    nonassistant_content_after_call = 0
    direct_call_result_pairs = 0
    rows_with_explicit_result_channel = 0
    rows_with_assistant_content_after_call = 0
    rows_with_output_like_assistant_content_after_call = 0
    candidate_examples: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        top_level_fields.update(row.keys())
        for field_path, value in walk(row):
            nested_paths[field_path] += 1
            if OBSERVATION_KEY_RE.search(field_path.split(".")[-1].replace("[]", "")):
                observation_named_paths[field_path] += 1

        prior_call_ids: set[str] = set()
        row_explicit = False
        row_assistant_after_call = False
        row_output_like = False
        for message_index, message in enumerate(row.get("messages") or []):
            role = str(message.get("role") or "<missing>")
            roles[role] += 1
            content = str(message.get("content") or "")
            if content.strip():
                content_by_role[role] += 1
            calls = message.get("tool_calls") or []
            if calls:
                tool_calls_by_role[role] += len(calls)
            message_call_id = message.get("tool_call_id")
            if str(message_call_id or "").strip():
                tool_call_ids_by_role[role] += 1
                row_explicit = True
                if str(message_call_id) in prior_call_ids:
                    direct_call_result_pairs += 1
            if role.lower() in {"tool", "function", "observation", "result"}:
                row_explicit = True
            message_observation_fields = [
                path
                for path, value in walk(message)
                if OBSERVATION_KEY_RE.search(path.split(".")[-1].replace("[]", ""))
                and value not in (None, "", [], {})
            ]
            if message_observation_fields:
                messages_with_observation_named_fields += 1
                row_explicit = True
            if prior_call_ids and content.strip():
                if role == "assistant":
                    assistant_content_after_call += 1
                    row_assistant_after_call = True
                    output_like = bool(OUTPUT_LIKE_RE.search(content))
                    assistant_output_like_after_call += int(output_like)
                    row_output_like |= output_like
                    if output_like and len(candidate_examples) < 10:
                        candidate_examples.append(
                            {
                                "row": row_index,
                                "task": row.get("task"),
                                "message_index": message_index,
                                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                                "content_prefix": content[:160].replace("\n", "\\n"),
                            }
                        )
                else:
                    nonassistant_content_after_call += 1
            for call in calls:
                if call.get("id") is not None:
                    prior_call_ids.add(str(call["id"]))
        rows_with_explicit_result_channel += int(row_explicit)
        rows_with_assistant_content_after_call += int(row_assistant_after_call)
        rows_with_output_like_assistant_content_after_call += int(row_output_like)

    call_argument_paths = {
        path: count
        for path, count in sorted(nested_paths.items())
        if "tool_calls[]" in path and path.endswith(("arguments", "name", "id", "type"))
    }
    result = {
        "review_version": 1,
        "dataset": "greghavens/kimi-k3-coding-and-debugging-traces",
        "revision": REVISION,
        "scope": "All locally pinned active shards; structural inspection only; no training or model scoring.",
        "labels": {
            "VERIFIED": [
                "All pinned manifest and Parquet file bytes match the recorded SHA-256 and size claims."
                if not checksum_failures
                else "One or more pinned files failed checksum verification.",
                f"Loaded {len(rows)} rows from {len(shard_paths)} active shards.",
                "No explicit environment-result channel was found in any record."
                if rows_with_explicit_result_channel == 0
                else "At least one explicit result-like channel was found and requires review.",
            ],
            "INFERRED": [
                "Assistant prose after a tool call can describe or imitate execution output, but the records provide no provenance that makes it an authentic environment observation.",
                "Tool-call arguments encode requested actions or payloads, not the environment response to those actions.",
            ],
            "UNKNOWN": [
                "Whether any assistant prose was copied from unretained tool output cannot be established from the pinned release.",
                "Whether an upstream artifact outside this pinned release retains authentic observations is not established here.",
            ],
            "RECOMMENDED": [
                "Keep the stop gate closed for observation-conditioned training.",
                "Only reopen after pinning and auditing a source with explicit, provenance-preserving environment result records.",
            ],
        },
        "integrity": {
            "checksum_files_checked": len(expected["files"]),
            "checksum_failures": checksum_failures,
            "manifest_active_shards": len(shard_paths),
            "manifest_rows": manifest["row_count"],
            "loaded_rows": len(rows),
            "uniform_parquet_schema": len(schemas) == 1,
            "distinct_schema_count": len(schemas),
        },
        "raw_schema": {
            "arrow_schema": str(table.schema),
            "top_level_field_presence": sorted_counter(top_level_fields),
            "nested_field_presence": sorted_counter(nested_paths),
            "observation_named_field_presence": sorted_counter(observation_named_paths),
            "tool_call_field_presence": call_argument_paths,
        },
        "adversarial_channels": {
            "message_roles": sorted_counter(roles),
            "nonempty_content_messages_by_role": sorted_counter(content_by_role),
            "tool_calls_by_message_role": sorted_counter(tool_calls_by_role),
            "tool_call_id_messages_by_role": sorted_counter(tool_call_ids_by_role),
            "messages_with_nonempty_observation_named_fields": messages_with_observation_named_fields,
            "matched_explicit_call_result_pairs": direct_call_result_pairs,
            "rows_with_explicit_result_channel": rows_with_explicit_result_channel,
            "assistant_content_messages_after_prior_call": assistant_content_after_call,
            "rows_with_assistant_content_after_prior_call": rows_with_assistant_content_after_call,
            "heuristic_output_like_assistant_messages_after_prior_call": assistant_output_like_after_call,
            "rows_with_heuristic_output_like_assistant_content_after_prior_call": rows_with_output_like_assistant_content_after_call,
            "nonassistant_content_messages_after_prior_call": nonassistant_content_after_call,
            "heuristic_candidate_examples": candidate_examples,
            "heuristic_limit": "Lexical candidates are not authenticated observations and are reported only to challenge false negatives.",
        },
        "verdict": {
            "stop_gate_justified": rows_with_explicit_result_channel == 0 and not checksum_failures,
            "honest_representation_recovers_environment_observations": False,
            "reason": "The pinned bytes contain actions and assistant-authored text but no explicit result role, linked result ID, result/output field, or other provenance-preserving environment channel.",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    OUTPUT_PATH.write_text(payload)
    log = (
        "# Kimi observation audit reproduction command log\n"
        "# Workspace boundary: /home/davidhatley/Projects/research/system_a\n"
        "$ .venv/bin/python scripts/reproduce_kimi_observation_audit.py\n"
        f"loaded_rows={len(rows)} active_shards={len(shard_paths)} checksum_failures={len(checksum_failures)}\n"
        f"explicit_result_rows={rows_with_explicit_result_channel} matched_pairs={direct_call_result_pairs}\n"
        f"assistant_after_call={assistant_content_after_call} heuristic_output_like={assistant_output_like_after_call}\n"
        f"output={OUTPUT_PATH.relative_to(ROOT)} sha256={hashlib.sha256(payload.encode()).hexdigest()}\n"
    )
    LOG_PATH.write_text(log)
    print(log, end="")
    return 0 if result["verdict"]["stop_gate_justified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
