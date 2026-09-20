#!/usr/bin/env python3
"""Bounded, reproducible audit of the Hermes Kimi candidate for Experiment 001."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hermes_audit"
OUT = ROOT / "artifacts/dataset_candidates/hermes"
REPO = "lambda/hermes-agent-reasoning-traces"
API = f"https://huggingface.co/api/datasets/{REPO}"
SOURCE_PATH = "data/kimi/train.parquet"
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.I | re.S)
RESULT_RE = re.compile(r"<tool_response\b[^>]*>(.*?)</tool_response\s*>", re.I | re.S)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]


def summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "mean": round(sum(values) / len(values), 2),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def parse_json_payload(payload: str) -> object | None:
    try:
        return json.loads(payload.strip())
    except (json.JSONDecodeError, TypeError):
        return None


def normalized_task(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    with urllib.request.urlopen(API, timeout=60) as response:
        metadata_raw = response.read()
    metadata = json.loads(metadata_raw)
    revision = metadata["sha"]
    sibling = next(item for item in metadata["siblings"] if item["rfilename"] == SOURCE_PATH)
    source_url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{SOURCE_PATH}"
    parquet_path = DATA / f"kimi-{revision}.parquet"
    if not parquet_path.exists():
        urllib.request.urlretrieve(source_url, parquet_path)

    source_sha = digest_bytes(parquet_path)
    lfs = sibling.get("lfs", {})
    expected_sha = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:") or None
    if expected_sha and source_sha != expected_sha:
        raise RuntimeError(f"checksum mismatch: expected {expected_sha}, got {source_sha}")

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    categories: Counter[str] = Counter()
    subcategories: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    action_names: Counter[str] = Counter()
    definition_names: Counter[str] = Counter()
    toolsets: Counter[str] = Counter()
    trajectory_hashes: Counter[str] = Counter()
    exact_tasks: Counter[str] = Counter()
    normalized_tasks: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    turn_counts: list[int] = []
    call_counts: list[int] = []
    result_counts: list[int] = []
    trajectory_chars: list[int] = []
    trajectory_words: list[int] = []
    prefix_chars: list[int] = []
    prefix_words: list[int] = []
    target_rows = 0
    response_only_targets = 0
    prior_observation_targets = 0
    calls_total = 0
    results_total = 0
    paired_adjacent = 0
    parse_failures = 0
    think_messages = 0
    residual_think_prefixes = 0
    tool_def_parse_failures = 0
    terminal_outcome_fields: Counter[str] = Counter()

    for row in rows:
        ids[row["id"]] += 1
        categories[row["category"]] += 1
        subcategories[row["subcategory"]] += 1
        exact_tasks[row["task"]] += 1
        normalized_tasks[normalized_task(row["task"])] += 1
        conversations = row["conversations"]
        roles.update(message["from"] for message in conversations)
        turn_counts.append(len(conversations))
        exact_tasks[row["task"]] += 0

        tools = parse_json_payload(row["tools"])
        if not isinstance(tools, list):
            tool_def_parse_failures += 1
            tools = []
        names = []
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = tool.get("name") if isinstance(tool, dict) else None
            name = name if isinstance(name, str) else function.get("name")
            if isinstance(name, str):
                definition_names[name] += 1
                names.append(name)
        toolsets[canonical(sorted(names))] += 1

        stripped_conversation = []
        trajectory_call_count = 0
        trajectory_result_count = 0
        observations_seen = 0
        for index, message in enumerate(conversations):
            value = message["value"] or ""
            if THINK_RE.search(value):
                think_messages += 1
            stripped = THINK_RE.sub("", value)
            stripped_conversation.append({"from": message["from"], "value": stripped})
            calls = CALL_RE.findall(value) if message["from"] == "gpt" else []
            results = RESULT_RE.findall(value) if message["from"] == "tool" else []
            trajectory_call_count += len(calls)
            trajectory_result_count += len(results)
            calls_total += len(calls)
            results_total += len(results)
            if results:
                observations_seen += len(results)
                for payload in results:
                    parsed = parse_json_payload(payload)
                    if isinstance(parsed, dict):
                        for key in ("exit_code", "error", "status", "success"):
                            if key in parsed or (isinstance(parsed.get("content"), dict) and key in parsed["content"]):
                                terminal_outcome_fields[key] += 1
            for payload in calls:
                parsed = parse_json_payload(payload)
                if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
                    action_names[parsed["name"]] += 1
                else:
                    parse_failures += 1
            if calls and index + 1 < len(conversations) and conversations[index + 1]["from"] == "tool":
                next_results = RESULT_RE.findall(conversations[index + 1]["value"] or "")
                paired_adjacent += min(len(calls), len(next_results))
            if message["from"] == "gpt":
                target_rows += 1
                if not calls:
                    response_only_targets += 1
                if observations_seen:
                    prior_observation_targets += 1
                prefix = canonical(stripped_conversation[:-1])
                if THINK_RE.search(prefix):
                    residual_think_prefixes += 1
                prefix_chars.append(len(prefix))
                prefix_words.append(len(prefix.split()))

        call_counts.append(trajectory_call_count)
        result_counts.append(trajectory_result_count)
        rendered = canonical(stripped_conversation)
        trajectory_hashes[digest_text(rendered)] += 1
        trajectory_chars.append(len(rendered))
        trajectory_words.append(len(rendered.split()))

    def duplicate_stats(counter: Counter[str]) -> dict[str, int]:
        return {
            "unique": len(counter),
            "duplicate_groups": sum(count > 1 for count in counter.values()),
            "excess_records": sum(count - 1 for count in counter.values() if count > 1),
            "max_group_size": max(counter.values()),
        }

    defined_names = set(definition_names)
    defined_action_calls = sum(count for name, count in action_names.items() if name in defined_names)
    undefined_action_calls = sum(count for name, count in action_names.items() if name not in defined_names)

    profile = {
        "audit_version": "hermes-exp001-candidate-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "complete Kimi-K2.5 config only; GLM metadata/card comparison only",
        "source": {
            "repo": REPO,
            "revision": revision,
            "config": "kimi",
            "split": "train",
            "path": SOURCE_PATH,
            "url": source_url,
            "license": metadata.get("cardData", {}).get("license"),
            "teacher": "moonshotai/Kimi-K2.5",
            "harness": "NousResearch/hermes-agent via hermes-agent-generator",
            "bytes": parquet_path.stat().st_size,
            "sha256": source_sha,
        },
        "schema": {field.name: str(field.type) for field in table.schema},
        "counts": {
            "trajectories": len(rows),
            "reconstructed_assistant_targets": target_rows,
            "response_only_targets": response_only_targets,
            "tool_call_targets": target_rows - response_only_targets,
            "targets_with_prior_tool_observation": prior_observation_targets,
            "tool_calls": calls_total,
            "tool_responses": results_total,
            "adjacent_call_response_pairs": paired_adjacent,
            "call_parse_failures": parse_failures,
            "tool_definition_parse_failures": tool_def_parse_failures,
            "think_bearing_messages": think_messages,
            "residual_think_in_reconstructed_prefixes": residual_think_prefixes,
        },
        "distributions": {
            "categories": dict(categories.most_common()),
            "subcategories": dict(subcategories.most_common()),
            "roles": dict(roles.most_common()),
            "trajectory_turns": summary(turn_counts),
            "calls_per_trajectory": summary(call_counts),
            "responses_per_trajectory": summary(result_counts),
        },
        "linkage": {
            "call_response_count_delta": calls_total - results_total,
            "adjacent_pair_rate_over_calls": round(paired_adjacent / calls_total, 6),
            "observation_bearing_target_rate": round(prior_observation_targets / target_rows, 6),
            "terminal_outcome_like_fields_inside_tool_responses": dict(terminal_outcome_fields),
            "dataset_level_task_outcome_or_reward_field": False,
        },
        "tools": {
            "definition_names_presence_by_trajectory": dict(definition_names.most_common()),
            "observed_action_calls": dict(action_names.most_common()),
            "distinct_definition_names": len(definition_names),
            "distinct_observed_action_names": len(action_names),
            "distinct_toolsets": len(toolsets),
            "defined_action_calls": defined_action_calls,
            "undefined_action_calls": undefined_action_calls,
            "undefined_action_names": sorted(name for name in action_names if name not in defined_names),
        },
        "duplicates": {
            "ids": duplicate_stats(ids),
            "reasoning_stripped_complete_trajectories": duplicate_stats(trajectory_hashes),
            "exact_task_text": duplicate_stats(exact_tasks),
            "normalized_task_text": duplicate_stats(normalized_tasks),
        },
        "lengths": {
            "reasoning_stripped_trajectory_characters": summary(trajectory_chars),
            "reasoning_stripped_trajectory_whitespace_tokens": summary(trajectory_words),
            "reconstructed_prefix_characters": summary(prefix_chars),
            "reconstructed_prefix_whitespace_tokens": summary(prefix_words),
            "model_token_counts": "UNKNOWN_NOT_MEASURED_NO_FROZEN_TOKENIZER",
        },
        "split_feasibility": {
            "source_split_count": 1,
            "source_split": "train only",
            "recommended_group_key": "normalized task text, after exact trajectory deduplication",
            "unique_normalized_task_groups": len(normalized_tasks),
            "task_group_split_possible": len(normalized_tasks) >= 3,
            "source_held_out_test_available": False,
        },
        "reasoning_exclusion": {
            "method": "remove complete <think>...</think> spans before rendering every prefix",
            "residual_span_assertion_passed": residual_think_prefixes == 0,
            "malformed_or_unclosed_think_tags": "UNKNOWN_NOT_STRUCTURALLY_PARSED",
        },
        "candidate_target_taxonomy": {
            "minimal": ["RESPOND_OR_FINISH", *definition_names.keys()],
            "semantics": "target assistant turn mode plus multi-label published Hermes tool-definition names",
            "arguments_excluded": True,
            "reasoning_excluded": True,
        },
    }

    manifest = {
        "generated_at_utc": profile["generated_at_utc"],
        "repo": REPO,
        "revision": revision,
        "files": [
            {
                "repo_path": SOURCE_PATH,
                "local_path": str(parquet_path.relative_to(ROOT)),
                "bytes": parquet_path.stat().st_size,
                "sha256": source_sha,
                "hub_lfs_sha256": expected_sha,
            }
        ],
        "metadata_api": API,
        "metadata_response_sha256": hashlib.sha256(metadata_raw).hexdigest(),
    }
    commands = "\n".join(
        [
            f"[{started}] .venv/bin/python scripts/audit_hermes_candidate.py",
            f"GET {API}",
            f"GET {source_url} -> {parquet_path.relative_to(ROOT)}",
            f"profiled complete kimi config: {len(rows)} trajectories; no training",
            f"source sha256: {source_sha}",
            f"completed: {profile['generated_at_utc']}",
            "",
        ]
    )
    (OUT / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "commands.log").write_text(commands)


if __name__ == "__main__":
    main()
