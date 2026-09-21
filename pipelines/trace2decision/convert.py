#!/usr/bin/env python3
"""Convert cumulative GLM-5.2 agent traces to typed next-action decisions."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ONTOLOGY = (
    "read",
    "search",
    "edit",
    "execute",
    "other_tool",
    "respond_or_finish",
)
QUESTION = {
    "type": "choice",
    "instructions": "Choose the next observable agent action type.",
    "criteria": {
        "read": "Read a known file or resource.",
        "search": "Search or list files, symbols, or resources.",
        "edit": "Create or modify files or structured content.",
        "execute": "Execute a shell command, program, test, or process.",
        "other_tool": "Call another tool not covered by the named action types.",
        "respond_or_finish": "Respond without an executable tool call or finish the task.",
    },
}
SPLIT_SEED = "trace2decision-i1-20260921"
MAX_TASK_CHARS = 6000
MAX_SYSTEM_CHARS = 2500
MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_CHARS = 14000

READ_NAMES = {"read", "read_file", "cat", "view", "open_file"}
SEARCH_NAMES = {"ls", "list", "glob", "grep", "find", "search", "search_files", "ripgrep"}
EDIT_NAMES = {"edit", "write", "write_file", "apply_patch", "patch", "replace"}
EXECUTE_NAMES = {"bash", "shell", "terminal", "exec", "execute", "execute_code", "run", "process"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded(text: str, limit: int) -> str:
    """Deterministically retain both ends when one field exceeds its budget."""
    if len(text) <= limit:
        return text
    marker = f"\n...[TRUNCATED {len(text) - limit} CHARS]...\n"
    room = limit - len(marker)
    left = room // 2
    return text[:left] + marker + text[-(room - left) :]


def normalize_task(text: str) -> str:
    return " ".join(text.lower().split())


def action_for_tool(name: str) -> str:
    normalized = name.lower().strip().replace("-", "_")
    if normalized in READ_NAMES or normalized.startswith("read_"):
        return "read"
    if normalized in SEARCH_NAMES or any(token in normalized for token in ("search", "grep", "glob", "find")):
        return "search"
    if normalized in EDIT_NAMES or any(token in normalized for token in ("edit", "write", "patch")):
        return "edit"
    if normalized in EXECUTE_NAMES or any(token in normalized for token in ("shell", "terminal", "execute")):
        return "execute"
    return "other_tool"


def target_action(target: dict[str, Any]) -> tuple[str, list[str]]:
    calls = target.get("tool_calls") or []
    names = [str(call.get("function", {}).get("name", "")) for call in calls]
    if not names:
        return "respond_or_finish", names
    # The contract requires one choice. For parallel bundles, the first serialized
    # executable call is the observed next action and later calls remain audit data.
    return action_for_tool(names[0]), names


def render_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "unknown")).upper()
    parts: list[str] = []
    content = str(message.get("content") or "")
    if content:
        parts.append(bounded(content, MAX_MESSAGE_CHARS))
    calls = message.get("tool_calls") or []
    for call in calls:
        function = call.get("function", {})
        parts.append(
            "TOOL_CALL "
            + canonical_json(
                {
                    "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", ""),
                }
            )
        )
    if message.get("tool_call_id"):
        parts.insert(0, f"TOOL_RESULT id={message['tool_call_id']}")
    if message.get("name"):
        parts.insert(0, f"name={message['name']}")
    return f"[{role}]\n" + ("\n".join(parts) if parts else "[EMPTY]")


def initial_fields(prefix: list[dict[str, Any]]) -> tuple[str, str, int]:
    system = ""
    task = ""
    task_index = -1
    for index, message in enumerate(prefix):
        role = message.get("role")
        if role == "system" and not system:
            system = str(message.get("content") or "")
        if role == "user" and not task:
            task = str(message.get("content") or "")
            task_index = index
    return system, task, task_index


def render_state(prefix: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    system, task, task_index = initial_fields(prefix)
    candidates = [
        bounded(render_message(message), MAX_MESSAGE_CHARS)
        for index, message in enumerate(prefix)
        if index != task_index and message.get("role") != "system"
    ]
    retained: list[str] = []
    used = 0
    for rendered in reversed(candidates):
        cost = len(rendered) + (2 if retained else 0)
        if used + cost > MAX_HISTORY_CHARS:
            break
        retained.append(rendered)
        used += cost
    retained.reverse()
    omitted = len(candidates) - len(retained)
    history = "\n\n".join(retained) if retained else "[NO PRIOR ACTION HISTORY]"
    if omitted:
        history = f"[OMITTED {omitted} OLDER MESSAGES]\n\n" + history
    state = (
        "TASK\n"
        + bounded(task, MAX_TASK_CHARS)
        + "\n\nSYSTEM CONSTRAINTS\n"
        + bounded(system, MAX_SYSTEM_CHARS)
        + "\n\nAVAILABLE NEXT-ACTION TYPES\n"
        + ", ".join(ONTOLOGY)
        + "\n\nOBSERVABLE HISTORY BEFORE TARGET\n"
        + history
    )
    return state, {
        "history_candidates": len(candidates),
        "history_retained": len(retained),
        "history_omitted": omitted,
        "latest_prefix_role": prefix[-1].get("role") if prefix else None,
        "has_prior_tool_result": any(message.get("role") == "tool" for message in prefix),
    }


def split_for(group_key: str) -> str:
    bucket = int(hashlib.sha256(f"{SPLIT_SEED}\n{group_key}".encode()).hexdigest()[:16], 16) % 10000
    if bucket < 8000:
        return "train"
    if bucket < 9000:
        return "dev"
    return "test"


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values, default=0),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values, default=0),
        "mean": round(sum(values) / len(values), 2) if values else 0,
    }


def load_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def convert(source: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    malformed: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    source_trajectory_rows: collections.Counter[str] = collections.Counter()
    source_split_rows: collections.Counter[str] = collections.Counter()
    source_tool_names: collections.Counter[str] = collections.Counter()
    parallel_targets = 0
    prior_observation_rows = 0
    target_mutation_failures = 0
    rows_with_unresolved_prior_call_ids = 0
    task_groups: dict[str, set[str]] = collections.defaultdict(set)
    trajectory_signatures: dict[str, set[str]] = collections.defaultdict(set)
    final_messages: dict[str, list[dict[str, Any]]] = {}

    for line_number, row in load_rows(source):
        source_rows += 1
        messages = row.get("messages")
        target_index = row.get("target_message_index")
        trajectory = str(row.get("source_trajectory_sha256") or "")
        if (
            not isinstance(messages, list)
            or not messages
            or not trajectory
            or target_index != len(messages) - 1
            or row.get("n_messages") != len(messages)
            or messages[-1].get("role") != "assistant"
        ):
            malformed.append({"line": line_number, "task": row.get("task"), "reason": "invalid cumulative target boundary"})
            continue
        target = messages[-1]
        prefix = messages[:-1]
        source_split_rows[str(row.get("split"))] += 1
        if int(row.get("assistant_step", 0)) == int(row.get("assistant_steps", -1)):
            trajectory_signatures[sha256_bytes(canonical_json(messages).encode())].add(trajectory)
            final_messages[trajectory] = messages
        system, task_text, _ = initial_fields(prefix)
        if not task_text:
            malformed.append({"line": line_number, "task": row.get("task"), "reason": "missing pre-target user task"})
            continue
        state, state_info = render_state(prefix)
        prior_call_ids = {
            str(call.get("id"))
            for message in prefix
            for call in (message.get("tool_calls") or [])
            if call.get("id")
        }
        prior_result_ids = {
            str(message.get("tool_call_id"))
            for message in prefix
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        unresolved_prior_call_ids = prior_call_ids - prior_result_ids
        rows_with_unresolved_prior_call_ids += bool(unresolved_prior_call_ids)
        if unresolved_prior_call_ids:
            continue
        mutated = dict(target)
        mutated["content"] = "TARGET_CANARY_9a31"
        mutated_messages = prefix + [mutated]
        if render_state(mutated_messages[:-1])[0] != state or "TARGET_CANARY_9a31" in state:
            target_mutation_failures += 1
        action, names = target_action(target)
        source_tool_names.update(names)
        parallel_targets += len(names) > 1
        prior_observation_rows += state_info["has_prior_tool_result"]
        group_key = sha256_bytes(normalize_task(task_text).encode())
        task_groups[group_key].add(trajectory)
        source_trajectory_rows[trajectory] += 1
        probabilities = {option: float(option == action) for option in ONTOLOGY}
        record = {
            "state": state,
            "questions": {"next_action": QUESTION},
            "gold": {"next_action": {"type": "choice", "label": action, "probabilities": probabilities}},
        }
        converted.append(
            {
                "record": record,
                "line": line_number,
                "task": str(row.get("task")),
                "task_group": group_key,
                "trajectory": trajectory,
                "assistant_step": int(row.get("assistant_step")),
                "action": action,
                "raw_tool_names": names,
                "source_split": str(row.get("split")),
                "state_chars": len(state),
                **state_info,
            }
        )

    converted.sort(key=lambda item: (item["task_group"], item["trajectory"], item["assistant_step"], item["line"]))
    duplicate_states = 0
    conflicting_states = 0
    seen_states: dict[str, str] = {}
    retained: list[dict[str, Any]] = []
    for item in converted:
        state_hash = sha256_bytes(item["record"]["state"].encode())
        if state_hash in seen_states:
            duplicate_states += 1
            conflicting_states += seen_states[state_hash] != item["action"]
            continue
        seen_states[state_hash] = item["action"]
        item["state_sha256"] = state_hash
        item["split"] = split_for(item["task_group"])
        retained.append(item)

    split_rows: collections.Counter[str] = collections.Counter()
    split_trajectories: dict[str, set[str]] = collections.defaultdict(set)
    split_groups: dict[str, set[str]] = collections.defaultdict(set)
    labels: collections.Counter[str] = collections.Counter()
    label_by_split: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    trajectory_splits: dict[str, set[str]] = collections.defaultdict(set)
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    state_splits: dict[str, set[str]] = collections.defaultdict(set)
    files: dict[str, Any] = {}

    unique_calls = 0
    unique_results = 0
    linked_results = 0
    orphan_results = 0
    missing_results = 0
    final_target_call_ids_without_result = 0
    for messages in final_messages.values():
        seen_call_ids: set[str] = set()
        result_ids: set[str] = set()
        for message_index, message in enumerate(messages):
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    call_id = str(call.get("id") or "")
                    unique_calls += 1
                    if call_id:
                        seen_call_ids.add(call_id)
                        if message_index == len(messages) - 1:
                            final_target_call_ids_without_result += 1
            if message.get("role") == "tool":
                unique_results += 1
                call_id = str(message.get("tool_call_id") or "")
                if call_id and call_id in seen_call_ids:
                    linked_results += 1
                    result_ids.add(call_id)
                else:
                    orphan_results += 1
        missing_results += len(seen_call_ids - result_ids)

    for split in ("train", "dev", "test"):
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in retained:
                if item["split"] != split:
                    continue
                handle.write(canonical_json(item["record"]) + "\n")
                split_rows[split] += 1
                split_trajectories[split].add(item["trajectory"])
                split_groups[split].add(item["task_group"])
                labels[item["action"]] += 1
                label_by_split[split][item["action"]] += 1
                trajectory_splits[item["trajectory"]].add(split)
                group_splits[item["task_group"]].add(split)
                state_splits[item["state_sha256"]].add(split)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": split_rows[split]}

    sample_items = sorted(retained, key=lambda item: sha256_bytes(f"sample\n{item['trajectory']}\n{item['assistant_step']}".encode()))[:12]
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sample_items:
            sample = {
                "source": {key: item[key] for key in ("task", "trajectory", "assistant_step", "split", "raw_tool_names")},
                "record": item["record"],
            }
            handle.write(canonical_json(sample) + "\n")
    files[samples_path.name] = {"bytes": samples_path.stat().st_size, "sha256": sha256_file(samples_path), "rows": len(sample_items)}

    report = {
        "schema_version": "trace2decision-i1-v1",
        "source": {
            "dataset": "11-47/glm-5.2-coding-and-debugging-traces",
            "revision": "1371ed38f8890d0520a53bc7ad850308eb4d7a22",
            "file": str(source),
            "sha256": sha256_file(source),
            "rows": source_rows,
            "source_split_rows": dict(sorted(source_split_rows.items())),
        },
        "contract": {
            "top_level_fields": ["state", "questions", "gold"],
            "question": QUESTION,
            "gold_semantics": "one-hot observed behavior; behavioral cloning, not optimality",
            "parallel_call_rule": "first serialized tool call determines the single next_action label",
            "reasoning_content_included": False,
            "future_derived_tools_used_included": False,
        },
        "truncation": {
            "task_chars": MAX_TASK_CHARS,
            "system_chars": MAX_SYSTEM_CHARS,
            "per_message_chars": MAX_MESSAGE_CHARS,
            "history_chars": MAX_HISTORY_CHARS,
            "field_policy": "head and tail with explicit marker",
            "history_policy": "newest complete rendered messages that fit; older messages omitted explicitly",
        },
        "conversion": {
            "structurally_valid_source_rows": source_rows - len(malformed),
            "eligible_rows_after_complete_prior_linkage": len(converted),
            "malformed_rows": len(malformed),
            "retained_rows": len(retained),
            "trajectories": len({item["trajectory"] for item in retained}),
            "task_prompt_groups": len({item["task_group"] for item in retained}),
            "rows_with_prior_tool_result": prior_observation_rows,
            "rows_with_unresolved_prior_call_ids": rows_with_unresolved_prior_call_ids,
            "parallel_call_targets": parallel_targets,
            "exact_rendered_state_duplicates_removed": duplicate_states,
            "duplicate_states_with_conflicting_labels": conflicting_states,
            "target_mutation_failures": target_mutation_failures,
            "unique_source_calls": unique_calls,
            "unique_source_tool_results": unique_results,
            "tool_results_linked_to_prior_call_id": linked_results,
            "orphan_tool_results": orphan_results,
            "prior_call_ids_without_result": missing_results,
            "final_target_call_ids_without_result": final_target_call_ids_without_result,
            "source_tool_names": dict(sorted(source_tool_names.items())),
            "malformed_examples": malformed[:20],
        },
        "distribution": {
            "labels": dict(sorted(labels.items())),
            "labels_by_split": {split: dict(sorted(label_by_split[split].items())) for split in ("train", "dev", "test")},
            "state_chars": distribution([item["state_chars"] for item in retained]),
            "history_messages_omitted": distribution([item["history_omitted"] for item in retained]),
            "rows_per_trajectory": distribution(list(source_trajectory_rows.values())),
        },
        "split": {
            "seed": SPLIT_SEED,
            "group": "sha256(normalized initial user task text)",
            "algorithm": "sha256(seed + newline + group), first 64 bits modulo 10000; 0-7999 train, 8000-8999 dev, 9000-9999 test",
            "rows": {split: split_rows[split] for split in ("train", "dev", "test")},
            "trajectories": {split: len(split_trajectories[split]) for split in ("train", "dev", "test")},
            "groups": {split: len(split_groups[split]) for split in ("train", "dev", "test")},
            "trajectory_crossings": sum(len(value) > 1 for value in trajectory_splits.values()),
            "task_group_crossings": sum(len(value) > 1 for value in group_splits.values()),
            "state_hash_crossings": sum(len(value) > 1 for value in state_splits.values()),
            "normalized_task_duplicate_groups": sum(len(value) > 1 for value in task_groups.values()),
            "exact_complete_trajectory_duplicate_groups": sum(len(value) > 1 for value in trajectory_signatures.values()),
        },
        "files": files,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = convert(args.source, args.output_dir)
    print(canonical_json({"rows": report["conversion"]["retained_rows"], "files": report["files"]}))


if __name__ == "__main__":
    main()
