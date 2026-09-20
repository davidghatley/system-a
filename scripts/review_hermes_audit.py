#!/usr/bin/env python3
"""Independent, local-only adversarial checks for the Hermes audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/hermes_audit/kimi-b92885e4f0161d4b2536512710e004d4892cac6e.parquet"
OUTPUT = ROOT / "artifacts/reviews/hermes_review.json"
REVISION = "b92885e4f0161d4b2536512710e004d4892cac6e"
CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.I | re.S)
RESPONSE_RE = re.compile(r"<tool_response\b[^>]*>(.*?)</tool_response\s*>", re.I | re.S)
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.I)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.I)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def parse(payload: str) -> object | None:
  try:
    return json.loads(payload.strip())
  except json.JSONDecodeError:
    return None


def normalize_task(value: str) -> str:
  return re.sub(r"\s+", " ", value.strip().lower())


def template_task(value: str) -> str:
  value = normalize_task(value)
  value = re.sub(r"\b\d+(?:\.\d+)*\b", "<num>", value)
  value = re.sub(r"`[^`]+`|'[^']+'|\"[^\"]+\"", "<quoted>", value)
  return value


def percentile(values: list[int], p: float) -> int:
  ordered = sorted(values)
  return ordered[int((len(ordered) - 1) * p)]


def distribution(values: list[int]) -> dict[str, int | float]:
  return {
    "min": min(values),
    "median": percentile(values, 0.5),
    "p95": percentile(values, 0.95),
    "max": max(values),
    "mean": round(sum(values) / len(values), 2),
  }


def main() -> None:
  rows = pq.read_table(SOURCE).to_pylist()
  call_spans = response_spans = parsed_calls = parsed_responses = 0
  invalid_calls = invalid_responses = 0
  adjacent_count_matches = adjacent_name_matches = 0
  adjacent_name_mismatches = adjacent_unparseable = 0
  response_ids_valid_shape = response_ids_unique_within_trajectory = 0
  calls_missing_explicit_id = 0
  calls_outside_row_toolset = 0
  targets_with_outside_row_toolset = set()
  targets_with_invalid_calls = set()
  targets_with_calls = targets_with_valid_calls = response_only_targets = 0
  mixed_text_and_call_targets = 0
  targets_with_prior_observation = 0
  target_rows = 0
  think_open = think_close = think_complete = malformed_think_messages = 0
  residual_reasoning_markers_after_strip = 0
  task_counts: Counter[str] = Counter()
  template_counts: Counter[str] = Counter()
  task_categories: defaultdict[str, set[str]] = defaultdict(set)
  task_toolsets: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
  toolset_counts: Counter[tuple[str, ...]] = Counter()
  system_hashes: Counter[str] = Counter()
  system_mentions_teacher = 0
  tool_definition_chars: list[int] = []
  task_chars: list[int] = []
  latest_response_chars: list[int] = []
  minimal_state_chars: list[int] = []
  rendered_prefix_hashes: Counter[str] = Counter()
  response_failure_like = response_success_like = 0

  for row_index, row in enumerate(rows):
    tools = parse(row["tools"])
    row_names = tuple(sorted(
      tool.get("function", {}).get("name")
      for tool in tools
      if isinstance(tool, dict) and isinstance(tool.get("function", {}).get("name"), str)
    ))
    toolset_counts[row_names] += 1
    tool_definition_chars.append(len(row["tools"]))
    task_chars.append(len(row["task"]))
    task_key = normalize_task(row["task"])
    task_counts[task_key] += 1
    template_counts[template_task(row["task"])] += 1
    task_categories[task_key].add(row["category"])
    task_toolsets[task_key].add(row_names)
    seen_response = False
    seen_response_value = ""
    response_ids: list[str] = []
    stripped_history: list[dict[str, str]] = []

    for message_index, message in enumerate(row["conversations"]):
      role = message["from"]
      value = message["value"] or ""
      opens = len(THINK_OPEN_RE.findall(value))
      closes = len(THINK_CLOSE_RE.findall(value))
      completes = len(THINK_RE.findall(value))
      think_open += opens
      think_close += closes
      think_complete += completes
      if opens != closes or completes != opens:
        malformed_think_messages += 1
      stripped = THINK_RE.sub("", value)
      if THINK_OPEN_RE.search(stripped) or THINK_CLOSE_RE.search(stripped):
        residual_reasoning_markers_after_strip += 1

      calls = CALL_RE.findall(value) if role == "gpt" else []
      responses = RESPONSE_RE.findall(value) if role == "tool" else []
      call_spans += len(calls)
      response_spans += len(responses)
      call_objects = [parse(payload) for payload in calls]
      response_objects = [parse(payload) for payload in responses]
      parsed_calls += sum(isinstance(obj, dict) for obj in call_objects)
      parsed_responses += sum(isinstance(obj, dict) for obj in response_objects)
      invalid_calls += sum(not isinstance(obj, dict) for obj in call_objects)
      invalid_responses += sum(not isinstance(obj, dict) for obj in response_objects)

      for obj in call_objects:
        if isinstance(obj, dict):
          calls_missing_explicit_id += int(not isinstance(obj.get("id"), str))
      for obj in response_objects:
        if isinstance(obj, dict):
          call_id = obj.get("tool_call_id")
          if isinstance(call_id, str):
            response_ids.append(call_id)
            response_ids_valid_shape += int(bool(re.fullmatch(r"functions\.[^:]+:\d+", call_id)))
          content = obj.get("content")
          text = json.dumps(content, sort_keys=True).lower()
          response_failure_like += int(any(marker in text for marker in ('"success": false', 'error', 'failed', 'not available')))
          response_success_like += int(any(marker in text for marker in ('"success": true', '"status": "success"', 'exit_code": 0')))

      if calls:
        valid_names = [obj.get("name") for obj in call_objects if isinstance(obj, dict) and isinstance(obj.get("name"), str)]
        outside = [name for name in valid_names if name not in row_names]
        calls_outside_row_toolset += len(outside)
        if outside:
          targets_with_outside_row_toolset.add((row_index, message_index))
        if len(valid_names) != len(calls):
          targets_with_invalid_calls.add((row_index, message_index))
        if message_index + 1 < len(row["conversations"]):
          next_message = row["conversations"][message_index + 1]
          next_payloads = RESPONSE_RE.findall(next_message["value"] or "") if next_message["from"] == "tool" else []
          next_objects = [parse(payload) for payload in next_payloads]
          adjacent_count_matches += int(len(calls) == len(next_payloads))
          if all(isinstance(obj, dict) for obj in call_objects + next_objects):
            call_names = [obj.get("name") for obj in call_objects]
            result_names = [obj.get("name") for obj in next_objects]
            if call_names == result_names:
              adjacent_name_matches += 1
            else:
              adjacent_name_mismatches += 1
          else:
            adjacent_unparseable += 1

      if role == "gpt":
        target_rows += 1
        key = (row_index, message_index)
        if calls:
          targets_with_calls += 1
          targets_with_valid_calls += int(key not in targets_with_invalid_calls and key not in targets_with_outside_row_toolset)
          outside_tags = CALL_RE.sub("", stripped).strip()
          mixed_text_and_call_targets += int(bool(outside_tags))
        else:
          response_only_targets += 1
        targets_with_prior_observation += int(seen_response)
        prefix = json.dumps(stripped_history, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        rendered_prefix_hashes[hashlib.sha256(prefix.encode()).hexdigest()] += 1
        if seen_response:
          latest_response_chars.append(len(seen_response_value))
          minimal_state_chars.append(len(row["task"]) + len(seen_response_value))

      if responses:
        seen_response = True
        seen_response_value = value
      stripped_history.append({"from": role, "value": stripped})

    response_ids_unique_within_trajectory += int(len(response_ids) == len(set(response_ids)))
    systems = [m["value"] or "" for m in row["conversations"] if m["from"] == "system"]
    for system in systems:
      system_hashes[hashlib.sha256(system.encode()).hexdigest()] += 1
      system_mentions_teacher += int("kimi" in system.lower() or "moonshot" in system.lower())

  duplicate_prefix_groups = sum(count > 1 for count in rendered_prefix_hashes.values())
  duplicate_prefix_excess = sum(count - 1 for count in rendered_prefix_hashes.values() if count > 1)
  exact_duplicate_groups = sum(count > 1 for count in task_counts.values())
  template_duplicate_groups = sum(count > 1 for count in template_counts.values())
  multi_category_task_groups = sum(len(values) > 1 for values in task_categories.values())
  multi_toolset_task_groups = sum(len(values) > 1 for values in task_toolsets.values())

  tokenizer_result: dict[str, object]
  snapshot = ROOT / "data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/tokenizer"
  try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    task_tokens = [len(tokenizer.encode(row["task"], add_special_tokens=False)) for row in rows]
    definition_tokens = [len(tokenizer.encode(row["tools"], add_special_tokens=False)) for row in rows]
    latest_tokens: list[int] = []
    minimal_tokens: list[int] = []
    for row in rows:
      latest = None
      for message in row["conversations"]:
        if message["from"] == "gpt" and latest is not None:
          latest_tokens.append(len(tokenizer.encode(latest, add_special_tokens=False)))
          minimal_tokens.append(len(tokenizer.encode(row["task"] + "\n" + latest, add_special_tokens=False)))
        if message["from"] == "tool":
          latest = message["value"] or ""
    tokenizer_result = {
      "tokenizer_path": str(snapshot.relative_to(ROOT)),
      "task_tokens": distribution(task_tokens),
      "tool_definition_tokens": distribution(definition_tokens),
      "latest_complete_tool_message_tokens": distribution(latest_tokens),
      "task_plus_latest_complete_tool_message_tokens": distribution(minimal_tokens),
      "task_plus_latest_observation_over_512": sum(value > 512 for value in minimal_tokens),
      "task_plus_latest_observation_count": len(minimal_tokens),
      "tool_definitions_over_512": sum(value > 512 for value in definition_tokens),
      "note": "Counts exclude Laya question/options/special-token overhead; usable state budget is therefore below 512.",
    }
  except Exception as exc:
    tokenizer_result = {"error": type(exc).__name__, "detail": str(exc)}

  output = {
    "review_version": "hermes-adversarial-review-v1",
    "decision": "NO-GO",
    "decision_scope": "training or freezing Hermes as the Experiment 001 replacement now",
    "source": {
      "repo": "lambda/hermes-agent-reasoning-traces",
      "revision": REVISION,
      "local_path": str(SOURCE.relative_to(ROOT)),
      "bytes": SOURCE.stat().st_size,
      "sha256": sha256(SOURCE),
      "rows": len(rows),
    },
    "verified": {
      "targets": {
        "all_gpt_turns": target_rows,
        "with_tool_call_span": targets_with_calls,
        "response_only_by_absence_of_complete_call_span": response_only_targets,
        "with_only_parseable_row_defined_calls": targets_with_valid_calls,
        "with_prior_tool_response": targets_with_prior_observation,
        "mixed_nonreasoning_text_and_tool_call": mixed_text_and_call_targets,
        "targets_with_unparseable_calls": len(targets_with_invalid_calls),
        "targets_with_calls_outside_own_row_toolset": len(targets_with_outside_row_toolset),
      },
      "linkage": {
        "call_spans": call_spans,
        "response_spans": response_spans,
        "parsed_calls": parsed_calls,
        "parsed_responses": parsed_responses,
        "invalid_calls": invalid_calls,
        "invalid_responses": invalid_responses,
        "calls_without_explicit_call_id": calls_missing_explicit_id,
        "assistant_turns_followed_by_equal_response_count": adjacent_count_matches,
        "assistant_turns_with_ordered_call_response_name_match": adjacent_name_matches,
        "assistant_turns_with_name_mismatch": adjacent_name_mismatches,
        "assistant_turns_unparseable_for_name_check": adjacent_unparseable,
        "response_ids_matching_functions_name_index_shape": response_ids_valid_shape,
        "trajectories_with_unique_response_ids": response_ids_unique_within_trajectory,
        "physical_execution_replayed": False,
      },
      "row_tool_taxonomy": {
        "distinct_toolsets": len(toolset_counts),
        "toolset_trajectory_counts": {"|".join(names): count for names, count in toolset_counts.most_common()},
        "calls_outside_own_row_toolset": calls_outside_row_toolset,
      },
      "reasoning": {
        "think_open_tags": think_open,
        "think_close_tags": think_close,
        "complete_think_spans": think_complete,
        "messages_with_unbalanced_or_noncanonical_think_structure": malformed_think_messages,
        "messages_with_residual_think_markers_after_complete_span_removal": residual_reasoning_markers_after_strip,
      },
      "duplicates_and_groups": {
        "unique_exact_normalized_tasks": len(task_counts),
        "exact_duplicate_task_groups": exact_duplicate_groups,
        "largest_exact_task_group": max(task_counts.values()),
        "heuristic_numeric_quoted_template_groups": len(template_counts),
        "heuristic_duplicate_template_groups": template_duplicate_groups,
        "exact_task_groups_crossing_categories": multi_category_task_groups,
        "exact_task_groups_crossing_toolsets": multi_toolset_task_groups,
        "duplicate_reasoning_stripped_prefix_groups_global": duplicate_prefix_groups,
        "duplicate_reasoning_stripped_prefix_excess_global": duplicate_prefix_excess,
      },
      "teacher_harness": {
        "distinct_system_prompt_hashes": len(system_hashes),
        "system_messages_mentioning_kimi_or_moonshot": system_mentions_teacher,
        "teacher_identity_present_as_row_field": False,
        "harness_identity_present_as_row_field": False,
        "browser_regime_trajectories": sum(count for names, count in toolset_counts.items() if any(name.startswith("browser_") for name in names)),
      },
      "outcome_signals": {
        "tool_responses_with_failure_like_text": response_failure_like,
        "tool_responses_with_success_like_text": response_success_like,
        "terminal_trajectory_success_field": False,
      },
      "character_lengths": {
        "tool_definitions": distribution(tool_definition_chars),
        "tasks": distribution(task_chars),
        "latest_complete_tool_messages": distribution(latest_response_chars),
        "task_plus_latest_complete_tool_message": distribution(minimal_state_chars),
      },
      "laya_tokenizer": tokenizer_result,
    },
    "inferred": [
      "Ordered adjacent name agreement supports serialization linkage, but calls carry no explicit IDs, so response tool_call_id values cannot be cryptographically or independently joined to call records.",
      "The publisher's real-execution claim is plausible from payload detail but is not independently authenticated by these bytes or replay.",
      "Six row-level toolsets and a distinct browser regime make a single global 26-name target vulnerable to availability and regime shortcuts.",
      "Exact normalized task grouping does not establish template or repository disjointness.",
    ],
    "unknown": [
      "Underlying licenses and redistribution rights for embedded repository, website, and task content.",
      "Whether every trajectory was generated by the named Kimi-K2.5 teacher and one invariant harness; neither identity is row-attested.",
      "Physical authenticity and fidelity of serialized tool responses without execution logs, signatures, or replay.",
      "Task-family and repository leakage after semantic or near-duplicate clustering.",
      "A frozen typed target, renderer, split manifest, and truncation policy that satisfy the amended protocol without shortcut leakage.",
    ],
    "mandatory_conditions": [
      "Refreeze the protocol for Hermes before training: recorded traces rather than successful traces; no outcome-aware claim; one source split; explicit group-split algorithm and manifests.",
      "Define one coherent typed bundle using row-level tool availability, OTHER_TOOL and malformed-call behavior, valid-bundle decoding, and train-only taxonomy decisions; do not treat RESPOND_OR_FINISH plus names as the complete Laya question specification.",
      "Validate and retain ordered call/response name linkage and response IDs; drop or quarantine every unparseable or row-undefined target and publish IDs/reasons.",
      "Cluster tasks by templates and repository/site identity in addition to exact normalized text; freeze group-disjoint train/development/calibration/test manifests and prove required trajectory and family support.",
      "Implement a target-isolation canary and renderer allowlist; remove all reasoning structurally and reject malformed tags.",
      "Freeze the local Laya tokenizer and a message-boundary renderer. Prove task, active-tool availability, and latest complete call/result survive within the actual state budget after question/options overhead; otherwise increase context under a separately measured protocol or reject the corpus.",
      "Resolve embedded-content provenance/license policy or constrain redistribution and publish a documented risk acceptance; Apache-2.0 repository metadata alone is insufficient.",
      "Treat teacher, harness, and real-execution attribution as publisher claims unless row-level provenance or independently verifiable execution evidence is obtained.",
    ],
  }
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
  main()
