#!/usr/bin/env python3
"""Local-only reconciliation profile for the narrow Hermes preprocessing prototype."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/hermes_audit/kimi-b92885e4f0161d4b2536512710e004d4892cac6e.parquet"
TOKENIZER = ROOT / "data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/tokenizer"
OUTPUT = ROOT / "artifacts/dataset_candidates/hermes/reconciled_profile.json"
LOG = ROOT / "artifacts/dataset_candidates/hermes/reconciliation_commands.log"
SOURCE_SHA256 = "d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02"
TOOLS = ("patch", "process", "read_file", "search_files", "terminal", "write_file")
CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.I | re.S)
RESPONSE_RE = re.compile(r"<tool_response\b[^>]*>(.*?)</tool_response\s*>", re.I | re.S)
THINK_TAG_RE = re.compile(r"<think\b[^>]*>|</think\s*>", re.I)
QUESTION_BLOCK = "\n".join(f"Question call_{name}: no | yes" for name in TOOLS)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def parse_json(value: str) -> object | None:
  try:
    return json.loads(value.strip())
  except (json.JSONDecodeError, TypeError):
    return None


def tool_names(raw: str) -> tuple[str, ...]:
  definitions = parse_json(raw)
  if not isinstance(definitions, list):
    return ()
  names = []
  for definition in definitions:
    if not isinstance(definition, dict):
      continue
    name = definition.get("function", {}).get("name")
    if isinstance(name, str):
      names.append(name)
  return tuple(sorted(names))


def strip_reasoning(value: str) -> tuple[str, bool]:
  """Remove canonical, non-nested think spans; reject unmatched or nested tags."""
  output: list[str] = []
  cursor = 0
  inside = False
  valid = True
  for match in THINK_TAG_RE.finditer(value):
    closing = match.group(0).lower().startswith("</")
    if closing:
      if not inside:
        valid = False
      else:
        inside = False
    else:
      if inside:
        valid = False
      elif not inside:
        output.append(value[cursor:match.start()])
        inside = True
    cursor = match.end()
  if inside:
    valid = False
  if not inside:
    output.append(value[cursor:])
  stripped = "".join(output)
  if THINK_TAG_RE.search(stripped):
    valid = False
  return stripped, valid


def normalize_task(value: str) -> str:
  return re.sub(r"\s+", " ", value.strip().lower())


def template_task(value: str) -> str:
  value = normalize_task(value)
  value = re.sub(r"\b\d+(?:\.\d+)*\b", "<num>", value)
  return re.sub(r"`[^`]+`|'[^']+'|\"[^\"]+\"", "<quoted>", value)


def split_for(group: str) -> str:
  bucket = int(hashlib.sha256(f"20260920\n{group}".encode()).hexdigest()[:16], 16) % 10_000
  if bucket < 7000:
    return "train"
  if bucket < 8000:
    return "development"
  if bucket < 9000:
    return "calibration"
  return "test"


def distribution(values: list[int]) -> dict[str, int | float]:
  ordered = sorted(values)
  return {
    "min": ordered[0],
    "median": ordered[(len(ordered) - 1) // 2],
    "p95": ordered[int((len(ordered) - 1) * 0.95)],
    "max": ordered[-1],
    "mean": round(sum(ordered) / len(ordered), 2),
  }


def main() -> None:
  started = datetime.now(timezone.utc).isoformat()
  if sha256_file(SOURCE) != SOURCE_SHA256:
    raise RuntimeError("Pinned source checksum mismatch")
  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
  rows = pq.read_table(SOURCE).to_pylist()

  selected_trajectories = 0
  malformed_reasoning_trajectories = 0
  malformed_reasoning_messages = 0
  candidate_targets = 0
  invalid_target_calls = 0
  linkage_exceptions = 0
  mixed_text_targets = 0
  over_budget = 0
  eligible: list[dict[str, object]] = []
  family_identity_fields = Counter()

  for row_index, row in enumerate(rows):
    if tool_names(row["tools"]) != TOOLS:
      continue
    selected_trajectories += 1
    stripped_messages = []
    trajectory_reasoning_valid = True
    for message in row["conversations"]:
      stripped, valid = strip_reasoning(message["value"] or "")
      malformed_reasoning_messages += int(not valid)
      trajectory_reasoning_valid &= valid
      stripped_messages.append({"from": message["from"], "value": stripped})
    if not trajectory_reasoning_valid:
      malformed_reasoning_trajectories += 1
      continue

    for message_index, message in enumerate(stripped_messages):
      if message["from"] != "gpt" or message_index < 2:
        continue
      previous_call = stripped_messages[message_index - 2]
      previous_result = stripped_messages[message_index - 1]
      if previous_call["from"] != "gpt" or previous_result["from"] != "tool":
        continue
      prior_calls = [parse_json(value) for value in CALL_RE.findall(previous_call["value"])]
      prior_results = [parse_json(value) for value in RESPONSE_RE.findall(previous_result["value"])]
      if not prior_calls or len(prior_calls) != len(prior_results):
        linkage_exceptions += 1
        continue
      prior_names = [value.get("name") for value in prior_calls if isinstance(value, dict)]
      result_names = [value.get("name") for value in prior_results if isinstance(value, dict)]
      if len(prior_names) != len(prior_calls) or prior_names != result_names:
        linkage_exceptions += 1
        continue

      candidate_targets += 1
      target_calls = [parse_json(value) for value in CALL_RE.findall(message["value"])]
      if any(not isinstance(value, dict) or value.get("name") not in TOOLS for value in target_calls):
        invalid_target_calls += 1
        continue
      target_names = sorted({value["name"] for value in target_calls})
      mixed_text_targets += int(bool(CALL_RE.sub("", message["value"]).strip()) and bool(target_calls))
      state = (
        f"Task:\n{row['task']}\n"
        f"Available tools: {', '.join(TOOLS)}\n"
        f"Previous assistant call:\n{previous_call['value']}\n"
        f"Latest tool response:\n{previous_result['value']}\n"
        f"{QUESTION_BLOCK}"
      )
      token_ids = tokenizer.encode(state, add_special_tokens=True)
      if len(token_ids) > 512:
        over_budget += 1
        continue
      task_group = template_task(row["task"])
      eligible.append({
        "row_index": row_index,
        "trajectory_id": row["id"],
        "message_index": message_index,
        "task_group": task_group,
        "split": split_for(task_group),
        "state_hash": hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
        "tokens": len(token_ids),
        "labels": target_names,
      })

  eligible.sort(key=lambda item: (str(item["trajectory_id"]), int(item["message_index"])))
  deduplicated = []
  seen_states = set()
  for item in eligible:
    if item["state_hash"] in seen_states:
      continue
    seen_states.add(item["state_hash"])
    deduplicated.append(item)

  split_rows = Counter(item["split"] for item in deduplicated)
  split_trajectories: defaultdict[str, set[str]] = defaultdict(set)
  split_groups: defaultdict[str, set[str]] = defaultdict(set)
  split_positive_rows: defaultdict[str, Counter[str]] = defaultdict(Counter)
  split_positive_trajectories: defaultdict[str, defaultdict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
  for item in deduplicated:
    split = str(item["split"])
    trajectory = str(item["trajectory_id"])
    split_trajectories[split].add(trajectory)
    split_groups[split].add(str(item["task_group"]))
    for label in item["labels"]:
      split_positive_rows[split][label] += 1
      split_positive_trajectories[split][label].add(trajectory)

  group_partitions: defaultdict[str, set[str]] = defaultdict(set)
  for item in deduplicated:
    group_partitions[str(item["task_group"])].add(str(item["split"]))
  trajectory_partitions: defaultdict[str, set[str]] = defaultdict(set)
  for item in deduplicated:
    trajectory_partitions[str(item["trajectory_id"])].add(str(item["split"]))

  canary = "TARGET_CANARY_6f20c2"
  canary_state = f"Task:\ncanary task\nAvailable tools: {', '.join(TOOLS)}\nPrevious assistant call:\ncall\nLatest tool response:\nresult\n{QUESTION_BLOCK}"
  canary_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(canary_state, add_special_tokens=True))
  profile = {
    "profile_version": "hermes-reconciliation-v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "decision": "GO_BOUNDED_PREPROCESSING_PROTOTYPE_ONLY",
    "main_training_decision": "NO_GO",
    "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": SOURCE_SHA256, "rows": len(rows)},
    "subset": {
      "exact_row_toolset": list(TOOLS),
      "source_trajectories": selected_trajectories,
      "requires_prior_immediately_adjacent_parseable_call_result_pair": True,
      "physical_execution_authenticated": False,
      "serialization_description": "publisher-attributed responses with measured ordered adjacency and name agreement",
    },
    "target": {
      "questions": [{"name": f"call_{name}", "primitive": "noul", "options": ["no", "yes"]} for name in TOOLS],
      "valid_bundle": "empty set means RESPOND_OR_FINISH; otherwise nonempty set of yes tools",
      "decoder": "independent scores are decoded to the highest-scoring member of the 64 valid six-bit bundles; mode is derived",
      "mixed_text_plus_call": "tool set is the target; non-reasoning text is not a separate label",
      "unknown_or_malformed": "exclude; never map to RESPOND_OR_FINISH; OTHER_TOOL is unreachable in this fixed-toolset prototype",
    },
    "renderer": {
      "fields": ["task", "fixed active-tool names", "latest complete assistant call", "latest complete tool response", "six frozen question strings"],
      "tokenizer_path": str(TOKENIZER.relative_to(ROOT)),
      "max_tokens_including_special_tokens_and_questions": 512,
      "truncation": "none; reject over-budget states so no call/result payload is split",
      "eligible_token_lengths": distribution([int(item["tokens"]) for item in deduplicated]),
      "target_canary_absent_after_tokenization": all(canary not in token for token in canary_tokens),
      "renderer_accepts_target_content": False,
    },
    "filters": {
      "malformed_reasoning_messages_in_selected_subset": malformed_reasoning_messages,
      "trajectories_rejected_for_any_malformed_reasoning": malformed_reasoning_trajectories,
      "candidate_observation_conditioned_targets": candidate_targets,
      "targets_rejected_for_invalid_or_out_of_toolset_target_calls": invalid_target_calls,
      "prior_linkage_exceptions": linkage_exceptions,
      "targets_rejected_over_512": over_budget,
      "mixed_text_plus_call_targets_before_budget_and_dedup": mixed_text_targets,
      "representable_before_exact_state_dedup": len(eligible),
      "exact_tokenized_state_duplicates_removed": len(eligible) - len(deduplicated),
      "final_rows": len(deduplicated),
      "final_trajectories": len({str(item["trajectory_id"]) for item in deduplicated}),
      "final_task_template_proxy_groups": len({str(item["task_group"]) for item in deduplicated}),
    },
    "split": {
      "group_key": "lowercase whitespace-normalized task with numeric and quoted literals replaced",
      "warning": "This is a deterministic template proxy, not authenticated repository/site/task-family identity.",
      "algorithm": "sha256(UTF-8 '20260920\\n' + group_key), first 64 bits modulo 10000; [0,7000) train, [7000,8000) development, [8000,9000) calibration, [9000,10000) test",
      "rows": dict(split_rows),
      "trajectories": {key: len(value) for key, value in split_trajectories.items()},
      "groups": {key: len(value) for key, value in split_groups.items()},
      "positive_rows_by_tool": {key: dict(value) for key, value in split_positive_rows.items()},
      "positive_trajectories_by_tool": {key: {tool: len(ids) for tool, ids in value.items()} for key, value in split_positive_trajectories.items()},
      "task_groups_crossing_splits": sum(len(value) > 1 for value in group_partitions.values()),
      "trajectories_crossing_splits": sum(len(value) > 1 for value in trajectory_partitions.values()),
      "state_hashes_unique": len(seen_states) == len(deduplicated),
    },
    "provenance": {
      "teacher_harness": "publisher-level attribution only; absent from row fields",
      "execution": "not externally authenticated; no replay, signature, call-side ID, or external log reference",
      "embedded_content_license": "unknown; repository Apache-2.0 metadata does not authenticate embedded-content rights",
      "repository_site_task_family_ids": "absent; not inferred as trusted identifiers",
    },
    "prototype_stop_gates": [
      "No training or official test access under this profile.",
      "Stop if any generated row contains a reasoning marker, target canary, incomplete call/result pair, non-six-tool label, or more than 512 tokens.",
      "Stop before training unless task-family/repository/site isolation is established from auditable evidence or the claim is narrowed to template-proxy-disjoint within-corpus imitation.",
      "Stop before redistribution until embedded-content provenance and license policy is accepted explicitly.",
      "Stop if every split lacks prespecified minimum independent trajectories and positive trajectories per retained tool; thresholds must be frozen in a revised protocol.",
      "Do not describe responses as authenticated execution, traces as successful, or teacher/harness identity as row-attested.",
    ],
  }
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
  LOG.write_text("\n".join([
    f"[{started}] .venv/bin/python scripts/reconcile_hermes_candidate.py",
    "local-only: no download, task replay, model inference, or training",
    f"source_sha256={SOURCE_SHA256}",
    f"tokenizer={TOKENIZER.relative_to(ROOT)}",
    f"completed={profile['generated_at_utc']}",
    "",
  ]))


if __name__ == "__main__":
  main()
