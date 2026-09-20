#!/usr/bin/env python3
"""Build the frozen exp_001b preprocessing artifacts. No model is loaded or run."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/exp_001b_hermes_preprocessing"
SOURCE = ROOT / "data/hermes_audit/kimi-b92885e4f0161d4b2536512710e004d4892cac6e.parquet"
TOKENIZER_PATH = ROOT / "data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/tokenizer"
DATA_OUT = ROOT / "data/exp_001b"
RESULTS = EXP / "results"
ANALYSIS = EXP / "analysis"
LOGS = EXP / "logs"
SOURCE_SHA256 = "d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02"
TOOLS = ("patch", "process", "read_file", "search_files", "terminal", "write_file")
SPLITS = ("train", "development", "calibration", "test")
QUESTION_BLOCK = "\n".join(f"Question call_{name}: no | yes" for name in TOOLS)
CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.I | re.S)
RESPONSE_RE = re.compile(r"<tool_response\b[^>]*>(.*?)</tool_response\s*>", re.I | re.S)
THINK_TAG_RE = re.compile(r"<think\b[^>]*>|</think\s*>", re.I)
REASONING_MARKER_RE = re.compile(r"</?think\b|<\|(?:begin|end)_of_thought\|>|(?:^|\n)\s*(?:reasoning|analysis)\s*:", re.I)


def canonical(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def parse_json(value: str) -> Any | None:
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
      return ()
    function = definition.get("function")
    name = function.get("name") if isinstance(function, dict) else definition.get("name")
    if not isinstance(name, str):
      return ()
    names.append(name)
  return tuple(sorted(names))


def strip_reasoning(value: str) -> tuple[str, bool]:
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
      else:
        output.append(value[cursor:match.start()])
        inside = True
    cursor = match.end()
  if inside:
    valid = False
  else:
    output.append(value[cursor:])
  stripped = "".join(output)
  return stripped, valid and not THINK_TAG_RE.search(stripped)


def proxy_group(task: str) -> str:
  value = re.sub(r"\s+", " ", task.strip().lower())
  value = re.sub(r"\b\d+(?:\.\d+)*\b", "<num>", value)
  return re.sub(r"`[^`]+`|'[^']+'|\"[^\"]+\"", "<quoted>", value)


def assign_split(group: str) -> str:
  bucket = int(hashlib.sha256(f"20260920\n{group}".encode()).hexdigest()[:16], 16) % 10_000
  return "train" if bucket < 7000 else "development" if bucket < 8000 else "calibration" if bucket < 9000 else "test"


def render(task: str, prior_call: str, prior_result: str) -> str:
  return (f"Task:\n{task}\nAvailable tools: {', '.join(TOOLS)}\n"
          f"Previous assistant call:\n{prior_call}\nLatest tool response:\n{prior_result}\n{QUESTION_BLOCK}")


def rate(numerator: int, denominator: int) -> float:
  return round(numerator / denominator, 6) if denominator else 0.0


def distribution(values: list[int]) -> dict[str, int | float]:
  ordered = sorted(values)
  return {"min": ordered[0], "median": ordered[(len(ordered) - 1) // 2],
          "p95": ordered[int((len(ordered) - 1) * 0.95)], "max": ordered[-1],
          "mean": round(sum(ordered) / len(ordered), 2)}


def main() -> None:
  for directory in (DATA_OUT, RESULTS, ANALYSIS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)
  source_sha = sha256_file(SOURCE)
  if source_sha != SOURCE_SHA256:
    raise RuntimeError(f"STOP: source checksum mismatch: {source_sha}")

  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
  rows = pq.read_table(SOURCE).to_pylist()
  ledger: list[dict[str, Any]] = []
  candidates: list[dict[str, Any]] = []
  counts = Counter()
  canary_failures = 0

  for row_index, source_row in enumerate(rows):
    trajectory_id = str(source_row["id"])
    if tool_names(source_row["tools"]) != TOOLS:
      ledger.append({"trajectory_id": trajectory_id, "row_index": row_index, "scope": "trajectory", "reason": "non_exact_toolset"})
      continue
    counts["exact_toolset_trajectories"] += 1
    stripped_messages = []
    malformed_indexes = []
    for message_index, message in enumerate(source_row["conversations"]):
      stripped, valid = strip_reasoning(message["value"] or "")
      if not valid:
        malformed_indexes.append(message_index)
      stripped_messages.append({"from": message["from"], "value": stripped})
    if malformed_indexes:
      ledger.append({"trajectory_id": trajectory_id, "row_index": row_index, "scope": "trajectory",
                     "reason": "malformed_or_nested_reasoning", "message_indexes": malformed_indexes})
      counts["malformed_reasoning_trajectories"] += 1
      continue

    for message_index, target_message in enumerate(stripped_messages):
      if target_message["from"] != "gpt" or message_index < 2:
        continue
      prior_call = stripped_messages[message_index - 2]
      prior_result = stripped_messages[message_index - 1]
      if prior_call["from"] != "gpt" or prior_result["from"] != "tool":
        continue
      raw_calls = CALL_RE.findall(prior_call["value"])
      raw_results = RESPONSE_RE.findall(prior_result["value"])
      if not raw_calls or not raw_results:
        continue
      counts["observation_conditioned_candidates"] += 1
      calls = [parse_json(value) for value in raw_calls]
      results = [parse_json(value) for value in raw_results]
      call_names = [value.get("name") for value in calls if isinstance(value, dict)]
      result_names = [value.get("name") for value in results if isinstance(value, dict)]
      if (len(raw_calls) != len(raw_results) or len(call_names) != len(raw_calls)
          or len(result_names) != len(raw_results) or call_names != result_names):
        ledger.append({"trajectory_id": trajectory_id, "message_index": message_index,
                       "scope": "candidate", "reason": "invalid_prior_call_result_linkage"})
        counts["linkage_exclusions"] += 1
        continue
      target_payloads = CALL_RE.findall(target_message["value"])
      target_calls = [parse_json(value) for value in target_payloads]
      if any(not isinstance(value, dict) or value.get("name") not in TOOLS for value in target_calls):
        ledger.append({"trajectory_id": trajectory_id, "message_index": message_index,
                       "scope": "candidate", "reason": "malformed_or_unavailable_target"})
        counts["invalid_target_exclusions"] += 1
        continue
      labels = sorted({value["name"] for value in target_calls})
      group = proxy_group(source_row["task"])
      split = assign_split(group)
      state = render(source_row["task"], prior_call["value"], prior_result["value"])
      token_ids = tokenizer.encode(state, add_special_tokens=True)

      # Target and excluded metadata are intentionally outside the renderer signature.
      canary_record = dict(source_row)
      for field in ("id", "category", "subcategory", "tools"):
        canary_record[field] = f"EXCLUDED_METADATA_CANARY_{field}"
      canary_labels = ["TARGET_CANARY"]
      canary_state = render(source_row["task"], prior_call["value"], prior_result["value"])
      if tokenizer.encode(canary_state, add_special_tokens=True) != token_ids or canary_labels == labels:
        canary_failures += 1

      candidate = {"trajectory_id": trajectory_id, "message_index": message_index, "category": source_row["category"],
                   "proxy_group": group, "split": split, "labels": labels, "token_count": len(token_ids),
                   "state": state, "token_ids": token_ids,
                   "state_hash": hashlib.sha256(canonical(token_ids).encode()).hexdigest()}
      candidates.append(candidate)
      if len(token_ids) > 512:
        ledger.append({"trajectory_id": trajectory_id, "message_index": message_index, "scope": "candidate",
                       "reason": "over_budget", "split": split, "category": source_row["category"],
                       "proxy_group_sha256": hashlib.sha256(group.encode()).hexdigest(), "labels": labels,
                       "token_count": len(token_ids)})

  valid_budget = [item for item in candidates if item["token_count"] <= 512]
  valid_budget.sort(key=lambda item: (item["trajectory_id"], item["message_index"]))
  retained = []
  seen: set[tuple[int, ...]] = set()
  for item in valid_budget:
    token_key = tuple(item["token_ids"])
    if token_key in seen:
      ledger.append({"trajectory_id": item["trajectory_id"], "message_index": item["message_index"],
                     "scope": "candidate", "reason": "duplicate_token_state", "state_hash": item["state_hash"]})
      continue
    seen.add(token_key)
    retained.append(item)

  split_stats: dict[str, Any] = {}
  support: dict[str, Any] = {}
  for split in SPLITS:
    split_candidates = [item for item in candidates if item["split"] == split]
    split_retained = [item for item in retained if item["split"] == split]
    over = sum(item["token_count"] > 512 for item in split_candidates)
    split_stats[split] = {"budget_eligible_candidates": len(split_candidates), "over_budget": over,
                          "over_budget_rate": rate(over, len(split_candidates)), "retained_rows": len(split_retained),
                          "retained_trajectories": len({item["trajectory_id"] for item in split_retained}),
                          "retained_proxy_groups": len({item["proxy_group"] for item in split_retained})}
    support[split] = {}
    for tool in TOOLS:
      positives = [item for item in split_retained if tool in item["labels"]]
      support[split][tool] = {"positive_rows": len(positives),
                              "positive_trajectories": len({item["trajectory_id"] for item in positives})}

  def bias_table(key: str) -> dict[str, Any]:
    buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
      buckets[str(item[key])].append(item)
    return {name: {"candidates": len(items), "over_budget": sum(item["token_count"] > 512 for item in items),
                   "over_budget_rate": rate(sum(item["token_count"] > 512 for item in items), len(items))}
            for name, items in sorted(buckets.items())}

  label_bias = {}
  label_rate_differences = {}
  for tool in TOOLS:
    positive = [item for item in candidates if tool in item["labels"]]
    negative = [item for item in candidates if tool not in item["labels"]]
    positive_rate = rate(sum(item["token_count"] > 512 for item in positive), len(positive))
    negative_rate = rate(sum(item["token_count"] > 512 for item in negative), len(negative))
    label_bias[tool] = {"positive": {"candidates": len(positive), "over_budget_rate": positive_rate},
                        "negative": {"candidates": len(negative), "over_budget_rate": negative_rate}}
    label_rate_differences[tool] = round(abs(positive_rate - negative_rate), 6)

  split_rates = [split_stats[name]["over_budget_rate"] for name in SPLITS]
  trajectory_splits: defaultdict[str, set[str]] = defaultdict(set)
  group_splits: defaultdict[str, set[str]] = defaultdict(set)
  for item in retained:
    trajectory_splits[item["trajectory_id"]].add(item["split"])
    group_splits[item["proxy_group"]].add(item["split"])
  marker_rows = sum(bool(REASONING_MARKER_RE.search(item["state"])) for item in retained)
  gates = {
    "source_checksum": source_sha == SOURCE_SHA256,
    "zero_residual_reasoning_markers": marker_rows == 0,
    "target_and_metadata_canary_invariance": canary_failures == 0,
    "complete_adjacent_linkage": counts["linkage_exclusions"] == 0,
    "valid_retained_labels": all(set(item["labels"]) <= set(TOOLS) for item in retained),
    "no_truncation_and_max_512": all(item["token_count"] <= 512 and tokenizer.encode(item["state"], add_special_tokens=True) == item["token_ids"] for item in retained),
    "exact_token_state_unique": len(seen) == len(retained),
    "split_isolation": all(len(value) == 1 for value in trajectory_splits.values()) and all(len(value) == 1 for value in group_splits.values()),
    "minimum_positive_trajectory_support": all(support[split][tool]["positive_trajectories"] >= 10 for split in SPLITS for tool in TOOLS),
    "split_over_budget_difference_lte_10pp": max(split_rates) - min(split_rates) <= 0.10,
    "label_over_budget_difference_lte_20pp": all(value <= 0.20 for value in label_rate_differences.values()),
  }
  disposition = "PASS" if all(gates.values()) else "FAIL"

  model_paths = {}
  for split in SPLITS:
    output = DATA_OUT / f"{split}.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8", newline="\n", mtime=0) as handle:
      for item in retained:
        if item["split"] == split:
          handle.write(canonical({key: item[key] for key in ("trajectory_id", "message_index", "proxy_group", "state", "token_ids", "state_hash", "labels")}) + "\n")
    model_paths[split] = output
    ids = sorted({item["trajectory_id"] for item in retained if item["split"] == split})
    (RESULTS / f"{split}_ids.txt").write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")

  with gzip.open(RESULTS / "exclusion_ledger.jsonl.gz", "wt", encoding="utf-8", newline="\n", mtime=0) as handle:
    for entry in sorted(ledger, key=lambda value: (value.get("row_index", -1), value["trajectory_id"], value.get("message_index", -1), value["reason"])):
      handle.write(canonical(entry) + "\n")

  profile = {
    "profile_version": "exp001b-preprocessing-v1", "disposition": disposition,
    "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": source_sha, "rows": len(rows)},
    "tokenizer": {"path": str(TOKENIZER_PATH.relative_to(ROOT)), "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982"},
    "counts": dict(sorted(counts.items())) | {"valid_before_budget": len(candidates), "over_budget_exclusions": len(candidates) - len(valid_budget),
                                                "dedup_exclusions": len(valid_budget) - len(retained), "retained_rows": len(retained)},
    "retained_token_lengths": distribution([item["token_count"] for item in retained]),
    "split": split_stats, "positive_support": support,
    "assertions": {"residual_reasoning_marker_rows": marker_rows, "canary_failures": canary_failures,
                   "trajectory_cross_split": sum(len(value) > 1 for value in trajectory_splits.values()),
                   "proxy_group_cross_split": sum(len(value) > 1 for value in group_splits.values())},
    "selection_bias": {"by_split": bias_table("split"), "by_proxy_group": bias_table("proxy_group"),
                       "by_category": bias_table("category"), "by_label": label_bias,
                       "split_max_absolute_rate_difference": round(max(split_rates) - min(split_rates), 6),
                       "label_absolute_rate_differences": label_rate_differences},
    "stop_gates": gates,
  }
  (RESULTS / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

  bias_lines = ["# Selection Bias", "", f"Disposition: **{disposition}**", "",
                "Rates measure exclusions above 512 tokens among structurally and target-valid candidates.", "",
                "| Split | Candidates | Over budget | Rate |", "|---|---:|---:|---:|"]
  for split in SPLITS:
    value = split_stats[split]
    bias_lines.append(f"| {split} | {value['budget_eligible_candidates']} | {value['over_budget']} | {value['over_budget_rate']:.2%} |")
  bias_lines += ["", f"Maximum split difference: {max(split_rates) - min(split_rates):.2%} (gate: <= 10.00%).", "",
                 "| Tool | Positive rate | Negative rate | Absolute difference |", "|---|---:|---:|---:|"]
  for tool in TOOLS:
    bias_lines.append(f"| {tool} | {label_bias[tool]['positive']['over_budget_rate']:.2%} | {label_bias[tool]['negative']['over_budget_rate']:.2%} | {label_rate_differences[tool]:.2%} |")
  bias_lines += ["", "Complete machine-readable tables by split, proxy group, source category, and label are in `../results/profile.json`.", ""]
  (ANALYSIS / "selection_bias.md").write_text("\n".join(bias_lines), encoding="utf-8")

  checksum_paths = [SOURCE, RESULTS / "profile.json", RESULTS / "exclusion_ledger.jsonl.gz", ANALYSIS / "selection_bias.md", *model_paths.values(),
                    *(RESULTS / f"{split}_ids.txt" for split in SPLITS)]
  checksums = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in checksum_paths}
  (RESULTS / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": checksums}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [name for name, passed in gates.items() if not passed]
  (LOGS / "build.log").write_text("\n".join(["command=.venv/bin/python experiments/exp_001b_hermes_preprocessing/scripts/build_dataset.py",
                                               "network=disabled_by_local_files_only", "model_loading=none", "inference=none", "training=none",
                                               f"source_sha256={source_sha}", f"disposition={disposition}", f"failed_gates={canonical(failed)}", ""]), encoding="utf-8")


if __name__ == "__main__":
  main()
