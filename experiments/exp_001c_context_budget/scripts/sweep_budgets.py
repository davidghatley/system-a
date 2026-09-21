#!/usr/bin/env python3
"""Run the frozen exp_001c train/development-only context-budget sweep."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/exp_001c_context_budget"
PRIOR_SCRIPTS = ROOT / "experiments/exp_001b_hermes_preprocessing/scripts"
sys.path.insert(0, str(PRIOR_SCRIPTS))
import build_dataset as prior

SOURCE = prior.SOURCE
TOKENIZER_PATH = prior.TOKENIZER_PATH
LAYA = prior.LAYA
PROTOCOL = EXP / "protocol.json"
ARTIFACTS = ROOT / "artifacts/exp_001c"
SOURCE_SHA256 = prior.SOURCE_SHA256
LAYA_REVISION = prior.LAYA_REVISION
TOKENIZER_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
PRIOR_CODE_SHA256 = "6e0671712ea8a23755fdb3d397272694823e6f7ebd9b277d4c830ec0cda5f6c2"
LAYA_COMMON_SHA256 = "f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2"
TOKENIZER_FILE_SHA256 = {
  "tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
  "tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
}
BUDGETS = (512, 640, 768, 1024)
ELIGIBLE_BUDGETS = (640, 768, 1024)
SELECTION_SPLITS = ("train", "development")
PROTECTED_SPLITS = ("calibration", "test")
TOOLS = prior.TOOLS
HEAD_MAX_LEN = prior.HEAD_MAX_LEN
QUESTIONS = prior.QUESTIONS


def canonical(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def rate(numerator: int, denominator: int) -> float:
  if not denominator:
    raise AssertionError("rate denominator must be positive")
  return numerator / denominator


class BudgetInputs:
  """Complete-state fit checks around the actual upstream Laya constructor."""

  def __init__(self, tokenizer, max_len: int):
    if max_len not in BUDGETS:
      raise ValueError(f"unfrozen budget: {max_len}")
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.empty = {
      tool: prior.build_sequence(tokenizer, "", question, max_len, HEAD_MAX_LEN)
      for tool, question in QUESTIONS.items()
    }
    self.rooms = {tool: max_len - len(ids) for tool, (ids, _) in self.empty.items()}

  def state_ids(self, state: str) -> list[int]:
    tokenizer = self.tokenizer
    return tokenizer(
      prior.serialize_state(state).replace(tokenizer.mask_token, " "),
      add_special_tokens=False,
      truncation=False,
    )["input_ids"]

  def build(self, state: str):
    full = self.state_ids(state)
    required_length = len(full) + max(len(ids) for ids, _ in self.empty.values())
    if any(len(full) > room for room in self.rooms.values()):
      return None, required_length
    sequences = []
    for tool, question in QUESTIONS.items():
      ids, markers = prior.build_sequence(
        self.tokenizer, state, question, self.max_len, HEAD_MAX_LEN
      )
      empty_ids, empty_markers = self.empty[tool]
      assert ids == empty_ids[:-1] + full + [self.tokenizer.sep_token_id], "upstream state loss"
      assert markers == empty_markers
      assert len(ids) <= self.max_len
      sequences.append({
        "tool": tool,
        "ids": ids,
        "markers": markers,
        "length": len(ids),
        "qtype": prior.QTYPES["noul"],
      })
    return sequences, required_length


def input_key(sequences) -> tuple:
  return tuple((tuple(sequence["ids"]), tuple(sequence["markers"])) for sequence in sequences)


def parse_selection_candidate(source_row: dict[str, Any], row_index: int, counts: Counter):
  """Return eligible candidates after split assignment; protected rows are never parsed."""
  group = prior.proxy_group(source_row["task"])
  split = prior.assign_split(group)
  if split in PROTECTED_SPLITS:
    counts[f"protected_{split}_rows_skipped_before_target_parse"] += 1
    return []
  assert split in SELECTION_SPLITS
  counts[f"selection_{split}_source_rows"] += 1
  if prior.tool_names(source_row["tools"]) != TOOLS:
    counts["non_exact_toolset_trajectories"] += 1
    return []
  counts["exact_toolset_trajectories"] += 1

  stripped_messages = []
  malformed_indexes = []
  for message_index, message in enumerate(source_row["conversations"]):
    stripped, valid = prior.strip_reasoning(message["value"] or "")
    if not valid:
      malformed_indexes.append(message_index)
    stripped_messages.append({"from": message["from"], "value": stripped})
  if malformed_indexes:
    counts["malformed_reasoning_trajectories"] += 1
    return []

  candidates = []
  trajectory_id = str(source_row["id"])
  for message_index, target_message in enumerate(stripped_messages):
    if target_message["from"] != "gpt" or message_index < 2:
      continue
    prior_call = stripped_messages[message_index - 2]
    prior_result = stripped_messages[message_index - 1]
    if prior_call["from"] != "gpt" or prior_result["from"] != "tool":
      continue
    has_call_wrapper = bool(prior.re.search(r"</?tool_call\b", prior_call["value"], prior.re.I))
    has_result_wrapper = bool(prior.re.search(r"</?tool_response\b", prior_result["value"], prior.re.I))
    if not has_call_wrapper and not has_result_wrapper:
      continue
    counts["observation_conditioned_candidates"] += 1
    calls = prior.payloads(prior_call["value"], "tool_call")
    results = prior.payloads(prior_result["value"], "tool_response")
    if (
      not calls
      or not results
      or len(calls) != len(results)
      or [payload["name"] for payload in calls] != [payload["name"] for payload in results]
      or any(payload["name"] not in TOOLS for payload in calls)
    ):
      counts["linkage_exclusions"] += 1
      continue
    target_calls = prior.payloads(target_message["value"], "tool_call")
    if target_calls is None or any(payload["name"] not in TOOLS for payload in target_calls):
      counts["invalid_target_exclusions"] += 1
      continue
    candidates.append({
      "trajectory_id": trajectory_id,
      "message_index": message_index,
      "proxy_group": group,
      "split": split,
      "labels": sorted({payload["name"] for payload in target_calls}),
      "state": prior.render_source(source_row, message_index),
      "source_row": source_row,
      "row_index": row_index,
      "linkage_valid": True,
    })
  return candidates


def check_full_canaries(candidates, largest_inputs: BudgetInputs) -> dict[str, int]:
  tested = 0
  failures = 0
  for index, candidate in enumerate(candidates):
    sequences, _ = largest_inputs.build(candidate["state"])
    if sequences is None:
      continue
    tested += 1
    try:
      prior.canary_checks(
        candidate["source_row"], candidate["message_index"], largest_inputs,
        candidate["state"], sequences,
      )
    except AssertionError:
      failures += 1
    if index and index % 10000 == 0:
      print(f"Canaries checked through candidate {index}/{len(candidates)}", file=sys.stderr)
  return {"representable_at_1024_tested": tested, "failures": failures}


def budget_result(candidates, inputs: BudgetInputs, common_integrity: dict[str, bool]):
  evaluated = []
  for candidate in candidates:
    sequences, required_length = inputs.build(candidate["state"])
    evaluated.append(candidate | {
      "representable": sequences is not None,
      "required_length": required_length,
      "sequences": sequences,
    })

  representable = [item for item in evaluated if item["representable"]]
  representable.sort(key=lambda item: (item["trajectory_id"], item["message_index"]))
  retained = []
  seen = set()
  for item in representable:
    key = input_key(item["sequences"])
    if key in seen:
      continue
    seen.add(key)
    retained.append(item)

  by_split = {}
  by_tool_split = {}
  for split in SELECTION_SPLITS:
    split_candidates = [item for item in evaluated if item["split"] == split]
    over_budget = sum(not item["representable"] for item in split_candidates)
    by_split[split] = {
      "candidates": len(split_candidates),
      "over_budget": over_budget,
      "over_budget_rate": rate(over_budget, len(split_candidates)),
    }
    by_tool_split[split] = {}
    split_retained = [item for item in retained if item["split"] == split]
    for tool in TOOLS:
      positive = [item for item in split_candidates if tool in item["labels"]]
      negative = [item for item in split_candidates if tool not in item["labels"]]
      positive_rate = rate(sum(not item["representable"] for item in positive), len(positive))
      negative_rate = rate(sum(not item["representable"] for item in negative), len(negative))
      retained_positive = [item for item in split_retained if tool in item["labels"]]
      by_tool_split[split][tool] = {
        "positive_candidates": len(positive),
        "negative_candidates": len(negative),
        "positive_over_budget": sum(not item["representable"] for item in positive),
        "negative_over_budget": sum(not item["representable"] for item in negative),
        "positive_over_budget_rate": positive_rate,
        "negative_over_budget_rate": negative_rate,
        "absolute_rate_difference": abs(positive_rate - negative_rate),
        "positive_retained_trajectories": len({item["trajectory_id"] for item in retained_positive}),
      }

  pooled_tools = {}
  for tool in TOOLS:
    positive = [item for item in evaluated if tool in item["labels"]]
    negative = [item for item in evaluated if tool not in item["labels"]]
    positive_rate = rate(sum(not item["representable"] for item in positive), len(positive))
    negative_rate = rate(sum(not item["representable"] for item in negative), len(negative))
    pooled_tools[tool] = {
      "positive_candidates": len(positive),
      "negative_candidates": len(negative),
      "positive_over_budget_rate": positive_rate,
      "negative_over_budget_rate": negative_rate,
      "absolute_rate_difference": abs(positive_rate - negative_rate),
      "positive_retained_trajectories": len({
        item["trajectory_id"] for item in retained if tool in item["labels"]
      }),
    }

  trajectory_splits: defaultdict[str, set[str]] = defaultdict(set)
  group_splits: defaultdict[str, set[str]] = defaultdict(set)
  for item in retained:
    trajectory_splits[item["trajectory_id"]].add(item["split"])
    group_splits[item["proxy_group"]].add(item["split"])
  residual_markers = sum(bool(prior.REASONING_MARKER_RE.search(item["state"])) for item in retained)
  retained_linkage_failures = sum(not item["linkage_valid"] for item in retained)
  exact_keys = [input_key(item["sequences"]) for item in retained]
  split_difference = abs(by_split["train"]["over_budget_rate"] - by_split["development"]["over_budget_rate"])

  gates = dict(common_integrity)
  gates.update({
    "zero_residual_reasoning_markers": residual_markers == 0,
    "zero_malformed_linkage_in_retained": retained_linkage_failures == 0,
    "valid_retained_labels": all(set(item["labels"]) <= set(TOOLS) for item in retained),
    "no_truncation": all(
      sequence["length"] <= inputs.max_len for item in retained for sequence in item["sequences"]
    ),
    "exact_representation_unique": len(exact_keys) == len(set(exact_keys)) == len(retained),
    "trajectory_split_isolation": all(len(splits) == 1 for splits in trajectory_splits.values()),
    "proxy_group_split_isolation": all(len(splits) == 1 for splits in group_splits.values()),
    "minimum_positive_trajectory_support": all(
      by_tool_split[split][tool]["positive_retained_trajectories"] >= 10
      for split in SELECTION_SPLITS for tool in TOOLS
    ),
    "split_over_budget_difference_lte_10pp": split_difference <= 0.10,
    "label_over_budget_difference_lte_20pp": all(
      by_tool_split[split][tool]["absolute_rate_difference"] <= 0.20
      for split in SELECTION_SPLITS for tool in TOOLS
    ),
  })
  overall = {
    "valid_pre_budget_candidates": len(evaluated),
    "representable_candidates": len(representable),
    "over_budget_exclusions": len(evaluated) - len(representable),
    "representable_fraction": len(representable) / len(evaluated),
    "duplicate_exclusions": len(representable) - len(retained),
    "retained_rows": len(retained),
    "retained_trajectories": len(trajectory_splits),
    "retained_proxy_groups": len(group_splits),
  }
  audit = {
    "residual_reasoning_marker_rows": residual_markers,
    "retained_linkage_failures": retained_linkage_failures,
    "trajectory_cross_split": sum(len(value) > 1 for value in trajectory_splits.values()),
    "proxy_group_cross_split": sum(len(value) > 1 for value in group_splits.values()),
    "max_retained_sequence_length": max(
      sequence["length"] for item in retained for sequence in item["sequences"]
    ),
  }
  return {
    "budget": inputs.max_len,
    "promotion_eligible": inputs.max_len in ELIGIBLE_BUDGETS,
    "overall": overall,
    "by_split": by_split,
    "by_tool_and_split": by_tool_split,
    "pooled_train_development_descriptive": pooled_tools,
    "split_absolute_rate_difference": split_difference,
    "state_rooms": inputs.rooms,
    "integrity_audit": audit,
    "gates": gates,
    "passes_all_data_gates": all(gates.values()),
  }


def render_markdown(output: dict[str, Any]) -> str:
  lines = [
    "# exp_001c Context-Budget Sweep",
    "",
    f"Protocol SHA-256: `{output['provenance']['protocol_sha256']}`",
    "",
    "Only train and development contributed candidates, targets, metrics, gates, and selection. Calibration and test rows were skipped from task-derived split assignment before conversation parsing.",
    "",
    "| Budget | Retained | Retention | Worst train tool bias | Worst dev tool bias | Pass |",
    "|---:|---:|---:|---:|---:|---|",
  ]
  for budget in BUDGETS:
    result = output["budgets"][str(budget)]
    train_worst = max(row["absolute_rate_difference"] for row in result["by_tool_and_split"]["train"].values())
    dev_worst = max(row["absolute_rate_difference"] for row in result["by_tool_and_split"]["development"].values())
    disposition = "reference" if budget == 512 else ("PASS" if result["passes_all_data_gates"] else "FAIL")
    lines.append(
      f"| {budget} | {result['overall']['retained_rows']:,} | {result['overall']['representable_fraction']:.2%} | {train_worst:.4%} | {dev_worst:.4%} | {disposition} |"
    )
  lines.extend(["", f"Selected budget: **{output['selection']['selected_budget']}**", ""])
  for budget in BUDGETS:
    result = output["budgets"][str(budget)]
    lines.extend([
      f"## Budget {budget}", "",
      "| Split | Candidates | Over budget | Rate |",
      "|---|---:|---:|---:|",
    ])
    for split in SELECTION_SPLITS:
      row = result["by_split"][split]
      lines.append(f"| {split} | {row['candidates']:,} | {row['over_budget']:,} | {row['over_budget_rate']:.4%} |")
    lines.extend(["", "| Split / tool | Positive | Negative | Positive rate | Negative rate | Difference | Positive retained trajectories |", "|---|---:|---:|---:|---:|---:|---:|"])
    for split in SELECTION_SPLITS:
      for tool in TOOLS:
        row = result["by_tool_and_split"][split][tool]
        lines.append(f"| {split} / {tool} | {row['positive_candidates']:,} | {row['negative_candidates']:,} | {row['positive_over_budget_rate']:.4%} | {row['negative_over_budget_rate']:.4%} | {row['absolute_rate_difference']:.4%} | {row['positive_retained_trajectories']:,} |")
    lines.append("")
  return "\n".join(lines)


def main() -> None:
  if ARTIFACTS.exists():
    raise SystemExit(f"Refusing to reuse execution directory: {ARTIFACTS}")
  ARTIFACTS.mkdir(parents=True)
  started = time.perf_counter()
  protocol = json.loads(PROTOCOL.read_text())
  if protocol["status"] != "FROZEN_BEFORE_EXECUTION":
    raise RuntimeError("protocol is not frozen")
  assert tuple(protocol["representation"]["candidate_total_lengths"]) == BUDGETS
  assert tuple(protocol["representation"]["eligible_total_lengths"]) == ELIGIBLE_BUDGETS
  source_sha = sha256_file(SOURCE)
  if source_sha != SOURCE_SHA256:
    raise RuntimeError(f"source checksum mismatch: {source_sha}")
  laya_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LAYA, text=True).strip()
  laya_dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=LAYA, text=True).strip()
  if laya_head != LAYA_REVISION or laya_dirty:
    raise RuntimeError(f"Laya checkout mismatch or dirty: head={laya_head} dirty={bool(laya_dirty)}")
  if sha256_file(PRIOR_SCRIPTS / "build_dataset.py") != PRIOR_CODE_SHA256:
    raise RuntimeError("imported exp_001b v0.2 implementation checksum mismatch")
  if sha256_file(LAYA / "laya/common.py") != LAYA_COMMON_SHA256:
    raise RuntimeError("Laya common.py checksum mismatch")
  actual_tokenizer_hashes = {
    name: sha256_file(TOKENIZER_PATH / name) for name in TOKENIZER_FILE_SHA256
  }
  if actual_tokenizer_hashes != TOKENIZER_FILE_SHA256:
    raise RuntimeError(f"tokenizer checksum mismatch: {actual_tokenizer_hashes}")

  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
  inputs = {budget: BudgetInputs(tokenizer, budget) for budget in BUDGETS}
  rows = pq.read_table(SOURCE).to_pylist()
  counts = Counter()
  candidates = []
  for row_index, source_row in enumerate(rows):
    candidates.extend(parse_selection_candidate(source_row, row_index, counts))
    if row_index and row_index % 1000 == 0:
      print(f"Scanned {row_index}/{len(rows)} source rows", file=sys.stderr)

  canaries = check_full_canaries(candidates, inputs[1024])
  common_integrity = {
    "source_checksum": source_sha == SOURCE_SHA256,
    "all_canaries_pass": canaries["failures"] == 0,
    "protected_splits_excluded_before_target_parse": (
      counts["protected_calibration_rows_skipped_before_target_parse"] > 0
      and counts["protected_test_rows_skipped_before_target_parse"] > 0
    ),
  }
  results = {}
  for budget in BUDGETS:
    print(f"Evaluating budget {budget}", file=sys.stderr)
    results[str(budget)] = budget_result(candidates, inputs[budget], common_integrity)

  passing = [budget for budget in ELIGIBLE_BUDGETS if results[str(budget)]["passes_all_data_gates"]]
  selected = min(passing) if passing else None
  provenance = {
    "protocol_sha256": sha256_file(PROTOCOL),
    "sweep_code_sha256": sha256_file(Path(__file__)),
    "repository_head_at_execution": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "source_sha256": source_sha,
    "laya_revision": laya_head,
    "imported_v0.2_code_sha256": sha256_file(PRIOR_SCRIPTS / "build_dataset.py"),
    "laya_common_sha256": sha256_file(LAYA / "laya/common.py"),
    "tokenizer_revision": TOKENIZER_REVISION,
    "tokenizer_files": actual_tokenizer_hashes,
    "command": ".venv/bin/python experiments/exp_001c_context_budget/scripts/sweep_budgets.py",
  }
  output = {
    "experiment_id": "exp_001c_context_budget",
    "disposition": "DATA_GATE_PASS" if selected is not None else "PREPROCESSING_REMEDIATION_FAIL",
    "provenance": provenance,
    "protected_data": {
      "selection_splits": list(SELECTION_SPLITS),
      "protected_splits": list(PROTECTED_SPLITS),
      "calibration_or_test_label_statistics_emitted": False,
      "prior_exp001b_aggregate_exposure_limitation": True,
      "skipped_rows": {split: counts[f"protected_{split}_rows_skipped_before_target_parse"] for split in PROTECTED_SPLITS},
    },
    "structural_counts_train_development": dict(sorted(counts.items())),
    "linkage_clarification": {
      "excluded_malformed_source_candidates": counts["linkage_exclusions"],
      "excluded_candidates_fail_integrity_gate": False,
      "required_retained_linkage_failures": 0,
    },
    "canaries": canaries,
    "budgets": results,
    "selection": {
      "eligible_budgets": list(ELIGIBLE_BUDGETS),
      "reference_only_budget": 512,
      "passing_eligible_budgets": passing,
      "selected_budget": selected,
      "rule": "smallest passing eligible budget",
    },
    "wall_seconds": time.perf_counter() - started,
  }
  (ARTIFACTS / "budget_sweep.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
  (ARTIFACTS / "budget_sweep.md").write_text(render_markdown(output) + "\n")
  checksums = {
    str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    for path in (PROTOCOL, Path(__file__), ARTIFACTS / "budget_sweep.json", ARTIFACTS / "budget_sweep.md")
  }
  (ARTIFACTS / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": checksums}, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"disposition": output["disposition"], "selected_budget": selected}, indent=2))


if __name__ == "__main__":
  if sys.flags.optimize:
    raise RuntimeError("Assertions must be enabled")
  main()
