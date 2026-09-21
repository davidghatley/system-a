#!/usr/bin/env python3
"""Assertion tests for exp_001c before its full deterministic sweep."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import sweep_budgets as sweep
import laya.common as upstream


def source_fixture(task: str, target: str | None = None):
  return {
    "id": task,
    "category": "fixture",
    "task": task,
    "tools": sweep.prior.canonical([
      {"type": "function", "function": {"name": tool}} for tool in sweep.TOOLS
    ]),
    "conversations": [
      {"from": "gpt", "value": '<think>secret</think><tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'},
      {"from": "tool", "value": '<tool_response>{"name":"read_file","result":"ok"}</tool_response>'},
      {"from": "gpt", "value": target or '<tool_call>{"name":"terminal","arguments":{}}</tool_call>'},
    ],
  }


def task_for_split(split: str) -> str:
  for index in range(10000):
    value = index
    letters = ""
    while True:
      letters = chr(ord("a") + value % 26) + letters
      value = value // 26 - 1
      if value < 0:
        break
    task = f"fixture task {split} {letters}"
    if sweep.prior.assign_split(sweep.prior.proxy_group(task)) == split:
      return task
  raise AssertionError(f"could not construct {split} fixture")


def main():
  tokenizer = sweep.AutoTokenizer.from_pretrained(sweep.TOKENIZER_PATH, local_files_only=True)
  assert sweep.prior.build_sequence is upstream.build_sequence
  inputs = {budget: sweep.BudgetInputs(tokenizer, budget) for budget in sweep.BUDGETS}

  short = "short complete state"
  semantic_ids = inputs[512].state_ids(short)
  assert all(wrapper.state_ids(short) == semantic_ids for wrapper in inputs.values())
  assert all(wrapper.build(short)[0] is not None for wrapper in inputs.values())

  boundary = None
  for repetitions in range(1, 1000):
    candidate = " x" * repetitions
    if inputs[512].build(candidate)[0] is None and inputs[640].build(candidate)[0] is not None:
      boundary = candidate
      break
  assert boundary is not None
  assert inputs[512].build(boundary)[0] is None
  accepted, _ = inputs[640].build(boundary)
  assert accepted is not None
  full = inputs[640].state_ids(boundary)
  for sequence in accepted:
    empty_ids = inputs[640].empty[sequence["tool"]][0]
    assert sequence["ids"] == empty_ids[:-1] + full + [tokenizer.sep_token_id]

  for budget, wrapper in inputs.items():
    with patch.object(sweep.prior, "build_sequence", wraps=upstream.build_sequence) as actual:
      assert wrapper.build(short)[0] is not None
      assert actual.call_count == 6
    too_long = " x" * (budget + 100)
    with patch.object(sweep.prior, "build_sequence", wraps=upstream.build_sequence) as actual:
      assert wrapper.build(too_long)[0] is None
      assert actual.call_count == 0

  train_source = source_fixture(task_for_split("train"))
  counts = sweep.Counter()
  candidates = sweep.parse_selection_candidate(train_source, 0, counts)
  assert len(candidates) == 1 and candidates[0]["linkage_valid"]
  state = candidates[0]["state"]
  sequences, _ = inputs[1024].build(state)
  sweep.prior.canary_checks(train_source, 2, inputs[1024], state, sequences)
  assert not sweep.prior.REASONING_MARKER_RE.search(state)

  malformed = deepcopy(train_source)
  malformed["conversations"][1]["value"] = '<tool_response>{"name":"terminal","result":"mismatch"}</tool_response>'
  malformed_counts = sweep.Counter()
  assert sweep.parse_selection_candidate(malformed, 0, malformed_counts) == []
  assert malformed_counts["linkage_exclusions"] == 1
  incomplete = deepcopy(train_source)
  incomplete["conversations"][1]["value"] = "missing response wrapper"
  incomplete_counts = sweep.Counter()
  assert sweep.parse_selection_candidate(incomplete, 0, incomplete_counts) == []
  assert incomplete_counts["linkage_exclusions"] == 1
  common = {"source_checksum": True, "all_canaries_pass": True, "protected_splits_excluded_before_target_parse": True}
  gate_candidates = []
  for split in sweep.SELECTION_SPLITS:
    for tool in sweep.TOOLS:
      for index in range(10):
        item = dict(candidates[0])
        item.update(
          trajectory_id=f"{split}-{tool}-{index}",
          proxy_group=f"{split}-{tool}-{index}",
          split=split,
          labels=[tool],
          state=f"{state}\nFixture uniqueness: {split} {tool} {'x' * (index + 1)}",
        )
        gate_candidates.append(item)
  fixture_result = sweep.budget_result(gate_candidates, inputs[512], common)
  assert fixture_result["gates"]["zero_malformed_linkage_in_retained"]

  protected = [
    source_fixture(task_for_split(split), target="PROTECTED_TARGET_SENTINEL<tool_call>not-json</tool_call>")
    for split in sweep.PROTECTED_SPLITS
  ]
  protected_counts = sweep.Counter()
  original_payloads = sweep.prior.payloads
  inspected_values = []
  def recording_payloads(value, tag):
    inspected_values.append(value)
    return original_payloads(value, tag)
  with patch.object(sweep.prior, "payloads", side_effect=recording_payloads):
    assert all(sweep.parse_selection_candidate(row, index, protected_counts) == [] for index, row in enumerate(protected))
  assert not inspected_values
  assert protected_counts["protected_calibration_rows_skipped_before_target_parse"] == 1
  assert protected_counts["protected_test_rows_skipped_before_target_parse"] == 1

  print("PASS: cross-budget semantics, 512/640 boundary, no truncation, upstream constructor, canaries, linkage exclusion semantics, protected-target isolation")


if __name__ == "__main__":
  assert not __import__("sys").flags.optimize
  main()
