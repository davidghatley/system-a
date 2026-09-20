#!/usr/bin/env python3
"""Profile a bounded, pinned sample of Exgentic agent traces."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


REPO = "Exgentic/agent-llm-traces"
REVISION = "70036b93a04e61b0ea2706a68b962f4f26774587"
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/exgentic_audit"
ARTIFACT_ROOT = ROOT / "artifacts/dataset_candidates/exgentic"
INITIAL_SHARDS = (0, 3, 4, 8, 9, 13, 22, 29, 35, 38)


def percentile(values: list[int], fraction: float) -> int:
  ordered = sorted(values)
  return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0


def parse_json(value: str | None) -> list | dict | None:
  if not value:
    return None
  try:
    return json.loads(value)
  except (TypeError, json.JSONDecodeError):
    return None


def parts(messages: list[dict] | None) -> list[dict]:
  return [part for message in messages or [] for part in message.get("parts", []) or []]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--download", action="store_true")
  args = parser.parse_args()
  info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
  assert info.sha == REVISION
  destination = DATA_ROOT / REVISION
  destination.mkdir(parents=True, exist_ok=True)
  ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

  remote_files = {item.rfilename: item for item in info.siblings}
  selected = ["README.md"] + [f"data/train-{index:05d}-of-00039.parquet" for index in INITIAL_SHARDS]
  if args.download:
    for name in selected:
      hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=destination)

  files = []
  for name in selected:
    path = destination / name
    if not path.exists():
      continue
    files.append({
      "path": name,
      "bytes": path.stat().st_size,
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "remote_bytes": remote_files[name].size,
    })

  counts = collections.Counter()
  combinations = collections.Counter()
  tools = collections.Counter()
  part_types = collections.Counter()
  tool_definition_counts = []
  input_lengths = []
  output_lengths = []
  duplicate_sessions = collections.Counter()
  duplicate_outputs = collections.Counter()
  examples = []
  shard_coverage = collections.defaultdict(collections.Counter)

  for file_record in files:
    if not file_record["path"].endswith(".parquet"):
      continue
    table = pq.read_table(destination / file_record["path"])
    for row in table.to_pylist():
      counts["sessions"] += 1
      duplicate_sessions[row["session_id"]] += 1
      models = tuple(row["models"] or [])
      combinations[(row["benchmark"], row["harness"], models)] += 1
      shard_coverage[file_record["path"]][(row["benchmark"], row["harness"])] += 1
      spans = sorted(row["spans"] or [], key=lambda span: (span["start_time"], span["span_id"]))
      counts["spans"] += len(spans)
      parsed_spans = []
      for span in spans:
        attributes = span["attributes"] or {}
        inputs_raw = attributes.get("gen_ai.input.messages") or ""
        outputs_raw = attributes.get("gen_ai.output.messages") or ""
        definitions_raw = attributes.get("gen_ai.tool.definitions") or ""
        inputs = parse_json(inputs_raw)
        outputs = parse_json(outputs_raw)
        definitions = parse_json(definitions_raw)
        input_lengths.append(len(inputs_raw))
        output_lengths.append(len(outputs_raw))
        if inputs is None:
          counts["unparseable_input_messages"] += 1
        if outputs is None:
          counts["unparseable_output_messages"] += 1
        if definitions is not None:
          counts["spans_with_tool_definitions"] += 1
          tool_definition_counts.append(len(definitions))
          for definition in definitions:
            tools[definition.get("name") or definition.get("function", {}).get("name") or "<unnamed>"] += 1
        input_parts = parts(inputs if isinstance(inputs, list) else [])
        output_parts = parts(outputs if isinstance(outputs, list) else [])
        parsed_spans.append((input_parts, output_parts))
        for part in input_parts + output_parts:
          part_types[part.get("type", "<missing>")] += 1
        calls = [part for part in output_parts if part.get("type") in {"tool_call", "function_call"}]
        results = [part for part in input_parts if part.get("type") in {"tool_call_response", "tool_result", "function_call_response"}]
        counts["output_tool_calls"] += len(calls)
        counts["input_tool_results"] += len(results)
        if calls:
          counts["spans_with_output_tool_calls"] += 1
        if results:
          counts["spans_with_input_tool_results"] += 1
        call_ids = {str(part.get("id") or part.get("tool_call_id") or "") for part in calls}
        result_ids = {str(part.get("id") or part.get("tool_call_id") or "") for part in results}
        counts["same_span_matched_call_result_ids"] += len((call_ids & result_ids) - {""})
        counts["thinking_parts"] += sum(part.get("type") == "thinking" for part in input_parts + output_parts)
        if outputs_raw:
          duplicate_outputs[hashlib.sha256(outputs_raw.encode()).hexdigest()] += 1
        if len(examples) < 12 and (calls or results):
          examples.append({
            "session_id": row["session_id"], "benchmark": row["benchmark"], "harness": row["harness"],
            "model": attributes.get("gen_ai.request.model"), "span_id": span["span_id"],
            "input_result_types": [part.get("type") for part in results],
            "output_call_names": [part.get("name") for part in calls],
          })
      for index, (_, output_parts) in enumerate(parsed_spans[:-1]):
        calls = [part for part in output_parts if part.get("type") in {"tool_call", "function_call"}]
        if not calls:
          continue
        next_input_parts = parsed_spans[index + 1][0]
        results = [part for part in next_input_parts if part.get("type") in {"tool_call_response", "tool_result", "function_call_response"}]
        call_ids = {str(part.get("id") or part.get("tool_call_id") or "") for part in calls} - {""}
        result_ids = {str(part.get("id") or part.get("tool_call_id") or "") for part in results} - {""}
        counts["action_spans_with_later_span"] += 1
        counts["action_spans_reconstructable_from_next_input"] += bool(call_ids and call_ids <= result_ids)
        counts["action_call_ids_with_later_span"] += len(call_ids)
        counts["action_call_ids_matched_in_next_input"] += len(call_ids & result_ids)

  profile = {
    "profile_version": 1,
    "dataset": REPO,
    "revision": REVISION,
    "license": "cdla-permissive-2.0",
    "scope": {"sampling": "bounded shard sample", "selected_shards": list(INITIAL_SHARDS), "downloaded_bytes": sum(f["bytes"] for f in files)},
    "source_claimed_totals": {"sessions": 1781, "shards": 39, "bytes": sum((f.size or 0) for n, f in remote_files.items() if n.endswith(".parquet"))},
    "files": files,
    "sample_counts": dict(counts),
    "coverage": [
      {"benchmark": key[0], "harness": key[1], "models": list(key[2]), "sessions": value}
      for key, value in sorted(combinations.items())
    ],
    "coverage_by_shard": {
      name: [{"benchmark": key[0], "harness": key[1], "sessions": value} for key, value in sorted(values.items())]
      for name, values in sorted(shard_coverage.items())
    },
    "message_part_types": dict(sorted(part_types.items())),
    "tool_definitions": {"unique_names": len(tools), "most_common": dict(tools.most_common(30))},
    "tool_definitions_per_span": {"min": min(tool_definition_counts, default=0), "median": statistics.median(tool_definition_counts) if tool_definition_counts else 0, "max": max(tool_definition_counts, default=0)},
    "serialized_context_characters": {
      "input": {"median": percentile(input_lengths, .5), "p95": percentile(input_lengths, .95), "max": max(input_lengths, default=0)},
      "output": {"median": percentile(output_lengths, .5), "p95": percentile(output_lengths, .95), "max": max(output_lengths, default=0)},
    },
    "duplicates": {
      "duplicate_session_ids": sum(1 for value in duplicate_sessions.values() if value > 1),
      "excess_exact_output_messages": sum(value - 1 for value in duplicate_outputs.values() if value > 1),
    },
    "examples": examples,
    "limitations": ["Sample measurements are not full-corpus estimates.", "Dataset has no task ID, task outcome, reward, or benchmark score column.", "Same-span ID matching is diagnostic only; results normally occur in later call inputs."],
  }
  (ARTIFACT_ROOT / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
  (ARTIFACT_ROOT / "source_manifest.json").write_text(json.dumps({"dataset": REPO, "revision": REVISION, "files": files}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
  main()
