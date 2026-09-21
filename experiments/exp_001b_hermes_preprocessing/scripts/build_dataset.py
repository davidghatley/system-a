#!/usr/bin/env python3
"""Build the frozen exp_001b preprocessing artifacts. No model is loaded or run."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/exp_001b_hermes_preprocessing"
SOURCE = ROOT / "data/hermes_audit/kimi-b92885e4f0161d4b2536512710e004d4892cac6e.parquet"
TOKENIZER_PATH = ROOT / "data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/tokenizer"
PROTOCOL = EXP / "protocol_v0.2.0.json"
RUN = ROOT / "artifacts/exp_001b/v0.2.0-run1"
DATA_OUT = RUN / "model_ready"
RESULTS = RUN
ANALYSIS = RUN
LOGS = RUN
LAYA = ROOT / "data/laya"
LAYA_REVISION = "d113dca2512fb3eaca313534bc54c7162d87c1d4"
sys.path.insert(0, str(LAYA))
from laya.common import build_sequence, serialize_state, QTYPES
SOURCE_SHA256 = "d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02"
TOOLS = ("patch", "process", "read_file", "search_files", "terminal", "write_file")
SPLITS = ("train", "development", "calibration", "test")
MAX_LEN = 512
HEAD_MAX_LEN = 192
CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.I | re.S)
RESPONSE_RE = re.compile(r"<tool_response\b[^>]*>(.*?)</tool_response\s*>", re.I | re.S)
THINK_TAG_RE = re.compile(r"<think\b[^>]*>|</think\s*>", re.I)
REASONING_MARKER_RE = re.compile(r"</?think\b|<\|(?:begin|end)_of_thought\|>|(?:^|\n)\s*(?:reasoning|analysis)\s*:", re.I)


QUESTIONS = {tool: {"t": "noul", "ins": f"Does the next assistant turn call the {tool} tool?", "crit": None}
             for tool in TOOLS}


class LayaInputs:
  """Fit precheck and assertions around the actual upstream constructor."""

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.empty = {tool: build_sequence(tokenizer, "", q, MAX_LEN, HEAD_MAX_LEN)
                  for tool, q in QUESTIONS.items()}
    self.rooms = {tool: MAX_LEN - len(ids) for tool, (ids, _) in self.empty.items()}

  def build(self, state):
    tok = self.tokenizer
    full = tok(serialize_state(state).replace(tok.mask_token, " "),
               add_special_tokens=False, truncation=False)["input_ids"]
    required_length = len(full) + max(len(ids) for ids, _ in self.empty.values())
    if any(len(full) > room for room in self.rooms.values()):
      return None, required_length
    sequences = []
    for tool, q in QUESTIONS.items():
      ids, markers = build_sequence(tok, state, q, MAX_LEN, HEAD_MAX_LEN)
      empty_ids, empty_markers = self.empty[tool]
      assert ids == empty_ids[:-1] + full + [tok.sep_token_id], "upstream state loss"
      assert markers == empty_markers and len(ids) <= MAX_LEN
      sequences.append({"tool": tool, "ids": ids, "markers": markers,
                        "length": len(ids), "qtype": QTYPES["noul"]})
    return sequences, required_length


def input_key(sequences):
  return tuple((tuple(seq["ids"]), tuple(seq["markers"])) for seq in sequences)


def canonical(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def deterministic_gzip_write(path: Path, content: str) -> None:
  """Write content to a gzip file with mtime=0 for deterministic output."""
  with path.open("wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=raw) as handle:
      handle.write(content.encode("utf-8"))


def parse_json(value: str) -> Any | None:
  try:
    return json.loads(value.strip())
  except (json.JSONDecodeError, TypeError):
    return None


def payloads(value: str, tag: str) -> list[dict] | None:
  """Require complete, nonnested wrappers; malformed markup is never no-call."""
  pattern = CALL_RE if tag == "tool_call" else RESPONSE_RE
  matches = list(pattern.finditer(value))
  marker = re.compile(r"</?" + tag + r"\b", re.I)
  outside = pattern.sub("", value)
  if marker.search(outside) or any(marker.search(m.group(1)) for m in matches):
    return None
  parsed = [parse_json(m.group(1)) for m in matches]
  if any(not isinstance(p, dict) or not isinstance(p.get("name"), str) for p in parsed):
    return None
  return parsed


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


def render_state(task: str, prior_call: str, prior_result: str) -> str:
  """Render the state portion for Laya (without questions)."""
  return (f"Task:\n{task}\nAvailable tools: {', '.join(TOOLS)}\n"
          f"Previous assistant call:\n{prior_call}\nLatest tool response:\n{prior_result}")


def render_source(source: dict, target_index: int) -> str:
  """Model-input allowlist: task + selected prior values only. No target read."""
  prior = source["conversations"][target_index - 2:target_index]
  assert [m["from"] for m in prior] == ["gpt", "tool"]
  values = [strip_reasoning(m["value"] or "") for m in prior]
  assert all(valid for _, valid in values), "malformed reasoning"
  return render_state(source["task"], values[0][0], values[1][0])


def canary_checks(source, target_index, inputs, state, sequences):
  expected = input_key(sequences)

  def equal(record):
    rendered = render_source(record, target_index)
    assert rendered.encode("utf-8") == state.encode("utf-8"), "canary changed rendered bytes"
    actual, _ = inputs.build(rendered)
    assert actual is not None and input_key(actual) == expected, "canary changed model input"

  target = dict(source)
  target["conversations"] = list(source["conversations"])
  target["conversations"][target_index] = {
    "from": "gpt", "value": 'TARGET_CANARY<tool_call>{"name":"patch","arguments":{"sentinel":"TARGET_CANARY"}}</tool_call>',
    "content": "TARGET_CANARY", "calls": ["TARGET_CANARY"], "arguments": {"x": "TARGET_CANARY"},
    "labels": ["TARGET_CANARY"], "reasoning_content": "TARGET_CANARY", "other_target_only": "TARGET_CANARY"}
  equal(target)

  metadata = {key: (value if key in ("task", "tools", "conversations") else "METADATA_CANARY")
              for key, value in source.items()}
  for key in ("id", "category", "subcategory", "teacher", "harness", "split", "provenance",
              "row_index", "shard", "labels", "unknown_metadata"):
    metadata[key] = "METADATA_CANARY"
  equal(metadata)

  # Differ only INSIDE the valid source reasoning span; strip through render_source.
  for secret in ("REASONING_A", 'REASONING_B <tool_call>{"name":"terminal"}</tool_call>'):
    reasoning = dict(source)
    reasoning["conversations"] = list(source["conversations"])
    message = dict(source["conversations"][target_index - 2])
    stripped, valid = strip_reasoning(message["value"] or "")
    assert valid
    message["value"] = f"<think>{secret}</think>" + stripped
    reasoning["conversations"][target_index - 2] = message
    equal(reasoning)


def write_model_outputs(retained, gates, directory):
  """No output directory or split files are created unless ALL gates pass."""
  if not all(gates.values()):
    return {}
  directory.mkdir(exist_ok=False)
  paths = {}
  for split in SPLITS:
    output = directory / f"{split}.jsonl.gz"
    content = "".join(canonical({key: item[key] for key in
                       ("trajectory_id", "message_index", "proxy_group", "state", "sequences", "labels")}) + "\n"
                      for item in retained if item["split"] == split)
    deterministic_gzip_write(output, content)
    paths[split] = output
  return paths


def rate(numerator: int, denominator: int) -> float:
  return round(numerator / denominator, 6) if denominator else 0.0


def distribution(values: list[int]) -> dict[str, int | float]:
  ordered = sorted(values)
  if not ordered:
    return {}
  return {"min": ordered[0], "median": ordered[(len(ordered) - 1) // 2],
          "p95": ordered[int((len(ordered) - 1) * 0.95)], "max": ordered[-1],
          "mean": round(sum(ordered) / len(ordered), 2)}


def main() -> None:
  RUN.mkdir(parents=True, exist_ok=False)
  (RUN / "status.json").write_text(canonical({"disposition": "RUNNING", "approved": False}) + "\n")
  protocol_sha = sha256_file(PROTOCOL)
  provenance = {"protocol_version": "0.2.0", "protocol_sha256": protocol_sha,
                "code_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_sha256": sha256_file(Path(__file__)), "laya_revision": LAYA_REVISION,
                "laya_common_sha256": sha256_file(LAYA / "laya/common.py"),
                "tokenizer_files": {p.name: sha256_file(p) for p in sorted(TOKENIZER_PATH.iterdir()) if p.is_file()}}
  (RUN / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
  assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LAYA, text=True).strip() == LAYA_REVISION
  assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=LAYA, text=True).strip()
  source_sha = sha256_file(SOURCE)
  if source_sha != SOURCE_SHA256:
    raise RuntimeError(f"STOP: source checksum mismatch: {source_sha}")

  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
  inputs = LayaInputs(tokenizer)
  rows = pq.read_table(SOURCE).to_pylist()
  ledger: list[dict[str, Any]] = []
  candidates: list[dict[str, Any]] = []
  counts = Counter()
  total_canary_failures = 0
  canary_tested = 0

  for row_index, source_row in enumerate(rows):
    if row_index % 1000 == 0 and row_index > 0:
      print(f"Processed {row_index}/{len(rows)} rows...", file=sys.stderr)
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
      if not re.search(r"</?tool_call\b", prior_call["value"], re.I) or not re.search(r"</?tool_response\b", prior_result["value"], re.I):
        continue
      counts["observation_conditioned_candidates"] += 1
      calls = payloads(prior_call["value"], "tool_call")
      results = payloads(prior_result["value"], "tool_response")
      if (not calls or not results or len(calls) != len(results)
          or [p["name"] for p in calls] != [p["name"] for p in results]
          or any(p["name"] not in TOOLS for p in calls)):
        ledger.append({"trajectory_id": trajectory_id, "message_index": message_index,
                       "scope": "candidate", "reason": "invalid_prior_call_result_linkage"})
        counts["linkage_exclusions"] += 1
        continue
      target_calls = payloads(target_message["value"], "tool_call")
      if target_calls is None or any(value["name"] not in TOOLS for value in target_calls):
        ledger.append({"trajectory_id": trajectory_id, "message_index": message_index,
                       "scope": "candidate", "reason": "malformed_or_unavailable_target"})
        counts["invalid_target_exclusions"] += 1
        continue
      labels = sorted({value["name"] for value in target_calls})
      group = proxy_group(source_row["task"])
      split = assign_split(group)
      state = render_source(source_row, message_index)
      sequences, required_length = inputs.build(state)
      all_within_budget = sequences is not None
      if all_within_budget:
        canary_tested += 1
        try:
          canary_checks(source_row, message_index, inputs, state, sequences)
        except AssertionError as error:
          total_canary_failures += 1
          ledger.append({"trajectory_id": trajectory_id, "message_index": message_index,
                         "scope": "candidate", "reason": "canary_failure", "detail": str(error)})

      candidate = {"trajectory_id": trajectory_id, "message_index": message_index, "category": source_row["category"],
                   "proxy_group": group, "split": split, "labels": labels, "sequences": sequences,
                    "all_within_budget": all_within_budget, "state": state if all_within_budget else None}
      candidates.append(candidate)
      if not all_within_budget:
        ledger.append({"trajectory_id": trajectory_id, "message_index": message_index, "scope": "candidate",
                       "reason": "over_budget", "split": split, "category": source_row["category"],
                       "proxy_group_sha256": hashlib.sha256(group.encode()).hexdigest(), "labels": labels,
                        "max_token_count": required_length})

  valid_budget = [item for item in candidates if item["all_within_budget"]]
  # For deduplication, use a hash of all 6 sequences combined
  valid_budget.sort(key=lambda item: (item["trajectory_id"], item["message_index"]))
  retained = []
  seen: set[tuple] = set()
  for item in valid_budget:
    token_key = input_key(item["sequences"])
    if token_key in seen:
      ledger.append({"trajectory_id": item["trajectory_id"], "message_index": item["message_index"],
                     "scope": "candidate", "reason": "duplicate_token_state",
                     "state_hash": hashlib.sha256(canonical(token_key).encode()).hexdigest()})
      continue
    seen.add(token_key)
    retained.append(item)

  split_stats: dict[str, Any] = {}
  support: dict[str, Any] = {}
  for split in SPLITS:
    split_candidates = [item for item in candidates if item["split"] == split]
    split_retained = [item for item in retained if item["split"] == split]
    over = sum(not item["all_within_budget"] for item in split_candidates)
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
    return {name: {"candidates": len(items), "over_budget": sum(not item["all_within_budget"] for item in items),
                   "over_budget_rate": rate(sum(not item["all_within_budget"] for item in items), len(items))}
             for name, items in sorted(buckets.items())}

  label_bias = {}
  label_rate_differences = {}
  for tool in TOOLS:
    positive = [item for item in candidates if tool in item["labels"]]
    negative = [item for item in candidates if tool not in item["labels"]]
    positive_rate = rate(sum(not item["all_within_budget"] for item in positive), len(positive))
    negative_rate = rate(sum(not item["all_within_budget"] for item in negative), len(negative))
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
    "target_and_metadata_canary_invariance": total_canary_failures == 0,
    "canary_coverage": canary_tested == len(valid_budget) and canary_tested > 0,
    "complete_adjacent_linkage": counts["linkage_exclusions"] == 0,
    "valid_retained_labels": all(set(item["labels"]) <= set(TOOLS) for item in retained),
    "no_truncation_and_max_512": all(item["all_within_budget"] for item in retained),
    "exact_token_state_unique": len(seen) == len(retained),
    "split_isolation": all(len(value) == 1 for value in trajectory_splits.values()) and all(len(value) == 1 for value in group_splits.values()),
    "minimum_positive_trajectory_support": all(support[split][tool]["positive_trajectories"] >= 10 for split in SPLITS for tool in TOOLS),
    "split_over_budget_difference_lte_10pp": max(split_rates) - min(split_rates) <= 0.10,
    "label_over_budget_difference_lte_20pp": all(value <= 0.20 for value in label_rate_differences.values()),
  }
  disposition = "PASS" if all(gates.values()) else "FAIL"

  model_paths = write_model_outputs(retained, gates, DATA_OUT)
  manifest = [{key: item[key] for key in ("trajectory_id", "message_index", "proxy_group", "split", "labels")} |
              {"input_sha256": hashlib.sha256(canonical(input_key(item["sequences"])).encode()).hexdigest(),
               "max_length": max(s["length"] for s in item["sequences"])} for item in retained]
  deterministic_gzip_write(RESULTS / "retained_manifest.jsonl.gz", "".join(canonical(item) + "\n" for item in manifest))
  (RESULTS / "split_manifest.json").write_text(json.dumps({split: {
    "trajectory_ids": sorted({item["trajectory_id"] for item in retained if item["split"] == split}),
    "proxy_groups": sorted({item["proxy_group"] for item in retained if item["split"] == split})}
    for split in SPLITS}, indent=2, sort_keys=True) + "\n")

  # Always write exclusion ledger (regardless of disposition)
  with open(RESULTS / "exclusion_ledger.jsonl.gz", "wb") as handle:
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=handle) as gz:
      for entry in sorted(ledger, key=lambda value: (value.get("row_index", -1), value["trajectory_id"], value.get("message_index", -1), value["reason"])):
        gz.write((canonical(entry) + "\n").encode("utf-8"))

  # For retained token lengths, use the max sequence length across all 6 questions
  retained_max_lengths = [max(s["length"] for s in item["sequences"]) for item in retained]
  retained_mean_lengths = [sum(s["length"] for s in item["sequences"]) / len(item["sequences"]) for item in retained]

  profile = {
    "profile_version": "exp001b-preprocessing-v0.2.0", "disposition": disposition,
    "execution": str(RUN.relative_to(ROOT)), "provenance": provenance, "state_rooms": inputs.rooms,
    "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": source_sha, "rows": len(rows)},
    "tokenizer": {"path": str(TOKENIZER_PATH.relative_to(ROOT)), "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982"},
    "counts": dict(sorted(counts.items())) | {"valid_before_budget": len(candidates), "over_budget_exclusions": len(candidates) - len(valid_budget),
                                                 "dedup_exclusions": len(valid_budget) - len(retained), "retained_rows": len(retained),
                                                 "retained_trajectories": len(trajectory_splits), "retained_proxy_groups": len(group_splits)},
    "retained_max_token_lengths": distribution(retained_max_lengths),
    "retained_mean_token_lengths": distribution(retained_mean_lengths),
    "split": split_stats, "positive_support": support,
    "assertions": {"residual_reasoning_marker_rows": marker_rows, "canary_failures": total_canary_failures,
                   "canary_candidates_tested": canary_tested, "target_checks": canary_tested,
                   "metadata_checks": canary_tested, "reasoning_variant_checks": 2 * canary_tested,
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

  checksum_paths = [SOURCE, PROTOCOL, Path(__file__), RESULTS / "provenance.json", RESULTS / "profile.json",
                    RESULTS / "exclusion_ledger.jsonl.gz", RESULTS / "retained_manifest.jsonl.gz",
                    RESULTS / "split_manifest.json", ANALYSIS / "selection_bias.md", *model_paths.values()]
  checksums = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in checksum_paths}
  (RESULTS / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": checksums}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [name for name, passed in gates.items() if not passed]
  (LOGS / "build.log").write_text("\n".join(["command=.venv/bin/python experiments/exp_001b_hermes_preprocessing/scripts/build_dataset.py",
                                                "network=disabled_by_local_files_only", "model_loading=none", "inference=none", "training=none",
                                                 f"source_sha256={source_sha}", f"protocol_sha256={protocol_sha}",
                                                 f"disposition={disposition}", f"failed_gates={canonical(failed)}", ""]), encoding="utf-8")
  (RUN / "status.json").write_text(canonical({"disposition": disposition, "approved": disposition == "PASS",
    "protocol_sha256": protocol_sha, "checksums_sha256": sha256_file(RESULTS / "checksums.json")}) + "\n")

  if disposition == "FAIL":
    sys.exit(1)


if __name__ == "__main__":
  if sys.flags.optimize:
    raise RuntimeError("Assertions must be enabled")
  # Existing executions are immutable, including their failure evidence.
  if RUN.exists():
    raise SystemExit(f"Refusing to reuse execution directory: {RUN}")
  try:
    main()
  except Exception as error:
    if RUN.exists():
      diagnostic = {"disposition": "ERROR", "approved": False, "error": repr(error)}
      (RUN / "status.json").write_text(canonical(diagnostic) + "\n")
      (RUN / "error.json").write_text(canonical(diagnostic) + "\n")
    raise
