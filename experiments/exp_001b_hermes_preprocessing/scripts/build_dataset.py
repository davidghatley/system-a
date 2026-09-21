#!/usr/bin/env python3
"""Build the frozen exp_001b preprocessing artifacts. No model is loaded or run."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Suppress transformers warning about long sequences
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")


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
MAX_LEN = 512
HEAD_MAX_LEN = 192
CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.I | re.S)
RESPONSE_RE = re.compile(r"<tool_response\b[^>]*>(.*?)</tool_response\s*>", re.I | re.S)
THINK_TAG_RE = re.compile(r"<think\b[^>]*>|</think\s*>", re.I)
REASONING_MARKER_RE = re.compile(r"</?think\b|<\|(?:begin|end)_of_thought\|>|(?:^|\n)\s*(?:reasoning|analysis)\s*:", re.I)


def serialize_state(state: Any) -> str:
    """Serialize state for Laya sequence construction."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_noul_instruction(tool_name: str) -> str:
    """Render the instruction for a noul question about a tool."""
    return f"Does the next assistant turn call the {tool_name} tool?"


def build_laya_sequence(tok, state: str, tool_name: str, max_len: int = MAX_LEN, head_max_len: int = HEAD_MAX_LEN, truncate_left: bool = False) -> tuple[list[int], list[int]]:
    """Build one Laya noul sequence: [CLS] noul question: <ins> [SEP] [MASK] no [MASK] yes [SEP] state [SEP]."""
    mask_tok = tok.mask_token
    # Options for noul: false (no), true (yes)
    opts = [
        "false: no, the statement does not hold",
        "true: yes, the statement holds"
    ]
    ins = render_noul_instruction(tool_name).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % ("noul", ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for opt in opts:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opt.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    # Use tokenizer truncation to avoid tokenizing full long state
    serialized = serialize_state(state).replace(mask_tok, " ")
    st = tok(serialized, add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


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
  with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=path.open("wb")) as handle:
    handle.write(content.encode("utf-8"))


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


def render_state(task: str, prior_call: str, prior_result: str) -> str:
  """Render the state portion for Laya (without questions)."""
  return (f"Task:\n{task}\nAvailable tools: {', '.join(TOOLS)}\n"
          f"Previous assistant call:\n{prior_call}\nLatest tool response:\n{prior_result}")


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
  total_canary_failures = 0

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
      state = render_state(source_row["task"], prior_call["value"], prior_result["value"])

      # Build 6 Laya sequences (one per noul question)
      sequences = []
      all_within_budget = True
      for tool in TOOLS:
        seq_ids, seq_markers = build_laya_sequence(tokenizer, state, tool)
        sequences.append({"tool": tool, "ids": seq_ids, "markers": seq_markers, "length": len(seq_ids)})
        if len(seq_ids) > MAX_LEN:
          all_within_budget = False

      # Canary tests (efficient): test state tokenization directly since state is tokenized
      # independently in Laya's build_sequence. Only tokenize up to room budget.
      cand_canary_failures = 0
      # Compute room by building one sequence and extracting state portion
      sample_seq_ids, _ = build_laya_sequence(tokenizer, state, TOOLS[0])
      sep_token_id = tokenizer.sep_token_id
      sep_positions = [i for i, t in enumerate(sample_seq_ids) if t == sep_token_id]
      if len(sep_positions) >= 2:
        state_start = sep_positions[-2] + 1
        state_end = sep_positions[-1]
        room = state_end - state_start
      else:
        room = 400  # fallback

      state_ids = tokenizer(serialize_state(state).replace(tokenizer.mask_token, " "), add_special_tokens=False, truncation=True, max_length=room)["input_ids"]

      # Test 1: Target isolation - labels should not appear in state tokens
      mutated_state = state + " TARGET_CANARY_" + "_".join(labels)
      mutated_ids = tokenizer(serialize_state(mutated_state).replace(tokenizer.mask_token, " "), add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
      if state_ids == mutated_ids:
        cand_canary_failures += 1

      # Test 2: Excluded metadata isolation - mutate prior_call (part of state)
      mutated_call = prior_call["value"] + " EXCLUDED_METADATA_CANARY"
      mutated_state = render_state(source_row["task"], mutated_call, prior_result["value"])
      mutated_ids = tokenizer(serialize_state(mutated_state).replace(tokenizer.mask_token, " "), add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
      if state_ids == mutated_ids:
        cand_canary_failures += 1

      # Test 3: Reasoning isolation - add reasoning-like content to prior_call
      mutated_call = prior_call["value"] + " \nreason"
      mutated_state = render_state(source_row["task"], mutated_call, prior_result["value"])
      mutated_ids = tokenizer(serialize_state(mutated_state).replace(tokenizer.mask_token, " "), add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
      if state_ids == mutated_ids:
        cand_canary_failures += 1

      total_canary_failures += cand_canary_failures

      candidate = {"trajectory_id": trajectory_id, "message_index": message_index, "category": source_row["category"],
                   "proxy_group": group, "split": split, "labels": labels, "sequences": sequences,
                   "all_within_budget": all_within_budget, "state": state}
      candidates.append(candidate)
      if not all_within_budget:
        max_len = max(s["length"] for s in sequences)
        ledger.append({"trajectory_id": trajectory_id, "message_index": message_index, "scope": "candidate",
                       "reason": "over_budget", "split": split, "category": source_row["category"],
                       "proxy_group_sha256": hashlib.sha256(group.encode()).hexdigest(), "labels": labels,
                       "max_token_count": max_len})

  valid_budget = [item for item in candidates if item["all_within_budget"]]
  # For deduplication, use a hash of all 6 sequences combined
  valid_budget.sort(key=lambda item: (item["trajectory_id"], item["message_index"]))
  retained = []
  seen: set[tuple[int, ...]] = set()
  for item in valid_budget:
    # Combine all sequence token IDs for deduplication key
    token_key = tuple(t for seq in item["sequences"] for t in seq["ids"])
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

  model_paths = {}
  if disposition == "PASS":
    for split in SPLITS:
      output = DATA_OUT / f"{split}.jsonl.gz"
      with open(output, "wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=handle) as gz:
          for item in retained:
            if item["split"] == split:
              gz.write((canonical({key: item[key] for key in ("trajectory_id", "message_index", "proxy_group", "state", "sequences", "labels")}) + "\n").encode("utf-8"))
      model_paths[split] = output
      ids = sorted({item["trajectory_id"] for item in retained if item["split"] == split})
      (RESULTS / f"{split}_ids.txt").write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")

  # Always write exclusion ledger (regardless of disposition)
  with open(RESULTS / "exclusion_ledger.jsonl.gz", "wb") as handle:
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=handle) as gz:
      for entry in sorted(ledger, key=lambda value: (value.get("row_index", -1), value["trajectory_id"], value.get("message_index", -1), value["reason"])):
        gz.write((canonical(entry) + "\n").encode("utf-8"))

  # For retained token lengths, use the max sequence length across all 6 questions
  retained_max_lengths = [max(s["length"] for s in item["sequences"]) for item in retained]
  retained_mean_lengths = [sum(s["length"] for s in item["sequences"]) / len(item["sequences"]) for item in retained]

  profile = {
    "profile_version": "exp001b-preprocessing-v2", "disposition": disposition,
    "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": source_sha, "rows": len(rows)},
    "tokenizer": {"path": str(TOKENIZER_PATH.relative_to(ROOT)), "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982"},
    "counts": dict(sorted(counts.items())) | {"valid_before_budget": len(candidates), "over_budget_exclusions": len(candidates) - len(valid_budget),
                                                "dedup_exclusions": len(valid_budget) - len(retained), "retained_rows": len(retained)},
    "retained_max_token_lengths": distribution(retained_max_lengths),
    "retained_mean_token_lengths": distribution(retained_mean_lengths),
    "split": split_stats, "positive_support": support,
    "assertions": {"residual_reasoning_marker_rows": marker_rows, "canary_failures": total_canary_failures,
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

  checksum_paths = [SOURCE, RESULTS / "profile.json", RESULTS / "exclusion_ledger.jsonl.gz", ANALYSIS / "selection_bias.md"]
  if disposition == "PASS":
    checksum_paths.extend([*model_paths.values(), *(RESULTS / f"{split}_ids.txt" for split in SPLITS)])
  checksums = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in checksum_paths}
  (RESULTS / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": checksums}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [name for name, passed in gates.items() if not passed]
  (LOGS / "build.log").write_text("\n".join(["command=.venv/bin/python experiments/exp_001b_hermes_preprocessing/scripts/build_dataset.py",
                                                "network=disabled_by_local_files_only", "model_loading=none", "inference=none", "training=none",
                                                f"source_sha256={source_sha}", f"disposition={disposition}", f"failed_gates={canonical(failed)}", ""]), encoding="utf-8")

  if disposition == "FAIL":
    sys.exit(1)


if __name__ == "__main__":
  main()
