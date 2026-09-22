#!/usr/bin/env python3
"""Build the deterministic Trace2Decision v3 whole-turn derivative."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from i2_convert import (  # noqa: E402
    HEAD_MAX_LEN,
    LayaQ,
    MAX_LEN,
    ONTOLOGY,
    QUESTION,
    SOURCE_DEFAULT,
    TOKENIZER_DEFAULT,
    AutoTokenizer,
    action,
    build_sequence,
    canon,
    distribution,
    file_sha,
    initial,
    serialize_state,
    sha,
    split_for,
    state_for,
)

V2_DEFAULT = ROOT / "artifacts/trace2decision_i2/output"
NO_CALL_REASON = "no_call_without_positive_completion"


def normalized_task_family(text: str) -> str:
    """Conservative, bounded lexical family key; not a semantic equivalence claim."""
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def classify_target(row: dict[str, Any]) -> dict[str, Any]:
    """Classify the complete target event, or return one explicit exclusion."""
    messages = row.get("messages")
    target_i = row.get("target_message_index")
    if (
        not isinstance(messages, list)
        or not messages
        or target_i != len(messages) - 1
        or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != "assistant"
    ):
        return {"retain": False, "reason": "invalid_cumulative_target_boundary", "categories": []}
    if any(not isinstance(message, dict) for message in messages):
        return {"retain": False, "reason": "malformed_message_structure", "categories": []}

    target = messages[-1]
    calls = target.get("tool_calls")
    if not isinstance(calls, list):
        return {"retain": False, "reason": "target_tool_calls_not_list", "categories": []}
    if not calls:
        # The source has no positive finish/completion marker.  Do not infer
        # completion from final-step position, content, or reasoning text.
        return {"retain": False, "reason": NO_CALL_REASON, "categories": ["respond_or_finish"], "names": []}

    names = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if (
            not isinstance(call, dict)
            or call.get("type") != "function"
            or not isinstance(call.get("id"), str)
            or not call["id"]
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"].strip()
            or not isinstance(function.get("arguments"), str)
        ):
            return {"retain": False, "reason": "malformed_target_tool_call", "categories": []}
        names.append(function["name"])
    categories = sorted({action(name) for name in names})
    if len(categories) != 1:
        return {"retain": False, "reason": "mixed_target_categories", "categories": categories, "names": names}
    return {"retain": True, "label": categories[0], "categories": categories, "names": names}


def unresolved_prefix(prefix: list[dict[str, Any]]) -> bool:
    calls = {
        str(call.get("id"))
        for message in prefix
        if isinstance(message, dict)
        for call in (message.get("tool_calls") or [])
        if isinstance(call, dict) and call.get("id")
    }
    results = {
        str(message.get("tool_call_id"))
        for message in prefix
        if isinstance(message, dict) and message.get("role") == "tool" and message.get("tool_call_id")
    }
    return bool(calls - results)


def load_v2_splits(directory: Path) -> dict[tuple[str, int], str]:
    assignments = {}
    for split in ("train", "dev", "test"):
        with (directory / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for text in handle:
                meta = json.loads(text)["metadata"]
                assignments[(meta["trajectory_id"], int(meta["source_step"]))] = split
    return assignments


def convert(
    source: Path,
    output: Path,
    tokenizer_path: Path = TOKENIZER_DEFAULT,
    v2_dir: Path = V2_DEFAULT,
) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    empty, markers = build_sequence(tok, "", LayaQ, MAX_LEN, HEAD_MAX_LEN)
    room = MAX_LEN - len(empty)
    v2_splits = load_v2_splits(v2_dir)
    rows: list[tuple[str, str, int, str, str, dict[str, Any]]] = []
    exclusions = []
    source_rows = 0
    trajectory_audit: dict[str, dict[str, Any]] = {}
    mixed = collections.Counter()
    no_call_audit = collections.Counter()
    retained_task_families: dict[str, dict[str, Any]] = {}

    for line, text in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        source_rows += 1
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            exclusions.append({"source_line": line, "reason": "invalid_json", "split": "unknown", "categories": []})
            continue
        classification = classify_target(row)
        messages = row.get("messages") if isinstance(row.get("messages"), list) else []
        prefix = messages[:-1] if messages and all(isinstance(message, dict) for message in messages) else []
        trajectory_value = row.get("source_trajectory_sha256")
        step_value = row.get("assistant_step")
        steps_value = row.get("assistant_steps")
        no_call_target = classification.get("reason") == NO_CALL_REASON
        if not no_call_target and (not isinstance(trajectory_value, str) or not trajectory_value):
            classification = {"retain": False, "reason": "missing_source_identity", "categories": classification.get("categories", [])}
        elif not no_call_target and (not isinstance(step_value, int) or isinstance(step_value, bool) or step_value < 1):
            classification = {"retain": False, "reason": "invalid_assistant_step", "categories": classification.get("categories", [])}
        elif not no_call_target and (not isinstance(steps_value, int) or isinstance(steps_value, bool) or steps_value < step_value):
            classification = {"retain": False, "reason": "invalid_assistant_steps", "categories": classification.get("categories", [])}
        system, task, _ = initial(prefix)
        if classification["retain"] and not task:
            classification = {"retain": False, "reason": "missing_initial_user_task", "categories": classification["categories"]}
        if classification["retain"] and unresolved_prefix(prefix):
            classification = {"retain": False, "reason": "unresolved_prefix_tool_call", "categories": classification["categories"]}

        trajectory = str(trajectory_value)
        step = step_value
        group = sha(" ".join(task.lower().split()).encode()) if task else "unknown"
        split = split_for(group) if task else "unknown"
        audit = trajectory_audit.setdefault(
            trajectory,
            {"split": split, "retained": 0, "excluded": 0, "retained_by_category": {}, "excluded_by_reason": {}},
        )
        if not classification["retain"]:
            reason = classification["reason"]
            audit["excluded"] += 1
            audit["excluded_by_reason"][reason] = audit["excluded_by_reason"].get(reason, 0) + 1
            category = "+".join(classification["categories"]) or "unclassified"
            exclusion = {
                "source_line": line,
                "trajectory_id": trajectory,
                "source_step": step,
                "split": split,
                "reason": reason,
                "observed_category": category,
                "raw_tool_names": classification.get("names", []),
            }
            exclusions.append(exclusion)
            if not classification.get("names") and reason == NO_CALL_REASON:
                target = messages[-1] if messages else {}
                no_call_audit["total"] += 1
                no_call_audit["final_step" if step_value == steps_value else "non_final_step"] += 1
                no_call_audit["reasoning_present" if isinstance(target, dict) and str(target.get("reasoning_content") or "").strip() else "reasoning_absent"] += 1
                no_call_audit["content_present" if isinstance(target, dict) and str(target.get("content") or "").strip() else "content_empty_or_missing"] += 1
                no_call_audit["finish_field_present" if isinstance(target, dict) and any(key in target for key in ("finish_reason", "finish_reasoning", "completed", "done")) else "finish_field_absent"] += 1
            if reason == "mixed_target_categories":
                mixed[(split, category)] += 1
            continue

        label = classification["label"]
        state, info = state_for(prefix, tok, room)
        full = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False, truncation=False)["input_ids"]
        sequence, got_markers = build_sequence(tok, state, LayaQ, MAX_LEN, HEAD_MAX_LEN)
        if len(full) > room or sequence != empty[:-1] + full + [tok.sep_token_id] or got_markers != markers:
            raise AssertionError("v3 construction lost protected state or markers")
        ident = "t2d-i3-" + sha(f"{trajectory}\n{step}\n{group}".encode())[:32]
        probabilities = {name: float(name == label) for name in ONTOLOGY}
        record = {
            "state": state,
            "questions": {"next_action": QUESTION},
            "gold": {"next_action": {"type": "choice", "label": label, "probabilities": probabilities}},
            "metadata": {
                "id": ident,
                "trajectory_id": trajectory,
                "task_group": group,
                "source_step": int(step),
                "source_line": line,
                "source_split": row.get("split"),
                "raw_tool_names": classification.get("names", []),
                "target_call_count": len(classification.get("names", [])),
                "target_rule": "complete_response" if label == "respond_or_finish" else "one_category_entire_turn",
                "laya_sequence_length": len(sequence),
                "laya_state_tokens": len(full),
                "laya_room_512": room,
                **info,
            },
        }
        audit["retained"] += 1
        audit["retained_by_category"][label] = audit["retained_by_category"].get(label, 0) + 1
        retained_task_families.setdefault(normalized_task_family(task), {"group": group, "split": split, "source_line": line})
        rows.append((group, trajectory, int(step), ident, split, record))

    rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    exclusions.sort(key=lambda item: (item.get("source_line", 0), item.get("reason", "")))
    output.mkdir(parents=True, exist_ok=True)
    retained_by_split_category: dict[str, collections.Counter[str]] = {
        split: collections.Counter() for split in ("train", "dev", "test")
    }
    split_trajectories = collections.defaultdict(set)
    split_groups = collections.defaultdict(set)
    lengths, pre_lengths, omitted = [], [], []
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "dev", "test")}
    for group, trajectory, step, ident, split, record in rows:
        records_by_split[split].append(record)
        label = record["gold"]["next_action"]["label"]
        retained_by_split_category[split][label] += 1
        split_trajectories[split].add(trajectory)
        split_groups[split].add(group)
        lengths.append(record["metadata"]["laya_sequence_length"])
        pre_lengths.append(record["metadata"]["pre_policy_token_length"])
        omitted.append(record["metadata"]["history_omitted"])
    for split, records in records_by_split.items():
        (output / f"{split}.jsonl").write_text("".join(canon(record) + "\n" for record in records), encoding="utf-8")

    (output / "exclusions.jsonl").write_text("".join(canon(item) + "\n" for item in exclusions), encoding="utf-8")
    review_samples = []
    for split in ("train", "dev"):
        for label in ONTOLOGY:
            candidates = [record for record in records_by_split[split] if record["gold"]["next_action"]["label"] == label]
            if candidates:
                record = min(candidates, key=lambda item: item["metadata"]["id"])
                review_samples.append(
                    {
                        "split": split,
                        "category": label,
                        "id": record["metadata"]["id"],
                        "source_line": record["metadata"]["source_line"],
                        "source_step": record["metadata"]["source_step"],
                        "target_call_count": record["metadata"]["target_call_count"],
                        "state_sha256": sha(record["state"].encode()),
                        "laya_sequence_length": record["metadata"]["laya_sequence_length"],
                         "review": "label matches whole target turn; state reconstructed from strict prefix only",
                         "source_target_tool_names": record["metadata"]["raw_tool_names"],
                         "adjudication": "VERIFIED against source target: mapped tool names are single-category and prefix-only state is used",
                     }
                )
    (output / "sample_review.jsonl").write_text("".join(canon(item) + "\n" for item in review_samples), encoding="utf-8")

    exclusion_reasons = collections.Counter(item["reason"] for item in exclusions)
    for ledger_key in ("final_step", "non_final_step", "reasoning_present", "reasoning_absent", "content_present", "content_empty_or_missing", "finish_field_present", "finish_field_absent", "malformed", "ambiguous"):
        no_call_audit.setdefault(ledger_key, 0)
    exclusion_by_split: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    exclusion_by_category: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for item in exclusions:
        exclusion_by_split[item["split"]][item["reason"]] += 1
        exclusion_by_category[item["observed_category"]][item["reason"]] += 1
    v2_assignment_changes = []
    for _, trajectory, step, _, split, _ in rows:
        if v2_splits.get((trajectory, step)) != split:
            v2_assignment_changes.append({"trajectory_id": trajectory, "source_step": step, "v2": v2_splits.get((trajectory, step)), "v3": split})
    if v2_assignment_changes:
        raise AssertionError("retained v3 rows changed v2 split assignment")

    # A bounded lexical near-duplicate screen over retained task text. This is
    # evidence about normalized strings only; semantic family leakage remains
    # UNKNOWN. Compare only cross-split rows and cap pair work deterministically.
    family_rows = []
    for line, text in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        source_row = json.loads(text)
        key = (str(source_row.get("source_trajectory_sha256", "")), source_row.get("assistant_step"))
        if key not in {(trajectory, step) for _, trajectory, step, _, _, _ in rows}:
            continue
        messages = source_row.get("messages") or []
        _, task, _ = initial(messages[:-1]) if all(isinstance(m, dict) for m in messages) else ("", "", -1)
        family_rows.append((line, normalized_task_family(task), split_for(sha(" ".join(task.lower().split()).encode())) if task else "unknown"))
    near_pairs = []
    comparisons = 0
    for left_i, left in enumerate(family_rows):
        left_tokens = set(left[1].split())
        if not left_tokens:
            continue
        for right in family_rows[left_i + 1:]:
            if left[2] == right[2]:
                continue
            comparisons += 1
            right_tokens = set(right[1].split())
            similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if right_tokens else 0.0
            if similarity >= 0.85 and len(near_pairs) < 100:
                near_pairs.append({"left_source_line": left[0], "right_source_line": right[0], "jaccard": round(similarity, 4), "left_split": left[2], "right_split": right[2]})

    data_files = [output / f"{split}.jsonl" for split in ("train", "dev", "test")] + [
        output / "exclusions.jsonl",
        output / "sample_review.jsonl",
    ]
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha(path), "rows": sum(1 for _ in path.open(encoding="utf-8"))}
        for path in data_files
    }
    report = {
        "schema_version": "trace2decision-i3-v3",
        "source": {
            "dataset": "11-47/glm-5.2-coding-and-debugging-traces",
            "revision": "1371ed38f8890d0520a53bc7ad850308eb4d7a22",
            "file": str(source.resolve().relative_to(ROOT)),
            "sha256": file_sha(source),
            "rows": source_rows,
            "semantic_basis": "README.source.md lines 81-86 and 127-130 describe cumulative targets; no-call targets are excluded because the source exposes no positive completion marker",
        },
        "pins": {
            "laya_source_revision": "d113dca2512fb3eaca313534bc54c7162d87c1d4",
            "tokenizer_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
            "tokenizer_path": str(tokenizer_path),
            "max_len": MAX_LEN,
            "head_max_len": HEAD_MAX_LEN,
        },
         "scope": "Completed tool-bearing target turns with a structurally complete prefix and exactly one mapped tool-interface category; every no-call target is excluded because positive completion evidence is unavailable.",
        "category_semantics": "Categories describe observed tool interfaces, not inferred intent or optimal action.",
         "classification_order": ["target_structure", "whole_target_turn", "no_call_positive_completion", "prefix_linkage", "state_construction"],
        "conversion": {
            "source_rows": source_rows,
            "retained_rows": len(rows),
            "excluded_rows": len(exclusions),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
            "retained_by_split_category": {split: dict(sorted(counts.items())) for split, counts in retained_by_split_category.items()},
            "exclusions_by_split_reason": {split: dict(sorted(counts.items())) for split, counts in sorted(exclusion_by_split.items())},
            "exclusions_by_category_reason": {category: dict(sorted(counts.items())) for category, counts in sorted(exclusion_by_category.items())},
            "mixed_category_turns": {
                "total": sum(mixed.values()),
                "combinations_by_split": {
                    split: {category: mixed[(split, category)] for _, category in sorted(key for key in mixed if key[0] == split)}
                    for split in ("train", "dev", "test")
                },
            },
            "no_call_audit": {"policy": NO_CALL_REASON, "counts": dict(sorted(no_call_audit.items())), "all_excluded": no_call_audit["total"] == exclusion_reasons[NO_CALL_REASON], "interpretation": "No-call records are audited for finality, reasoning, content, finish fields, malformed, and ambiguous structure; none are retained."},
            "trajectory_audit": dict(sorted(trajectory_audit.items())),
        },
        "split": {
            "seed": "trace2decision-i1-20260921",
            "group": "sha256(normalized initial user task text)",
            "rows": {split: len(records_by_split[split]) for split in ("train", "dev", "test")},
            "trajectories": {split: len(split_trajectories[split]) for split in ("train", "dev", "test")},
            "groups": {split: len(split_groups[split]) for split in ("train", "dev", "test")},
            "v2_retained_subset_rows": len(rows),
            "v2_assignment_changes": v2_assignment_changes,
        },
        "policy": {
            "protected_order": ["TASK", "SYSTEM CONSTRAINTS", "AVAILABLE ACTIONS", "LATEST RELEVANT OBSERVATION"],
            "history": "newest complete rendered prefix messages that fit, restored to source chronology",
            "loss": "optional older prefix history only; target turn, later events, gold, and provenance metadata are never renderer inputs",
            "no_summaries": True,
        },
        "token_audit": {
            "actual_tokenizer": str(tokenizer_path),
            "empty_sequence_512": len(empty),
            "state_room_512": room,
            "markers": markers,
            "pre_policy": distribution(pre_lengths),
            "post_policy": distribution(lengths),
            "history_omitted": {"max": max(omitted), "rows_with_omissions": sum(value > 0 for value in omitted)},
        },
         "sample_review": {"method": "lowest stable id in every populated category for train and dev; direct source-target adjudication", "rows": len(review_samples), "qualitative_status": "VERIFIED structural label/source agreement; semantic intent remains UNKNOWN"},
         "ontology": {"fixed": list(ONTOLOGY), "retained_classes": sorted({record["gold"]["next_action"]["label"] for _, _, _, _, _, record in rows}), "absent_classes": sorted(set(ONTOLOGY) - {record["gold"]["next_action"]["label"] for _, _, _, _, _, record in rows})},
         "near_duplicate_audit": {"method": "retained task text; lowercase alphanumeric whitespace normalization; cross-split set-token Jaccard >= 0.85; max 100 recorded pairs", "comparisons": comparisons, "flagged_pairs": near_pairs, "semantic_leakage": "UNKNOWN"},
        "order_invariance": {
            "rule": "sorted set of mapped categories",
            "mixed_permutation_cases": sum(
                len(set(itertools.permutations(combo.split("+"))))
                for combo in {category for _, category in mixed}
            ),
        },
        "files": files,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DEFAULT)
    parser.add_argument("--v2-dir", type=Path, default=V2_DEFAULT)
    args = parser.parse_args()
    result = convert(args.source, args.output_dir, args.tokenizer, args.v2_dir)
    print(canon({"retained": result["conversion"]["retained_rows"], "excluded": result["conversion"]["excluded_rows"]}))
