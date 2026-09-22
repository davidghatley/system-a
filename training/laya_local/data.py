"""Conversion from native typed-decisions records to actual Laya items."""

from __future__ import annotations

from typing import Any

from shared.typed_decisions import parse_record


def to_laya_items(record: dict[str, Any], tokenizer: Any, *, max_len: int, head_max_len: int) -> list[dict[str, Any]]:
    """Convert each question without reordering or hardening its gold distribution."""
    from laya.common import QTYPES, build_sequence, render_options

    record = parse_record(record)
    metadata = record.get("metadata")
    record_id = record.get("id")
    if record_id is None and isinstance(metadata, dict):
        record_id = metadata.get("id")
    items = []
    for question_id, public_question in record["questions"].items():
        internal = {
            "t": public_question["type"],
            "ins": public_question["instructions"],
            "crit": public_question.get("criteria"),
        }
        ids, markers = build_sequence(tokenizer, record["state"], internal, max_len, head_max_len, truncate_left=True)
        options = render_options(internal)
        if len(markers) != len(options):
            raise ValueError(f"question {question_id!r} options exceed head_max_len={head_max_len}")
        probability_map = record["gold"][question_id]["probabilities"]
        keys = list(public_question["criteria"]) if internal["t"] == "choice" else (
            [str(i) for i in range(len(public_question["criteria"]))] if internal["t"] == "score" else ["false", "true"]
        )
        target = [float(probability_map[key]) for key in keys]
        native_label = keys.index(record["gold"][question_id]["label"])
        items.append({
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES[internal["t"]],
            "target": target,
            # Exact-choice accuracy is deliberately the native hard label.  The
            # distribution argmax remains useful diagnostic information only.
            "label": native_label,
            "native_label": native_label,
            "gold_argmax": max(range(len(target)), key=target.__getitem__),
            "record_id": record_id,
            "metadata": metadata,
            "question_id": question_id,
            "question_type": internal["t"],
            "option_keys": keys,
        })
    return items
