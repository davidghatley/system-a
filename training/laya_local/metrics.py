"""Metric helpers whose targets are reconstructed from native JSONL records."""

from __future__ import annotations

from typing import Any

from shared.typed_decisions import parse_record


def native_choice_label(record: dict[str, Any], question_id: str) -> int:
    """Return the index of the declared native gold label, not a probability argmax."""
    record = parse_record(record)
    question = record["questions"][question_id]
    keys = list(question["criteria"])
    return keys.index(record["gold"][question_id]["label"])


def reconstruct_choice_counts(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, int]:
    """Independently join predictions to native records and count hard labels."""
    native = {
        (record.get("id"), qid): native_choice_label(record, qid)
        for record in records
        for qid, question in record["questions"].items()
        if question["type"] == "choice"
    }
    rows = [row for row in predictions if row["type"] == "choice"]
    correct = sum(int(row["prediction"] == native[(row["record_id"], row["question_id"])]) for row in rows)
    return {"choice_correct": correct, "choice_total": len(rows)}
