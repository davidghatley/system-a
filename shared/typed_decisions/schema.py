"""Minimal validation for native state/questions/gold typed-decision records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


class ValidationError(ValueError):
    """A native typed-decisions record is malformed."""


def _decoded(value: Any, field: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{field} is not valid JSON: {exc}") from exc
    return value


def parse_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Decode Hub string columns and validate one public native record."""
    if not isinstance(raw, dict):
        raise ValidationError("record must be an object")
    record = dict(raw)
    for field in ("state", "questions", "gold"):
        if field not in record:
            raise ValidationError(f"missing required field: {field}")
        record[field] = _decoded(record[field], field)
    validate_record(record)
    return record


def _option_keys(question: dict[str, Any]) -> list[str]:
    qtype = question.get("type")
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2 or not all(isinstance(k, str) for k in criteria):
            raise ValidationError("choice criteria must be an object with at least two string keys")
        return list(criteria)
    if qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValidationError("score criteria must be a list with at least two levels")
        return [str(i) for i in range(len(criteria))]
    if qtype == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria).issubset({"false", "true"})):
            raise ValidationError("noul criteria may only describe false and true")
        return ["false", "true"]
    raise ValidationError(f"unsupported question type: {qtype!r}")


def validate_record(record: dict[str, Any]) -> None:
    """Validate state, questions, and probability-distribution semantics."""
    if not isinstance(record.get("state"), (str, dict, list)):
        raise ValidationError("state must be a string, object, or list")
    questions = record.get("questions")
    gold = record.get("gold")
    if not isinstance(questions, dict) or not questions:
        raise ValidationError("questions must be a non-empty object")
    if not isinstance(gold, dict) or set(gold) != set(questions):
        raise ValidationError("gold keys must exactly match question keys")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not isinstance(question, dict):
            raise ValidationError("question ids must be strings and definitions must be objects")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValidationError(f"question {qid!r} requires non-empty instructions")
        keys = _option_keys(question)
        answer = gold[qid]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValidationError(f"gold type mismatch for {qid!r}")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or list(probabilities) != keys:
            raise ValidationError(f"gold probability keys/order mismatch for {qid!r}: expected {keys}")
        values = list(probabilities.values())
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
            raise ValidationError(f"gold probabilities must be finite and nonnegative for {qid!r}")
        if not math.isclose(sum(values), 1.0, abs_tol=2e-5):
            raise ValidationError(f"gold probabilities must sum to one for {qid!r}")
        if answer.get("label") not in keys:
            raise ValidationError(f"gold label is not an option for {qid!r}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate native records from JSON Lines."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(parse_record(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValidationError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValidationError(f"{path}: no records")
    return records


def dump_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Validate and write records in the public native representation."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            parsed = parse_record(record)
            handle.write(json.dumps(parsed, sort_keys=True, separators=(",", ":")) + "\n")
