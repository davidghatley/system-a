"""Dependency-free record-level classification metrics for Iteration 3."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


class MetricsError(ValueError):
    """Prediction input violates the frozen metric contract."""


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    nll_floor: float = 1e-12,
) -> dict[str, Any]:
    """Score one probability distribution per whole-turn record.

    Every input record must contain exactly ``record_id``, ``gold_label`` and a
    ``probabilities`` mapping in the fixed label insertion order. Missing
    classes remain in macro-F1 and receive zero support/recall/F1.
    """
    fixed = list(labels)
    if len(fixed) < 2 or len(set(fixed)) != len(fixed) or not all(isinstance(x, str) and x for x in fixed):
        raise MetricsError("labels must be at least two unique non-empty strings")
    if not isinstance(nll_floor, (int, float)) or not math.isfinite(nll_floor) or not 0 < nll_floor < 1:
        raise MetricsError("nll_floor must be finite and strictly between zero and one")

    rows = list(records)
    if not rows:
        raise MetricsError("at least one record is required")
    label_index = {label: index for index, label in enumerate(fixed)}
    confusion = [[0 for _ in fixed] for _ in fixed]
    seen: set[str] = set()
    nll_sum = brier_sum = 0.0

    for position, row in enumerate(rows):
        if set(row) != {"record_id", "gold_label", "probabilities"}:
            raise MetricsError(f"record {position} must contain only record_id, gold_label, and probabilities")
        record_id = row["record_id"]
        gold = row["gold_label"]
        probabilities = row["probabilities"]
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise MetricsError(f"record {position} has a missing or duplicate record_id")
        seen.add(record_id)
        if gold not in label_index:
            raise MetricsError(f"record {record_id} has an unknown gold_label")
        if not isinstance(probabilities, Mapping) or list(probabilities) != fixed:
            raise MetricsError(f"record {record_id} probabilities must use the fixed label order")
        values = list(probabilities.values())
        if not all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values):
            raise MetricsError(f"record {record_id} probabilities must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=2e-6):
            raise MetricsError(f"record {record_id} probabilities must sum to one")
        predicted_index = max(range(len(fixed)), key=values.__getitem__)
        gold_index = label_index[gold]
        confusion[gold_index][predicted_index] += 1
        nll_sum -= math.log(max(float(values[gold_index]), nll_floor))
        brier_sum += sum((float(value) - float(index == gold_index)) ** 2 for index, value in enumerate(values))

    per_class = {}
    f1_values = []
    correct = 0
    for index, label in enumerate(fixed):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        correct += true_positive
        f1_values.append(f1)
        per_class[label] = {"support": support, "recall": recall, "precision": precision, "f1": f1}

    result = {
        "unit": "whole_turn_record",
        "labels": fixed,
        "records": len(rows),
        "macro_f1": sum(f1_values) / len(fixed),
        "accuracy": correct / len(rows),
        "per_class": per_class,
        "confusion_matrix": {"rows": "gold", "columns": "prediction", "labels": fixed, "values": confusion},
        "nll": nll_sum / len(rows),
        "brier": brier_sum / len(rows),
        "nll_probability_floor": nll_floor,
    }
    if not _finite_tree(result):
        raise MetricsError("metric computation produced a non-finite value")
    return result
