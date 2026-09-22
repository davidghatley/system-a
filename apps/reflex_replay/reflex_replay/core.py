"""Validation and evaluation for stored next-action predictions."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time
from typing import Any


LABELS = ("edit", "execute", "other_tool", "read", "respond_or_finish", "search")
AMBIGUOUS_MARGIN = 0.05
ADVISORY_WARNING = (
    "Experimental replay only. Agreement predicts an observed action; it is not "
    "a control signal, reliable recommendation, optimality judgment, or task-success result."
)
PROVENANCE_WARNING = (
    "Trace provenance is incomplete: physical execution, teacher identity, task success, "
    "action optimality, and source/model overlap are unverified or unknown."
)


class ReplayError(Exception):
    """An actionable replay input error."""


def _json_lines(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReplayError(f"cannot read {path}: {exc}") from exc
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ReplayError(f"{path}:{number}: expected a JSON object")
        result.append(value)
    return result


def _source_line(root: pathlib.Path, relative: str, line_number: int) -> tuple[dict[str, Any], str]:
    path = root / relative
    try:
        raw = path.read_text(encoding="utf-8").splitlines()[line_number - 1]
    except (OSError, IndexError) as exc:
        raise ReplayError(f"cannot load frozen source {relative}:{line_number}: {exc}") from exc
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReplayError(f"frozen source {relative}:{line_number} is invalid JSON") from exc
    return record, hashlib.sha256((raw + "\n").encode()).hexdigest()


def _canonical(value: Any) -> bytes:
    """The bytes covered by the local, tamper-evident prediction envelope."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _compact_source_sha256(record: dict[str, Any]) -> str:
    try:
        compact = record["state"]
    except KeyError as exc:
        raise ReplayError("frozen source record has no compact state") from exc
    return hashlib.sha256(_canonical(compact)).hexdigest()


def _prediction_body(prediction: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in prediction.items() if key != "integrity"}
    return {"artifact": artifact, "prediction": payload}


def _artifact_for_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": case["source"],
        "source_line": case["source_line"],
        "source_line_sha256": case["source_line_sha256"],
        "compact_source_sha256": _compact_source_sha256(case["record"]),
        "record_id": case["record_id"],
        "observed_label": case["observed_label"],
    }


def _verify_prediction_integrity(prediction: dict[str, Any], case: dict[str, Any]) -> None:
    integrity = prediction.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "SHA-256" or integrity.get("canonicalization") != "json-sort-keys-utf8-v1":
        raise ReplayError(f"prediction for {case['record_id']} lacks a supported integrity envelope")
    artifact = integrity.get("artifact")
    expected_artifact = _artifact_for_case(case)
    if artifact != expected_artifact:
        raise ReplayError(f"prediction for {case['record_id']} has mismatched artifact identity")
    digest = integrity.get("sha256")
    if not isinstance(digest, str) or digest != hashlib.sha256(_canonical(_prediction_body(prediction, artifact))).hexdigest():
        raise ReplayError(f"prediction for {case['record_id']} failed SHA-256 integrity verification")
    if prediction.get("record_id") != case["record_id"] or prediction.get("reference") != case["observed_label"]:
        raise ReplayError(f"prediction for {case['record_id']} has mismatched bound reference")


def load_suite(root: pathlib.Path, manifest_path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot load suite {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReplayError("suite must be a JSON object")
    if manifest.get("schema_version") != "reflex-replay-suite-v1":
        raise ReplayError("suite.schema_version must be 'reflex-replay-suite-v1'")
    cases = []
    manifest_cases = manifest.get("cases", [])
    if not isinstance(manifest_cases, list):
        raise ReplayError("suite.cases must be an array")
    for index, case in enumerate(manifest_cases, 1):
        try:
            if not isinstance(case["source_line"], int) or isinstance(case["source_line"], bool) or case["source_line"] < 1:
                raise TypeError
            record, digest = _source_line(root, case["source"], case["source_line"])
            record_id = record["metadata"]["id"]
            observed = record["gold"]["next_action"]["label"]
        except (KeyError, TypeError) as exc:
            raise ReplayError(f"suite case {index}: malformed case or source record") from exc
        if digest != case["source_line_sha256"]:
            raise ReplayError(f"suite case {index}: source hash mismatch for {record_id}")
        if record_id != case["record_id"] or observed != case["observed_label"]:
            raise ReplayError(f"suite case {index}: frozen identity/reference mismatch")
        cases.append({**case, "record": record})
    if not cases:
        raise ReplayError("suite must contain at least one case")
    return cases


def evaluate_prediction(prediction: dict[str, Any], observed: str) -> dict[str, Any]:
    try:
        probabilities = prediction["probabilities"]
        model = prediction["model"]
        elapsed = float(prediction["elapsed_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayError("prediction requires model, probabilities, and elapsed_seconds") from exc
    if not isinstance(model, dict) or not all(isinstance(model.get(key), str) and model[key].strip() for key in ("id", "revision")):
        raise ReplayError("prediction.model requires non-empty string id and revision")
    if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
        raise ReplayError(f"prediction probabilities must contain exactly {list(LABELS)}")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ReplayError("prediction.elapsed_seconds must be finite and non-negative")
    values: dict[str, float] = {}
    for label in LABELS:
        value = probabilities[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReplayError(f"probability for {label} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ReplayError(f"probability for {label} must be in [0, 1]")
        values[label] = value
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        raise ReplayError("prediction probabilities must sum to 1")
    ranked = sorted(values, key=lambda label: (-values[label], LABELS.index(label)))
    selected = ranked[0]
    margin = values[ranked[0]] - values[ranked[1]]
    return {
        "model": model,
        "probabilities": values,
        "selected": selected,
        "observed": observed,
        "correct": selected == observed,
        "top_two_margin": margin,
        "ambiguous_low_margin": margin <= AMBIGUOUS_MARGIN,
        "elapsed_seconds": elapsed,
        "elapsed_scope": prediction.get("elapsed_scope", "unspecified"),
        "provenance": prediction.get("provenance", {}),
    }


def _majority_prediction(record_id: str) -> dict[str, Any]:
    counts = {"edit": 178, "execute": 540, "other_tool": 0, "read": 241, "respond_or_finish": 133, "search": 143}
    total = sum(counts.values())
    started = time.perf_counter()
    probabilities = {label: counts[label] / total for label in LABELS}
    elapsed = time.perf_counter() - started
    return {
        "record_id": record_id,
        "model": {"id": "trace2decision-training-majority", "revision": "i2-v2-train-counts"},
        "probabilities": probabilities,
        "elapsed_seconds": elapsed,
        "elapsed_scope": "local rule construction only",
        "provenance": {"training_rows": total, "counts": counts},
    }


def replay(cases: list[dict[str, Any]], stored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_record: dict[str, list[dict[str, Any]]] = {}
    for prediction in stored:
        record_id = prediction.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ReplayError("each stored prediction requires record_id")
        by_record.setdefault(record_id, []).append(prediction)
    known = {case["record_id"] for case in cases}
    extras = sorted(set(by_record) - known)
    if extras:
        raise ReplayError(f"predictions contain record IDs outside the suite: {', '.join(extras)}")
    output = []
    for case in cases:
        observed = case["observed_label"]
        predictions = [evaluate_prediction(_majority_prediction(case["record_id"]), observed)]
        checked = []
        for item in by_record.get(case["record_id"], []):
            _verify_prediction_integrity(item, case)
            checked.append(item)
        predictions.extend(evaluate_prediction(item, observed) for item in checked)
        output.append({**case, "predictions": predictions})
    return output


def load_predictions(path: pathlib.Path) -> list[dict[str, Any]]:
    return _json_lines(path)
