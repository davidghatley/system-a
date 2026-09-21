"""Validated API around the public Laya typed-decisions checkpoint."""

from __future__ import annotations

import math
import pathlib
import stat
import subprocess
import sys
from typing import Any, Mapping


CHECKPOINT_ID = "convaiinnovations/laya"
CHECKPOINT_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
LAYA_SOURCE_REVISION = "d113dca2512fb3eaca313534bc54c7162d87c1d4"
_NATIVE_MODULE_SUFFIXES = {".dll", ".dylib", ".pyd", ".so"}


class ReflexError(Exception):
    """Base class for actionable Reflex errors."""


class ReflexInputError(ReflexError):
    """The state/question record is malformed."""


class ReflexModelError(ReflexError):
    """The pinned checkpoint or inference runtime is unavailable."""


def _fail(path: str, message: str) -> None:
    raise ReflexInputError(f"{path}: {message}")


def validate_record(record: Any) -> dict[str, Any]:
    """Validate and return a native ``state + questions (+ gold)`` record."""
    if not isinstance(record, dict):
        _fail("input", "expected a JSON object")
    if "state" not in record:
        _fail("state", "required field is missing")
    if not isinstance(record["state"], (str, dict, list)):
        _fail("state", "expected a string, object, or array")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        _fail("questions", "expected a non-empty object")

    option_names: dict[str, list[str]] = {}
    for qid, question in questions.items():
        path = f"questions.{qid}"
        if not isinstance(qid, str) or not qid:
            _fail("questions", "question IDs must be non-empty strings")
        if not isinstance(question, dict):
            _fail(path, "expected an object")
        qtype = question.get("type")
        if qtype not in ("choice", "score", "noul"):
            _fail(f"{path}.type", "expected choice, score, or noul")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            _fail(f"{path}.instructions", "expected a non-empty string")
        criteria = question.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                _fail(f"{path}.criteria", "choice requires an object with at least two options")
            if any(not isinstance(key, str) or not key for key in criteria):
                _fail(f"{path}.criteria", "option names must be non-empty strings")
            option_names[qid] = list(criteria)
        elif qtype == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                _fail(f"{path}.criteria", "score requires an array with at least two levels")
            option_names[qid] = [str(index) for index in range(len(criteria))]
        else:
            if criteria is not None and (
                not isinstance(criteria, dict) or any(key not in ("false", "true") for key in criteria)
            ):
                _fail(f"{path}.criteria", "noul criteria may only describe false and true")
            option_names[qid] = ["false", "true"]

    gold = record.get("gold")
    if gold is not None:
        if not isinstance(gold, dict):
            _fail("gold", "expected an object keyed by question ID")
        unknown = set(gold) - set(questions)
        if unknown:
            _fail("gold", f"unknown question IDs: {', '.join(sorted(unknown))}")
        for qid, distribution in gold.items():
            path = f"gold.{qid}"
            if not isinstance(distribution, dict) or set(distribution) != set(option_names[qid]):
                _fail(path, f"expected exactly these options: {', '.join(option_names[qid])}")
            values = list(distribution.values())
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
                _fail(path, "probabilities must be finite non-negative numbers")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
                _fail(path, "probabilities must sum to 1 (within 1e-6)")
    return record


def resolve_checkpoint(cache_dir: pathlib.Path, offline: bool) -> pathlib.Path:
    """Resolve only the immutable checkpoint revision in the repository-local cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ReflexModelError("huggingface_hub is unavailable; install apps/reflex/requirements.txt") from exc
    try:
        path = pathlib.Path(
            snapshot_download(
                CHECKPOINT_ID,
                revision=CHECKPOINT_REVISION,
                cache_dir=str(cache_dir),
                local_files_only=offline,
            )
        ).resolve()
    except Exception as exc:
        mode = "offline cache" if offline else "Hugging Face download/cache"
        raise ReflexModelError(
            f"Pinned checkpoint {CHECKPOINT_ID}@{CHECKPOINT_REVISION} is unavailable in the {mode}. "
            "Run once without --offline with network access, then retry offline. "
            f"Underlying error: {exc}"
        ) from exc
    required = ("rl_agent_config.json", "model.safetensors", "encoder", "tokenizer")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise ReflexModelError(f"Pinned checkpoint snapshot is incomplete; missing: {', '.join(missing)}")
    return path


def _is_import_payload(path: pathlib.Path) -> bool:
    """Return whether a path can alter Python imports or native loading."""
    if path.suffix.lower() == ".py" or path.suffix.lower() in _NATIVE_MODULE_SUFFIXES:
        return True
    try:
        return bool(path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except FileNotFoundError:
        return False


def _validate_laya_source_integrity(source: pathlib.Path, status: str) -> None:
    """Reject changed import payloads, but tolerate generated bytecode caches."""
    violations: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        relative = line[3:]
        if " -> " in relative:
            relative = relative.rsplit(" -> ", 1)[1]
        path = pathlib.Path(relative)
        if "__pycache__" in path.parts and (
            path.suffix.lower() in {".pyc", ".pyo"} or path.name == "__pycache__"
        ):
            continue
        if _is_import_payload(source / path):
            violations.append(relative)
    if violations:
        raise ReflexModelError(
            f"Laya source checkout at {source} has changed import payload(s): {', '.join(sorted(violations))}; "
            f"restore it to commit {LAYA_SOURCE_REVISION} before loading"
        )


class Reflex:
    """Run deterministic, observational typed decisions over an agent state."""

    def __init__(self, agent: Any, checkpoint_path: pathlib.Path | None = None):
        self._agent = agent
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(
        cls,
        *,
        cache_dir: str | pathlib.Path = "data/cache/huggingface/hub",
        laya_source: str | pathlib.Path = "data/laya",
        device: str | None = None,
        offline: bool = False,
    ) -> "Reflex":
        source = pathlib.Path(laya_source).resolve()
        if not (source / "laya" / "agent.py").is_file():
            raise ReflexModelError(
                f"Pinned Laya source is unavailable at {source}. Clone NandhaKishorM/laya at "
                f"commit {LAYA_SOURCE_REVISION} there."
            )
        try:
            source_revision = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise ReflexModelError(f"Could not verify the Laya source revision at {source}") from exc
        if source_revision != LAYA_SOURCE_REVISION:
            raise ReflexModelError(
                f"Laya source revision mismatch: expected {LAYA_SOURCE_REVISION}, found {source_revision}"
            )
        try:
            status = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise ReflexModelError(f"Could not verify the Laya source tree integrity at {source}") from exc
        _validate_laya_source_integrity(source, status)
        checkpoint = resolve_checkpoint(pathlib.Path(cache_dir), offline)
        try:
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            from laya.agent import Agent

            agent = Agent(str(checkpoint), device=device)
            if device is not None and str(agent.device) != str(device):
                raise RuntimeError(f"requested device {device}, but Laya selected {agent.device}")
        except Exception as exc:
            raise ReflexModelError(
                f"Could not load pinned checkpoint on device {device or 'auto'}: {type(exc).__name__}: {exc}"
            ) from exc
        return cls(agent, checkpoint)

    def decide(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate questions and expose each complete probability distribution."""
        validated = validate_record(record)
        try:
            raw = self._agent.system_one(validated["state"], validated["questions"])
            answers_raw = _required_mapping(raw, "response")
            answers_raw = _required_mapping(answers_raw.get("answers"), "response.answers")
            requested = set(validated["questions"])
            if set(answers_raw) != requested:
                missing = sorted(requested - set(answers_raw))
                extra = sorted(set(answers_raw) - requested)
                detail = f"missing question IDs: {', '.join(missing)}" if missing else f"unexpected question IDs: {', '.join(extra)}"
                raise ValueError(detail)

            answers: dict[str, Any] = {}
            for qid, question in validated["questions"].items():
                source = _required_mapping(answers_raw[qid], f"response.answers.{qid}")
                qtype = question["type"]
                if source.get("type") != qtype:
                    raise ValueError(f"response.answers.{qid}.type must be {qtype!r}")
                confidence = _finite_number(source.get("confidence"), f"response.answers.{qid}.confidence")
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"response.answers.{qid}.confidence must be in [0, 1]")
                if qtype == "noul":
                    true_probability = _probability(source.get("noul"), f"response.answers.{qid}.noul")
                    displayed_true = round(true_probability, 4)
                    probabilities = {"false": round(1.0 - displayed_true, 4), "true": displayed_true}
                    selected = "true" if true_probability >= 0.5 else "false"
                else:
                    option_names = list(question["criteria"]) if qtype == "choice" else [str(i) for i in range(len(question["criteria"]))]
                    probabilities = _probabilities(source.get("probabilities"), option_names, qid)
                    selected = source.get("choice") if qtype == "choice" else max(probabilities, key=probabilities.get)
                    if qtype == "choice" and selected not in option_names:
                        raise ValueError(f"response.answers.{qid}.choice must name one of {option_names}")
                answer = {"type": qtype, "selected": selected, "probabilities": probabilities, "confidence": confidence}
                if qtype == "score":
                    answer["expected_score"] = _finite_number(source.get("score"), f"response.answers.{qid}.score")
                answers[qid] = answer
        except ReflexModelError:
            raise
        except Exception as exc:
            raise ReflexModelError(
                f"Laya returned an invalid response: {exc}. Check the pinned checkpoint/source versions and model output contract."
            ) from exc

        device = str(getattr(self._agent, "device", "unknown"))
        result = {
            "checkpoint": {"id": CHECKPOINT_ID, "revision": CHECKPOINT_REVISION},
            "device": device,
            "answers": answers,
            "usage": raw.get("usage", {}),
            "warning": (
                "Observational checkpoint scores, not a universal policy or safety verdict; "
                "probabilities are not established as calibrated for this agent-state domain."
            ),
        }
        if validated.get("gold"):
            agreement = {}
            for qid, distribution in validated["gold"].items():
                gold_selected = max(distribution, key=distribution.get)
                agreement[qid] = {
                    "gold_argmax": gold_selected,
                    "model_argmax": answers[qid]["selected"],
                    "argmax_agrees": answers[qid]["selected"] == gold_selected,
                }
            result["reference_evaluation"] = agreement
        return result


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _probability(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be in [0, 1]")
    return result


def _probabilities(value: Any, option_names: list[str], qid: str) -> dict[str, float]:
    probabilities = _required_mapping(value, f"response.answers.{qid}.probabilities")
    if set(probabilities) != set(option_names):
        raise ValueError(f"response.answers.{qid}.probabilities must contain exactly {option_names}")
    result = {name: _probability(probabilities[name], f"response.answers.{qid}.probabilities.{name}") for name in option_names}
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"response.answers.{qid}.probabilities must sum to 1")
    return result
