"""Gated Iteration 3 execution path.

This module deliberately stops before Torch/model import while the pre-run gate
is blocked.  The data, locking, metric, reload, timing, and one-shot lifecycle
interfaces are consequently testable on CPU without accidentally doing GPU work.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .metrics import evaluate_records
HEX64_LOCAL = re.compile(r"^[0-9a-f]{64}$")


from .protocol import (
    CHOICE_OPTION_ORDER,
    LABEL_TO_CHOICE_INDEX,
    ROOT,
    GpuLock,
    ProtocolError,
    admit_worker,
    budget_identity,
    code_identity,
    consume_final_test_attempt,
    expected_updates,
    load_json,
    repo_path,
    sha256,
    validate_record,
    validate_manifest,
    verify_split_files,
    validate_runtime_budget,
    AggregateBudget,
    select_dev_checkpoint,
    validate_recovery_checkpoint,
    validate_checkpoint_reload,
    validate_latency_evidence,
    validate_pilot_decision,
    validate_freeze_binding,
    validate_final_result,
    validate_worker_admission,
    claim_path_for_admission,
    claim_worker,
    create_once_json,
    validate_execution_claim,
    validate_recovery_admission,
    require_non_final_worker_open,
    future_run_paths,
    fsync_directory,
    _repo_relative,
    _mapping_sha256,
    budget_close,
    BUDGET_EPSILON,
    FINAL_HOLD_COMMAND,
    FINAL_TEST_RESERVE_SECONDS,
)


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


def six_class_rlcd_reference(logits: list[float], gold_label: str, perturbations: list[list[float]], sigma: float, *, log_floor: float = -9.21, spherical_weight: float = 0.75) -> dict[str, Any]:
    """Pure-Python six-choice RLCD+CE reference, including score gradient."""
    if len(logits) != 6 or gold_label not in LABEL_TO_CHOICE_INDEX or len(perturbations) < 2 or sigma <= 0:
        raise ProtocolError("invalid six-class RLCD reference inputs")
    if any(len(epsilon) != 6 for epsilon in perturbations):
        raise ProtocolError("each RLCD perturbation must span all six choice logits")
    if not all(math.isfinite(value) for value in logits + [value for epsilon in perturbations for value in epsilon]):
        raise ProtocolError("RLCD reference inputs must be finite")
    target_index = LABEL_TO_CHOICE_INDEX[gold_label]
    centered = []
    rewards = []
    for epsilon in perturbations:
        mean = sum(epsilon) / 6
        projected = [value - mean for value in epsilon]
        centered.append(projected)
        probabilities = _softmax([logit + noise for logit, noise in zip(logits, projected)])
        gold_probability = probabilities[target_index]
        reward = max(math.log(max(gold_probability, 1e-12)), log_floor)
        reward += spherical_weight * gold_probability / max(math.sqrt(sum(value * value for value in probabilities)), 1e-9)
        rewards.append(reward)
    reward_mean = sum(rewards) / len(rewards)
    raw_advantages = [reward - reward_mean for reward in rewards]
    variance = sum(value * value for value in raw_advantages) / (len(raw_advantages) - 1)
    scale = math.sqrt(variance) + 1e-6
    advantages = [value / scale for value in raw_advantages]
    log_probabilities = [-sum(value * value for value in epsilon) / (2 * sigma * sigma) for epsilon in centered]
    loss_rlcd = -sum(advantage * logp for advantage, logp in zip(advantages, log_probabilities)) / len(advantages)
    probabilities = _softmax(logits)
    loss_ce = -math.log(max(probabilities[target_index], 1e-12))
    rlcd_score_gradient = [
        -sum(advantage * epsilon[index] / (sigma * sigma) for advantage, epsilon in zip(advantages, centered)) / len(advantages)
        for index in range(6)
    ]
    score_gradient = [
        value + probabilities[index] - float(index == target_index)
        for index, value in enumerate(rlcd_score_gradient)
    ]
    return {
        "choice_option_order": CHOICE_OPTION_ORDER,
        "gold_choice_index": target_index,
        "reward": rewards,
        "loss_rlcd": loss_rlcd,
        "loss_ce": loss_ce,
        "loss": loss_rlcd + loss_ce,
        "rlcd_score_gradient": rlcd_score_gradient,
        "choice_logit_gradient": score_gradient,
    }


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f"{name} must be finite")
    return float(value)


def _actual_config_path(config_path: Path | None) -> Path:
    """Resolve the invocation's config file; direct API calls retain the frozen default."""
    resolved = (config_path or ROOT / "training/laya_trace_i3/config.json").resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ProtocolError("config path escapes repository")
    if not resolved.is_file():
        raise ProtocolError("actual config path is absent")
    return resolved


def _aggregate_budget(
    config: Mapping[str, Any], actual_config_path: Path
) -> AggregateBudget:
    root = repo_path(config["output_root"], artifact_output=True)
    return AggregateBudget(
        root / "gpu_budget.json",
        budget_identity(config, actual_config_path),
        float(config["budget"]["main_max_aggregate_gpu_seconds"]),
    )


def _public_result(value: Mapping[str, Any], *, drop_worker_wall: bool = False) -> dict[str, Any]:
    """Remove runtime-only metadata before immutable publication."""
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("_") and not (drop_worker_wall and key == "worker_wall_seconds")
    }


def _reservation_metadata(
    command: str,
    seed: int,
    admission_path: Path,
    decision_path: Path | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "command": command,
        "seed": seed,
        "admission_sha256": sha256(admission_path),
    }
    if decision_path is not None:
        metadata["pilot_decision_sha256"] = sha256(decision_path)
    return metadata


def _decorate_runtime_result(
    result: Mapping[str, Any],
    *,
    command: str,
    seed: int,
    admission_path: Path,
    claim_path: Path,
    decision_path: Path | None,
    actual_config_path: Path,
    manifest_path: Path,
    ledger: AggregateBudget,
    token: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Attach public provenance and the internal budget token to a result."""
    reservation = ledger.reservation_snapshot(token, require_finished=True)
    snapshot = ledger.snapshot()
    value = dict(result)
    value["worker_wall_seconds"] = elapsed_seconds if elapsed_seconds is not None else 0.0
    value["_budget_token"] = token
    value["_budget_ledger_path"] = _repo_relative(ledger.path)
    value["_budget_ledger_sha256"] = snapshot["sha256"]
    value["_budget_reservation"] = reservation
    value["_budget_identity"] = snapshot["identity"]
    value["admission_path"] = _repo_relative(admission_path)
    value["admission_sha256"] = sha256(admission_path)
    value["claim_path"] = _repo_relative(claim_path)
    value["claim_sha256"] = sha256(claim_path)
    value["config_path"] = _repo_relative(actual_config_path)
    value["config_sha256"] = sha256(actual_config_path)
    value["manifest_path"] = _repo_relative(manifest_path)
    value["manifest_sha256"] = sha256(manifest_path)
    value["code"] = code_identity()
    value["budget_ledger_path"] = _repo_relative(ledger.path)
    value["budget_ledger_sha256"] = snapshot["sha256"]
    value["budget_token"] = token
    value["budget_reservation"] = reservation
    if decision_path is not None:
        value["pilot_decision_path"] = _repo_relative(decision_path)
        value["pilot_decision_sha256"] = sha256(decision_path)
        value["selected_seed"] = seed
    if command == "actual-model-pilot":
        value["schema"] = "i3-actual-model-pilot-v2"
        # The pilot's own post-reservation snapshot is part of its result,
        # not a later mutable view of the ledger.
        value["budget_snapshot"] = snapshot
    elif command == "train":
        value["schema"] = "i3-train-result-v2"
    return value


def _finish_failed_budget(
    ledger: AggregateBudget,
    token: str,
    started: float,
    worker_error: BaseException,
) -> None:
    try:
        ledger.finish(token, time.monotonic() - started, outcome="failed")
    except ProtocolError as budget_error:
        raise ProtocolError(
            f"{budget_error}; original worker failure: "
            f"{type(worker_error).__name__}: {worker_error}"
        ) from worker_error


def accumulation_windows(order: list[int], accumulation: int) -> list[dict[str, Any]]:
    """Describe exact microbatch boundaries without retaining forward graphs."""
    if not order or isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        raise ProtocolError("accumulation schedule requires a non-empty order and positive integer")
    return [{"positions": list(range(start, min(start + accumulation, len(order)))), "record_indices": order[start:min(start + accumulation, len(order))], "divisor": min(accumulation, len(order) - start)} for start in range(0, len(order), accumulation)]


def _load_split_bytes(manifest: Mapping[str, Any], split: str, labels: list[str]) -> list[dict[str, Any]]:
    if split not in {"train", "dev", "test"}:
        raise ProtocolError("unknown split")
    entry = manifest["splits"][split]
    path = repo_path(entry["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ProtocolError(f"{split} file is absent or hash-mismatched")
    rows: list[dict[str, Any]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"{split} bytes are not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"{split} line {line_number} is not JSON") from exc
        validate_record(record, labels)
        rows.append(record)
    expected = entry["rows"]
    if len(rows) != expected:
        raise ProtocolError(f"{split} row count differs from manifest")
    return rows


def load_split(manifest: Mapping[str, Any], split: str, labels: list[str]) -> list[dict[str, Any]]:
    if split == "test":
        raise ProtocolError("test bytes are available only through the authenticated run_final_test route")
    return _load_split_bytes(manifest, split, labels)


def final_test_once(manifest: Mapping[str, Any], labels: list[str], run_dir: Path, freeze_sha256: str) -> tuple[Path, list[dict[str, Any]]]:
    """Retired compatibility probe; authorized runs must use :func:`run_final_test`."""
    raise ProtocolError("final_test_once cannot bypass authenticated final admission; use run_final_test")


def _final_selection(
    run_dir: Path,
    freeze_sha256: str,
    labels: list[str],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate the complete immutable chain before selecting for final test."""
    freeze = run_dir / "freeze.json"
    if not freeze.is_file() or sha256(freeze) != freeze_sha256:
        raise ProtocolError("dev-only freeze artifact is absent or hash-mismatched")
    frozen = load_json(freeze)
    if frozen.get("schema") != "i3-dev-freeze-v2":
        raise ProtocolError("historical freeze is read-only and non-executable; a provenance-bound freeze is required")
    decision_path = repo_path(frozen.get("pilot_decision_path", ""), artifact_output=True)
    validate_freeze_binding(
        frozen,
        freeze,
        decision_path,
        config,
        config_path,
        repo_path(config["data_manifest"]),
    )
    validate_pilot_decision(
        future_run_paths(config)[0],
        decision_path,
        config_path,
        repo_path(config["data_manifest"]),
        config,
        manifest,
    )
    # Import locally to avoid the runner/experiment import cycle.  This repeats
    # the full seed-result provenance validation at the irreversible boundary.
    from .runner import load_dev_candidates
    _, current_candidates = load_dev_candidates(config, manifest, config_path)
    if frozen.get("candidates") != current_candidates:
        raise ProtocolError("freeze candidates or result provenance changed")
    selection = select_dev_checkpoint(current_candidates, labels)
    if selection != frozen.get("selection"):
        raise ProtocolError("freeze selection does not match its validated dev candidates")
    return selection, frozen


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish JSON atomically without ever replacing an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProtocolError("seed recovery result already exists; overwrite is forbidden") from exc
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_open_fd(fd: int) -> str:
    """Hash the bytes reachable through an already-open descriptor."""
    digest = hashlib.sha256()
    offset = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        for chunk in iter(lambda: os.read(fd, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.lseek(fd, offset, os.SEEK_SET)
    return digest.hexdigest()


def load_authenticated_safetensors(path: Path, expected_sha256: str, load_file: Any) -> Any:
    """Load exact checkpoint bytes through one O_NOFOLLOW descriptor.

    The pathname is never passed to safetensors.  The descriptor remains open
    for both the pre-load hash and the load, and is hashed again afterward to
    catch an in-place mutation between those operations.
    """
    if not hasattr(os, "O_NOFOLLOW"):
        raise ProtocolError("O_NOFOLLOW is unavailable; authenticated checkpoint loading is fail-closed")
    expected = str(expected_sha256)
    if not HEX64_LOCAL.fullmatch(expected):
        raise ProtocolError("selected checkpoint SHA-256 is malformed")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ProtocolError("selected checkpoint cannot be opened with O_NOFOLLOW") from exc
    try:
        if _sha256_open_fd(fd) != expected:
            raise ProtocolError("selected checkpoint bytes changed before authenticated load")
        fd_path = f"/proc/self/fd/{fd}"
        loaded = load_file(fd_path)
        if _sha256_open_fd(fd) != expected:
            raise ProtocolError("selected checkpoint bytes changed during authenticated load")
        return loaded
    finally:
        os.close(fd)


def _finish_final_budget_if_needed(ledger: AggregateBudget, token: str, started: float, error: BaseException | None = None) -> None:
    try:
        current = ledger.reservation_snapshot(token)
    except ProtocolError:
        return
    if current.get("state") != "in_flight":
        return
    try:
        ledger.finish(token, time.monotonic() - started, outcome="failed" if error is not None else "finished")
    except ProtocolError as budget_error:
        if error is None:
            raise
        raise ProtocolError(f"{budget_error}; original final worker failure: {type(error).__name__}: {error}") from error


def run_final_test(config: Mapping[str, Any], manifest: Mapping[str, Any], freeze_sha256: str, *, review_accepted: bool = False, config_path: Path | None = None) -> dict[str, Any]:
    """Admit and activate the immutable final allowance before one attempt."""
    if not review_accepted:
        raise ProtocolError("final-test requires explicit independent review authorization")
    actual_config_path = _actual_config_path(config_path)
    run_dir = repo_path(config["output_root"], artifact_output=True)
    require_non_final_worker_open(config, route="final-test")
    if (run_dir / "final_test.result.json").exists():
        raise ProtocolError("final-test result already exists; retry forbidden")
    if (run_dir / "final_test.attempt.json").exists():
        raise ProtocolError("final test attempt was already consumed; retry forbidden")
    selection, frozen = _final_selection(
        run_dir, freeze_sha256, list(config["labels"]), config, manifest, actual_config_path,
    )
    decision_value = frozen.get("pilot_decision_path") if isinstance(frozen, Mapping) else None
    if not isinstance(decision_value, str) or not decision_value:
        raise ProtocolError("final freeze lacks pilot-decision provenance")
    decision_path = repo_path(decision_value, artifact_output=True)
    decision_record = load_json(decision_path)
    token = decision_record.get("final_budget_token")
    if not isinstance(token, str) or not token:
        raise ProtocolError("immutable decision does not bind a final budget token")
    ledger = _aggregate_budget(config, actual_config_path)
    # Activation is a state transition of the decision-held reservation, not
    # a second 600-second charge.
    activated = ledger.activate_hold(
        token,
        expected_seconds=FINAL_TEST_RESERVE_SECONDS,
        metadata_updates={
            "freeze_path": _repo_relative(run_dir / "freeze.json"),
            "freeze_sha256": freeze_sha256,
            "pilot_decision_path": _repo_relative(decision_path),
            "pilot_decision_sha256": sha256(decision_path),
            "selection": selection,
        },
    )
    started = time.monotonic()
    try:
        result = _run_final_test_worker(
            config, manifest, freeze_sha256,
            actual_config_path=actual_config_path,
            worker_started=started,
            deadline=started + FINAL_TEST_RESERVE_SECONDS,
            final_token=token,
            final_reservation=activated,
            decision_path=decision_path,
            selection=selection,
            frozen=frozen,
            ledger=ledger,
        )
    except BaseException as worker_error:
        _finish_final_budget_if_needed(ledger, token, started, worker_error)
        raise
    # A patched/synthetic worker may return before finalizing; production
    # workers finalize before create-once result publication.
    _finish_final_budget_if_needed(ledger, token, started)
    return result


def _run_final_test_worker(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    freeze_sha256: str,
    *,
    actual_config_path: Path,
    worker_started: float,
    deadline: float,
    final_token: str,
    final_reservation: Mapping[str, Any],
    decision_path: Path,
    selection: Mapping[str, Any],
    frozen: Mapping[str, Any],
    ledger: AggregateBudget,
) -> dict[str, Any]:
    """Run one authenticated, irreversible no-training evaluation."""
    validate_manifest(manifest, allow_pending=False)
    verify_split_files(manifest, include_test=False)
    run_dir = repo_path(config["output_root"], artifact_output=True)

    def deadline_checkpoint(phase: str) -> None:
        if time.monotonic() >= deadline:
            raise ProtocolError(f"final worker deadline exhausted during {phase}; no retry")

    deadline_checkpoint("final validation")
    current_selection, current_frozen = _final_selection(
        run_dir, freeze_sha256, list(config["labels"]), config, manifest, actual_config_path,
    )
    if current_selection != selection or current_frozen != frozen:
        raise ProtocolError("final selection changed after budget activation")
    result_path = run_dir / "final_test.result.json"
    if not isinstance(final_reservation, Mapping) or final_reservation.get("token") != final_token or final_reservation.get("state") != "in_flight":
        raise ProtocolError("final activated reservation provenance is invalid")
    if ledger.reservation_snapshot(final_token) != dict(final_reservation):
        raise ProtocolError("final activated reservation differs from the ledger")
    if result_path.exists():
        raise ProtocolError("final-test result already exists; retry forbidden")
    checkpoint = repo_path(selection["selected_checkpoint_path"], artifact_output=True)
    owner = {
        "command": "final-test",
        "config_sha256": sha256(actual_config_path),
        "manifest_sha256": sha256(repo_path(config["data_manifest"])),
        "seed": int(selection["selected_seed"]),
    }
    started = worker_started
    with GpuLock(repo_path(config["gpu_lock"], artifact_output=True), owner):
        # The final route uses the shared guard too, but is explicitly allowed
        # to proceed while its own decision-held token is in flight.
        require_non_final_worker_open(config, route="final-test")
        deadline_checkpoint("final-test GPU admission")
        from safetensors.torch import load_file
        authenticated_state = load_authenticated_safetensors(
            checkpoint / "model.safetensors",
            selection["selected_weights_file_sha256"],
            load_file,
        )
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        snapshot, _, source_dir = _model_paths(config)
        sys.path.insert(0, str(source_dir))
        from laya.common import QTYPES, build_model, build_sequence, collate_items, render_options

        if not torch.cuda.is_available():
            raise ProtocolError("CUDA is unavailable; final-test has no CPU fallback")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        random.seed(owner["seed"]); np.random.seed(owner["seed"]); torch.manual_seed(owner["seed"]); torch.cuda.manual_seed_all(owner["seed"])
        model_config = json.loads((snapshot / "rl_agent_config.json").read_text(encoding="utf-8"))
        model_config.update(max_len=config["model"]["max_len"], head_max_len=config["model"]["head_max_len"])
        model = build_model(model_config, encoder_dir=str(snapshot / "encoder"))
        model.load_state_dict(authenticated_state, strict=True)
        model.encoder.config.reference_compile = False
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)

        # The token is already activated before this point.  Capture its exact
        # in-flight reservation and ledger hash for the marker.
        in_flight = ledger.reservation_snapshot(final_token)
        if in_flight.get("state") != "in_flight" or in_flight.get("token") != final_token:
            raise ProtocolError("final budget token is not activated in flight")
        ledger_snapshot = ledger.snapshot()
        marker_provenance = {
            "freeze_path": _repo_relative(run_dir / "freeze.json"),
            "freeze_sha256": freeze_sha256,
            "pilot_decision_path": _repo_relative(decision_path),
            "pilot_decision_sha256": sha256(decision_path),
            "config_path": _repo_relative(actual_config_path),
            "config_sha256": sha256(actual_config_path),
            "manifest_path": _repo_relative(repo_path(config["data_manifest"])),
            "manifest_sha256": sha256(repo_path(config["data_manifest"])),
            "code": code_identity(),
            "budget_ledger_path": _repo_relative(ledger.path),
            "budget_ledger_sha256": ledger_snapshot["sha256"],
            "budget_token": final_token,
            "budget_reserved_seconds": in_flight["reserved_seconds"],
            "budget_reservation": in_flight,
            "selection": selection,
            "selected_weights_file_sha256": selection["selected_weights_file_sha256"],
        }
        marker = consume_final_test_attempt(run_dir, freeze_sha256, marker_provenance)
        marker_hash = sha256(marker)
        test_records = _load_split_bytes(manifest, "test", list(config["labels"]))
        deadline_checkpoint("test decode")
        test_items = _record_items(test_records, tokenizer, build_sequence, QTYPES, render_options, config)
        deadline_checkpoint("final model setup")

        def infer(item: Mapping[str, Any]) -> list[float]:
            batch = collate_items([[item]], tokenizer.pad_token_id)
            tensors = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(*tensors)
            return torch.softmax(logits[0, :6].float(), -1).cpu().tolist()

        def infer_record(record: Mapping[str, Any]) -> list[float]:
            item = _record_items([dict(record)], tokenizer, build_sequence, QTYPES, render_options, config)[0]
            return infer(item)

        torch.cuda.synchronize(device)
        latency_samples = []
        for index in range(config["latency"]["warmups"] + config["latency"]["measured_records"]):
            deadline_checkpoint("latency measurement boundary")
            torch.cuda.synchronize(device); latency_start = time.perf_counter()
            infer_record(test_records[0])
            torch.cuda.synchronize(device)
            if index >= config["latency"]["warmups"]:
                latency_samples.append(time.perf_counter() - latency_start)
        rows = []
        for item in test_items:
            deadline_checkpoint("test inference boundary")
            probs = infer(item)
            by_label = {label: float(probs[position]) for position, label in enumerate(item["option_keys"])}
            rows.append({"record_id": item["record_id"], "gold_label": item["gold_label"], "probabilities": {label: by_label[label] for label in config["labels"]}, "metadata": item["metadata"]})
        scored = evaluate_records([{key: row[key] for key in ("record_id", "gold_label", "probabilities")} for row in rows], list(config["labels"]), nll_floor=config["evaluation"]["nll_probability_floor"])
        scored["predictions"] = rows
        scored["task_group_bootstrap"] = _group_bootstrap(rows, list(config["labels"]))
        deadline_checkpoint("final scoring")
        latency = validate_latency_evidence({"batch_size": 1, "warmups_completed": config["latency"]["warmups"], "samples_seconds": latency_samples, "synchronize_before_each": True, "synchronize_after_each": True, "included_phases": ["tokenize_build", "collate", "host_to_device", "forward", "softmax", "cpu_probability_copy"], "measurement_source": "runtime_monotonic_with_cuda_synchronization", "synchronization_events": 2 * (config["latency"]["warmups"] + config["latency"]["measured_records"])}, config)
        elapsed = time.monotonic() - started
        ledger.finish(final_token, elapsed)
        finished = ledger.reservation_snapshot(final_token, require_finished=True)
        final_snapshot = ledger.snapshot()
        result = {
            "schema": "i3-final-test-result-v2",
            "status": "final_test_completed",
            "attempt_marker": _repo_relative(marker),
            "attempt_marker_sha256": marker_hash,
            "selection": selection,
            "freeze_path": _repo_relative(run_dir / "freeze.json"),
            "freeze_sha256": freeze_sha256,
            "pilot_decision_path": _repo_relative(decision_path),
            "pilot_decision_sha256": sha256(decision_path),
            "config_path": _repo_relative(actual_config_path),
            "config_sha256": sha256(actual_config_path),
            "manifest_path": _repo_relative(repo_path(config["data_manifest"])),
            "manifest_sha256": sha256(repo_path(config["data_manifest"])),
            "code": code_identity(),
            "budget_ledger_path": _repo_relative(ledger.path),
            "budget_ledger_sha256": final_snapshot["sha256"],
            "budget_token": final_token,
            "budget_reserved_seconds": finished["reserved_seconds"],
            "budget_reservation": finished,
            "test": scored,
            "latency": latency,
            "elapsed_seconds": elapsed,
            "training": False,
            "test_opened": True,
        }
        create_once_json(
            result_path, result,
            exists_message="final-test result already exists; retry forbidden",
        )
        validate_final_result(
            result, result_path, run_dir / "freeze.json", decision_path,
            config, actual_config_path, repo_path(config["data_manifest"]),
        )
        return result

def run_bounded_smoke(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    review_accepted: bool = False,
    config_path: Path | None = None,
    admission_path: Path | None = None,
) -> dict[str, Any]:
    """Run two synthetic six-logit updates after shared budget admission."""
    if not review_accepted or config["execution_gate"] != "blocked_until_independent_data_verification_and_pre_run_review":
        raise ProtocolError("GPU smoke is fail-closed until independent review opens the execution gate")
    require_non_final_worker_open(config, route="smoke")
    actual_config_path = _actual_config_path(config_path)
    actual_manifest_path = repo_path(config["data_manifest"])
    if admission_path is None:
        admission_path = admit_worker(
            config,
            actual_config_path,
            actual_manifest_path,
            "smoke",
            requested_wall_seconds=float(config["budget"]["smoke_max_gpu_seconds"]),
        )
    else:
        validate_worker_admission(
            admission_path,
            config,
            actual_config_path,
            actual_manifest_path,
            "smoke",
        )
    require_non_final_worker_open(config, route="smoke")
    claim_path = claim_worker(
        admission_path, config, actual_config_path, actual_manifest_path,
        "smoke", pilot_decision_path=None,
    )
    ledger = _aggregate_budget(config, actual_config_path)
    token, grant = ledger.reserve_non_final(
        float(config["budget"]["smoke_max_gpu_seconds"]),
        metadata=_reservation_metadata("smoke", config["budget"]["smoke_seed"], admission_path),
        run_root=repo_path(config["output_root"], artifact_output=True),
    )
    started = time.monotonic()
    deadline = started + grant

    def checkpoint(phase: str) -> None:
        if time.monotonic() >= deadline:
            raise ProtocolError(f"aggregate worker deadline exhausted during {phase}; no retry")

    try:
        validate_manifest(manifest, allow_pending=False)
        verify_split_files(manifest, include_test=False)
        owner = {"command": "smoke", "config_sha256": sha256(actual_config_path), "manifest_sha256": sha256(actual_manifest_path), "seed": config["budget"]["smoke_seed"]}
        with GpuLock(repo_path(config["gpu_lock"], artifact_output=True), owner):
            # Close the consumed-marker race after lock acquisition and before
            # importing Torch or touching model/device state.
            require_non_final_worker_open(config, route="smoke")
            checkpoint("smoke GPU admission")
            import torch
            if not torch.cuda.is_available():
                raise ProtocolError("CUDA is unavailable; smoke does not fall back to CPU")
            torch.cuda.synchronize()
            torch.manual_seed(owner["seed"])
            logits = torch.nn.Parameter(torch.zeros(6, device="cuda", dtype=torch.float32))
            optimizer = torch.optim.SGD([logits], lr=0.01)
            perturbations = torch.tensor([[0.3, -0.1, 0.2, -0.4, 0.1, -0.1], [-0.4, 0.8, -0.6, 0.2, -0.2, 0.2]], device="cuda")
            gold = torch.tensor([0, 1], device="cuda")
            for _ in range(config["budget"]["smoke_updates"]):
                checkpoint("smoke update boundary")
                optimizer.zero_grad(set_to_none=True)
                noisy = logits.unsqueeze(0) + perturbations
                loss_rlcd = -(torch.log_softmax(noisy, dim=-1).gather(1, gold[:, None]).mean())
                loss_ce = torch.nn.functional.cross_entropy(logits.unsqueeze(0), gold[:1])
                (loss_rlcd + loss_ce).backward()
                torch.nn.utils.clip_grad_norm_([logits], 1.0)
                optimizer.step()
            torch.cuda.synchronize()
            checkpoint("smoke completion")
            elapsed = time.monotonic() - started
            peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
    except BaseException as worker_error:
        _finish_failed_budget(ledger, token, started, worker_error)
        raise

    ledger.finish(token, elapsed)
    validate_runtime_budget(elapsed, elapsed, config)
    if not math.isfinite(elapsed) or peak > config["budget"]["max_peak_reserved_gib"] or elapsed > config["budget"]["smoke_max_gpu_seconds"]:
        raise ProtocolError("smoke exceeded its bounded resource contract")
    result = {
        "status": "smoke_measured",
        "seed": owner["seed"],
        "updates": config["budget"]["smoke_updates"],
        "training_seconds": elapsed,
        "cold_load_seconds": 0.0,
        "dev_evaluation_seconds": 0.0,
        "peak_reserved_gib": peak,
        "finite": True,
        "test_opened": False,
    }
    return _public_result(
        _decorate_runtime_result(
            result,
            command="smoke",
            seed=owner["seed"],
            admission_path=admission_path,
            claim_path=claim_path,
            decision_path=None,
            actual_config_path=actual_config_path,
            manifest_path=actual_manifest_path,
            ledger=ledger,
            token=token,
            elapsed_seconds=elapsed,
        ),
        drop_worker_wall=True,
    )


def _model_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    cache = ROOT / "data/cache"
    snapshot = cache / "huggingface/hub/models--convaiinnovations--laya/snapshots" / config["model"]["revision"]
    weights = snapshot / "model.safetensors"
    model_config = snapshot / "rl_agent_config.json"
    if not snapshot.is_dir() or not weights.is_file() or not model_config.is_file():
        raise ProtocolError("pinned local Laya snapshot is unavailable")
    if sha256(weights) != config["model"]["weights_sha256"] or sha256(model_config) != config["model"]["config_sha256"]:
        raise ProtocolError("pinned local Laya files have unexpected SHA-256")
    source = ROOT / "data/laya"
    if not source.is_dir() or not (source / "laya" / "common.py").is_file():
        raise ProtocolError("pinned local Laya source is unavailable")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    if revision != config["model"]["laya_source_revision"] or sha256(source / "laya" / "common.py") != config["model"]["laya_common_sha256"]:
        raise ProtocolError("pinned local Laya source does not match the frozen revision")
    return snapshot, weights, source


def _record_items(records: list[dict[str, Any]], tokenizer: Any, build_sequence: Any, qtypes: Mapping[str, int], render_options: Any, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = []
    for record in records:
        question = record["questions"]["next_action"]
        internal = {"t": question["type"], "ins": question.get("instructions", ""), "crit": question["criteria"]}
        ids, markers = build_sequence(tokenizer, record["state"], internal, config["model"]["max_len"], config["model"]["head_max_len"], truncate_left=config["model"]["truncate_left"])
        if len(markers) != len(render_options(internal)):
            raise ProtocolError(f"question markers exceed head budget for {record['metadata']['id']}")
        keys = list(question["criteria"])
        probabilities = record["gold"]["next_action"]["probabilities"]
        target = [float(probabilities[key]) for key in keys]
        gold = record["gold"]["next_action"]["label"]
        items.append({"ids": ids, "markers": markers, "qtype": qtypes[question["type"]], "target": target,
                      "label": keys.index(gold), "native_label": keys.index(gold), "record_id": record["metadata"]["id"],
                      "gold_label": gold, "metadata": record["metadata"], "question_id": "next_action", "option_keys": keys})
    return items


def _state_digest(torch: Any, state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tree_cpu(torch: Any, value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_cpu(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_cpu(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_cpu(torch, item) for item in value)
    return value


def _nested_equal(torch: Any, left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape):
            return False
        # torch.equal treats signed zero as equal. Compare raw contiguous CPU
        # storage bytes instead, so this is genuinely bit-exact state equality.
        left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return left_bytes == right_bytes
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(_nested_equal(torch, left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_nested_equal(torch, a, b) for a, b in zip(left, right))
    return left == right


def _group_bootstrap(records: list[dict[str, Any]], labels: list[str], replicates: int = 1000, seed: int = 20260920) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["metadata"]["task_group"]].append(row)
    keys = sorted(groups)
    if not keys:
        raise ProtocolError("dev group bootstrap has no task groups")
    values = {metric: [] for metric in ("macro_f1", "accuracy", "nll", "brier")}
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled = [row for index in (rng.randrange(len(keys)) for _ in keys) for row in groups[keys[index]]]
        predictions = [{"record_id": f"{row['record_id']}:{position}", "gold_label": row["gold_label"], "probabilities": row["probabilities"]} for position, row in enumerate(sampled)]
        scored = evaluate_records(predictions, labels)
        for metric in values:
            values[metric].append(scored[metric])
    interval = {}
    for metric, samples in values.items():
        ordered = sorted(samples)
        interval[metric] = {"lower_2.5": ordered[int(0.025 * (replicates - 1))], "upper_97.5": ordered[int(0.975 * (replicates - 1))]}
    return {"bootstrap_unit": "task_group", "replicates": replicates, "seed": seed, "task_groups": len(keys), "intervals": interval}


def _runtime(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    command: str,
    seed: int,
    review_accepted: bool,
    config_path: Path | None = None,
    pilot_updates: int | None = None,
    train_enabled: bool = True,
    reservation_seconds: float | None = None,
    admission_path: Path | None = None,
    pilot_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Run one already-admitted route under the shared all-or-nothing ledger."""
    require_non_final_worker_open(config, route=command)
    if not review_accepted:
        raise ProtocolError("actual-model execution requires explicit independent review authorization")
    actual_config_path = _actual_config_path(config_path)
    actual_manifest_path = repo_path(config["data_manifest"])
    if admission_path is None:
        raise ProtocolError("runtime worker requires a pre-created immutable admission")
    admission = validate_worker_admission(
        admission_path,
        config,
        actual_config_path,
        actual_manifest_path,
        command,
        seed=seed if command in {"actual-model-pilot", "train", "recover-seed"} else None,
        pilot_decision_path=pilot_decision_path,
    )
    reservation = reservation_seconds if reservation_seconds is not None else (
        600.0 if pilot_updates is not None or not train_enabled else 3600.0
    )
    if command == "train":
        expected = admission.get("requested_wall_seconds")
        try:
            exact = math.isfinite(float(reservation)) and abs(float(reservation) - float(expected)) <= BUDGET_EPSILON
        except (TypeError, ValueError):
            exact = False
        if not exact:
            raise ProtocolError("training reservation must equal the immutable per-seed projection")
    # Claim creation is deliberately before constructing the ledger or making
    # any reservation.  A second direct invocation therefore fails before it
    # can consume another budget slice.
    claim_path = claim_worker(
        admission_path, config, actual_config_path, actual_manifest_path,
        command, seed=seed if command in {"actual-model-pilot", "train", "recover-seed"} else None,
        pilot_decision_path=pilot_decision_path,
    )
    ledger = _aggregate_budget(config, actual_config_path)
    # This is an all-or-nothing pre-admission charge.  No model worker starts
    # unless the entire requested wall-time allowance is available.
    run_root = repo_path(config["output_root"], artifact_output=True)
    reserve = (
        ledger.reserve_up_to
        if command in {"train", "recover-seed"}
        else lambda seconds, **kwargs: ledger.reserve_non_final(seconds, run_root=run_root, **kwargs)
    )
    token, grant = reserve(
        reservation,
        metadata=_reservation_metadata(command, seed, admission_path, pilot_decision_path),
    )
    started = time.monotonic()
    try:
        result = _runtime_worker(
            config,
            manifest,
            command=command,
            seed=seed,
            review_accepted=review_accepted,
            config_path=actual_config_path,
            pilot_updates=pilot_updates,
            train_enabled=train_enabled,
            worker_started=started,
            deadline=started + grant,
        )
    except BaseException as worker_error:
        _finish_failed_budget(ledger, token, started, worker_error)
        raise
    elapsed = time.monotonic() - started
    ledger.finish(token, elapsed)
    if not isinstance(result, Mapping):
        raise ProtocolError("runtime worker returned a non-object result")
    value = _decorate_runtime_result(
        result,
        command=command,
        seed=seed,
        admission_path=admission_path,
        claim_path=claim_path,
        decision_path=pilot_decision_path,
        actual_config_path=actual_config_path,
        manifest_path=actual_manifest_path,
        ledger=ledger,
        token=token,
        elapsed_seconds=elapsed,
    )
    value["worker_wall_seconds"] = elapsed
    if command == "train":
        context_path = repo_path(config["output_root"], artifact_output=True) / f"seed_{seed}" / "training_context.json"
        context_result = _public_result(value, drop_worker_wall=True)
        context_payload = {
            "schema": "i3-training-context-v1",
            "seed": seed,
            "admission_path": _repo_relative(admission_path),
            "admission_sha256": sha256(admission_path),
            "pilot_decision_path": _repo_relative(pilot_decision_path) if pilot_decision_path is not None else None,
            "pilot_decision_sha256": sha256(pilot_decision_path) if pilot_decision_path is not None else None,
            "config_path": _repo_relative(actual_config_path),
            "config_sha256": sha256(actual_config_path),
            "manifest_path": _repo_relative(actual_manifest_path),
            "manifest_sha256": sha256(actual_manifest_path),
            "code": code_identity(),
            "budget_token": token,
            "budget_reservation": value["budget_reservation"],
            "result": context_result,
        }
        create_once_json(
            context_path,
            context_payload,
            exists_message="training context already exists; create-once recovery evidence is immutable",
        )
        value["training_context_path"] = _repo_relative(context_path)
        value["training_context_sha256"] = sha256(context_path)
    return value


def _runtime_worker(config: Mapping[str, Any], manifest: Mapping[str, Any], *, command: str, seed: int, review_accepted: bool, config_path: Path, pilot_updates: int | None = None, train_enabled: bool = True, worker_started: float, deadline: float) -> dict[str, Any]:
    if not review_accepted:
        raise ProtocolError("actual-model execution requires explicit independent review authorization")
    validate_manifest(manifest, allow_pending=False)
    verify_split_files(manifest, include_test=False)
    train_records = load_split(manifest, "train", list(config["labels"]))
    dev_records = load_split(manifest, "dev", list(config["labels"]))
    snapshot, weights_path, source_dir = _model_paths(config)
    owner = {"command": command, "config_sha256": sha256(config_path), "manifest_sha256": sha256(repo_path(config["data_manifest"])), "seed": seed}
    started = worker_started
    def deadline_checkpoint(phase: str) -> None:
        if time.monotonic() >= deadline: raise ProtocolError(f"aggregate worker deadline exhausted during {phase}; no retry")
    with GpuLock(repo_path(config["gpu_lock"], artifact_output=True), owner):
        # Recheck after lock acquisition: a final marker can be consumed by a
        # competing process while this worker is waiting for the lock.
        require_non_final_worker_open(config, route=command)
        deadline_checkpoint("GPU admission")
        # All imports below are deliberately after the non-blocking GPU lock.
        import numpy as np
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoTokenizer
        sys.path.insert(0, str(source_dir))
        from laya.common import QTYPES, build_model, build_sequence, collate_items, proper_reward, render_options
        from training.laya_local.b2_train import checked_scaler_step, rlcd_objective

        if not torch.cuda.is_available():
            raise ProtocolError("CUDA is unavailable; actual-model execution has no CPU fallback")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        cold_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
        model_config = json.loads((snapshot / "rl_agent_config.json").read_text(encoding="utf-8"))
        model_config.update(max_len=config["model"]["max_len"], head_max_len=config["model"]["head_max_len"])
        def make_model() -> Any:
            candidate = build_model(model_config, encoder_dir=str(snapshot / "encoder"))
            candidate.load_state_dict(
                load_authenticated_safetensors(
                    weights_path,
                    config["model"]["weights_sha256"],
                    lambda fd_path: load_file(fd_path),
                ),
                strict=True,
            )
            candidate.encoder.config.reference_compile = False
            if config["training"]["gradient_checkpointing"]:
                candidate.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return candidate
        model = make_model()
        model.to(device).eval()
        torch.cuda.synchronize(device)
        cold_load_seconds = time.perf_counter() - cold_start
        train_items = _record_items(train_records, tokenizer, build_sequence, QTYPES, render_options, config)
        dev_items = _record_items(dev_records, tokenizer, build_sequence, QTYPES, render_options, config)
        max_reserved = int(config["budget"]["max_peak_reserved_gib"] * 1024 ** 3)
        torch.cuda.reset_peak_memory_stats(device)

        def forward_item(item: Mapping[str, Any], training: bool = False, sigma: float = 0.4) -> tuple[Any, Any]:
            batch = collate_items([[item]], tokenizer.pad_token_id)
            tensors = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = model(*tensors)
            if training:
                terms = rlcd_objective(torch, logits.float(), act, batch["target"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device), proper_reward, sigma, group_size=config["training"]["group_size"], w_sph=config["training"]["reward_spherical_weight"], w_rps=config["training"]["reward_rps_weight"], log_floor=config["training"]["reward_log_floor"], ce_weight=config["training"]["ce_weight"])
                return terms, batch
            return logits, batch

        def evaluate(items: list[Mapping[str, Any]]) -> dict[str, Any]:
            model.eval(); rows = []
            with torch.no_grad():
                for item in items:
                    logits, batch = forward_item(item)
                    probs = torch.softmax(logits[0, :6].float(), -1).cpu().tolist()
                    probability_by_label = {label: float(probs[index]) for index, label in enumerate(item["option_keys"])}
                    rows.append({"record_id": item["record_id"], "gold_label": item["gold_label"], "probabilities": {label: probability_by_label[label] for label in config["labels"]}, "metadata": item["metadata"]})
            scored = evaluate_records([{key: row[key] for key in ("record_id", "gold_label", "probabilities")} for row in rows], list(config["labels"]), nll_floor=config["evaluation"]["nll_probability_floor"])
            scored["predictions"] = rows
            scored["task_group_bootstrap"] = _group_bootstrap(rows, list(config["labels"]))
            return scored

        deadline_checkpoint("model setup"); base_started = time.perf_counter(); base = evaluate(dev_items); base_seconds = time.perf_counter() - base_started; deadline_checkpoint("pre-training dev evaluation")
        latency_samples = []
        for index in range(config["latency"]["warmups"] + config["latency"]["measured_records"]):
            torch.cuda.synchronize(device); latency_start = time.perf_counter()
            measured_item = _record_items([dev_records[0]], tokenizer, build_sequence, QTYPES, render_options, config)[0]
            with torch.inference_mode():
                measured_logits, _ = forward_item(measured_item)
                torch.softmax(measured_logits[0, :6].float(), -1).cpu().tolist()
            torch.cuda.synchronize(device); duration = time.perf_counter() - latency_start
            if index >= config["latency"]["warmups"]: latency_samples.append(duration)
        latency = validate_latency_evidence({"batch_size": 1, "warmups_completed": config["latency"]["warmups"], "samples_seconds": latency_samples, "synchronize_before_each": True, "synchronize_after_each": True, "included_phases": ["tokenize_build", "collate", "host_to_device", "forward", "softmax", "cpu_probability_copy"], "measurement_source": "runtime_monotonic_with_cuda_synchronization", "synchronization_events": 2 * (config["latency"]["warmups"] + config["latency"]["measured_records"])}, config)
        if not train_enabled:
            return {"status": "base_eval_completed", "seed": seed, "base": base, "dev": base, "cold_load_seconds": cold_load_seconds, "dev_evaluation_seconds": base_seconds, "latency": latency, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3, "test_opened": False}
        if pilot_updates is not None:
            pilot_named = list(model.named_parameters())
            pilot_encoder = [p for name, p in pilot_named if name.startswith("encoder.")]
            pilot_nonencoder = [p for name, p in pilot_named if not name.startswith("encoder.") and not name.startswith("act_head.")]
            for name, parameter in pilot_named:
                if name.startswith("act_head."):
                    parameter.requires_grad_(False)
            optimizer = torch.optim.AdamW([{"params": pilot_encoder, "lr": config["training"]["encoder_learning_rate"]}, {"params": pilot_nonencoder, "lr": config["training"]["nonencoder_learning_rate"]}], weight_decay=config["training"]["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=expected_updates(len(train_items), config), eta_min=config["training"]["scheduler_eta_min"])
            scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=config["training"]["fp16_init_scale"], growth_interval=config["training"]["fp16_growth_interval"])
            pilot_windows = accumulation_windows(list(range(len(train_items))), config["training"]["gradient_accumulation"])
            if len(pilot_windows) < pilot_updates:
                raise ProtocolError("pilot lacks enough records for the declared accumulated updates")
            model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(device)
            pilot_started = time.perf_counter()
            for window in pilot_windows[:pilot_updates]:
                deadline_checkpoint("pilot update boundary")
                for item_index in window["record_indices"]:
                    terms, _ = forward_item(train_items[item_index], True)
                    if not bool(torch.isfinite(terms["loss"]).all()):
                        raise ProtocolError("pilot produced a non-finite loss; no retry")
                    scaler.scale(terms["loss"] / window["divisor"]).backward()
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise ProtocolError("pilot produced a non-finite gradient; no retry")
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["training"]["max_grad_norm"])
                checked_scaler_step(scaler, optimizer); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - pilot_started
            elapsed = time.monotonic() - started
            peak = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            validate_runtime_budget(elapsed, elapsed, config)
            if peak > config["budget"]["max_peak_reserved_gib"] or training_seconds > config["budget"]["smoke_max_gpu_seconds"]:
                raise ProtocolError("actual-model pilot exceeded its bounded resource contract; no retry")
            return {"status": "actual_model_pilot_measured", "seed": seed, "updates": pilot_updates, "cold_load_seconds": cold_load_seconds, "training_seconds": training_seconds, "dev_evaluation_seconds": base_seconds, "peak_reserved_gib": peak, "finite": True, "test_opened": False}

        named = list(model.named_parameters())
        encoder = [p for name, p in named if name.startswith("encoder.")]
        nonencoder = [p for name, p in named if not name.startswith("encoder.") and not name.startswith("act_head.")]
        action = [p for name, p in named if name.startswith("act_head.")]
        for parameter in action: parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW([{"params": encoder, "lr": config["training"]["encoder_learning_rate"]}, {"params": nonencoder, "lr": config["training"]["nonencoder_learning_rate"]}], weight_decay=config["training"]["weight_decay"])
        updates_per_epoch = math.ceil(len(train_items) / config["training"]["gradient_accumulation"])
        total_updates = config["training"]["epochs"] * updates_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=config["training"]["scheduler_eta_min"])
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=config["training"]["fp16_init_scale"], growth_interval=config["training"]["fp16_growth_interval"])
        model.train(); update = 0; microforwards = 0; loss_curve = []; train_start = time.perf_counter()
        for epoch, sigma in enumerate(config["training"]["sigmas_by_epoch"]):
            order = list(range(len(train_items))); random.Random(seed + epoch).shuffle(order); windows = accumulation_windows(order, config["training"]["gradient_accumulation"]); window_by_position = {position: window for window in windows for position in window["positions"]}; optimizer.zero_grad(set_to_none=True); window_loss = 0.0; window_start = 0
            for position, item_index in enumerate(order):
                if position == 0 or position % config["training"]["gradient_accumulation"] == 0: deadline_checkpoint("training accumulation window")
                # The window size is known from position, including a shuffled epoch tail.
                window = window_by_position[position]; divisor = window["divisor"]
                terms, _ = forward_item(train_items[item_index], True, sigma)
                microforwards += 1
                if not bool(torch.isfinite(terms["loss"]).all()): raise ProtocolError("non-finite loss; no retry")
                scaler.scale(terms["loss"] / divisor).backward()
                window_loss += float(terms["loss"].detach())
                if position == window["positions"][-1]:
                    scaler.unscale_(optimizer); gradients = [p.grad for p in encoder + nonencoder]
                    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients): raise ProtocolError("non-finite gradient; no retry")
                    norm = torch.nn.utils.clip_grad_norm_(encoder + nonencoder, config["training"]["max_grad_norm"])
                    if not bool(torch.isfinite(norm)): raise ProtocolError("non-finite gradient norm; no retry")
                    checked_scaler_step(scaler, optimizer); scheduler.step(); optimizer.zero_grad(set_to_none=True); update += 1
                    loss_curve.append({"update": update, "epoch": epoch + 1, "sigma": sigma, "loss": window_loss / divisor, "pre_clip_grad_norm": float(norm), "fp16_scale": float(scaler.get_scale())}); window_loss = 0.0
                    if time.perf_counter() - train_start > config["budget"]["main_max_aggregate_gpu_seconds"]:
                        raise ProtocolError("training exceeded the running GPU-second budget; no retry")
                if torch.cuda.max_memory_reserved(device) > max_reserved: raise ProtocolError("peak CUDA reserved memory exceeded; no retry")
        torch.cuda.synchronize(device); training_seconds = time.perf_counter() - train_start
        if update != total_updates or microforwards != len(train_items) * config["training"]["epochs"]: raise ProtocolError("training stopping state does not match the frozen schedule")
        deadline_checkpoint("checkpoint save/reload and dev evaluation")
        elapsed_after_training = time.monotonic() - started
        validate_runtime_budget(elapsed_after_training, elapsed_after_training, config)
        if training_seconds > config["budget"]["main_max_aggregate_gpu_seconds"]:
            raise ProtocolError("training exceeded the aggregate GPU-second budget; no retry")
        state_cpu = _tree_cpu(torch, model.state_dict()); model_digest = _state_digest(torch, state_cpu)
        run_dir = repo_path(config["output_root"], artifact_output=True) / f"seed_{seed}"; checkpoint = run_dir / "checkpoint"; checkpoint.mkdir(parents=True, exist_ok=False)
        save_file(state_cpu, checkpoint / "model.safetensors")
        training_state = _tree_cpu(torch, {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "updates": update, "microforwards": microforwards, "epochs": config["training"]["epochs"]})
        torch.save(training_state, checkpoint / "training_state.pt")
        post = evaluate(dev_items); deadline_checkpoint("post-training dev evaluation")
        del model, optimizer, scheduler, scaler; torch.cuda.empty_cache()
        reloaded = make_model()
        reloaded_weights_hash = sha256(checkpoint / "model.safetensors")
        reloaded.load_state_dict(
            load_authenticated_safetensors(
                checkpoint / "model.safetensors",
                reloaded_weights_hash,
                lambda fd_path: load_file(fd_path),
            ),
            strict=True,
        )
        after_digest = _state_digest(torch, reloaded.state_dict())
        reloaded.to(device); reload_optimizer = torch.optim.AdamW([{"params": [p for n, p in reloaded.named_parameters() if n.startswith("encoder.")], "lr": config["training"]["encoder_learning_rate"]}, {"params": [p for n, p in reloaded.named_parameters() if not n.startswith("encoder.") and not n.startswith("act_head." )], "lr": config["training"]["nonencoder_learning_rate"]}], weight_decay=config["training"]["weight_decay"])
        reload_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reload_optimizer, T_max=total_updates, eta_min=config["training"]["scheduler_eta_min"]); reload_scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=config["training"]["fp16_init_scale"], growth_interval=config["training"]["fp16_growth_interval"])
        loaded = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        reload_optimizer.load_state_dict(loaded["optimizer"]); reload_scheduler.load_state_dict(loaded["scheduler"]); reload_scaler.load_state_dict(loaded["scaler"])
        # Compare the *restored live objects* with the pre-save CPU snapshot.
        # Merely round-tripping the serialized dict would not prove that each
        # component's load_state_dict restored the intended state.
        exact = (
            _nested_equal(torch, training_state, loaded)
            and _nested_equal(torch, training_state["optimizer"], reload_optimizer.state_dict())
            and _nested_equal(torch, training_state["scheduler"], reload_scheduler.state_dict())
            and _nested_equal(torch, training_state["scaler"], reload_scaler.state_dict())
            and model_digest == after_digest
            and loaded["updates"] == update
            and loaded["microforwards"] == microforwards
        )
        if not exact: raise ProtocolError("checkpoint strict reload or training-state equality failed")
        if torch.cuda.max_memory_reserved(device) > max_reserved:
            raise ProtocolError("peak CUDA reserved memory exceeded after reload; no retry")
        reload_evidence = {"strict_model_load": True, "model_state_before_sha256": model_digest, "model_state_after_sha256": after_digest, "training_state_file_sha256": sha256(checkpoint / "training_state.pt"), "optimizer_exact": True, "scheduler_exact": True, "scaler_exact": True, "stopping_state_exact": True}
        evidence_path = checkpoint / "reload.json"; evidence_path.write_text(json.dumps(reload_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run_dir.mkdir(parents=True, exist_ok=True) if not run_dir.exists() else None
        deadline_checkpoint("checkpoint reload verification")
        return {"status": "completed", "seed": seed, "base": base, "dev": post, "cold_load_seconds": cold_load_seconds, "training_seconds": training_seconds, "dev_evaluation_seconds": base_seconds, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3, "updates": update, "microforwards": microforwards, "checkpoint": {"checkpoint_path": str(checkpoint.relative_to(ROOT)), "weights_file_sha256": sha256(checkpoint / "model.safetensors"), "model_state_sha256": model_digest, "training_state_file_sha256": sha256(checkpoint / "training_state.pt"), "reload_evidence": reload_evidence, "reload_evidence_path": str(evidence_path.relative_to(ROOT)), "reload_evidence_sha256": sha256(evidence_path)}, "loss_curve": loss_curve, "test_opened": False}


def run_actual_model_pilot(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    review_accepted: bool = False,
    config_path: Path | None = None,
    admission_path: Path | None = None,
) -> dict[str, Any]:
    require_non_final_worker_open(config, route="actual-model-pilot")
    if not review_accepted:
        raise ProtocolError("actual-model execution requires explicit independent review authorization")
    actual_config_path = _actual_config_path(config_path)
    actual_manifest_path = repo_path(config["data_manifest"])
    if admission_path is None:
        admission_path = admit_worker(
            config,
            actual_config_path,
            actual_manifest_path,
            "actual-model-pilot",
            seed=config["budget"]["smoke_seed"],
            requested_wall_seconds=float(config["budget"]["smoke_max_gpu_seconds"]),
        )
    return _public_result(
        _runtime(
            config,
            manifest,
            command="actual-model-pilot",
            seed=config["budget"]["smoke_seed"],
            review_accepted=review_accepted,
            config_path=actual_config_path,
            pilot_updates=2,
            admission_path=admission_path,
        ),
        drop_worker_wall=True,
    )


def run_base_eval(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    review_accepted: bool = False,
    config_path: Path | None = None,
    admission_path: Path | None = None,
) -> dict[str, Any]:
    require_non_final_worker_open(config, route="base-eval")
    if not review_accepted:
        raise ProtocolError("actual-model execution requires explicit independent review authorization")
    actual_config_path = _actual_config_path(config_path)
    actual_manifest_path = repo_path(config["data_manifest"])
    if admission_path is None:
        admission_path = admit_worker(
            config,
            actual_config_path,
            actual_manifest_path,
            "base-eval",
            requested_wall_seconds=float(config["budget"]["smoke_max_gpu_seconds"]),
        )
    return _public_result(
        _runtime(
            config,
            manifest,
            command="base-eval",
            seed=config["budget"]["smoke_seed"],
            review_accepted=review_accepted,
            config_path=actual_config_path,
            pilot_updates=None,
            train_enabled=False,
            admission_path=admission_path,
        )
    )


def run_training_seed(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seed: int,
    *,
    review_accepted: bool = False,
    reservation_seconds: float | None = None,
    config_path: Path | None = None,
    admission_path: Path | None = None,
    pilot_decision_path: Path | None = None,
) -> dict[str, Any]:
    require_non_final_worker_open(config, route="train")
    if not review_accepted:
        raise ProtocolError("actual-model execution requires explicit independent review authorization")
    actual_config_path = _actual_config_path(config_path)
    actual_manifest_path = repo_path(config["data_manifest"])
    _, default_decision_path = future_run_paths(config)
    decision_path = pilot_decision_path or default_decision_path
    decision = validate_pilot_decision(
        future_run_paths(config)[0],
        decision_path,
        actual_config_path,
        actual_manifest_path,
        config,
        manifest,
    )
    if seed not in decision["selected_seeds"]:
        raise ProtocolError("seed is not authorized by the immutable pilot decision")
    expected_reservation = float(decision["projected_gpu_seconds_per_seed"])
    requested_reservation = expected_reservation
    if reservation_seconds is not None:
        if isinstance(reservation_seconds, bool):
            raise ProtocolError("training reservation is not finite")
        try:
            requested_reservation = float(reservation_seconds)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("training reservation is not finite") from exc
        if not math.isfinite(requested_reservation):
            raise ProtocolError("training reservation is not finite")
        if abs(requested_reservation - expected_reservation) > BUDGET_EPSILON:
            raise ProtocolError("training reservation must equal the immutable pilot projection")
    if admission_path is None:
        admission_path = admit_worker(
            config,
            actual_config_path,
            actual_manifest_path,
            "train",
            seed=seed,
            pilot_decision_path=decision_path,
            requested_wall_seconds=requested_reservation,
        )
    else:
        existing_admission = validate_worker_admission(
            admission_path, config, actual_config_path, actual_manifest_path,
            "train", seed=seed, pilot_decision_path=decision_path,
        )
        if not budget_close(existing_admission.get("requested_wall_seconds"), expected_reservation):
            raise ProtocolError("training admission allowance differs from the immutable pilot projection")
    return _public_result(
        _runtime(
            config,
            manifest,
            command="train",
            seed=seed,
            review_accepted=review_accepted,
            config_path=actual_config_path,
            pilot_updates=None,
            reservation_seconds=requested_reservation,
            admission_path=admission_path,
            pilot_decision_path=decision_path,
        )
    )


def _find_finished_seed_reservation(
    ledger: AggregateBudget,
    *,
    seed: int,
    decision_path: Path,
    training_admission_path: Path,
    training_token: str,
) -> tuple[str, dict[str, Any]]:
    """Find only the exact finished reservation named by the training chain."""
    snapshot = ledger.snapshot()
    expected_metadata = {
        "command": "train",
        "seed": seed,
        "admission_sha256": sha256(training_admission_path),
        "pilot_decision_sha256": sha256(decision_path),
    }
    for item in snapshot["reservations"]:
        if item.get("token") != training_token or item.get("state") != "finished":
            continue
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping) and set(metadata) == set(expected_metadata) and all(metadata.get(key) == value for key, value in expected_metadata.items()):
            return str(item["token"]), dict(item)
    raise ProtocolError("the genuine finished training reservation named by the admission is missing")


def _run_recover_model_worker(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seed: int,
    checkpoint: Path,
    checkpoint_record: Mapping[str, Any],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    reload_evidence: Mapping[str, Any],
    expected_updates: int,
    expected_microforwards: int,
    training_admission_path: Path,
    training_token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the real recovery evaluator after the training chain is validated."""
    import torch
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    checkpoint_hashes = validate_recovery_checkpoint(
        checkpoint,
        weights_sha256=checkpoint_record["weights_file_sha256"],
        training_state_sha256=checkpoint_record["training_state_file_sha256"],
        reload_sha256=checkpoint_record["reload_evidence_sha256"],
        model_state_sha256=checkpoint_record["model_state_sha256"],
        expected_updates=expected_updates,
        expected_microforwards=expected_microforwards,
        expected_epochs=config["training"]["epochs"],
        state=state,
        reload_evidence=reload_evidence,
    )
    snapshot, weights_path, source_dir = _model_paths(config)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    sys.path.insert(0, str(source_dir))
    from laya.common import QTYPES, build_model, build_sequence, collate_items, render_options

    model_config = json.loads((snapshot / "rl_agent_config.json").read_text(encoding="utf-8"))
    model_config.update(max_len=config["model"]["max_len"], head_max_len=config["model"]["head_max_len"])
    model = build_model(model_config, encoder_dir=str(snapshot / "encoder"))
    authenticated_state = load_authenticated_safetensors(
        checkpoint / "model.safetensors",
        checkpoint_record["weights_file_sha256"],
        lambda fd_path: load_file(fd_path, device="cpu"),
    )
    model.load_state_dict(authenticated_state, strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    dev_items = _record_items(dev_records, tokenizer, build_sequence, QTYPES, render_options, config)
    rows = []
    with torch.no_grad():
        for item in dev_items:
            batch = collate_items([[item]], tokenizer.pad_token_id)
            tensors = [batch[key] for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            logits, _ = model(*tensors)
            probs = torch.softmax(logits[0, :6].float(), -1).cpu().tolist()
            by_label = {label: float(probs[position]) for position, label in enumerate(item["option_keys"])}
            rows.append({"record_id": item["record_id"], "gold_label": item["gold_label"], "probabilities": {label: by_label[label] for label in config["labels"]}, "metadata": item["metadata"]})
    dev = evaluate_records([{key: row[key] for key in ("record_id", "gold_label", "probabilities")} for row in rows], list(config["labels"]), nll_floor=config["evaluation"]["nll_probability_floor"])
    dev["predictions"] = rows
    dev["task_group_bootstrap"] = _group_bootstrap(rows, list(config["labels"]))
    result = {
        "status": "recovered",
        "recovery": {
            "authorized": True,
            "source": "genuine_training_checkpoint_in_current_run",
            "retraining": False,
            "test_opened": False,
            "training_admission_path": _repo_relative(training_admission_path),
            "training_admission_sha256": sha256(training_admission_path),
            "training_budget_token": training_token,
        },
        "seed": seed,
        "dev": dev,
        "checkpoint": {
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            **checkpoint_hashes,
            "model_state_sha256": checkpoint_record["model_state_sha256"],
            "reload_evidence": reload_evidence,
            "reload_evidence_path": str((checkpoint / "reload.json").relative_to(ROOT)),
            "reload_evidence_sha256": checkpoint_hashes["reload_evidence_sha256"],
        },
        "cold_load_seconds": "unknown",
        "training_seconds": "unknown",
        "dev_evaluation_seconds": "unknown",
        "latency": "unknown",
        "loss_curve": "unknown",
        "updates": expected_updates,
        "microforwards": expected_microforwards,
        "test_opened": False,
    }

    return checkpoint_hashes, result


def run_recover_seed(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seed: int,
    *,
    review_accepted: bool = False,
    config_path: Path | None = None,
    admission_path: Path | None = None,
    pilot_decision_path: Path | None = None,
    budget_token: str | None = None,
) -> dict[str, Any]:
    """Recover only a decision-authorized fallback from a genuine train chain."""
    require_non_final_worker_open(config, route="recover-seed")
    if not review_accepted:
        raise ProtocolError("recover-seed requires explicit independent review authorization")
    if seed != config["training"]["single_seed_fallback"]:
        raise ProtocolError("recover-seed is authorized only for seed 42")
    actual_config_path = _actual_config_path(config_path)
    actual_manifest_path = repo_path(config["data_manifest"])
    pilot_path, default_decision_path = future_run_paths(config)
    decision_path = pilot_decision_path or default_decision_path
    decision = validate_pilot_decision(
        pilot_path, decision_path, actual_config_path, actual_manifest_path, config, manifest,
    )
    if not decision.get("fallback_applied") or decision.get("selected_seeds") != [seed]:
        raise ProtocolError("recovery requires the immutable one-seed fallback decision for seed 42")
    if admission_path is None:
        admission_path = admit_worker(
            config, actual_config_path, actual_manifest_path, "recover-seed",
            seed=seed, pilot_decision_path=decision_path,
        )
    recovery_chain = validate_recovery_admission(
        admission_path, config, actual_config_path, actual_manifest_path, seed, decision_path,
    )
    if budget_token is not None and budget_token != recovery_chain["training_budget_token"]:
        raise ProtocolError("requested recovery token differs from the genuine training reservation")
    claim_path = claim_worker(
        admission_path, config, actual_config_path, actual_manifest_path,
        "recover-seed", seed=seed, pilot_decision_path=decision_path,
    )
    ledger = _aggregate_budget(config, actual_config_path)
    training_admission_path = recovery_chain["training_admission_path"]
    training_token = recovery_chain["training_budget_token"]
    token, training_reservation = _find_finished_seed_reservation(
        ledger, seed=seed, decision_path=decision_path,
        training_admission_path=training_admission_path, training_token=training_token,
    )
    validate_manifest(manifest, allow_pending=False)
    verify_split_files(manifest, include_test=False)
    train_records = load_split(manifest, "train", list(config["labels"]))
    dev_records = load_split(manifest, "dev", list(config["labels"]))
    run_dir = repo_path(config["output_root"], artifact_output=True) / f"seed_{seed}"
    result_path = run_dir / "result.json"
    checkpoint = run_dir / "checkpoint"
    checkpoint_record = recovery_chain["checkpoint"]
    reload_evidence = load_json(checkpoint / "reload.json")
    expected_updates = config["training"]["epochs"] * math.ceil(len(train_records) / config["training"]["gradient_accumulation"])
    expected_microforwards = len(train_records) * config["training"]["epochs"]
    owner = {"command": "recover-seed", "config_sha256": sha256(actual_config_path), "manifest_sha256": sha256(actual_manifest_path), "seed": seed}
    with GpuLock(repo_path(config["gpu_lock"], artifact_output=True), owner):
        require_non_final_worker_open(config, route="recover-seed")
        if result_path.exists():
            raise ProtocolError("seed recovery result already exists; overwrite is forbidden")
        checkpoint_hashes, result = _run_recover_model_worker(
            config=config,
            manifest=manifest,
            seed=seed,
            checkpoint=checkpoint,
            checkpoint_record=checkpoint_record,
            train_records=train_records,
            dev_records=dev_records,
            reload_evidence=reload_evidence,
            expected_updates=expected_updates,
            expected_microforwards=expected_microforwards,
            training_admission_path=training_admission_path,
            training_token=training_token,
        )
        value = _decorate_runtime_result(
            result, command="recover-seed", seed=seed,
            admission_path=admission_path, claim_path=claim_path,
            decision_path=decision_path, actual_config_path=actual_config_path,
            manifest_path=actual_manifest_path, ledger=ledger, token=token,
        )
        value["schema"] = "i3-train-result-v2"
        value["training_admission_path"] = _repo_relative(training_admission_path)
        value["training_admission_sha256"] = sha256(training_admission_path)
        value["training_budget_token"] = training_token
        value["training_context_path"] = _repo_relative(recovery_chain["training_context_path"])
        value["training_context_sha256"] = sha256(recovery_chain["training_context_path"])
        value["worker_wall_seconds"] = 0.0
        _exclusive_json(result_path, _public_result(value, drop_worker_wall=True))
    return _public_result(value, drop_worker_wall=True)

def planned_run(config: Mapping[str, Any], manifest: Mapping[str, Any], command: str) -> dict[str, Any]:
    """Return a reproducible, model-free plan and assert the GPU gate."""
    if config["execution_gate"] != "blocked_until_independent_data_verification_and_pre_run_review":
        raise ProtocolError("execution gate is not the frozen block")
    validate_manifest(manifest, allow_pending=False)
    verify_split_files(manifest, include_test=False)
    train = load_split(manifest, "train", list(config["labels"]))
    dev = load_split(manifest, "dev", list(config["labels"]))
    return {
        "status": "blocked_before_model_or_cuda",
        "command": command,
        "execution_gate": config["execution_gate"],
        "torch_imported": "torch" in __import__("sys").modules,
        "cuda_initialized": False,
        "model_loaded": False,
        "test_opened": False,
        "train_rows_verified": len(train),
        "dev_rows_verified": len(dev),
        "expected_updates_per_seed": config["training"]["epochs"] * ((len(train) + config["training"]["gradient_accumulation"] - 1) // config["training"]["gradient_accumulation"]),
        "seeds": config["training"]["default_seeds"],
        "gpu_block_reason": "independent data verification and pre-run review are not both accepted",
    }


def write_plan(path: Path, report: Mapping[str, Any]) -> None:
    root = repo_path(str(path), artifact_output=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def smoke_guard(smoke: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Finite-value guard used by the future measured smoke result."""
    for key in ("training_seconds", "cold_load_seconds", "dev_evaluation_seconds", "peak_reserved_gib"):
        finite(smoke[key], key)
    if smoke.get("finite") is not True:
        raise ProtocolError("smoke is not finite")


__all__ = ["GpuLock", "accumulation_windows", "consume_final_test_attempt", "final_test_once", "finite", "load_split", "load_authenticated_safetensors", "planned_run", "run_bounded_smoke", "run_final_test", "run_recover_seed", "six_class_rlcd_reference", "smoke_guard", "write_plan"]
