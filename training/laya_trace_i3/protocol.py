"""Pre-model validation and lifecycle guards for the Iteration 3 experiment."""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
OWNED_ARTIFACT_ROOT = ROOT / "artifacts/experiment_i3/preflight"
HISTORICAL_RUN_ROOT = (ROOT / "artifacts/experiment_i3/preflight/runs").resolve()
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FIXED_LABELS = ["read", "search", "edit", "execute", "other_tool", "respond_or_finish"]
CHOICE_OPTION_ORDER = ["edit", "execute", "other_tool", "read", "respond_or_finish", "search"]
LABEL_TO_CHOICE_INDEX = {label: CHOICE_OPTION_ORDER.index(label) for label in FIXED_LABELS}
FROZEN_ITERATION_SECONDS = 14400
FROZEN_GPU_SECONDS = 7200
FINAL_TEST_RESERVE_SECONDS = 600.0
# Every budget boundary check uses this one tolerance.  Keeping the tolerance
# in the protocol layer prevents a decision, a ledger, and a worker admission
# from disagreeing by a different last-bit epsilon.
BUDGET_EPSILON = 1e-9
PILOT_SCHEMA = "i3-actual-model-pilot-v2"
PILOT_DECISION_SCHEMA = "i3-future-pilot-decision-v2"
TRAIN_RESULT_SCHEMA = "i3-train-result-v2"
TRAINING_CONTEXT_SCHEMA = "i3-training-context-v1"
FREEZE_SCHEMA = "i3-dev-freeze-v2"
CLAIM_SCHEMA = "i3-execution-claim-v1"
FINAL_ATTEMPT_SCHEMA = "i3-final-test-attempt-v2"
FINAL_RESULT_SCHEMA = "i3-final-test-result-v2"
# Keep the worker-admission schema name stable; the pilot/decision/result
# schemas are the provenance-versioned artifacts.
WORKER_ADMISSION_SCHEMA = "i3-worker-admission-v1"
FINAL_HOLD_COMMAND = "final-test"
FROZEN_LATENCY_PHASES = ["tokenize_build", "collate", "host_to_device", "forward", "softmax", "cpu_probability_copy"]


class ProtocolError(ValueError):
    """Configuration or manifest violates the predeclared protocol."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """Make a create/rename in *path* durable on filesystems supporting fsync."""
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def create_once_json(path: Path, value: Mapping[str, Any], *, exists_message: str) -> None:
    """Atomically publish and fsync one immutable JSON artifact and its directory."""
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProtocolError(exists_message) from exc
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)


def repo_path(value: str, *, artifact_output: bool = False) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ProtocolError("paths must be non-empty repository-relative strings")
    resolved = (ROOT / value).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ProtocolError(f"path escapes repository: {value}")
    if artifact_output and resolved != OWNED_ARTIFACT_ROOT and OWNED_ARTIFACT_ROOT not in resolved.parents:
        raise ProtocolError(f"output path escapes owned artifact root: {value}")
    return resolved


def _ledger_has_final_hold(run_dir: Path) -> bool:
    """Return whether a finalized plan has already claimed final headroom.

    The ledger is deliberately inspected conservatively.  A malformed ledger
    is not treated as an open budget: callers fail closed rather than allowing
    a route to race a partially written hold.
    """
    return _final_hold_state(run_dir) == "held"


def _final_hold_state(run_dir: Path) -> str | None:
    ledger = run_dir / "gpu_budget.json"
    if not ledger.exists():
        return None
    try:
        record = load_json(ledger)
        reservations = record.get("reservations")
        if not isinstance(reservations, list):
            raise ProtocolError("malformed aggregate budget ledger; admission denied")
        states = [
            item.get("state") for item in reservations
            if isinstance(item, Mapping)
            and isinstance(item.get("metadata", {}), Mapping)
            and item.get("metadata", {}).get("command") == FINAL_HOLD_COMMAND
        ]
        if not states:
            return None
        if len(set(states)) != 1:
            raise ProtocolError("multiple final budget holds are ambiguous; admission denied")
        return str(states[0])
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ProtocolError("malformed aggregate budget ledger; admission denied") from exc


def require_non_final_worker_open(
    config: Mapping[str, Any],
    *,
    route: str | None = None,
    path_resolver: Any = None,
) -> Path:
    """Apply cumulative worker guards before admission, budget, or model work.

    Once a pilot decision exists (or its final allowance has been held), only
    the seed-training and recovery routes may proceed.  The default route is
    intentionally restrictive so a direct caller cannot consume the reserved
    training headroom by omitting a command name.  ``validation`` is an
    internal read-only route used while reopening the immutable decision and
    pilot paths.
    """
    resolver = path_resolver or repo_path
    run_dir = resolver(config["output_root"], artifact_output=True)
    if run_dir == HISTORICAL_RUN_ROOT or HISTORICAL_RUN_ROOT in run_dir.parents:
        raise ProtocolError(
            "consumed historical experiment root is permanently blocked; "
            "a separate future run is independent and not a continuation"
        )
    if (run_dir / "final_test.attempt.json").exists():
        raise ProtocolError("Iteration 3 final test was consumed; additional model compute is forbidden")
    decision_exists = (run_dir / "pilot_decision.json").exists()
    hold_state = _final_hold_state(run_dir)
    hold_exists = hold_state is not None
    if (decision_exists or hold_exists) and route not in {"train", "recover-seed", "validation", "final-test"}:
        raise ProtocolError(
            "an immutable pilot decision/final budget hold already exists; "
            "only authorized training or recovery may consume this run's headroom"
        )
    if route in {"train", "recover-seed"} and hold_state not in {None, "held"}:
        raise ProtocolError("final evaluation has already activated or finalized its held budget; no further training/recovery")
    return run_dir


# Descriptive aliases keep the shared lifecycle primitive easy to discover
# without duplicating its implementation in callers.
require_worker_open = require_non_final_worker_open
require_compute_open = require_non_final_worker_open


def future_run_paths(
    config: Mapping[str, Any],
    *,
    path_resolver: Any = None,
) -> tuple[Path, Path]:
    """Return the two provenance-bound pilot paths for a future run."""
    root = require_non_final_worker_open(config, route="validation", path_resolver=path_resolver)
    return root / "future_actual_model_pilot.json", root / "pilot_decision.json"


def _admission_layout(config: Mapping[str, Any], command: str, seed: int | None) -> tuple[Path, tuple[Path, ...]]:
    root = require_non_final_worker_open(config, route=command)
    if command == "actual-model-pilot":
        return root / "future_actual_model_pilot.admission.json", (root / "future_actual_model_pilot.json",)
    if command == "base-eval":
        return root / "base_eval.admission.json", (root / "base_eval.result.json",)
    if command == "smoke":
        return root / "smoke.admission.json", (root / "smoke.result.json",)
    if command == "train":
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ProtocolError("seed admission requires an integer seed")
        run_dir = root / f"seed_{seed}"
        return run_dir / "admission.json", (run_dir / "result.json", run_dir / "checkpoint")
    if command == "recover-seed":
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ProtocolError("seed admission requires an integer seed")
        run_dir = root / f"seed_{seed}"
        return run_dir / "recovery_admission.json", (run_dir / "result.json",)
    raise ProtocolError(f"unsupported worker admission command: {command}")


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ProtocolError(f"path escapes repository: {path}")
    return str(resolved.relative_to(ROOT))


def _actual_repo_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ProtocolError(f"{label} path escapes repository")
    if not resolved.is_file():
        raise ProtocolError(f"{label} path is absent")
    return resolved


def budget_identity(
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Return the complete, current identity used by every shared ledger."""
    actual_config = _actual_repo_file(config_path.resolve(), "config")
    actual_manifest = _actual_repo_file(
        (manifest_path or repo_path(config["data_manifest"])).resolve(), "manifest"
    )
    code = code_identity()
    config_hash = sha256(actual_config)
    manifest_hash = sha256(actual_manifest)
    root = repo_path(config["output_root"], artifact_output=True)
    return {
        "run_root": _repo_relative(root),
        "config_path": _repo_relative(actual_config),
        "config_sha256": config_hash,
        "manifest_path": _repo_relative(actual_manifest),
        "manifest_sha256": manifest_hash,
        "declared_seeds": list(config["training"]["default_seeds"]),
        "code": code,
        # Retain the short aliases used by the original ledger record while
        # making the complete path/hash/code identity authoritative.
        "config": config_hash,
        "manifest": manifest_hash,
        "sources": code,
    }


def _validate_admission_record(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    command: str,
    seed: int | None,
    decision_path: Path | None,
) -> None:
    if not isinstance(record, Mapping) or record.get("schema") not in {WORKER_ADMISSION_SCHEMA, "i3-worker-admission-v2"}:
        raise ProtocolError("worker admission schema is invalid")
    required = {
        "command", "state", "run_root", "run_classification",
        "continues_consumed_historical_experiment", "config_path",
        "config_sha256", "manifest_path", "manifest_sha256", "code",
        "budget_ledger_path", "retry_policy",
    }
    if not required <= set(record):
        raise ProtocolError("worker admission provenance is incomplete")
    if record["command"] != command or record["state"] != "admitted_before_worker":
        raise ProtocolError("worker admission command/state does not match")
    if seed is not None and record.get("seed") != seed:
        raise ProtocolError("worker admission seed does not match")
    if seed is None and "seed" in record:
        raise ProtocolError("non-seed worker admission unexpectedly contains a seed")
    root = _repo_relative(repo_path(config["output_root"], artifact_output=True))
    if record["run_root"] != root or record["run_classification"] != "independent_future_run" or record["continues_consumed_historical_experiment"] is not False:
        raise ProtocolError("worker admission run classification is invalid")
    actual_config = _actual_repo_file(config_path.resolve(), "config")
    actual_manifest = _actual_repo_file(manifest_path.resolve(), "manifest")
    if record["config_path"] != _repo_relative(actual_config) or record["config_sha256"] != sha256(actual_config):
        raise ProtocolError("worker admission config provenance is stale or foreign")
    if record["manifest_path"] != _repo_relative(actual_manifest) or record["manifest_sha256"] != sha256(actual_manifest):
        raise ProtocolError("worker admission manifest provenance is stale or foreign")
    _validate_code_identity(record["code"])
    expected_ledger = _repo_relative(repo_path(config["output_root"], artifact_output=True) / "gpu_budget.json")
    if record["budget_ledger_path"] != expected_ledger:
        raise ProtocolError("worker admission budget ledger path is invalid")
    if command in {"actual-model-pilot", "base-eval", "smoke"}:
        if record.get("requested_wall_seconds") != float(config["budget"]["smoke_max_gpu_seconds"]):
            raise ProtocolError("fixed-route worker admission allowance is invalid")
    if command == "train":
        requested = record.get("requested_wall_seconds")
        if isinstance(requested, bool) or not isinstance(requested, (int, float)) or not math.isfinite(requested) or requested <= 0:
            raise ProtocolError("training admission allowance is invalid")
        if decision_path is not None:
            decision = load_json(decision_path)
            projection = decision.get("decision", {}).get("projected_gpu_seconds_per_seed")
            if not budget_close(requested, projection):
                raise ProtocolError("training admission allowance differs from immutable pilot projection")
    if command == "recover-seed":
        recovery_fields = {
            "training_admission_path", "training_admission_sha256", "training_budget_token",
            "training_context_path", "training_context_sha256",
        }
        if not recovery_fields <= set(record):
            raise ProtocolError("recovery admission training-chain provenance is incomplete")
    if decision_path is None:
        if "pilot_decision_path" in record or "pilot_decision_sha256" in record:
            raise ProtocolError("non-training worker admission unexpectedly binds a pilot decision")
    else:
        actual_decision = _actual_repo_file(decision_path.resolve(), "pilot decision")
        if record.get("pilot_decision_path") != _repo_relative(actual_decision) or record.get("pilot_decision_sha256") != sha256(actual_decision):
            raise ProtocolError("worker admission pilot-decision provenance is stale or foreign")


def _training_chain_for_recovery(
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    seed: int,
    decision_path: Path,
) -> dict[str, Any]:
    """Validate the complete, genuine training chain needed by recovery.

    Recovery is intentionally not allowed to discover a token by scanning the
    ledger.  It must start from the create-once training admission, then bind
    that admission to its claim, context, checkpoint bytes, and the one exact
    finished training reservation.
    """
    root = repo_path(config["output_root"], artifact_output=True)
    run_dir = root / f"seed_{seed}"
    training_admission_path = run_dir / "admission.json"
    training_admission = validate_worker_admission(
        training_admission_path, config, config_path, manifest_path, "train",
        seed=seed, pilot_decision_path=decision_path,
    )
    training_claim_path = claim_path_for_admission(training_admission_path)
    validate_execution_claim(
        training_claim_path, training_admission_path, config, config_path,
        manifest_path, "train", seed=seed, pilot_decision_path=decision_path,
    )
    context_path = run_dir / "training_context.json"
    if not context_path.is_file():
        raise ProtocolError("training context is missing; recovery cannot fabricate a result context")
    context = load_json(context_path)
    context_fields = {
        "schema", "seed", "admission_path", "admission_sha256",
        "pilot_decision_path", "pilot_decision_sha256", "config_path",
        "config_sha256", "manifest_path", "manifest_sha256", "code",
        "budget_token", "budget_reservation", "result",
    }
    if set(context) != context_fields:
        raise ProtocolError("training context fields are incomplete or unknown")
    if context.get("schema") != TRAINING_CONTEXT_SCHEMA or context.get("seed") != seed:
        raise ProtocolError("training context schema/seed is invalid")
    if context.get("admission_path") != _repo_relative(training_admission_path) or context.get("admission_sha256") != sha256(training_admission_path):
        raise ProtocolError("training context admission binding differs")
    if context.get("pilot_decision_path") != _repo_relative(decision_path) or context.get("pilot_decision_sha256") != sha256(decision_path):
        raise ProtocolError("training context decision binding differs")
    if context.get("config_path") != _repo_relative(config_path) or context.get("config_sha256") != sha256(config_path):
        raise ProtocolError("training context config provenance differs")
    if context.get("manifest_path") != _repo_relative(manifest_path) or context.get("manifest_sha256") != sha256(manifest_path):
        raise ProtocolError("training context manifest provenance differs")
    _validate_code_identity(context.get("code"))
    token = context.get("budget_token")
    if not isinstance(token, str) or not token:
        raise ProtocolError("training context budget token is missing")
    result = context.get("result")
    if not isinstance(result, Mapping) or context.get("budget_reservation") != result.get("budget_reservation"):
        raise ProtocolError("training context budget reservation differs from result")
    if result.get("status") != "completed" or result.get("seed") != seed:
        raise ProtocolError("training context does not contain a completed seed result")
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ProtocolError("training context checkpoint provenance is missing")
    expected_checkpoint = (run_dir / "checkpoint").resolve()
    checkpoint_path = repo_path(checkpoint.get("checkpoint_path", ""), artifact_output=True)
    if checkpoint_path != expected_checkpoint:
        raise ProtocolError("training context checkpoint is foreign")
    weights = checkpoint_path / "model.safetensors"
    state = checkpoint_path / "training_state.pt"
    evidence_path = checkpoint_path / "reload.json"
    if not weights.is_file() or not state.is_file() or not evidence_path.is_file():
        raise ProtocolError("training context checkpoint bytes are incomplete")
    if checkpoint.get("weights_file_sha256") != sha256(weights) or checkpoint.get("training_state_file_sha256") != sha256(state):
        raise ProtocolError("training context checkpoint bytes are hash-mismatched")
    if checkpoint.get("reload_evidence_path") != _repo_relative(evidence_path) or checkpoint.get("reload_evidence_sha256") != sha256(evidence_path):
        raise ProtocolError("training context reload evidence provenance differs")
    evidence = load_json(evidence_path)
    if evidence != checkpoint.get("reload_evidence"):
        raise ProtocolError("training context reload evidence differs")
    validate_checkpoint_reload(evidence, {
        "model_state_sha256": checkpoint.get("model_state_sha256"),
        "training_state_file_sha256": checkpoint.get("training_state_file_sha256"),
    })
    ledger = AggregateBudget(
        _ledger_path(config), budget_identity(config, config_path, manifest_path),
        float(config["budget"]["main_max_aggregate_gpu_seconds"]),
    )
    snapshot = ledger.snapshot()
    reservation = _validate_finished_reservation(
        snapshot, token, result.get("budget_reservation"),
        expected_metadata={
            "command": "train",
            "seed": seed,
            "admission_sha256": sha256(training_admission_path),
            "pilot_decision_sha256": sha256(decision_path),
        },
    )
    metadata = reservation.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != {"command", "seed", "admission_sha256", "pilot_decision_sha256"}:
        raise ProtocolError("training reservation metadata is not exact")
    if not budget_close(reservation.get("reserved_seconds"), training_admission.get("requested_wall_seconds")):
        raise ProtocolError("training reservation differs from its admission allowance")
    return {
        "training_admission_path": training_admission_path,
        "training_admission": training_admission,
        "training_claim_path": training_claim_path,
        "training_context_path": context_path,
        "training_context": context,
        "training_budget_token": token,
        "training_reservation": reservation,
        "checkpoint": checkpoint,
        "result": result,
    }


def admit_worker(
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    command: str,
    *,
    seed: int | None = None,
    pilot_decision_path: Path | None = None,
    requested_wall_seconds: float | None = None,
    allow_existing: bool = False,
) -> Path:
    """Create (or validate) the durable one-shot admission before a worker."""
    if command == "actual-model-pilot" and seed is None:
        seed = config["budget"]["smoke_seed"]
    admission_path, existing_artifacts = _admission_layout(config, command, seed)
    for artifact in existing_artifacts:
        if artifact.exists():
            raise ProtocolError(
                f"{command} already has immutable artifact {_repo_relative(artifact)}; worker admission denied"
            )
    actual_config = _actual_repo_file(config_path.resolve(), "config")
    actual_manifest = _actual_repo_file(manifest_path.resolve(), "manifest")
    if pilot_decision_path is not None:
        pilot_decision_path = _actual_repo_file(pilot_decision_path.resolve(), "pilot decision")
    if admission_path.exists():
        if not allow_existing:
            raise ProtocolError(
                f"{command} worker admission already exists; an unresolved or completed attempt "
                "forbids automatic repeat"
            )
        record = load_json(admission_path)
        _validate_admission_record(record, config, actual_config, actual_manifest, command, seed, pilot_decision_path)
        if command == "recover-seed":
            _validate_recovery_chain_record(record, config, config_path, manifest_path, seed, pilot_decision_path)
        return admission_path
    root = repo_path(config["output_root"], artifact_output=True)
    payload: dict[str, Any] = {
        "schema": WORKER_ADMISSION_SCHEMA,
        "command": command,
        "state": "admitted_before_worker",
        "run_root": _repo_relative(root),
        "run_classification": "independent_future_run",
        "continues_consumed_historical_experiment": False,
        "config_path": _repo_relative(actual_config),
        "config_sha256": sha256(actual_config),
        "manifest_path": _repo_relative(actual_manifest),
        "manifest_sha256": sha256(actual_manifest),
        "code": code_identity(),
        "budget_ledger_path": _repo_relative(root / "gpu_budget.json"),
        "admitted_unix_seconds": time.time(),
        "admitted_pid": os.getpid(),
        "retry_policy": "create-once; a crash or budget rejection remains unresolved and forbids automatic repeat",
    }
    if seed is not None:
        payload["seed"] = seed
    if pilot_decision_path is not None:
        payload["pilot_decision_path"] = _repo_relative(pilot_decision_path)
        payload["pilot_decision_sha256"] = sha256(pilot_decision_path)
    if requested_wall_seconds is not None:
        payload["requested_wall_seconds"] = finite_runtime(requested_wall_seconds, "requested worker wall seconds")
    if command == "recover-seed":
        if seed is None or pilot_decision_path is None:
            raise ProtocolError("recovery admission requires a seed and decision")
        chain = _training_chain_for_recovery(config, actual_config, actual_manifest, seed, pilot_decision_path)
        payload.update({
            "training_admission_path": _repo_relative(chain["training_admission_path"]),
            "training_admission_sha256": sha256(chain["training_admission_path"]),
            "training_budget_token": chain["training_budget_token"],
            "training_context_path": _repo_relative(chain["training_context_path"]),
            "training_context_sha256": sha256(chain["training_context_path"]),
        })
    _validate_code_identity(payload["code"])
    create_once_json(
        admission_path,
        payload,
        exists_message=(
            f"{command} worker admission already exists; an unresolved or completed attempt "
            "forbids automatic repeat"
        ),
    )
    return admission_path


def validate_worker_admission(
    admission_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    command: str,
    *,
    seed: int | None = None,
    pilot_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Validate an already-created admission without weakening one-shot state."""
    if not admission_path.is_file():
        raise ProtocolError("required worker admission is missing")
    record = load_json(admission_path)
    _validate_admission_record(record, config, config_path, manifest_path, command, seed, pilot_decision_path)
    return record


def claim_path_for_admission(admission_path: Path) -> Path:
    """Return the immutable execution-claim path paired with an admission."""
    return admission_path.with_name(f"{admission_path.stem}.claim.json")


def _claim_seed(config: Mapping[str, Any], command: str, seed: int | None) -> int:
    if command in {"actual-model-pilot", "base-eval", "smoke"}:
        return int(config["budget"]["smoke_seed"])
    if command in {"train", "recover-seed"}:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ProtocolError("seed worker claim requires an integer seed")
        return seed
    raise ProtocolError(f"unsupported execution claim command: {command}")


def _validate_claim_record(
    claim: Mapping[str, Any],
    claim_path: Path,
    admission_path: Path,
    admission: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    command: str,
    seed: int | None,
    pilot_decision_path: Path | None,
) -> None:
    required = {
        "schema", "state", "admission_path", "admission_sha256", "command",
        "seed", "config_path", "config_sha256", "manifest_path",
        "manifest_sha256", "code", "created_unix_seconds", "created_pid",
    }
    if not isinstance(claim, Mapping) or not required <= set(claim):
        raise ProtocolError("execution claim provenance is incomplete")
    optional = {"pilot_decision_path", "pilot_decision_sha256"} if pilot_decision_path is not None else set()
    if set(claim) != required | optional:
        raise ProtocolError("execution claim provenance contains unknown fields")
    if claim["schema"] != CLAIM_SCHEMA or claim["state"] != "claimed_before_budget":
        raise ProtocolError("execution claim schema/state is invalid")
    if claim["command"] != command or claim["seed"] != _claim_seed(config, command, seed):
        raise ProtocolError("execution claim command/seed does not match")
    if claim["admission_path"] != _repo_relative(admission_path) or claim["admission_sha256"] != sha256(admission_path):
        raise ProtocolError("execution claim admission binding differs")
    actual_config = _actual_repo_file(config_path.resolve(), "config")
    actual_manifest = _actual_repo_file(manifest_path.resolve(), "manifest")
    if claim["config_path"] != _repo_relative(actual_config) or claim["config_sha256"] != sha256(actual_config):
        raise ProtocolError("execution claim config provenance differs")
    if claim["manifest_path"] != _repo_relative(actual_manifest) or claim["manifest_sha256"] != sha256(actual_manifest):
        raise ProtocolError("execution claim manifest provenance differs")
    _validate_code_identity(claim["code"])
    if pilot_decision_path is None:
        if "pilot_decision_path" in claim or "pilot_decision_sha256" in claim:
            raise ProtocolError("non-training claim unexpectedly binds a decision")
    else:
        decision = _actual_repo_file(pilot_decision_path.resolve(), "pilot decision")
        if claim.get("pilot_decision_path") != _repo_relative(decision) or claim.get("pilot_decision_sha256") != sha256(decision):
            raise ProtocolError("execution claim decision provenance differs")
    if not isinstance(claim["created_unix_seconds"], (int, float)) or not math.isfinite(float(claim["created_unix_seconds"])):
        raise ProtocolError("execution claim timestamp is invalid")
    if isinstance(claim["created_pid"], bool) or not isinstance(claim["created_pid"], int):
        raise ProtocolError("execution claim PID is invalid")
    if claim_path.parent != admission_path.parent:
        raise ProtocolError("execution claim is not paired with its admission")


def claim_worker(
    admission_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    command: str,
    *,
    seed: int | None = None,
    pilot_decision_path: Path | None = None,
) -> Path:
    """Atomically claim one admission before any budget reservation.

    There is intentionally no ``allow_existing`` escape hatch.  A claim is a
    one-shot execution right, not a reusable validation token; a retry must be
    rejected before it can reserve budget or enter a worker.
    """
    require_non_final_worker_open(config, route=command)
    admission = validate_worker_admission(
        admission_path, config, config_path, manifest_path, command,
        seed=seed, pilot_decision_path=pilot_decision_path,
    )
    claim_path = claim_path_for_admission(admission_path)
    if claim_path.exists():
        raise ProtocolError("execution claim already exists; one-shot admission is consumed")
    if pilot_decision_path is not None:
        pilot_decision_path = _actual_repo_file(pilot_decision_path.resolve(), "pilot decision")
    payload: dict[str, Any] = {
        "schema": CLAIM_SCHEMA,
        "state": "claimed_before_budget",
        "admission_path": _repo_relative(admission_path),
        "admission_sha256": sha256(admission_path),
        "command": command,
        "seed": _claim_seed(config, command, seed),
        "config_path": _repo_relative(_actual_repo_file(config_path.resolve(), "config")),
        "config_sha256": sha256(config_path),
        "manifest_path": _repo_relative(_actual_repo_file(manifest_path.resolve(), "manifest")),
        "manifest_sha256": sha256(manifest_path),
        "code": code_identity(),
        "created_unix_seconds": time.time(),
        "created_pid": os.getpid(),
    }
    if pilot_decision_path is not None:
        payload["pilot_decision_path"] = _repo_relative(pilot_decision_path)
        payload["pilot_decision_sha256"] = sha256(pilot_decision_path)
    create_once_json(
        claim_path,
        payload,
        exists_message="execution claim already exists; one-shot admission is consumed",
    )
    # A claim is useful only if its own bytes are immediately validated.  This
    # also catches a concurrent replacement before the caller reserves budget.
    _validate_claim_record(
        load_json(claim_path), claim_path, admission_path, admission,
        config, config_path, manifest_path, command, seed, pilot_decision_path,
    )
    return claim_path


def validate_execution_claim(
    claim_path: Path,
    admission_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    command: str,
    *,
    seed: int | None = None,
    pilot_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a claim without granting a second execution right."""
    if not claim_path.is_file():
        raise ProtocolError("required execution claim is missing")
    claim = load_json(claim_path)
    admission = validate_worker_admission(
        admission_path, config, config_path, manifest_path, command,
        seed=seed, pilot_decision_path=pilot_decision_path,
    )
    _validate_claim_record(
        claim, claim_path, admission_path, admission, config, config_path,
        manifest_path, command, seed, pilot_decision_path,
    )
    return claim


# Public descriptive aliases for integrations that do not use the CLI.
execution_claim_path = claim_path_for_admission
create_execution_claim = claim_worker


def _validate_recovery_chain_record(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    seed: int | None,
    decision_path: Path | None,
) -> dict[str, Any]:
    if seed is None or decision_path is None:
        raise ProtocolError("recovery chain validation requires seed and decision")
    chain = _training_chain_for_recovery(config, config_path, manifest_path, seed, decision_path)
    expected = {
        "training_admission_path": _repo_relative(chain["training_admission_path"]),
        "training_admission_sha256": sha256(chain["training_admission_path"]),
        "training_budget_token": chain["training_budget_token"],
        "training_context_path": _repo_relative(chain["training_context_path"]),
        "training_context_sha256": sha256(chain["training_context_path"]),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ProtocolError("recovery admission training-chain binding differs")
    return chain


def validate_recovery_admission(
    admission_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    seed: int,
    decision_path: Path,
) -> dict[str, Any]:
    """Public validation for the separate recovery admission lifecycle."""
    if admission_path.name != "recovery_admission.json":
        raise ProtocolError("recovery admission path is not the dedicated recovery path")
    record = validate_worker_admission(
        admission_path, config, config_path, manifest_path, "recover-seed",
        seed=seed, pilot_decision_path=decision_path,
    )
    return _validate_recovery_chain_record(record, config, config_path, manifest_path, seed, decision_path)


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"schema_version", "experiment_name", "starting_commit", "data_manifest", "output_root", "gpu_lock", "model", "labels", "question_id", "record_contract", "training", "budget", "evaluation", "latency", "checkpoint", "execution_gate", "baselines"}
    if set(config) != required:
        raise ProtocolError(f"config keys differ: missing={sorted(required - set(config))}, extra={sorted(set(config) - required)}")
    if config["schema_version"] != "laya-trace-i3-config-v1" or config["starting_commit"] != "2d4408de85587a046c3f14fec1835dd9e69940c9":
        raise ProtocolError("config schema or starting commit changed")
    if config["labels"] != FIXED_LABELS or config["question_id"] != "next_action":
        raise ProtocolError("fixed label list or question ID changed")
    model = config["model"]
    pins = {
        "id": "convaiinnovations/laya",
        "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "weights_sha256": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        "config_sha256": "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
        "laya_source_revision": "d113dca2512fb3eaca313534bc54c7162d87c1d4",
        "laya_common_sha256": "f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2",
        "notebook_sha256": "2c37036054dfb92073b518e994cb66c8a9009fb8d15b060d9bdfc6d4f94b1d39",
        "max_len": 512,
        "head_max_len": 192,
        "truncate_left": False,
    }
    if any(model.get(key) != value for key, value in pins.items()):
        raise ProtocolError("pinned B2 model or sequence contract changed")
    training = config["training"]
    exact_training = {
        "default_seeds": [42, 314159], "single_seed_fallback": 42, "epochs": 4,
        "micro_batch_records": 1, "gradient_accumulation": 64, "group_size": 4,
        "sigmas_by_epoch": [0.4, 0.3, 0.2, 0.1], "optimizer": "AdamW",
        "encoder_learning_rate": 2.5e-5, "nonencoder_learning_rate": 1e-4,
        "weight_decay": 0.01, "scheduler": "CosineAnnealingLR", "scheduler_eta_min": 1e-6,
        "amp_dtype": "fp16", "fp16_init_scale": 64.0, "fp16_growth_interval": 1000,
        "max_grad_norm": 1.0, "reward_log_floor": -9.21, "reward_spherical_weight": 0.75,
        "reward_rps_weight": 1.0, "ce_weight": 1.0, "gradient_checkpointing": True,
        "calibration": "none; raw softmax probabilities",
    }
    mismatch = {key: (training.get(key), value) for key, value in exact_training.items() if training.get(key) != value}
    if mismatch:
        raise ProtocolError(f"demonstrated B2 recipe changed: {mismatch}")
    if training.get("choice_option_order") != CHOICE_OPTION_ORDER or training.get("label_to_choice_index") != LABEL_TO_CHOICE_INDEX:
        raise ProtocolError("six-class choice-logit mapping changed")
    if training.get("action_objective") != "loss_rlcd(choice_logits[6]) + 1.0 * cross_entropy(choice_logits[6], one_hot_gold); ancillary binary act_head is excluded":
        raise ProtocolError("six-class RLCD+CE action objective changed")
    budget = config["budget"]
    if budget.get("cpu_threads") != 4 or budget.get("smoke_updates") != 2 or budget.get("main_max_aggregate_gpu_seconds") != FROZEN_GPU_SECONDS or budget.get("iteration_max_elapsed_seconds") != FROZEN_ITERATION_SECONDS:
        raise ProtocolError("CPU, smoke, aggregate GPU, or fallback budget changed")
    if budget.get("projection_multiplier", 0) < 1 or budget.get("max_peak_reserved_gib") != 11.5 or budget.get("no_oom_retry") is not True:
        raise ProtocolError("resource guard changed")
    evaluation = config["evaluation"]
    if evaluation.get("primary_metric") != "macro_f1" or evaluation.get("checkpoint_selection_split") != "dev" or evaluation.get("fixed_labels_include_absent_classes") is not True:
        raise ProtocolError("evaluation or checkpoint-selection contract changed")
    if evaluation.get("practical_added_value_absolute_macro_f1") != 0.05 or evaluation.get("nll_probability_floor") != 1e-12:
        raise ProtocolError("comparison threshold or NLL contract changed")
    if evaluation.get("final_test_policy") != "after a dev-only freeze, one atomic test-attempt marker is created before test rows or predictions are opened; no retry":
        raise ProtocolError("final-test lifecycle policy changed")
    latency = config["latency"]
    if latency.get("warm_definition") != "CUDA synchronize; tokenize/build, collate, host-to-device copy, forward, raw softmax and CPU probability copy; CUDA synchronize" or latency.get("cold_load_definition") != "process-local timer immediately before local config/tokenizer/model construction through strict weights load, device transfer, eval mode and first CUDA synchronize" or latency.get("training_definition") != "first training microforward through final optimizer update and CUDA synchronize" or latency.get("no_cuda_events_as_wall_time") is not True:
        raise ProtocolError("latency definitions changed")
    contract = config["record_contract"]
    if contract.get("questions_per_record") != 1 or contract.get("unit") != "one whole target assistant turn per record":
        raise ProtocolError("record-level whole-turn contract changed")
    repo_path(config["data_manifest"])
    repo_path(config["output_root"], artifact_output=True)
    repo_path(config["gpu_lock"], artifact_output=True)
    if config["execution_gate"] != "blocked_until_independent_data_verification_and_pre_run_review":
        raise ProtocolError("GPU execution gate must remain blocked")
    if config["baselines"] != {
        "majority": "train-label frequencies; fixed six-label probabilities",
        "transition": "rendered-state-only previous-action cue with declared initial fallback",
        "tfidf_logistic": "TF-IDF on exact rendered state plus question; dev budget frozen before test",
        "primary_comparator": "highest dev macro_f1, then lowest dev nll, then lexical method name",
    }:
        raise ProtocolError("baseline interface or selection budget changed")


def expected_updates(train_rows: int, config: Mapping[str, Any]) -> int:
    if isinstance(train_rows, bool) or not isinstance(train_rows, int) or train_rows <= 0:
        raise ProtocolError("accepted train row count must be a positive integer")
    training = config["training"]
    return training["epochs"] * math.ceil(train_rows / training["gradient_accumulation"])


class GpuLock:
    """Single-host, non-blocking GPU lock acquired before Torch is imported."""

    def __init__(self, path: Path, owner: Mapping[str, Any]):
        resolved = path.resolve()
        if resolved != OWNED_ARTIFACT_ROOT and OWNED_ARTIFACT_ROOT not in resolved.parents:
            raise ProtocolError("GPU lock must remain under the owned artifact root")
        required = {"command", "config_sha256", "manifest_sha256", "seed"}
        if set(owner) != required:
            raise ProtocolError("GPU lock owner metadata is incomplete")
        if not isinstance(owner["command"], str) or not owner["command"]:
            raise ProtocolError("GPU lock command is invalid")
        if not all(HEX64.fullmatch(str(owner[key])) for key in ("config_sha256", "manifest_sha256")):
            raise ProtocolError("GPU lock owner hashes are invalid")
        if isinstance(owner["seed"], bool) or not isinstance(owner["seed"], int):
            raise ProtocolError("GPU lock seed is invalid")
        self.path = resolved
        self.owner = dict(owner)
        self.handle: Any = None

    def __enter__(self) -> "GpuLock":
        if "torch" in sys.modules:
            raise ProtocolError("GPU lock must be acquired before Torch import")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise ProtocolError(f"GPU lock is already held: {self.path.relative_to(ROOT)}") from exc
        record = {**self.owner, "pid": os.getpid(), "acquired_unix_seconds": time.time(), "state": "held"}
        self.handle.seek(0)
        self.handle.truncate()
        json.dump(record, self.handle, indent=2, sort_keys=True)
        self.handle.write("\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def select_dev_checkpoint(candidates: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    """Select only from complete declared-seed dev summaries."""
    if not candidates:
        raise ProtocolError("at least one dev checkpoint candidate is required")
    selected = []
    seen_seeds = set()
    for candidate in candidates:
        required = {"seed", "split", "macro_f1", "nll", "checkpoint_path", "weights_file_sha256", "model_state_sha256", "training_state_file_sha256", "labels", "reload_evidence", "reload_evidence_path", "reload_evidence_sha256"}
        if set(candidate) != required or candidate["split"] != "dev" or candidate["labels"] != list(labels):
            raise ProtocolError("checkpoint candidates must be fixed-label dev results")
        if candidate["seed"] in seen_seeds:
            raise ProtocolError("candidate seeds must be unique")
        seen_seeds.add(candidate["seed"])
        if not all(HEX64.fullmatch(str(candidate[key])) for key in ("weights_file_sha256", "model_state_sha256", "training_state_file_sha256")):
            raise ProtocolError("candidate checkpoint hashes are missing")
        if not all(not isinstance(candidate[key], bool) and isinstance(candidate[key], (int, float)) and math.isfinite(candidate[key]) for key in ("macro_f1", "nll")):
            raise ProtocolError("candidate dev metrics must be finite")
        checkpoint = repo_path(candidate["checkpoint_path"], artifact_output=True)
        if not checkpoint.is_dir():
            raise ProtocolError("checkpoint directory is absent")
        weights = checkpoint / "model.safetensors"
        state = checkpoint / "training_state.pt"
        if not weights.is_file() or not state.is_file() or sha256(weights) != candidate["weights_file_sha256"] or sha256(state) != candidate["training_state_file_sha256"]:
            raise ProtocolError("checkpoint bytes are absent or hash-mismatched")
        evidence_path = repo_path(candidate["reload_evidence_path"], artifact_output=True)
        if not evidence_path.is_file() or sha256(evidence_path) != candidate["reload_evidence_sha256"]:
            raise ProtocolError("reload evidence artifact is absent or hash-mismatched")
        try:
            evidence = load_json(evidence_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError("reload evidence artifact is not valid JSON") from exc
        if evidence != candidate["reload_evidence"]:
            raise ProtocolError("reload evidence does not match its recorded artifact")
        validate_checkpoint_reload(evidence, candidate)
        selected.append(dict(candidate))
    winner = min(selected, key=lambda item: (-item["macro_f1"], item["nll"], item["seed"]))
    return {
        "selection_split": "dev",
        "selection_rule": "highest macro_f1, then lowest nll, then lowest declared seed",
        "candidate_count": len(selected),
        "selected_seed": winner["seed"],
        "selected_checkpoint_path": winner["checkpoint_path"],
        "selected_weights_file_sha256": winner["weights_file_sha256"],
    }


def require_exact_seed_set(result_seeds: Sequence[int], selected_seeds: Sequence[int]) -> None:
    """Require result artifacts for exactly the seeds selected by the pilot gate."""
    actual = set(result_seeds)
    expected = set(selected_seeds)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ProtocolError(f"dev result seeds do not match pilot selection (missing={missing}, unexpected={unexpected})")


def consume_final_test_attempt(
    run_dir: Path,
    freeze_sha256: str,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically consume the sole attempt before test/model bytes are opened.

    The optional provenance mapping is the authenticated final marker payload;
    the two-argument form remains a compatibility helper for the historical
    low-level unit probe, but the production final route always supplies it.
    """
    resolved = run_dir.resolve()
    if resolved != OWNED_ARTIFACT_ROOT and OWNED_ARTIFACT_ROOT not in resolved.parents:
        raise ProtocolError("run directory escapes owned artifact root")
    freeze = resolved / "freeze.json"
    if not HEX64.fullmatch(freeze_sha256) or not freeze.is_file() or sha256(freeze) != freeze_sha256:
        raise ProtocolError("dev-only freeze artifact is absent or hash-mismatched")
    marker = resolved / "final_test.attempt.json"
    if provenance is None:
        raise ProtocolError("authenticated final provenance is required; use run_final_test")
    else:
        if not isinstance(provenance, Mapping):
            raise ProtocolError("final attempt provenance must be a mapping")
        reservation = provenance.get("budget_reservation")
        if not isinstance(reservation, Mapping) or reservation.get("state") != "in_flight":
            raise ProtocolError("final attempt requires the activated in-flight budget reservation")
        if provenance.get("budget_token") != reservation.get("token"):
            raise ProtocolError("final attempt budget token/reservation differs")
        payload = dict(provenance)
        payload.update({"schema": FINAL_ATTEMPT_SCHEMA, "freeze_sha256": freeze_sha256, "consumed_unix_seconds": time.time(), "policy": "single authenticated attempt; no retry"})
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolError("final test attempt was already consumed; retry forbidden") from exc
    fsync_directory(resolved)
    return marker


def budget_close(actual: Any, expected: Any) -> bool:
    """Compare budget values using the single protocol-wide epsilon."""
    try:
        left = finite_runtime(actual, "budget value")
        right = finite_runtime(expected, "budget value")
    except ProtocolError:
        return False
    return abs(left - right) <= BUDGET_EPSILON


def finite_runtime(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ProtocolError(f"{name} must be a finite non-negative number")
    return float(value)


def validate_runtime_budget(elapsed_seconds: Any, aggregate_gpu_seconds: Any, config: Mapping[str, Any]) -> None:
    elapsed = finite_runtime(elapsed_seconds, "iteration elapsed seconds")
    aggregate = finite_runtime(aggregate_gpu_seconds, "aggregate GPU seconds")
    if elapsed > FROZEN_ITERATION_SECONDS or aggregate > FROZEN_GPU_SECONDS:
        raise ProtocolError("frozen iteration or aggregate GPU budget exceeded; stop without retry")


def validate_manifest(manifest: Mapping[str, Any], *, allow_pending: bool) -> str:
    required = {"schema_version", "status", "acceptance", "dataset", "renderer", "labels", "splits"}
    if set(manifest) != required or manifest.get("schema_version") != "laya-trace-i3-manifest-v1":
        raise ProtocolError("manifest top-level contract changed")
    if manifest.get("labels") != FIXED_LABELS:
        raise ProtocolError("manifest label order differs from the fixed experiment labels")
    status = manifest.get("status")
    if status == "pending":
        if not allow_pending:
            raise ProtocolError("data manifest is pending independent acceptance")
        if manifest.get("acceptance") is not None:
            raise ProtocolError("pending manifest must not claim acceptance")
        return "pending"
    if status != "accepted":
        raise ProtocolError("manifest status must be pending or accepted")
    acceptance = manifest.get("acceptance")
    if not isinstance(acceptance, dict) or set(acceptance) != {"evidence_path", "evidence_sha256"}:
        raise ProtocolError("accepted manifest needs exact acceptance evidence")
    evidence_path = repo_path(acceptance["evidence_path"])
    if not HEX64.fullmatch(str(acceptance["evidence_sha256"])) or not evidence_path.is_file() or sha256(evidence_path) != acceptance["evidence_sha256"]:
        raise ProtocolError("acceptance evidence is absent or hash-mismatched")
    dataset = manifest.get("dataset")
    renderer = manifest.get("renderer")
    for name, value in (("dataset", dataset), ("renderer", renderer)):
        if not isinstance(value, dict) or not value:
            raise ProtocolError(f"accepted manifest requires {name} provenance")
    if renderer.get("schema_version") != "trace2decision-i3-v3" or not HEX64.fullmatch(str(renderer.get("implementation_sha256", ""))):
        raise ProtocolError("renderer schema/hash is not frozen to accepted v3")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "dev", "test"}:
        raise ProtocolError("manifest requires exactly train/dev/test splits")
    paths = []
    for split in ("train", "dev", "test"):
        entry = splits[split]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "rows", "trajectories", "task_groups"}:
            raise ProtocolError(f"{split} manifest entry has wrong fields")
        path = repo_path(entry["path"])
        paths.append(path)
        if not HEX64.fullmatch(str(entry["sha256"])):
            raise ProtocolError(f"{split} SHA-256 is not frozen")
        for field in ("rows", "trajectories", "task_groups"):
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProtocolError(f"{split} {field} must be a positive integer")
    if len(set(paths)) != 3:
        raise ProtocolError("split paths must be distinct")
    return "accepted"


def validate_record(record: Mapping[str, Any], labels: Sequence[str]) -> None:
    """Validate the model/evaluation boundary without importing a model stack."""
    required = {"state", "questions", "gold", "metadata"}
    if set(record) != required or not isinstance(record["state"], str) or not record["state"]:
        raise ProtocolError("record must contain state, questions, gold, and metadata")
    questions = record["questions"]
    if not isinstance(questions, Mapping) or set(questions) != {"next_action"}:
        raise ProtocolError("each record must contain exactly one next_action question")
    question = questions["next_action"]
    if not isinstance(question, Mapping) or question.get("type") != "choice":
        raise ProtocolError("next_action must be a choice question")
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping) or list(criteria) != CHOICE_OPTION_ORDER:
        raise ProtocolError("next_action criteria must use the frozen six-choice logit order")
    gold = record["gold"]
    if not isinstance(gold, Mapping) or set(gold) != {"next_action"}:
        raise ProtocolError("gold must contain exactly one next_action answer")
    answer = gold["next_action"]
    if not isinstance(answer, Mapping) or answer.get("label") not in labels:
        raise ProtocolError("gold label is outside the fixed label list")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(labels):
        raise ProtocolError("gold probabilities must contain the fixed labels")
    values = list(probabilities.values())
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values):
        raise ProtocolError("gold probabilities must be finite")
    if not math.isclose(sum(values), 1.0, abs_tol=2e-6, rel_tol=0.0):
        raise ProtocolError("gold probabilities must sum to one")
    metadata = record["metadata"]
    if not isinstance(metadata, Mapping):
        raise ProtocolError("metadata must contain bootstrap identifiers")
    if not re.fullmatch(r"t2d-i3-[0-9a-f]{32}", str(metadata.get("id", ""))):
        raise ProtocolError("record bootstrap ID is missing or malformed")
    if not all(HEX64.fullmatch(str(metadata.get(key, ""))) for key in ("trajectory_id", "task_group")):
        raise ProtocolError("trajectory/task-group bootstrap ID is missing or malformed")


def group_bootstrap_ids(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove stable row IDs and a one-group-per-trajectory bootstrap relation."""
    if not records:
        raise ProtocolError("bootstrap records must be non-empty")
    row_ids = set()
    trajectory_groups: dict[str, str] = {}
    group_rows: dict[str, int] = {}
    for record in records:
        validate_record(record, FIXED_LABELS)
        metadata = record["metadata"]
        row_id = metadata["id"]
        if row_id in row_ids:
            raise ProtocolError("bootstrap record IDs must be unique")
        row_ids.add(row_id)
        trajectory = metadata["trajectory_id"]
        group = metadata["task_group"]
        if trajectory in trajectory_groups and trajectory_groups[trajectory] != group:
            raise ProtocolError("a trajectory maps to multiple task groups")
        trajectory_groups[trajectory] = group
        group_rows[group] = group_rows.get(group, 0) + 1
    return {
        "bootstrap_unit": "task_group",
        "rows": len(row_ids),
        "trajectories": len(trajectory_groups),
        "task_groups": len(group_rows),
        "group_ids": sorted(group_rows),
        "row_counts_by_group": dict(sorted(group_rows.items())),
    }


def validate_checkpoint_reload(evidence: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Require runtime-produced strict model and complete training-state reload proof."""
    required = {"strict_model_load", "model_state_before_sha256", "model_state_after_sha256", "training_state_file_sha256", "optimizer_exact", "scheduler_exact", "scaler_exact", "stopping_state_exact"}
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        raise ProtocolError("checkpoint reload evidence is incomplete")
    before = str(evidence["model_state_before_sha256"])
    after = str(evidence["model_state_after_sha256"])
    state_file = str(evidence["training_state_file_sha256"])
    if not all(HEX64.fullmatch(value) for value in (before, after, state_file)):
        raise ProtocolError("checkpoint reload evidence hashes are malformed")
    if before != after or before != candidate["model_state_sha256"] or state_file != candidate["training_state_file_sha256"]:
        raise ProtocolError("checkpoint reload hashes do not match candidate bytes/state")
    for key in ("strict_model_load", "optimizer_exact", "scheduler_exact", "scaler_exact", "stopping_state_exact"):
        if evidence[key] is not True:
            raise ProtocolError(f"checkpoint reload guard failed: {key}")


def validate_recovery_stopping_state(state: Mapping[str, Any], *, expected_updates: int, expected_microforwards: int, expected_epochs: int) -> None:
    """Require the serialized training state to describe the frozen stop point."""
    if not isinstance(state, Mapping):
        raise ProtocolError("training state is not a mapping")
    if state.get("updates") != expected_updates or state.get("microforwards") != expected_microforwards or state.get("epochs") != expected_epochs:
        raise ProtocolError("training state does not match the exact frozen stopping state")


def validate_recovery_checkpoint(checkpoint: Path, *, weights_sha256: str, training_state_sha256: str, reload_sha256: str, model_state_sha256: str, expected_updates: int, expected_microforwards: int, expected_epochs: int, state: Mapping[str, Any], reload_evidence: Mapping[str, Any]) -> dict[str, str]:
    """Validate recovery bytes and state without importing Torch or opening data."""
    weights = checkpoint / "model.safetensors"
    training_state = checkpoint / "training_state.pt"
    reload_path = checkpoint / "reload.json"
    if not checkpoint.is_dir() or not weights.is_file() or not training_state.is_file() or not reload_path.is_file():
        raise ProtocolError("completed recovery checkpoint is incomplete")
    actual = {"weights_file_sha256": sha256(weights), "training_state_file_sha256": sha256(training_state), "reload_evidence_sha256": sha256(reload_path)}
    expected = {"weights_file_sha256": weights_sha256, "training_state_file_sha256": training_state_sha256, "reload_evidence_sha256": reload_sha256}
    if actual != expected:
        raise ProtocolError("completed recovery checkpoint bytes are hash-mismatched")
    recorded = load_json(reload_path)
    candidate = {"model_state_sha256": model_state_sha256, "training_state_file_sha256": training_state_sha256}
    if recorded != dict(reload_evidence):
        raise ProtocolError("recovery reload evidence does not match its recorded bytes")
    validate_checkpoint_reload(recorded, candidate)
    validate_recovery_stopping_state(state, expected_updates=expected_updates, expected_microforwards=expected_microforwards, expected_epochs=expected_epochs)
    return actual


def validate_latency_evidence(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, float]:
    """Validate synchronized wall-clock samples; it never performs model work."""
    required = {"batch_size", "warmups_completed", "samples_seconds", "synchronize_before_each", "synchronize_after_each", "included_phases", "measurement_source", "synchronization_events"}
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        raise ProtocolError("latency evidence is incomplete")
    latency = config["latency"]
    phases = FROZEN_LATENCY_PHASES
    samples = evidence["samples_seconds"]
    if evidence["batch_size"] != latency["batch_size"] or evidence["warmups_completed"] != latency["warmups"]:
        raise ProtocolError("latency batch size or warmup count changed")
    if evidence["included_phases"] != phases or evidence["synchronize_before_each"] is not True or evidence["synchronize_after_each"] is not True or evidence["measurement_source"] != "runtime_monotonic_with_cuda_synchronization" or evidence["synchronization_events"] != 2 * (latency["warmups"] + latency["measured_records"]):
        raise ProtocolError("latency synchronization or measured boundary changed")
    if not isinstance(samples, list) or len(samples) != latency["measured_records"] or not all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples):
        raise ProtocolError("latency samples must be finite positive measured records")
    ordered = sorted(float(x) for x in samples)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    middle = len(ordered) // 2
    median = (ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0 else ordered[middle]
    return {"median_seconds": median, "p95_seconds": ordered[p95_index]}


def verify_split_files(manifest: Mapping[str, Any], *, include_test: bool = False) -> dict[str, Any]:
    """Verify bytes/counts without decoding labels; test is opt-in and hash-only."""
    checked = {}
    for split in ("train", "dev") + (("test",) if include_test else ()):
        entry = manifest["splits"][split]
        path = repo_path(entry["path"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ProtocolError(f"{split} file is absent or hash-mismatched")
        rows = None
        if split != "test":
            with path.open(encoding="utf-8") as handle:
                rows = sum(1 for line in handle if line.strip())
            if rows != entry["rows"]:
                raise ProtocolError(f"{split} row count differs from manifest")
        checked[split] = {"path": entry["path"], "sha256": entry["sha256"], "rows_verified": rows}
    return checked


PILOT_MEASUREMENT_FIELDS = {
    "seed", "updates", "training_seconds", "cold_load_seconds",
    "dev_evaluation_seconds", "peak_reserved_gib", "finite", "status",
    "test_opened",
}
PILOT_PROVENANCE_FIELDS = {
    "admission_path", "admission_sha256", "claim_path", "claim_sha256",
    "config_path", "config_sha256", "manifest_path", "manifest_sha256",
    "code", "budget_ledger_path", "budget_ledger_sha256", "budget_token",
    "budget_reservation", "budget_snapshot",
}
RESULT_PROVENANCE_FIELDS = {
    "admission_path", "admission_sha256", "claim_path", "claim_sha256",
    "pilot_decision_path", "pilot_decision_sha256", "selected_seed",
    "config_path", "config_sha256", "manifest_path", "manifest_sha256",
    "code", "budget_ledger_path", "budget_ledger_sha256", "budget_token",
    "budget_reservation", "training_context_path", "training_context_sha256",
}
DECISION_IDENTITY_FIELDS = {
    "pilot_path", "pilot_sha256", "pilot_admission_path",
    "pilot_admission_sha256", "config_path", "config_sha256",
    "manifest_path", "manifest_sha256", "code", "budget_ledger_path",
    "budget_ledger_sha256", "planning_snapshot", "budget_snapshot", "final_budget_token",
    "final_budget_reservation", "final_budget_reservation_sha256",
    "final_budget_ledger_sha256", "final_budget_snapshot", "declared_seeds",
}
PILOT_DECISION_FIELDS = DECISION_IDENTITY_FIELDS | {
    "schema", "pilot_admission", "decision",
}


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("mapping is not JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ProtocolError(f"{name} must be a SHA-256 digest")
    return value


def _validate_pilot_measurements(
    pilot: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Validate the measured pilot envelope, accepting only legacy unit shape
    when the caller explicitly requests the compatibility path.
    """
    is_new = "schema" in pilot
    required = set(PILOT_MEASUREMENT_FIELDS)
    if is_new:
        if pilot.get("schema") != PILOT_SCHEMA:
            raise ProtocolError("pilot schema is invalid")
        if set(pilot) != required | {"schema"} | PILOT_PROVENANCE_FIELDS:
            raise ProtocolError("pilot provenance fields are incomplete or unknown")
    elif set(pilot) != required:
        raise ProtocolError("smoke evidence does not match the predeclared profile")
    if (
        pilot["status"] != "actual_model_pilot_measured"
        or pilot["test_opened"] is not False
        or pilot["seed"] != config["budget"]["smoke_seed"]
        or pilot["updates"] != config["budget"]["smoke_updates"]
    ):
        raise ProtocolError("smoke evidence does not match the predeclared profile")
    numeric = [
        pilot[key]
        for key in ("training_seconds", "cold_load_seconds", "dev_evaluation_seconds", "peak_reserved_gib")
    ]
    if pilot["finite"] is not True or not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        for value in numeric
    ):
        raise ProtocolError("smoke evidence is non-finite or unsuccessful")
    if (
        pilot["peak_reserved_gib"] > config["budget"]["max_peak_reserved_gib"]
        or pilot["training_seconds"] > config["budget"]["smoke_max_gpu_seconds"]
    ):
        raise ProtocolError("smoke exceeded memory or time budget; main training is forbidden")


def _ledger_path(config: Mapping[str, Any]) -> Path:
    return repo_path(config["output_root"], artifact_output=True) / "gpu_budget.json"


def current_budget_snapshot(
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Read one locked, fully validated aggregate-ledger snapshot."""
    identity = budget_identity(config, config_path, manifest_path)
    selected_ledger = (ledger_path or _ledger_path(config)).resolve()
    if selected_ledger != _ledger_path(config).resolve():
        raise ProtocolError("budget ledger path is foreign to the configured run")
    return AggregateBudget(
        selected_ledger,
        identity,
        float(config["budget"]["main_max_aggregate_gpu_seconds"]),
    ).snapshot()


def _reservation_from_snapshot(snapshot: Mapping[str, Any], token: str) -> dict[str, Any] | None:
    for item in snapshot.get("reservations", []):
        if item.get("token") == token:
            return dict(item)
    return None


def validate_budget_snapshot_prefix(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    mutable_tokens: Sequence[str] = (),
) -> None:
    """Allow later appends, but reject edits to the bound prefix.

    A final allowance is deliberately mutable only along its explicit
    ``held -> in_flight -> finished/failed`` lifecycle and only for the token
    named by the immutable decision.  All other prefix bytes remain exact.
    """
    required = {"path", "sha256", "identity", "limit", "charged_seconds", "reservations"}
    if not isinstance(recorded, Mapping) or not required <= set(recorded):
        raise ProtocolError("budget snapshot is incomplete")
    if (
        recorded.get("path") != current.get("path")
        or recorded["identity"] != current["identity"]
        or recorded["limit"] != current["limit"]
    ):
        raise ProtocolError("budget snapshot identity changed")
    old = recorded["reservations"]
    new = current["reservations"]
    if not isinstance(old, list) or not isinstance(new, list) or len(new) < len(old):
        raise ProtocolError("budget ledger reservation history changed")
    mutable = set(mutable_tokens)
    mutable_changed = False
    for index, old_item in enumerate(old):
        new_item = new[index]
        if new_item == old_item:
            continue
        if (
            not isinstance(old_item, Mapping)
            or old_item.get("token") not in mutable
            or old_item.get("state") != "held"
            or not isinstance(new_item, Mapping)
            or new_item.get("token") != old_item.get("token")
            or new_item.get("reserved_seconds") != old_item.get("reserved_seconds")
            or new_item.get("state") not in {"in_flight", "finished", "failed", "overrun"}
        ):
            raise ProtocolError("budget ledger reservation history changed")
        old_metadata = old_item.get("metadata", {})
        new_metadata = new_item.get("metadata", {})
        if not isinstance(old_metadata, Mapping) or not isinstance(new_metadata, Mapping):
            raise ProtocolError("budget ledger metadata changed")
        for key, value in old_metadata.items():
            if new_metadata.get(key) != value:
                raise ProtocolError("budget ledger bound metadata changed")
        mutable_changed = True
    if not budget_close(recorded["charged_seconds"], AggregateBudget._charged_seconds({"reservations": old})):
        raise ProtocolError("budget snapshot charged seconds are inconsistent")
    if len(new) == len(old) and not mutable_changed and current.get("sha256") != recorded.get("sha256"):
        raise ProtocolError("budget ledger bytes changed despite unchanged prefix")
    if recorded["sha256"] is not None:
        _validate_sha(recorded["sha256"], "budget snapshot ledger SHA-256")


def _validate_finished_reservation(
    snapshot: Mapping[str, Any],
    token: Any,
    recorded: Any,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(token, str) or not token or not isinstance(recorded, Mapping):
        raise ProtocolError("finished budget reservation is missing")
    item = _reservation_from_snapshot(snapshot, token)
    if item is None or item.get("state") != "finished" or item != dict(recorded):
        raise ProtocolError("finished budget reservation does not match the ledger")
    if expected_metadata is not None:
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping) or any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ProtocolError("finished budget reservation metadata does not match")
    return item


def _validate_provenance_paths(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    expected_command: str,
    expected_seed: int | None,
    expected_decision_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    actual_config = _actual_repo_file(config_path.resolve(), "config")
    actual_manifest = _actual_repo_file(manifest_path.resolve(), "manifest")
    root = repo_path(config["output_root"], artifact_output=True)
    admission_value = record.get("admission_path")
    if not isinstance(admission_value, str):
        raise ProtocolError("result admission path is missing")
    admission_path = repo_path(admission_value, artifact_output=True)
    if admission_path.parent != root and not (admission_path.parent.parent == root and admission_path.parent.name.startswith("seed_")):
        raise ProtocolError("result admission path escapes its run root")
    admission = load_json(admission_path)
    _validate_admission_record(
        admission, config, actual_config, actual_manifest,
        expected_command, expected_seed, expected_decision_path,
    )
    if _validate_sha(record.get("admission_sha256"), "admission SHA-256") != sha256(admission_path):
        raise ProtocolError("result admission SHA-256 does not match")
    if record.get("config_path") != _repo_relative(actual_config) or record.get("config_sha256") != sha256(actual_config):
        raise ProtocolError("result config provenance does not match")
    if record.get("manifest_path") != _repo_relative(actual_manifest) or record.get("manifest_sha256") != sha256(actual_manifest):
        raise ProtocolError("result manifest provenance does not match")
    _validate_code_identity(record.get("code"))
    claim_path = record.get("claim_path")
    claim_hash = record.get("claim_sha256")
    if not isinstance(claim_path, str):
        raise ProtocolError("result execution claim path is missing")
    actual_claim = repo_path(claim_path, artifact_output=True)
    if actual_claim != claim_path_for_admission(admission_path):
        raise ProtocolError("result execution claim path is foreign")
    if _validate_sha(claim_hash, "execution claim SHA-256") != sha256(actual_claim):
        raise ProtocolError("result execution claim SHA-256 does not match")
    claim = load_json(actual_claim)
    _validate_claim_record(
        claim, actual_claim, admission_path, admission, config, actual_config,
        actual_manifest, expected_command, expected_seed, expected_decision_path,
    )
    expected_ledger = _repo_relative(_ledger_path(config))
    if record.get("budget_ledger_path") != expected_ledger:
        raise ProtocolError("result budget ledger path does not match")
    _validate_sha(record.get("budget_ledger_sha256"), "budget ledger SHA-256")
    if record.get("budget_token") is None:
        raise ProtocolError("result budget token is missing")
    return admission_path, admission


def _validate_pilot_artifact(
    pilot: Mapping[str, Any],
    pilot_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    require_current_ledger_hash: bool = False,
) -> dict[str, Any]:
    """Validate a real, admitted and finished pilot before it can authorize work."""
    _validate_pilot_measurements(pilot, config)
    if pilot.get("schema") != PILOT_SCHEMA:
        raise ProtocolError("pilot provenance is required for an actual-model pilot")
    root = repo_path(config["output_root"], artifact_output=True)
    if pilot_path.resolve().parent != root:
        raise ProtocolError("pilot path is not in the configured run root")
    admission_path, admission = _validate_provenance_paths(
        pilot,
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        expected_command="actual-model-pilot",
        expected_seed=config["budget"]["smoke_seed"],
    )
    if admission_path != root / "future_actual_model_pilot.admission.json":
        raise ProtocolError("pilot admission path does not match the pilot route")
    if admission.get("requested_wall_seconds") != float(config["budget"]["smoke_max_gpu_seconds"]):
        raise ProtocolError("pilot admission request does not match the fixed pilot allowance")
    identity = budget_identity(config, config_path, manifest_path)
    current_snapshot = AggregateBudget(
        _ledger_path(config), identity, float(config["budget"]["main_max_aggregate_gpu_seconds"])
    ).snapshot()
    recorded_snapshot = pilot.get("budget_snapshot")
    if not isinstance(recorded_snapshot, Mapping):
        raise ProtocolError("pilot budget snapshot is missing")
    validate_budget_snapshot_prefix(recorded_snapshot, current_snapshot)
    if pilot.get("budget_ledger_sha256") != recorded_snapshot.get("sha256"):
        raise ProtocolError("pilot budget ledger hash does not match its attached snapshot")
    if _validate_sha(pilot.get("budget_ledger_sha256"), "pilot budget ledger SHA-256") != recorded_snapshot.get("sha256"):
        raise ProtocolError("pilot budget ledger hash is malformed")
    reservation = _validate_finished_reservation(
        recorded_snapshot,
        pilot.get("budget_token"),
        pilot.get("budget_reservation"),
    )
    metadata = reservation.get("metadata")
    expected_metadata = {
        "command": "actual-model-pilot",
        "seed": config["budget"]["smoke_seed"],
        "admission_sha256": pilot["admission_sha256"],
    }
    if not isinstance(metadata, Mapping) or set(metadata) != set(expected_metadata) or any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ProtocolError("pilot budget reservation metadata is missing, inexact, or does not match")
    if not budget_close(reservation["reserved_seconds"], config["budget"]["smoke_max_gpu_seconds"]):
        raise ProtocolError("pilot budget reservation is not the fixed 600-second allowance")
    if require_current_ledger_hash and current_snapshot.get("sha256") != recorded_snapshot.get("sha256"):
        # At decision creation no later append is allowed between pilot
        # validation and the atomic hold.  Existing appends are handled by the
        # prefix check; this stricter check is intentionally scoped to the
        # creation boundary.
        raise ProtocolError("pilot budget ledger changed before decision creation")
    return {
        "pilot": pilot, "admission": admission, "admission_path": admission_path,
        "snapshot": current_snapshot, "recorded_snapshot": dict(recorded_snapshot),
        "reservation": reservation, "identity": identity,
    }


def _decision_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping) or not DECISION_IDENTITY_FIELDS <= set(record):
        raise ProtocolError("pilot decision identity is incomplete")
    return {key: record[key] for key in sorted(DECISION_IDENTITY_FIELDS)}


def choose_seed_count(
    smoke: Mapping[str, Any],
    config: Mapping[str, Any],
    train_rows: int,
    *,
    config_path: Path | None = None,
    manifest_path: Path | None = None,
    budget_ledger_path: Path | None = None,
    charged_seconds: float | None = None,
    budget_snapshot: Mapping[str, Any] | None = None,
    require_provenance: bool = False,
) -> dict[str, Any]:
    """Choose the seed set using the *cumulative* locked GPU composition.

    ``charged_seconds`` is supplied when validating an immutable decision.  At
    creation time it is read from the shared ledger while holding the ledger
    lock.  The compatibility path for old CPU unit fixtures has no admission
    and therefore cannot authorize a real decision.
    """
    _validate_pilot_measurements(smoke, config)
    if require_provenance and smoke.get("schema") != PILOT_SCHEMA:
        raise ProtocolError("a provenance-bound actual-model pilot is required")
    actual_config_path = (config_path or ROOT / "training/laya_trace_i3/config.json").resolve()
    actual_manifest_path = (manifest_path or repo_path(config["data_manifest"])).resolve()
    if charged_seconds is None and budget_snapshot is not None:
        if not isinstance(budget_snapshot, Mapping) or "charged_seconds" not in budget_snapshot:
            raise ProtocolError("budget snapshot charged seconds are missing")
        charged_seconds = finite_runtime(budget_snapshot["charged_seconds"], "already charged GPU seconds")
        snapshot = dict(budget_snapshot)
    elif charged_seconds is None:
        try:
            snapshot = current_budget_snapshot(
                config,
                actual_config_path,
                actual_manifest_path,
                budget_ledger_path,
            )
            charged_seconds = float(snapshot["charged_seconds"])
        except ProtocolError:
            if require_provenance:
                raise
            # Legacy measurement-only unit tests predate the shared ledger;
            # their result is never accepted by create/validate decision.
            snapshot = None
            charged_seconds = 0.0
    else:
        charged_seconds = finite_runtime(charged_seconds, "already charged GPU seconds")
        snapshot = dict(budget_snapshot) if budget_snapshot is not None else None
    final_reserve = finite_runtime(FINAL_TEST_RESERVE_SECONDS, "final-test reserve seconds")
    limit = finite_runtime(config["budget"]["main_max_aggregate_gpu_seconds"], "aggregate GPU limit")
    if charged_seconds > limit:
        raise ProtocolError("current aggregate ledger charge exceeds its limit")
    updates = expected_updates(train_rows, config)
    per_seed = (
        smoke["training_seconds"] / smoke["updates"] * updates
        + smoke["cold_load_seconds"]
        + smoke["dev_evaluation_seconds"]
    ) * config["budget"]["projection_multiplier"]
    two_seed_projection = per_seed * len(config["training"]["default_seeds"])
    available_for_two = limit - charged_seconds - final_reserve
    fits_two = two_seed_projection <= available_for_two + BUDGET_EPSILON
    fallback = not fits_two
    one_seed_projection = per_seed
    if charged_seconds + one_seed_projection + final_reserve > limit + BUDGET_EPSILON:
        raise ProtocolError("aggregate composition cannot fit even the authorized fallback seed")
    total_two = charged_seconds + two_seed_projection + final_reserve
    total_one = charged_seconds + one_seed_projection + final_reserve
    composition = {
        "already_charged_gpu_seconds": charged_seconds,
        "planned_training_gpu_seconds_for_two_seeds": two_seed_projection,
        "planned_training_gpu_seconds_for_one_seed": one_seed_projection,
        "fixed_final_test_reserve_seconds": final_reserve,
        "available_gpu_seconds_after_current_charges_and_final_reserve": available_for_two,
        "aggregate_limit_seconds": limit,
        "two_seed_total_with_final_reserve_seconds": total_two,
        "one_seed_total_with_final_reserve_seconds": total_one,
        "fits_declared_seed_set": fits_two,
    }
    return {
        "expected_updates_per_seed": updates,
        "projected_gpu_seconds_per_seed": per_seed,
        "projected_gpu_seconds_two_seeds": two_seed_projection,
        "charged_seconds_before_training": charged_seconds,
        "fixed_final_test_reserve_seconds": final_reserve,
        "aggregate_composition": composition,
        "fallback_applied": fallback,
        "selected_seeds": [config["training"]["single_seed_fallback"]] if fallback else list(config["training"]["default_seeds"]),
        "rule": "select the declared pair only when current charged GPU seconds plus projected training plus the fixed 600-second final-test reserve is at most 7200; otherwise use seed 42",
    }


def code_identity() -> dict[str, str]:
    """Identity of every source file that can affect the gated result."""
    paths = {
        "runner.py": ROOT / "training/laya_trace_i3/runner.py",
        "experiment.py": ROOT / "training/laya_trace_i3/experiment.py",
        "protocol.py": ROOT / "training/laya_trace_i3/protocol.py",
        "metrics.py": ROOT / "training/laya_trace_i3/metrics.py",
        "b2_train.py": ROOT / "training/laya_local/b2_train.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


# Set exactly once after all definitions have loaded.  Validators compare an
# artifact with both this imported-code snapshot and the current disk identity;
# changing either one therefore fails closed instead of silently relabeling
# the already-imported implementation.
_IMPORT_TIME_CODE_IDENTITY: dict[str, str] | None = None
IMPORT_TIME_CODE_IDENTITY: dict[str, str] | None = None


def import_time_code_identity() -> dict[str, str]:
    if _IMPORT_TIME_CODE_IDENTITY is None or IMPORT_TIME_CODE_IDENTITY is None:
        raise ProtocolError("import-time code identity snapshot is unavailable")
    if _IMPORT_TIME_CODE_IDENTITY != IMPORT_TIME_CODE_IDENTITY:
        raise ProtocolError("import-time code identity snapshot aliases disagree")
    return dict(_IMPORT_TIME_CODE_IDENTITY)


def _validate_code_identity(recorded: Any) -> dict[str, str]:
    current = code_identity()
    imported = import_time_code_identity()
    if current != imported:
        raise ProtocolError("loaded code identity differs from the import-time code snapshot")
    if recorded != current:
        raise ProtocolError("code identity does not match current and import-time sources")
    return current


validate_loaded_code_identity = _validate_code_identity


def create_pilot_decision(
    pilot_path: Path,
    decision_path: Path,
    config_path: Path,
    manifest_path: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a decision and its final allowance as one fail-closed plan."""
    validate_manifest(manifest, allow_pending=False)
    root = require_non_final_worker_open(config, route="validation")
    expected_manifest_path = repo_path(config["data_manifest"]).resolve()
    if manifest_path.resolve() != expected_manifest_path:
        raise ProtocolError("pilot manifest path is foreign to the supplied config")
    if pilot_path.resolve().parent != root or decision_path.resolve().parent != root:
        raise ProtocolError("pilot and decision paths must remain in the configured run root")
    if decision_path.exists():
        raise ProtocolError("pilot decision already exists; immutable decisions cannot be replaced")
    if _final_hold_state(root) is not None:
        raise ProtocolError("orphan final budget hold already exists; fail closed")
    if not pilot_path.is_file():
        raise ProtocolError("a provenance-bound actual-model pilot is required")
    pilot = load_json(pilot_path)
    evidence = _validate_pilot_artifact(
        pilot, pilot_path, config, config_path, manifest_path, manifest,
        require_current_ledger_hash=True,
    )
    snapshot = evidence["snapshot"]
    decision = choose_seed_count(
        pilot, config, manifest["splits"]["train"]["rows"],
        config_path=config_path, manifest_path=manifest_path,
        charged_seconds=snapshot["charged_seconds"], budget_snapshot=snapshot,
        require_provenance=True,
    )
    identity = evidence["identity"]
    _validate_code_identity(identity["code"])
    binding_body = {
        "pilot_path": _repo_relative(pilot_path.resolve()),
        "pilot_sha256": sha256(pilot_path),
        "pilot_admission_path": _repo_relative(evidence["admission_path"]),
        "pilot_admission_sha256": sha256(evidence["admission_path"]),
        "config_path": identity["config_path"],
        "config_sha256": identity["config_sha256"],
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "code": code_identity(),
        "budget_snapshot": snapshot,
        "decision": decision,
    }
    binding_sha = _mapping_sha256(binding_body)
    ledger = AggregateBudget(
        _ledger_path(config), identity,
        float(config["budget"]["main_max_aggregate_gpu_seconds"]),
    )
    hold_token, hold_snapshot = ledger.reserve_hold(
        FINAL_TEST_RESERVE_SECONDS,
        metadata={
            "command": FINAL_HOLD_COMMAND,
            "binding_sha256": binding_sha,
            "pilot_decision_path": _repo_relative(decision_path),
            "pilot_path": _repo_relative(pilot_path.resolve()),
            "pilot_sha256": sha256(pilot_path),
        },
        expected_snapshot=snapshot,
        required_additional_seconds=(
            decision["projected_gpu_seconds_per_seed"] * len(decision["selected_seeds"])
        ),
        decision_path=decision_path,
    )
    # Recheck the complete composition after the hold is durable.  The hold
    # remains charged even if this check fails.
    post_hold_decision = choose_seed_count(
        pilot, config, manifest["splits"]["train"]["rows"],
        config_path=config_path, manifest_path=manifest_path,
        charged_seconds=snapshot["charged_seconds"], budget_snapshot=snapshot,
        require_provenance=True,
    )
    if post_hold_decision != decision:
        raise ProtocolError("post-hold decision composition changed; immutable decision denied")
    total_after_hold = (
        hold_snapshot["charged_seconds"]
        + decision["projected_gpu_seconds_per_seed"] * len(decision["selected_seeds"])
    )
    if total_after_hold > float(config["budget"]["main_max_aggregate_gpu_seconds"]) + BUDGET_EPSILON:
        raise ProtocolError("post-hold full aggregate composition exceeds the limit; hold retained fail closed")
    hold_reservation = next(
        item for item in hold_snapshot["reservations"] if item.get("token") == hold_token
    )
    record = {
        "schema": PILOT_DECISION_SCHEMA,
        "pilot_path": binding_body["pilot_path"],
        "pilot_sha256": binding_body["pilot_sha256"],
        "pilot_admission_path": binding_body["pilot_admission_path"],
        "pilot_admission_sha256": binding_body["pilot_admission_sha256"],
        "config_path": identity["config_path"],
        "config_sha256": identity["config_sha256"],
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "code": code_identity(),
        "declared_seeds": list(config["training"]["default_seeds"]),
        "budget_ledger_path": _repo_relative(_ledger_path(config)),
        "budget_ledger_sha256": hold_snapshot["sha256"],
        "planning_snapshot": snapshot,
        "budget_snapshot": hold_snapshot,
        "final_budget_token": hold_token,
        "final_budget_reservation": hold_reservation,
        "final_budget_reservation_sha256": _mapping_sha256(hold_reservation),
        "final_budget_ledger_sha256": hold_snapshot["sha256"],
        "final_budget_snapshot": hold_snapshot,
        "pilot_admission": evidence["admission"],
        "decision": decision,
    }
    create_once_json(
        decision_path, record,
        exists_message="pilot decision already exists; immutable decisions cannot be replaced",
    )
    return record


def _validate_final_hold_decision(
    recorded: Mapping[str, Any], current_snapshot: Mapping[str, Any], decision_path: Path
) -> dict[str, Any]:
    token = recorded.get("final_budget_token")
    if not isinstance(token, str) or not token:
        raise ProtocolError("decision final budget token is missing")
    final_snapshot = recorded.get("final_budget_snapshot")
    if not isinstance(final_snapshot, Mapping):
        raise ProtocolError("decision final budget snapshot is missing")
    validate_budget_snapshot_prefix(final_snapshot, current_snapshot, mutable_tokens=(token,))
    if recorded.get("final_budget_ledger_sha256") != final_snapshot.get("sha256"):
        raise ProtocolError("decision final budget ledger binding differs")
    _validate_sha(recorded.get("final_budget_reservation_sha256"), "final reservation SHA-256")
    item = _reservation_from_snapshot(final_snapshot, token)
    if item is None or not budget_close(item.get("reserved_seconds"), FINAL_TEST_RESERVE_SECONDS):
        raise ProtocolError("decision final budget reservation is missing or not 600 seconds")
    if item.get("state") != "held":
        raise ProtocolError("decision final budget reservation was not held at creation")
    metadata = item.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("command") != FINAL_HOLD_COMMAND
        or metadata.get("pilot_decision_path") != _repo_relative(decision_path)
        or not HEX64.fullmatch(str(metadata.get("binding_sha256", "")))
    ):
        raise ProtocolError("decision final budget metadata is invalid")
    if recorded.get("final_budget_reservation") != item:
        raise ProtocolError("decision final budget reservation binding differs")
    if recorded.get("final_budget_reservation_sha256") != _mapping_sha256(item):
        raise ProtocolError("decision final budget reservation hash differs")
    return item


def validate_pilot_decision(
    pilot_path: Path,
    decision_path: Path,
    config_path: Path,
    manifest_path: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate every immutable input and cumulative budget composition."""
    validate_manifest(manifest, allow_pending=False)
    root = require_non_final_worker_open(config, route="validation")
    expected_manifest_path = repo_path(config["data_manifest"]).resolve()
    if manifest_path.resolve() != expected_manifest_path:
        raise ProtocolError("pilot manifest path is foreign to the supplied config")
    if pilot_path.resolve().parent != root or decision_path.resolve().parent != root:
        raise ProtocolError("pilot and decision paths must remain in the configured run root")
    if not pilot_path.is_file() or not decision_path.is_file():
        raise ProtocolError("future-run pilot or immutable decision is missing")
    recorded = load_json(decision_path)
    if set(recorded) != PILOT_DECISION_FIELDS or recorded.get("schema") != PILOT_DECISION_SCHEMA:
        raise ProtocolError("pilot decision schema is stale or foreign")
    pilot = load_json(pilot_path)
    evidence = _validate_pilot_artifact(pilot, pilot_path, config, config_path, manifest_path, manifest)
    identity = evidence["identity"]
    _validate_code_identity(recorded.get("code"))
    static = {
        "pilot_path": _repo_relative(pilot_path.resolve()),
        "pilot_sha256": sha256(pilot_path),
        "pilot_admission_path": _repo_relative(evidence["admission_path"]),
        "pilot_admission_sha256": sha256(evidence["admission_path"]),
        "config_path": identity["config_path"],
        "config_sha256": identity["config_sha256"],
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "code": code_identity(),
        "declared_seeds": list(config["training"]["default_seeds"]),
        "budget_ledger_path": _repo_relative(_ledger_path(config)),
    }
    for key, value in static.items():
        if recorded.get(key) != value:
            raise ProtocolError("pilot decision is stale or bound inputs differ")
    if recorded.get("pilot_admission") != evidence["admission"]:
        raise ProtocolError("pilot decision admission binding differs")
    planning_snapshot = recorded.get("planning_snapshot")
    recorded_snapshot = recorded.get("budget_snapshot")
    if not isinstance(planning_snapshot, Mapping) or not isinstance(recorded_snapshot, Mapping):
        raise ProtocolError("pilot decision budget/planning snapshot is missing")
    _validate_sha(recorded.get("budget_ledger_sha256"), "pilot decision budget ledger SHA-256")
    _validate_sha(recorded_snapshot.get("sha256"), "pilot decision budget snapshot SHA-256")
    validate_budget_snapshot_prefix(planning_snapshot, evidence["snapshot"])
    validate_budget_snapshot_prefix(
        recorded_snapshot, evidence["snapshot"], mutable_tokens=(recorded.get("final_budget_token"),),
    )
    if recorded.get("budget_ledger_sha256") != recorded_snapshot.get("sha256"):
        raise ProtocolError("pilot decision budget ledger binding differs")
    _validate_final_hold_decision(recorded, evidence["snapshot"], decision_path)
    expected_decision = choose_seed_count(
        pilot, config, manifest["splits"]["train"]["rows"],
        config_path=config_path, manifest_path=manifest_path,
        charged_seconds=planning_snapshot["charged_seconds"],
        budget_snapshot=planning_snapshot, require_provenance=True,
    )
    if recorded.get("decision") != expected_decision:
        raise ProtocolError("pilot decision is stale or its aggregate composition changed")
    return recorded["decision"]


def decision_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable identity portion bound into a freeze."""
    return _decision_identity(record)


def validate_seed_result(
    result: Mapping[str, Any],
    result_path: Path,
    decision: Mapping[str, Any],
    decision_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Reject stale/foreign seed evidence before checkpoint selection."""
    if manifest_path.resolve() != repo_path(config["data_manifest"]).resolve():
        raise ProtocolError("seed result manifest path is foreign to the supplied config")
    if not isinstance(result, Mapping) or result.get("schema") != TRAIN_RESULT_SCHEMA:
        raise ProtocolError("seed result schema is stale or foreign")
    if any(str(key).startswith("_") for key in result):
        raise ProtocolError("persisted seed result contains runtime-only metadata")
    seed = result.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or result.get("status") not in {"completed", "recovered"}:
        raise ProtocolError("seed result is not an ordinary or recovered completion")
    if seed not in decision.get("selected_seeds", []):
        raise ProtocolError("seed result is not authorized by the immutable pilot decision")
    if not RESULT_PROVENANCE_FIELDS <= set(result):
        raise ProtocolError("seed result provenance is incomplete")
    if result.get("selected_seed") != seed:
        raise ProtocolError("seed result selected-seed binding differs")
    if result.get("pilot_decision_path") != _repo_relative(decision_path.resolve()) or result.get("pilot_decision_sha256") != sha256(decision_path):
        raise ProtocolError("seed result pilot-decision provenance is stale or foreign")
    _validate_code_identity(result.get("code"))
    root = repo_path(config["output_root"], artifact_output=True)
    run_dir = root / f"seed_{seed}"
    if result_path.parent != run_dir:
        raise ProtocolError("seed result path is foreign")
    status = result["status"]
    expected_admission = run_dir / ("admission.json" if status == "completed" else "recovery_admission.json")
    if result.get("admission_path") != _repo_relative(expected_admission):
        raise ProtocolError("seed result admission path is foreign")
    admission_path = repo_path(result["admission_path"], artifact_output=True)
    admission = load_json(admission_path)
    admission_path, admission = _validate_provenance_paths(
        result, config=config, config_path=config_path, manifest_path=manifest_path,
        expected_command="train" if status == "completed" else "recover-seed",
        expected_seed=seed, expected_decision_path=decision_path,
    )
    if admission.get("pilot_decision_sha256") != sha256(decision_path):
        raise ProtocolError("seed admission decision binding differs")
    if status == "completed":
        requested = admission.get("requested_wall_seconds")
        if not budget_close(requested, decision.get("projected_gpu_seconds_per_seed")):
            raise ProtocolError("training admission allowance differs from the immutable pilot projection")
        context_path = run_dir / "training_context.json"
        if result.get("training_context_path") != _repo_relative(context_path) or result.get("training_context_sha256") != sha256(context_path):
            raise ProtocolError("seed result training-context provenance differs")
        if not context_path.is_file():
            raise ProtocolError("training context is missing")
        context = load_json(context_path)
        context_result = dict(result)
        context_result.pop("training_context_path", None)
        context_result.pop("training_context_sha256", None)
        if context.get("schema") != TRAINING_CONTEXT_SCHEMA or context.get("result") != context_result:
            raise ProtocolError("training context does not match the completed result")
        training_token = result["budget_token"]
        training_admission_sha = result["admission_sha256"]
    else:
        if not isinstance(result.get("recovery"), Mapping) or result["recovery"].get("authorized") is not True or result["recovery"].get("retraining") is not False:
            raise ProtocolError("recovered result lacks authorized non-retraining provenance")
        chain = _validate_recovery_chain_record(
            admission, config, config_path, manifest_path, seed, decision_path,
        )
        context_path = chain["training_context_path"]
        if result.get("training_context_path") != _repo_relative(context_path) or result.get("training_context_sha256") != sha256(context_path):
            raise ProtocolError("recovered result training-context provenance differs")
        training_token = chain["training_budget_token"]
        training_admission_sha = sha256(chain["training_admission_path"])
        if result.get("training_admission_path") != _repo_relative(chain["training_admission_path"]) or result.get("training_admission_sha256") != training_admission_sha or result.get("training_budget_token") != training_token:
            raise ProtocolError("recovered result training admission/token binding differs")
        result_checkpoint = result.get("checkpoint")
        if not isinstance(result_checkpoint, Mapping) or any(
            result_checkpoint.get(key) != chain["checkpoint"].get(key)
            for key in ("checkpoint_path", "weights_file_sha256", "training_state_file_sha256", "model_state_sha256", "reload_evidence_path", "reload_evidence_sha256")
        ):
            raise ProtocolError("recovered result checkpoint provenance differs from training chain")
    identity = budget_identity(config, config_path, manifest_path)
    snapshot = AggregateBudget(
        _ledger_path(config), identity, float(config["budget"]["main_max_aggregate_gpu_seconds"]),
    ).snapshot()
    metadata = result["budget_reservation"].get("metadata", {}) if isinstance(result.get("budget_reservation"), Mapping) else {}
    expected_metadata = {
        "command": "train",
        "seed": seed,
        "admission_sha256": training_admission_sha,
        "pilot_decision_sha256": sha256(decision_path),
    }
    if not isinstance(metadata, Mapping) or set(metadata) != set(expected_metadata) or any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ProtocolError("seed result budget reservation metadata is not exact")
    reservation = _validate_finished_reservation(
        snapshot, training_token, result["budget_reservation"],
        expected_metadata=expected_metadata,
    )
    if status == "completed" and not budget_close(reservation["reserved_seconds"], decision.get("projected_gpu_seconds_per_seed")):
        raise ProtocolError("seed result budget reservation differs from the pilot projection")
    if status == "recovered" and reservation.get("token") != training_token:
        raise ProtocolError("recovered result did not retain the genuine training token")
    return dict(result)


def validate_freeze_binding(
    frozen: Mapping[str, Any],
    freeze_path: Path,
    decision_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate the non-result portion of a dev freeze before selection."""
    if not isinstance(frozen, Mapping) or frozen.get("schema") != FREEZE_SCHEMA:
        raise ProtocolError("historical freeze is read-only and non-executable; a provenance-bound freeze is required")
    if manifest_path.resolve() != repo_path(config["data_manifest"]).resolve():
        raise ProtocolError("freeze manifest path is foreign to the supplied config")
    if frozen.get("test_opened") is not False or not isinstance(frozen.get("candidates"), list):
        raise ProtocolError("freeze is not a dev-only checkpoint selection")
    if frozen.get("config_path") != _repo_relative(config_path.resolve()) or frozen.get("config_sha256") != sha256(config_path):
        raise ProtocolError("freeze config provenance does not match")
    if frozen.get("manifest_path") != _repo_relative(manifest_path.resolve()) or frozen.get("manifest_sha256") != sha256(manifest_path):
        raise ProtocolError("freeze manifest provenance does not match")
    _validate_code_identity(frozen.get("code"))
    if frozen.get("pilot_decision_path") != _repo_relative(decision_path.resolve()) or frozen.get("pilot_decision_sha256") != sha256(decision_path):
        raise ProtocolError("freeze pilot-decision provenance does not match")
    recorded = load_json(decision_path)
    if frozen.get("decision_identity") != decision_identity(recorded):
        raise ProtocolError("freeze decision identity does not match")
    result_hashes = frozen.get("result_artifact_sha256")
    selected = recorded.get("decision", {}).get("selected_seeds", [])
    if not isinstance(result_hashes, Mapping) or set(result_hashes) != {str(seed) for seed in selected}:
        raise ProtocolError("freeze result hash set does not match the decision")
    root = repo_path(config["output_root"], artifact_output=True)
    for seed in selected:
        path = root / f"seed_{seed}" / "result.json"
        if not path.is_file() or result_hashes.get(str(seed)) != sha256(path):
            raise ProtocolError("freeze exact result hash is stale or foreign")
    return dict(frozen)


def validate_final_result(
    result: Mapping[str, Any],
    result_path: Path,
    freeze_path: Path,
    decision_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate authenticated final provenance and its finished hold."""
    if not isinstance(result, Mapping) or result.get("schema") != FINAL_RESULT_SCHEMA:
        raise ProtocolError("final result schema is invalid")
    if result_path.name != "final_test.result.json" or result_path.parent != repo_path(config["output_root"], artifact_output=True):
        raise ProtocolError("final result path is foreign")
    required = {
        "status", "attempt_marker", "attempt_marker_sha256", "selection",
        "freeze_path", "freeze_sha256", "pilot_decision_path", "pilot_decision_sha256",
        "config_path", "config_sha256", "manifest_path", "manifest_sha256", "code",
        "budget_ledger_path", "budget_ledger_sha256", "budget_token", "budget_reservation",
    }
    if not required <= set(result) or result.get("status") != "final_test_completed":
        raise ProtocolError("final result provenance is incomplete")
    if result.get("freeze_path") != _repo_relative(freeze_path) or result.get("freeze_sha256") != sha256(freeze_path):
        raise ProtocolError("final result freeze binding differs")
    if result.get("pilot_decision_path") != _repo_relative(decision_path) or result.get("pilot_decision_sha256") != sha256(decision_path):
        raise ProtocolError("final result decision binding differs")
    _validate_code_identity(result.get("code"))
    if result.get("config_path") != _repo_relative(config_path) or result.get("config_sha256") != sha256(config_path):
        raise ProtocolError("final result config binding differs")
    if result.get("manifest_path") != _repo_relative(manifest_path) or result.get("manifest_sha256") != sha256(manifest_path):
        raise ProtocolError("final result manifest binding differs")
    marker = repo_path(result["attempt_marker"], artifact_output=True)
    if marker.name != "final_test.attempt.json" or not marker.is_file() or result.get("attempt_marker_sha256") != sha256(marker):
        raise ProtocolError("final attempt marker hash is stale or foreign")
    marker_record = load_json(marker)
    for key in ("freeze_sha256", "pilot_decision_path", "pilot_decision_sha256", "config_path", "config_sha256", "manifest_path", "manifest_sha256", "code", "budget_ledger_path", "budget_token", "selection"):
        if marker_record.get(key) != result.get(key):
            raise ProtocolError(f"final marker/result provenance differs for {key}")
    marker_reservation = marker_record.get("budget_reservation")
    if not isinstance(marker_reservation, Mapping) or marker_reservation.get("state") != "in_flight":
        raise ProtocolError("final marker does not bind the activated in-flight reservation")
    ledger = AggregateBudget(
        _ledger_path(config), budget_identity(config, config_path, manifest_path),
        float(config["budget"]["main_max_aggregate_gpu_seconds"]),
    )
    snapshot = ledger.snapshot()
    reservation = _validate_finished_reservation(
        snapshot, result["budget_token"], result["budget_reservation"],
        expected_metadata={
            "command": FINAL_HOLD_COMMAND,
            "freeze_sha256": result["freeze_sha256"],
            "pilot_decision_path": result["pilot_decision_path"],
            "pilot_decision_sha256": result["pilot_decision_sha256"],
        },
    )
    if result.get("budget_ledger_path") != _repo_relative(ledger.path) or result.get("budget_ledger_sha256") != snapshot["sha256"]:
        raise ProtocolError("final result ledger binding differs")
    if marker_reservation.get("token") != result["budget_token"]:
        raise ProtocolError("final marker token differs from result")
    if not isinstance(result.get("selection"), Mapping) or result["selection"] != marker_record.get("selection"):
        raise ProtocolError("final selection binding differs")
    return dict(result)


class AggregateBudget:
    """Conservative pre-admission wall-time accounting under a cross-process lock.

    A reservation is charged in full before a worker starts and is never reduced
    after success or failure.  ``reserve_up_to`` never makes a partial grant: it
    returns the complete request or rejects admission.  This is not a hard
    wall-clock kill switch; a blocking operation already in progress can only be
    stopped at a later safe boundary and its actual overrun is recorded.
    """

    def __init__(self, path: Path, identity: Mapping[str, Any], limit: float = FROZEN_GPU_SECONDS):
        self.path, self.identity, self.limit = path, dict(identity), limit
        self.limit = finite_runtime(self.limit, "aggregate budget limit")
        if self.limit <= 0:
            raise ProtocolError("aggregate budget limit must be positive")
        self.lock = path.with_suffix(path.suffix + ".lock")

    def reserve(self, seconds: float, *, metadata: Mapping[str, Any] | None = None) -> str:
        token, _ = self._reserve_full(seconds, metadata=metadata)
        return token

    def reserve_up_to(self, requested: float, *, metadata: Mapping[str, Any] | None = None) -> tuple[str, float]:
        # Retain the historical API name, but use all-or-nothing pre-admission.
        return self._reserve_full(requested, metadata=metadata)

    def reserve_non_final(
        self,
        requested: float,
        *,
        metadata: Mapping[str, Any] | None = None,
        run_root: Path,
    ) -> tuple[str, float]:
        """Reserve only while the run has no decision/final hold, atomically."""
        return self._reserve_full(requested, metadata=metadata, blocked_root=run_root)

    def reserve_hold(
        self,
        seconds: float,
        *,
        metadata: Mapping[str, Any] | None = None,
        expected_snapshot: Mapping[str, Any] | None = None,
        required_additional_seconds: float = 0.0,
        decision_path: Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Atomically create a permanent held charge for a future route.

        ``expected_snapshot`` is a compare-and-swap guard.  If another route
        appended a charge after planning, no hold is created.  Once appended,
        the hold is deliberately left in place even if the post-hold
        composition check fails: an ambiguous plan must never release capacity
        that another process may already have observed as reserved.
        """
        requested = finite_runtime(seconds, "held reservation")
        additional = finite_runtime(required_additional_seconds, "held reservation composition")
        if requested <= 0 or additional < 0:
            raise ProtocolError("held reservation and composition must be valid")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if decision_path is not None and decision_path.exists():
                raise ProtocolError("pilot decision already exists; immutable decision cannot be replaced")
            data = self._load_locked()
            self._validate(data)
            if data["identity"] != self.identity or data["limit"] != self.limit:
                raise ProtocolError("aggregate budget identity changed; refuse admission")
            current = self._snapshot_locked(data)
            if expected_snapshot is not None:
                expected = dict(expected_snapshot)
                if (
                    current.get("identity") != expected.get("identity")
                    or current.get("limit") != expected.get("limit")
                    or current.get("reservations") != expected.get("reservations")
                    or not budget_close(current.get("charged_seconds"), expected.get("charged_seconds"))
                ):
                    raise ProtocolError("budget changed while creating the immutable decision; fail closed")
            used = self._charged_seconds(data)
            if used + requested > self.limit + BUDGET_EPSILON:
                raise ProtocolError("aggregate GPU-second budget cannot hold the final reserve")
            import uuid
            token = uuid.uuid4().hex
            reservation: dict[str, Any] = {
                "token": token,
                "reserved_seconds": requested,
                "state": "held",
            }
            if metadata is not None:
                if not isinstance(metadata, Mapping):
                    raise ProtocolError("held reservation metadata must be a mapping")
                reservation["metadata"] = dict(metadata)
            data["reservations"].append(reservation)
            self._write(data)
            after = self._snapshot_locked(self._load_locked())
            # Recheck the complete composition after the hold is durable.  A
            # failure intentionally leaves the held charge in place.
            if after["charged_seconds"] + additional > self.limit + BUDGET_EPSILON:
                raise ProtocolError("post-hold aggregate composition is ambiguous or over limit; hold retained fail closed")
            return token, after

    # Descriptive aliases used by callers and review probes.
    hold = reserve_hold
    reserve_final_hold = reserve_hold

    def activate_hold(
        self,
        token: str,
        *,
        expected_seconds: float = FINAL_TEST_RESERVE_SECONDS,
        metadata_updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transition one held charge to in-flight without adding a charge."""
        if not isinstance(token, str) or not token:
            raise ProtocolError("final budget hold token is invalid")
        expected_seconds = finite_runtime(expected_seconds, "final budget hold seconds")
        with self.lock.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = self._load_locked()
            self._validate(data)
            if data["identity"] != self.identity or data["limit"] != self.limit:
                raise ProtocolError("aggregate budget identity changed")
            for item in data["reservations"]:
                if item.get("token") != token:
                    continue
                if item.get("state") != "held":
                    raise ProtocolError("final budget hold is not held; activation/retry is forbidden")
                if not budget_close(item.get("reserved_seconds"), expected_seconds):
                    raise ProtocolError("final budget hold allowance is not the fixed 600 seconds")
                if metadata_updates is not None:
                    if not isinstance(metadata_updates, Mapping):
                        raise ProtocolError("final budget metadata update must be a mapping")
                    metadata = dict(item.get("metadata", {}))
                    updates = dict(metadata_updates)
                    if any(key in metadata and metadata[key] != value for key, value in updates.items()):
                        raise ProtocolError("final budget metadata cannot rewrite bound provenance")
                    metadata.update(updates)
                    try:
                        json.dumps(metadata, sort_keys=True)
                    except (TypeError, ValueError) as exc:
                        raise ProtocolError("final budget metadata is not JSON serializable") from exc
                    item["metadata"] = metadata
                item["state"] = "in_flight"
                self._write(data)
                return next(dict(value) for value in data["reservations"] if value.get("token") == token)
            raise ProtocolError("final budget hold token is unknown")

    def _snapshot_locked(self, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "path": _repo_relative(self.path.resolve()),
            "sha256": sha256(self.path) if self.path.is_file() else None,
            "identity": dict(data["identity"]),
            "limit": data["limit"],
            "charged_seconds": self._charged_seconds(data),
            "reservations": [dict(item) for item in data["reservations"]],
        }

    def _reserve_full(
        self,
        requested: float,
        *,
        metadata: Mapping[str, Any] | None = None,
        blocked_root: Path | None = None,
    ) -> tuple[str, float]:
        import uuid

        requested = finite_runtime(requested, "requested reservation")
        if requested <= 0 or requested > self.limit:
            raise ProtocolError("reservation must be positive and no greater than the aggregate limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = self._load_locked()
            self._validate(data)
            if data["identity"] != self.identity or data["limit"] != self.limit:
                raise ProtocolError("aggregate budget identity changed; refuse admission")
            if blocked_root is not None:
                blocked_root = blocked_root.resolve()
                if (blocked_root / "pilot_decision.json").exists() or _ledger_has_final_hold(blocked_root):
                    raise ProtocolError("immutable pilot decision/final budget hold already exists; non-final route denied")
            used = self._charged_seconds(data)
            remaining = self.limit - used
            if remaining < requested and not budget_close(remaining, requested):
                raise ProtocolError(
                    f"aggregate GPU-second budget exhausted or insufficient: {remaining:.6f} seconds remain; "
                    f"full request of {requested:.6f} seconds denied; no partial grant or retry"
                )
            token = uuid.uuid4().hex
            reservation: dict[str, Any] = {
                "token": token,
                "reserved_seconds": requested,
                "state": "in_flight",
            }
            if metadata is not None:
                if not isinstance(metadata, Mapping):
                    raise ProtocolError("budget reservation metadata must be a mapping")
                try:
                    json.dumps(dict(metadata), sort_keys=True)
                except (TypeError, ValueError) as exc:
                    raise ProtocolError("budget reservation metadata is not JSON serializable") from exc
                reservation["metadata"] = dict(metadata)
            data["reservations"].append(reservation)
            self._write(data)
            return token, requested

    def finish(self, token: str, actual_seconds: float, *, outcome: str = "finished") -> None:
        actual_seconds = finite_runtime(actual_seconds, "actual worker seconds")
        if outcome not in {"finished", "failed"}:
            raise ProtocolError("invalid reservation outcome")
        with self.lock.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = self._load_locked()
            self._validate(data)
            if data["identity"] != self.identity or data["limit"] != self.limit:
                raise ProtocolError("aggregate budget identity changed")
            for item in data["reservations"]:
                if item["token"] == token:
                    if item["state"] != "in_flight":
                        raise ProtocolError("reservation already finalized")
                    state = "overrun" if actual_seconds > item["reserved_seconds"] else outcome
                    item.update(state=state, actual_seconds=actual_seconds)
                    self._write(data)
                    if state == "overrun":
                        raise ProtocolError("worker exceeded its full reservation; overrun durably charged")
                    return
            raise ProtocolError("unknown aggregate reservation")

    def snapshot(self) -> dict[str, Any]:
        """Return a locked semantic snapshot of the current ledger."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = self._load_locked()
            self._validate(data)
            if data["identity"] != self.identity or data["limit"] != self.limit:
                raise ProtocolError("aggregate budget identity changed")
            return self._snapshot_locked(data)

    def reservation_snapshot(self, token: str, *, require_finished: bool = False) -> dict[str, Any]:
        """Return one exact reservation, optionally requiring finalization."""
        if not isinstance(token, str) or not token:
            raise ProtocolError("aggregate reservation token is invalid")
        with self.lock.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = self._load_locked()
            self._validate(data)
            if data["identity"] != self.identity or data["limit"] != self.limit:
                raise ProtocolError("aggregate budget identity changed")
            for item in data["reservations"]:
                if item["token"] == token:
                    if require_finished and item["state"] != "finished":
                        raise ProtocolError("aggregate reservation is not finished")
                    return dict(item)
            raise ProtocolError("unknown aggregate reservation")

    @staticmethod
    def _charged_seconds(data: Mapping[str, Any]) -> float:
        # Unresolved (in_flight) reservations remain fully charged.  Once an
        # actual duration is known, an overrun replaces—not adds to—the grant.
        return sum(
            max(item["reserved_seconds"], item.get("actual_seconds", 0.0))
            for item in data["reservations"]
        )

    def _load_locked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema": "i3-gpu-budget-v1",
                "identity": self.identity,
                "limit": self.limit,
                "reservations": [],
            }
        try:
            data = load_json(self.path)
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise ProtocolError("malformed aggregate budget ledger; admission denied") from exc
        return data

    def _write(self, data: Mapping[str, Any]) -> None:
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate(self, data: Any) -> None:
        if (
            not isinstance(data, dict)
            or set(data) != {"schema", "identity", "limit", "reservations"}
            or data["schema"] != "i3-gpu-budget-v1"
            or not isinstance(data["identity"], dict)
            or not isinstance(data["reservations"], list)
        ):
            raise ProtocolError("malformed aggregate budget ledger; admission denied")
        if (
            isinstance(data["limit"], bool)
            or not isinstance(data["limit"], (int, float))
            or not math.isfinite(data["limit"])
            or data["limit"] != self.limit
        ):
            raise ProtocolError("invalid ledger limit")
        tokens: set[str] = set()
        total = 0.0
        for item in data["reservations"]:
            if (
                not isinstance(item, dict)
                or set(item) - {"token", "reserved_seconds", "state", "actual_seconds", "metadata"}
                or not {"token", "reserved_seconds", "state"} <= set(item)
                or item["state"] not in {"held", "in_flight", "finished", "failed", "overrun"}
                or ("metadata" in item and not isinstance(item["metadata"], dict))
            ):
                raise ProtocolError("malformed reservation")
            if not isinstance(item["token"], str) or not item["token"] or item["token"] in tokens:
                raise ProtocolError("duplicate or malformed reservation token")
            tokens.add(item["token"])
            reserve = finite_runtime(item["reserved_seconds"], "ledger reservation")
            actual = finite_runtime(item.get("actual_seconds", 0.0), "ledger actual")
            has_actual = "actual_seconds" in item
            if (
                reserve <= 0
                or (item["state"] in {"held", "in_flight"} and has_actual)
                or (item["state"] not in {"held", "in_flight"} and not has_actual)
                or (item["state"] in {"finished", "failed"} and actual > reserve)
                or (item["state"] == "overrun" and actual <= reserve)
            ):
                raise ProtocolError("inconsistent reservation state")
            total += max(reserve, actual)
        if total > self.limit + BUDGET_EPSILON and not any(item["state"] == "overrun" for item in data["reservations"]):
            raise ProtocolError("ledger is over limit; admission denied")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"{path} must contain a JSON object")
    return value


# Capture the loaded source set once, after the module has been fully defined.
# The public alias is intentionally mutable only for isolated adversarial tests;
# production callers should use ``import_time_code_identity()``.
_IMPORT_TIME_CODE_IDENTITY = code_identity()
IMPORT_TIME_CODE_IDENTITY = dict(_IMPORT_TIME_CODE_IDENTITY)
