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
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FIXED_LABELS = ["read", "search", "edit", "execute", "other_tool", "respond_or_finish"]
CHOICE_OPTION_ORDER = ["edit", "execute", "other_tool", "read", "respond_or_finish", "search"]
LABEL_TO_CHOICE_INDEX = {label: CHOICE_OPTION_ORDER.index(label) for label in FIXED_LABELS}
FROZEN_ITERATION_SECONDS = 14400
FROZEN_GPU_SECONDS = 7200
FROZEN_LATENCY_PHASES = ["tokenize_build", "collate", "host_to_device", "forward", "softmax", "cpu_probability_copy"]


class ProtocolError(ValueError):
    """Configuration or manifest violates the predeclared protocol."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str, *, artifact_output: bool = False) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ProtocolError("paths must be non-empty repository-relative strings")
    resolved = (ROOT / value).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ProtocolError(f"path escapes repository: {value}")
    if artifact_output and resolved != OWNED_ARTIFACT_ROOT and OWNED_ARTIFACT_ROOT not in resolved.parents:
        raise ProtocolError(f"output path escapes owned artifact root: {value}")
    return resolved


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


def consume_final_test_attempt(run_dir: Path, freeze_sha256: str) -> Path:
    """Atomically consume the sole test attempt before callers may open test."""
    resolved = run_dir.resolve()
    if resolved != OWNED_ARTIFACT_ROOT and OWNED_ARTIFACT_ROOT not in resolved.parents:
        raise ProtocolError("run directory escapes owned artifact root")
    freeze = resolved / "freeze.json"
    if not HEX64.fullmatch(freeze_sha256) or not freeze.is_file() or sha256(freeze) != freeze_sha256:
        raise ProtocolError("dev-only freeze artifact is absent or hash-mismatched")
    marker = resolved / "final_test.attempt.json"
    payload = {"freeze_sha256": freeze_sha256, "consumed_unix_seconds": time.time(), "policy": "single attempt; no retry"}
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolError("final test attempt was already consumed; retry forbidden") from exc
    return marker


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


def choose_seed_count(smoke: Mapping[str, Any], config: Mapping[str, Any], train_rows: int) -> dict[str, Any]:
    """Apply the sole predeclared fallback using a completed measured smoke."""
    required = {"seed", "updates", "training_seconds", "cold_load_seconds", "dev_evaluation_seconds", "peak_reserved_gib", "finite", "status", "test_opened"}
    if (
        set(smoke) != required
        or smoke["status"] != "actual_model_pilot_measured"
        or smoke["test_opened"] is not False
        or smoke["seed"] != config["budget"]["smoke_seed"]
        or smoke["updates"] != config["budget"]["smoke_updates"]
    ):
        raise ProtocolError("smoke evidence does not match the predeclared profile")
    numeric = [smoke[key] for key in ("training_seconds", "cold_load_seconds", "dev_evaluation_seconds", "peak_reserved_gib")]
    if smoke["finite"] is not True or not all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in numeric):
        raise ProtocolError("smoke evidence is non-finite or unsuccessful")
    if smoke["peak_reserved_gib"] > config["budget"]["max_peak_reserved_gib"] or smoke["training_seconds"] > config["budget"]["smoke_max_gpu_seconds"]:
        raise ProtocolError("smoke exceeded memory or time budget; main training is forbidden")
    updates = expected_updates(train_rows, config)
    per_seed = (smoke["training_seconds"] / smoke["updates"] * updates + smoke["cold_load_seconds"] + smoke["dev_evaluation_seconds"]) * config["budget"]["projection_multiplier"]
    two_seed_projection = per_seed * len(config["training"]["default_seeds"])
    fallback = two_seed_projection > config["budget"]["main_max_aggregate_gpu_seconds"]
    return {
        "expected_updates_per_seed": updates,
        "projected_gpu_seconds_per_seed": per_seed,
        "projected_gpu_seconds_two_seeds": two_seed_projection,
        "fallback_applied": fallback,
        "selected_seeds": [config["training"]["single_seed_fallback"]] if fallback else config["training"]["default_seeds"],
        "rule": "single seed only when measured conservative two-seed projection exceeds 7200 GPU-seconds",
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"{path} must contain a JSON object")
    return value
