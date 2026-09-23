#!/usr/bin/env python3
"""CPU-only preflight entry point for the gated Iteration 3 Laya experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

BOOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOT_ROOT))

from training.laya_trace_i3.protocol import (
    ROOT,
    HISTORICAL_RUN_ROOT,
    ProtocolError,
    admit_worker as _protocol_admit_worker,
    code_identity,
    create_pilot_decision,
    create_once_json,
    decision_identity,
    expected_updates,
    future_run_paths as _protocol_future_run_paths,
    group_bootstrap_ids,
    load_json,
    repo_path,
    require_exact_seed_set,
    require_non_final_worker_open,
    select_dev_checkpoint,
    sha256,
    validate_config,
    validate_freeze_binding,
    validate_manifest,
    validate_pilot_decision,
    validate_runtime_budget,
    validate_seed_result,
    verify_split_files,
)
from training.laya_trace_i3.experiment import (
    final_test_once, load_split, planned_run, run_actual_model_pilot, run_base_eval,
    run_bounded_smoke, run_final_test, run_recover_seed, run_training_seed, six_class_rlcd_reference, write_plan,
)
DEFAULT_CONFIG = ROOT / "training/laya_trace_i3/config.json"


def future_run_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    return _protocol_future_run_paths(config, path_resolver=repo_path)


def require_compute_open(config: dict[str, Any], *, route: str | None = None) -> None:
    """Compatibility wrapper around the shared cumulative worker guard."""
    require_non_final_worker_open(config, route=route, path_resolver=repo_path)


def admit_worker(
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    command: str,
    *,
    seed: int | None = None,
    pilot_decision_path: Path | None = None,
    requested_wall_seconds: float | None = None,
) -> Path:
    return _protocol_admit_worker(
        config,
        config_path,
        manifest_path,
        command,
        seed=seed,
        pilot_decision_path=pilot_decision_path,
        requested_wall_seconds=requested_wall_seconds,
    )


def persist_seed_result(run_dir: Path, result: dict[str, Any]) -> None:
    """Publish a seed result once; never replace an existing result identity."""
    create_once_json(
        run_dir / "result.json",
        result,
        exists_message="seed result already exists; replacement/retry is forbidden",
    )


def load_dev_candidates(
    config: dict[str, Any],
    manifest: dict[str, Any],
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load only complete, provenance-bound results selected by the decision."""
    pilot_path, decision_path = future_run_paths(config)
    manifest_path = repo_path(config["data_manifest"])
    decision = validate_pilot_decision(
        pilot_path, decision_path, config_path, manifest_path, config, manifest
    )
    runs_root = repo_path(config["output_root"], artifact_output=True)
    result_paths = list(runs_root.glob("seed_*/result.json"))
    result_by_seed: dict[int, Path] = {}
    for result_path in result_paths:
        try:
            seed = int(result_path.parent.name.removeprefix("seed_"))
        except ValueError as exc:
            raise ProtocolError(f"unexpected dev result directory: {result_path.relative_to(ROOT)}") from exc
        if seed in result_by_seed:
            raise ProtocolError(f"duplicate dev result seed: {seed}")
        result_by_seed[seed] = result_path
    require_exact_seed_set(result_by_seed, decision["selected_seeds"])

    candidates = []
    for seed in decision["selected_seeds"]:
        result_path = result_by_seed[seed]
        result = validate_seed_result(
            load_json(result_path),
            result_path,
            decision,
            decision_path,
            config,
            config_path,
            manifest_path,
        )
        checkpoint = result.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ProtocolError("seed result checkpoint provenance is incomplete")
        expected_seed_root = (runs_root / f"seed_{seed}").resolve()
        checkpoint_path = repo_path(checkpoint["checkpoint_path"], artifact_output=True)
        if checkpoint_path != expected_seed_root / "checkpoint":
            raise ProtocolError("result checkpoint must belong to its selected run seed")
        candidates.append({"seed": seed, "split": "dev", "macro_f1": result["dev"]["macro_f1"], "nll": result["dev"]["nll"], "checkpoint_path": checkpoint["checkpoint_path"], "weights_file_sha256": checkpoint["weights_file_sha256"], "model_state_sha256": checkpoint["model_state_sha256"], "training_state_file_sha256": checkpoint["training_state_file_sha256"], "labels": config["labels"], "reload_evidence": checkpoint["reload_evidence"], "reload_evidence_path": checkpoint["reload_evidence_path"], "reload_evidence_sha256": checkpoint["reload_evidence_sha256"]})
    return decision, candidates


def build_report(config_path: Path, *, verify_accepted_files: bool = False, supplied_manifest_hash: str | None = None) -> dict[str, Any]:
    if "torch" in sys.modules:
        raise ProtocolError("preflight must run before Torch is imported")
    config = load_json(config_path)
    validate_config(config)
    manifest_path = repo_path(config["data_manifest"])
    manifest = load_json(manifest_path)
    status = validate_manifest(manifest, allow_pending=not verify_accepted_files)
    report: dict[str, Any] = {
        "status": "blocked_pending_data_acceptance" if status == "pending" else "validated_no_model_load",
        "config_path": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "manifest_status": status,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "model_loaded": False,
        "test_opened": False,
        "fixed_labels": config["labels"],
        "default_seeds": config["training"]["default_seeds"],
        "single_seed_fallback_rule": config["budget"]["fallback_rule"],
    }
    if verify_accepted_files:
        if supplied_manifest_hash != report["manifest_sha256"]:
            raise ProtocolError("accepted manifest SHA-256 must be supplied explicitly and match")
        report["verified_files"] = verify_split_files(manifest, include_test=False)
        report["expected_updates_per_seed"] = expected_updates(manifest["splits"]["train"]["rows"], config)
    return report


def build_repair_report(config_path: Path) -> dict[str, Any]:
    """Capture CPU-only repair proofs without opening test or importing Torch."""
    config = load_json(config_path)
    validate_config(config)
    manifest = load_json(repo_path(config["data_manifest"]))
    validate_manifest(manifest, allow_pending=False)
    perturbations = [
        [0.3, -0.1, 0.2, -0.4, 0.1, -0.1],
        [-0.4, 0.8, -0.6, 0.2, -0.2, 0.2],
        [0.05, 0.1, -0.2, -0.05, 0.15, -0.05],
        [-0.9, -0.6, 0.3, 1.2, -0.3, 0.3],
    ]
    objective = {}
    for label in config["labels"]:
        proof = six_class_rlcd_reference([0.2, -0.1, 0.4, 0.0, -0.3, 0.1], label, perturbations, 0.4)
        objective[label] = {
            "gold_choice_index": proof["gold_choice_index"],
            "loss": proof["loss"],
            "loss_rlcd": proof["loss_rlcd"],
            "rlcd_gradient_l1": sum(abs(value) for value in proof["rlcd_score_gradient"]),
            "gradient_l1": sum(abs(value) for value in proof["choice_logit_gradient"]),
        }
    groups = {}
    for split in ("train", "dev"):
        proof = group_bootstrap_ids(load_split(manifest, split, list(config["labels"])))
        encoded_ids = "\n".join(proof["group_ids"]).encode()
        groups[split] = {
            "bootstrap_unit": proof["bootstrap_unit"],
            "rows": proof["rows"],
            "trajectories": proof["trajectories"],
            "task_groups": proof["task_groups"],
            "sorted_group_ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
        }
    return {
        "status": "repair_preflight_passed",
        "config_sha256": sha256(config_path),
        "manifest_sha256": sha256(repo_path(config["data_manifest"])),
        "choice_option_order": config["training"]["choice_option_order"],
        "label_to_choice_index": config["training"]["label_to_choice_index"],
        "objective_proof": objective,
        "group_bootstrap_proof": groups,
        "torch_imported": "torch" in sys.modules,
        "model_loaded": False,
        "test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-config", "dry-run", "validate-accepted-manifest", "repair-preflight", "actual-model-pilot", "record-pilot-decision", "base-eval", "train", "recover-seed", "dev-select", "final-test", "smoke"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--review-accepted", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    if args.command in {"actual-model-pilot", "base-eval", "train", "recover-seed", "smoke"} and not args.review_accepted:
        raise ProtocolError(f"{args.command} requires explicit independent review authorization")
    if args.command == "validate-config":
        config = load_json(config_path)
        validate_config(config)
        report = {"status": "config_valid", "config_sha256": sha256(config_path), "torch_imported": False, "model_loaded": False}
    elif args.command in {"dry-run", "validate-accepted-manifest"}:
        report = build_report(config_path, verify_accepted_files=args.command == "validate-accepted-manifest", supplied_manifest_hash=args.manifest_sha256)
    elif args.command == "repair-preflight":
        report = build_repair_report(config_path)
    elif args.command == "actual-model-pilot":
        config = load_json(config_path)
        validate_config(config)
        require_compute_open(config, route="actual-model-pilot")
        manifest_path = repo_path(config["data_manifest"])
        manifest = load_json(manifest_path)
        pilot_path, _ = future_run_paths(config)
        admission_path = admit_worker(
            config,
            config_path,
            manifest_path,
            "actual-model-pilot",
            seed=config["budget"]["smoke_seed"],
            requested_wall_seconds=600.0,
        )
        report = run_actual_model_pilot(
            config,
            manifest,
            review_accepted=args.review_accepted,
            config_path=config_path,
            admission_path=admission_path,
        )
        report = {
            key: value
            for key, value in report.items()
            if not key.startswith("_") and key != "worker_wall_seconds"
        }
        create_once_json(
            pilot_path,
            report,
            exists_message="future actual-model pilot already exists; immutable result cannot be replaced",
        )
    elif args.command == "record-pilot-decision":
        config = load_json(config_path)
        validate_config(config)
        require_compute_open(config, route="validation")
        manifest_path = repo_path(config["data_manifest"])
        manifest = load_json(manifest_path)
        pilot_path, decision_path = future_run_paths(config)
        record = create_pilot_decision(pilot_path, decision_path, config_path, manifest_path, config, manifest)
        report = {
            "status": "pilot_decision_frozen",
            "selected_seeds": record["decision"]["selected_seeds"],
            "charged_seconds_before_training": record["decision"]["charged_seconds_before_training"],
            "fixed_final_test_reserve_seconds": record["decision"]["fixed_final_test_reserve_seconds"],
            "aggregate_composition": record["decision"]["aggregate_composition"],
            "budget_ledger_sha256": record["budget_ledger_sha256"],
            "test_opened": False,
        }
    elif args.command == "base-eval":
        config = load_json(config_path)
        validate_config(config)
        require_compute_open(config, route="base-eval")
        manifest_path = repo_path(config["data_manifest"])
        manifest = load_json(manifest_path)
        admission_path = admit_worker(config, config_path, manifest_path, "base-eval", requested_wall_seconds=600.0)
        report = run_base_eval(
            config,
            manifest,
            review_accepted=args.review_accepted,
            config_path=config_path,
            admission_path=admission_path,
        )
        report = {key: value for key, value in report.items() if not key.startswith("_") and key != "worker_wall_seconds"}
        create_once_json(
            repo_path(config["output_root"], artifact_output=True) / "base_eval.result.json",
            report,
            exists_message="base evaluation result already exists; retry is forbidden",
        )
    elif args.command == "train":
        config = load_json(config_path)
        validate_config(config)
        require_compute_open(config, route="train")
        if args.seed is None:
            raise ProtocolError("train requires exactly one explicit frozen --seed per process")
        manifest_path = repo_path(config["data_manifest"])
        manifest = load_json(manifest_path)
        pilot_path, decision_path = future_run_paths(config)
        decision = validate_pilot_decision(
            pilot_path,
            decision_path,
            config_path,
            manifest_path,
            config,
            manifest,
        )
        if args.seed not in decision["selected_seeds"]:
            raise ProtocolError("seed is not authorized by the immutable pilot decision")
        admission_path = admit_worker(
            config,
            config_path,
            manifest_path,
            "train",
            seed=args.seed,
            pilot_decision_path=decision_path,
            requested_wall_seconds=decision["projected_gpu_seconds_per_seed"],
        )
        item = run_training_seed(
            config,
            manifest,
            args.seed,
            review_accepted=args.review_accepted,
            reservation_seconds=decision["projected_gpu_seconds_per_seed"],
            config_path=config_path,
            admission_path=admission_path,
            pilot_decision_path=decision_path,
        )
        if item.get("seed") != args.seed or item.get("status") not in {"completed", "recovered"}:
            raise ProtocolError("training worker returned an invalid seed result")
        run_dir = repo_path(config["output_root"], artifact_output=True) / f"seed_{args.seed}"
        public_item = {key: value for key, value in item.items() if not key.startswith("_") and key != "worker_wall_seconds"}
        persist_seed_result(run_dir, public_item)
        report = {"status": "training_completed", "seeds": [args.seed], "reports": [public_item], "test_opened": False}
    elif args.command == "recover-seed":
        if args.seed is None:
            raise ProtocolError("recover-seed requires an explicit --seed")
        config = load_json(config_path)
        validate_config(config)
        require_compute_open(config, route="recover-seed")
        if args.seed != config["training"]["single_seed_fallback"]:
            raise ProtocolError("recover-seed is authorized only for seed 42")
        manifest_path = repo_path(config["data_manifest"])
        manifest = load_json(manifest_path)
        pilot_path, decision_path = future_run_paths(config)
        decision = validate_pilot_decision(
            pilot_path, decision_path, config_path, manifest_path, config, manifest
        )
        if (
            not decision.get("fallback_applied")
            or decision.get("selected_seeds") != [args.seed]
        ):
            raise ProtocolError("recovery requires the immutable one-seed fallback decision for seed 42")
        admission_path = admit_worker(
            config,
            config_path,
            manifest_path,
            "recover-seed",
            seed=args.seed,
            pilot_decision_path=decision_path,
        )
        report = run_recover_seed(
            config,
            manifest,
            args.seed,
            review_accepted=args.review_accepted,
            config_path=config_path,
            admission_path=admission_path,
            pilot_decision_path=decision_path,
        )
    elif args.command == "dev-select":
        config = load_json(config_path)
        validate_config(config)
        require_compute_open(config, route="validation")
        manifest_path = repo_path(config["data_manifest"])
        manifest = load_json(manifest_path)
        decision, candidates = load_dev_candidates(config, manifest, config_path)
        selection = select_dev_checkpoint(candidates, config["labels"])
        freeze = repo_path(config["output_root"], artifact_output=True) / "freeze.json"
        if freeze.exists():
            raise ProtocolError("dev freeze already exists; immutable freeze cannot be replaced")
        result_root = repo_path(config["output_root"], artifact_output=True)
        result_hashes = {
            str(seed): sha256(result_root / f"seed_{seed}/result.json")
            for seed in decision["selected_seeds"]
        }
        decision_record = load_json(repo_path(config["output_root"], artifact_output=True) / "pilot_decision.json")
        create_once_json(
            freeze,
            {
                "schema": "i3-dev-freeze-v2",
                "selection": selection,
                "candidates": candidates,
                "result_artifact_sha256": result_hashes,
                "pilot_decision_path": str((repo_path(config["output_root"], artifact_output=True) / "pilot_decision.json").relative_to(ROOT)),
                "pilot_decision_sha256": sha256(repo_path(config["output_root"], artifact_output=True) / "pilot_decision.json"),
                "decision_identity": decision_identity(decision_record),
                "config_path": str(config_path.resolve().relative_to(ROOT)),
                "config_sha256": sha256(config_path),
                "manifest_path": str(manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha256(manifest_path),
                "code": code_identity(),
                "test_opened": False,
            },
            exists_message="dev freeze already exists; immutable freeze cannot be replaced",
        )
        report = {"status": "dev_checkpoint_selected", "seed_decision": decision, "selection": selection, "freeze_path": str(freeze.relative_to(ROOT)), "freeze_sha256": sha256(freeze), "test_opened": False}
    elif args.command == "smoke":
        config = load_json(config_path)
        validate_config(config)
        require_compute_open(config, route="smoke")
        manifest_path = repo_path(config["data_manifest"])
        manifest = load_json(manifest_path)
        admission_path = admit_worker(
            config,
            config_path,
            manifest_path,
            "smoke",
            requested_wall_seconds=float(config["budget"]["smoke_max_gpu_seconds"]),
        )
        report = run_bounded_smoke(
            config,
            manifest,
            review_accepted=args.review_accepted,
            config_path=config_path,
            admission_path=admission_path,
        )
        report = {key: value for key, value in report.items() if not key.startswith("_") and key != "worker_wall_seconds"}
        create_once_json(
            repo_path(config["output_root"], artifact_output=True) / "smoke.result.json",
            report,
            exists_message="bounded smoke result already exists; retry is forbidden",
        )
    elif args.command == "final-test":
        if not args.freeze_sha256:
            raise ProtocolError("final-test requires the dev-freeze SHA-256")
        config = load_json(config_path)
        validate_config(config)
        manifest = load_json(repo_path(config["data_manifest"]))
        # The final route is intentionally separate from the non-final worker
        # guard; run_final_test activates the immutable held budget token and
        # owns the one create-once authenticated marker.
        report = run_final_test(
            config,
            manifest,
            args.freeze_sha256,
            review_accepted=args.review_accepted,
            config_path=config_path,
        )
    else:
        config = load_json(config_path)
        validate_config(config)
        manifest = load_json(repo_path(config["data_manifest"]))
        report = planned_run(config, manifest, args.command)
    if args.report:
        report_path = args.report.resolve()
        allowed = (ROOT / "artifacts/experiment_i3").resolve()
        if report_path != allowed and allowed not in report_path.parents:
            raise ProtocolError("reports must remain under artifacts/experiment_i3/preflight")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
