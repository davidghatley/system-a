#!/usr/bin/env python3
"""CPU-only preflight entry point for the gated Iteration 3 Laya experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

BOOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOT_ROOT))

from training.laya_trace_i3.protocol import (
    ROOT,
    ProtocolError,
    expected_updates,
    group_bootstrap_ids,
    load_json,
    repo_path,
    sha256,
    validate_config,
    validate_manifest,
    validate_runtime_budget,
    verify_split_files,
)
from training.laya_trace_i3.experiment import (
    final_test_once, load_split, planned_run, run_actual_model_pilot, run_base_eval,
    run_bounded_smoke, run_final_test, run_training_seed, six_class_rlcd_reference, write_plan,
)
from training.laya_trace_i3.protocol import select_dev_checkpoint


DEFAULT_CONFIG = ROOT / "training/laya_trace_i3/config.json"


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
    parser.add_argument("command", choices=("validate-config", "dry-run", "validate-accepted-manifest", "repair-preflight", "actual-model-pilot", "base-eval", "train", "dev-select", "final-test", "smoke"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--review-accepted", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
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
        manifest = load_json(repo_path(config["data_manifest"]))
        report = run_actual_model_pilot(config, manifest, review_accepted=args.review_accepted)
    elif args.command == "base-eval":
        config = load_json(config_path)
        validate_config(config)
        manifest = load_json(repo_path(config["data_manifest"]))
        report = run_base_eval(config, manifest, review_accepted=args.review_accepted)
    elif args.command == "train":
        config = load_json(config_path)
        validate_config(config)
        manifest = load_json(repo_path(config["data_manifest"]))
        seeds = [args.seed] if args.seed is not None else config["training"]["default_seeds"]
        if any(seed not in config["training"]["default_seeds"] for seed in seeds):
            raise ProtocolError("seed is not declared by the frozen training config")
        training_started = time.monotonic()
        reports = [run_training_seed(config, manifest, seed, review_accepted=args.review_accepted) for seed in seeds]
        aggregate_gpu_seconds = sum(float(item["cold_load_seconds"]) + float(item["training_seconds"]) + float(item["dev_evaluation_seconds"]) for item in reports)
        validate_runtime_budget(time.monotonic() - training_started, aggregate_gpu_seconds, config)
        for item in reports:
            run_dir = repo_path(config["output_root"], artifact_output=True) / f"seed_{item['seed']}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = {"status": "training_completed", "seeds": [item["seed"] for item in reports], "reports": reports, "test_opened": False}
    elif args.command == "dev-select":
        config = load_json(config_path)
        validate_config(config)
        manifest = load_json(repo_path(config["data_manifest"]))
        candidates = []
        for seed in config["training"]["default_seeds"]:
            result_path = repo_path(config["output_root"], artifact_output=True) / f"seed_{seed}" / "result.json"
            if not result_path.is_file():
                raise ProtocolError(f"missing completed seed result: {result_path.relative_to(ROOT)}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = result["checkpoint"]
            candidates.append({"seed": seed, "split": "dev", "macro_f1": result["dev"]["macro_f1"], "nll": result["dev"]["nll"], "checkpoint_path": checkpoint["checkpoint_path"], "weights_file_sha256": checkpoint["weights_file_sha256"], "model_state_sha256": checkpoint["model_state_sha256"], "training_state_file_sha256": checkpoint["training_state_file_sha256"], "labels": config["labels"], "reload_evidence": checkpoint["reload_evidence"], "reload_evidence_path": checkpoint["reload_evidence_path"], "reload_evidence_sha256": checkpoint["reload_evidence_sha256"]})
        selection = select_dev_checkpoint(candidates, config["labels"])
        freeze = repo_path(config["output_root"], artifact_output=True) / "freeze.json"
        freeze.write_text(json.dumps({"selection": selection, "candidates": candidates, "config_sha256": sha256(config_path), "manifest_sha256": sha256(repo_path(config["data_manifest"])), "test_opened": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = {"status": "dev_checkpoint_selected", "selection": selection, "freeze_path": str(freeze.relative_to(ROOT)), "freeze_sha256": sha256(freeze), "test_opened": False}
    elif args.command == "smoke":
        config = load_json(config_path)
        validate_config(config)
        manifest = load_json(repo_path(config["data_manifest"]))
        report = run_bounded_smoke(config, manifest, review_accepted=args.review_accepted)
    elif args.command == "final-test":
        if not args.freeze_sha256:
            raise ProtocolError("final-test requires the dev-freeze SHA-256")
        config = load_json(config_path)
        validate_config(config)
        manifest = load_json(repo_path(config["data_manifest"]))
        report = run_final_test(config, manifest, args.freeze_sha256, review_accepted=args.review_accepted)
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
