from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

from training.laya_trace_i3.metrics import MetricsError, evaluate_records
from training.laya_trace_i3.protocol import (
    CHOICE_OPTION_ORDER,
    FIXED_LABELS,
    ROOT,
    ProtocolError,
    choose_seed_count,
    expected_updates,
    group_bootstrap_ids,
    select_dev_checkpoint,
    sha256,
    validate_latency_evidence,
    validate_config,
    validate_manifest,
    validate_runtime_budget,
)
from training.laya_trace_i3.runner import build_repair_report, build_report
from training.laya_trace_i3.experiment import GpuLock, accumulation_windows, load_split, planned_run, run_actual_model_pilot, run_bounded_smoke, run_final_test, six_class_rlcd_reference


CONFIG_PATH = ROOT / "training/laya_trace_i3/config.json"
MANIFEST_PATH = ROOT / "artifacts/experiment_i3/preflight/data_manifest.v3.json"
PENDING_MANIFEST_PATH = ROOT / "artifacts/experiment_i3/preflight/data_manifest.pending.json"
PILOT_PATH = ROOT / "artifacts/experiment_i3/preflight/actual_model_pilot.json"


def probabilities(**overrides: float) -> dict[str, float]:
    values = {label: 0.0 for label in FIXED_LABELS}
    values.update(overrides)
    return values


class MetricsTest(unittest.TestCase):
    def test_record_metrics_fixed_labels_missing_classes_and_confusion(self):
        rows = [
            {"record_id": "r1", "gold_label": "read", "probabilities": probabilities(read=0.8, search=0.2)},
            {"record_id": "r2", "gold_label": "search", "probabilities": probabilities(read=0.6, search=0.4)},
        ]
        result = evaluate_records(rows, FIXED_LABELS)
        self.assertEqual(result["unit"], "whole_turn_record")
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3) / 6)
        self.assertEqual(result["per_class"]["other_tool"]["support"], 0)
        self.assertEqual(result["per_class"]["other_tool"]["recall"], 0.0)
        self.assertEqual(result["confusion_matrix"]["values"][0][0], 1)
        self.assertEqual(result["confusion_matrix"]["values"][1][0], 1)
        self.assertAlmostEqual(result["nll"], (-math.log(0.8) - math.log(0.4)) / 2)
        self.assertAlmostEqual(result["brier"], (0.08 + 0.72) / 2)

    def test_perfect_six_class_metrics(self):
        rows = [
            {"record_id": f"r{index}", "gold_label": label, "probabilities": probabilities(**{label: 1.0})}
            for index, label in enumerate(FIXED_LABELS)
        ]
        result = evaluate_records(rows, FIXED_LABELS)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["nll"], 0.0)
        self.assertEqual(result["brier"], 0.0)

    def test_rejects_duplicate_nonfinite_wrong_order_and_extra_question_fields(self):
        valid = {"record_id": "r", "gold_label": "read", "probabilities": probabilities(read=1.0)}
        with self.assertRaisesRegex(MetricsError, "duplicate"):
            evaluate_records([valid, valid], FIXED_LABELS)
        bad = copy.deepcopy(valid)
        bad["probabilities"]["read"] = float("nan")
        with self.assertRaisesRegex(MetricsError, "finite"):
            evaluate_records([bad], FIXED_LABELS)
        bad = copy.deepcopy(valid)
        bad["probabilities"] = dict(reversed(list(bad["probabilities"].items())))
        with self.assertRaisesRegex(MetricsError, "fixed label order"):
            evaluate_records([bad], FIXED_LABELS)
        bad = dict(valid, question_id="another_item")
        with self.assertRaisesRegex(MetricsError, "must contain only"):
            evaluate_records([bad], FIXED_LABELS)


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text())
        cls.manifest = json.loads(MANIFEST_PATH.read_text())
        cls.pending = json.loads(PENDING_MANIFEST_PATH.read_text())
        cls.pilot = json.loads(PILOT_PATH.read_text())

    def test_config_and_accepted_plan_do_not_load_model_or_test(self):
        validate_config(self.config)
        self.assertEqual(validate_manifest(self.manifest, allow_pending=False), "accepted")
        report = planned_run(self.config, self.manifest, "base-eval")
        self.assertEqual(report["status"], "blocked_before_model_or_cuda")
        self.assertFalse(report["torch_imported"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["test_opened"])

    def test_pending_manifest_cannot_authorize_execution(self):
        self.assertEqual(validate_manifest(self.pending, allow_pending=True), "pending")
        with self.assertRaisesRegex(ProtocolError, "pending independent acceptance"):
            validate_manifest(self.pending, allow_pending=False)

    def test_record_contract_is_one_question_and_train_dev_are_readable(self):
        train = load_split(self.manifest, "train", FIXED_LABELS)
        dev = load_split(self.manifest, "dev", FIXED_LABELS)
        self.assertEqual((len(train), len(dev)), (1051, 140))
        with self.assertRaisesRegex(ProtocolError, "test bytes"):
            load_split(self.manifest, "test", FIXED_LABELS)

    def test_six_class_mapping_reward_and_nonzero_objective(self):
        perturbations = [
            [0.3, -0.1, 0.2, -0.4, 0.1, -0.1],
            [-0.4, 0.8, -0.6, 0.2, -0.2, 0.2],
            [0.05, 0.1, -0.2, -0.05, 0.15, -0.05],
            [-0.9, -0.6, 0.3, 1.2, -0.3, 0.3],
        ]
        for label in FIXED_LABELS:
            result = six_class_rlcd_reference([0.2, -0.1, 0.4, 0.0, -0.3, 0.1], label, perturbations, 0.4)
            self.assertEqual(CHOICE_OPTION_ORDER[result["gold_choice_index"]], label)
            self.assertTrue(math.isfinite(result["loss"]))
            self.assertGreater(sum(abs(value) for value in result["rlcd_score_gradient"]), 1e-6)
            self.assertGreater(sum(abs(value) for value in result["choice_logit_gradient"]), 0.0)
            self.assertGreater(abs(result["loss_rlcd"]), 1e-6)

    def test_group_bootstrap_ids_are_stable_and_grouped(self):
        train = load_split(self.manifest, "train", FIXED_LABELS)
        dev = load_split(self.manifest, "dev", FIXED_LABELS)
        self.assertEqual((group_bootstrap_ids(train)["rows"], group_bootstrap_ids(train)["task_groups"]), (1051, 163))
        self.assertEqual((group_bootstrap_ids(dev)["rows"], group_bootstrap_ids(dev)["task_groups"]), (140, 20))

    def test_repair_report_is_model_free_and_test_closed(self):
        report = build_repair_report(CONFIG_PATH)
        self.assertEqual(report["status"], "repair_preflight_passed")
        self.assertFalse(report["torch_imported"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["test_opened"])

    def test_nonblocking_gpu_lock_contention(self):
        lock_path = ROOT / "artifacts/experiment_i3/preflight/guard_probe/gpu.lock"
        owner = {"command": "unit-lock-probe", "config_sha256": "a" * 64, "manifest_sha256": "b" * 64, "seed": 42}
        with GpuLock(lock_path, owner):
            with self.assertRaisesRegex(ProtocolError, "already held"):
                with GpuLock(lock_path, owner):
                    self.fail("contended lock was acquired")
        with mock.patch.dict(sys.modules, {"torch": object()}):
            with self.assertRaisesRegex(ProtocolError, "before Torch import"):
                with GpuLock(lock_path, owner):
                    self.fail("lock was acquired after Torch import")

    def test_latency_guard_requires_synchronized_complete_samples(self):
        evidence = {
            "batch_size": 1,
            "warmups_completed": 3,
            "samples_seconds": [0.01 + index / 1000 for index in range(10)],
            "synchronize_before_each": True,
            "synchronize_after_each": True,
            "included_phases": ["tokenize_build", "collate", "host_to_device", "forward", "softmax", "cpu_probability_copy"],
            "measurement_source": "runtime_monotonic_with_cuda_synchronization",
            "synchronization_events": 26,
        }
        summary = validate_latency_evidence(evidence, self.config)
        self.assertEqual(summary, {"median_seconds": 0.014499999999999999, "p95_seconds": 0.019})
        evidence["synchronize_after_each"] = False
        with self.assertRaisesRegex(ProtocolError, "synchronization"):
            validate_latency_evidence(evidence, self.config)

    def test_schedule_is_derived_only_after_accepted_count(self):
        self.assertEqual(expected_updates(1235, self.config), 80)
        with self.assertRaisesRegex(ProtocolError, "positive integer"):
            expected_updates(0, self.config)

    def test_microbatch_boundaries_use_positions_and_scale_short_tail(self):
        shuffled = [8, 2, 9, 1, 7, 0, 6, 3, 5, 4]
        windows = accumulation_windows(shuffled, 4)
        self.assertEqual([window["record_indices"] for window in windows], [[8, 2, 9, 1], [7, 0, 6, 3], [5, 4]])
        self.assertEqual([window["divisor"] for window in windows], [4, 4, 2])
        self.assertEqual(windows[-1]["positions"], [8, 9])

    def test_final_test_validates_before_irreversible_order_and_test_decode(self):
        events = []
        selection = {"selected_seed": 42, "selected_checkpoint_path": "artifacts/experiment_i3/preflight/guard_probe/checkpoint", "selected_weights_file_sha256": "a" * 64}
        probe = ROOT / "artifacts/experiment_i3/preflight/guard_probe/final_test_probe"
        probe.mkdir(parents=True, exist_ok=True)

        class LockProbe:
            def __init__(self, *args):
                pass

            def __enter__(self):
                events.append("lock")
                return self

            def __exit__(self, *args):
                events.append("unlock")

        def stop_before_rows(*args):
            events.append("test")
            raise RuntimeError("stopped before real test rows")

        with mock.patch("training.laya_trace_i3.experiment._final_selection", return_value=(selection, {})), mock.patch("training.laya_trace_i3.experiment.GpuLock", LockProbe), mock.patch("training.laya_trace_i3.experiment.consume_final_test_attempt", side_effect=lambda *args: events.append("marker") or probe / "marker.json"), mock.patch("training.laya_trace_i3.experiment._load_split_bytes", side_effect=stop_before_rows):
            with self.assertRaisesRegex(RuntimeError, "before real test rows"):
                run_final_test(self.config, self.manifest, "b" * 64, review_accepted=True)
        self.assertEqual(events[:3], ["lock", "marker", "test"])

    def test_final_test_rejects_selection_before_lock_or_test(self):
        events = []
        with mock.patch("training.laya_trace_i3.experiment._final_selection", side_effect=ProtocolError("bad frozen selection")), mock.patch("training.laya_trace_i3.experiment.GpuLock", side_effect=lambda *args: events.append("lock")), mock.patch("training.laya_trace_i3.experiment._load_split_bytes", side_effect=lambda *args: events.append("test")):
            with self.assertRaisesRegex(ProtocolError, "bad frozen selection"):
                run_final_test(self.config, self.manifest, "c" * 64, review_accepted=True)
        self.assertEqual(events, [])

    def test_two_seed_default_and_only_declared_fallback(self):
        smoke = dict(self.pilot, training_seconds=10.0, cold_load_seconds=2.0, dev_evaluation_seconds=3.0, peak_reserved_gib=8.0)
        decision = choose_seed_count(smoke, self.config, 1280)
        self.assertEqual(decision["selected_seeds"], [42, 314159])
        slow = dict(smoke, training_seconds=100.0)
        decision = choose_seed_count(slow, self.config, 1280)
        self.assertTrue(decision["fallback_applied"])
        self.assertEqual(decision["selected_seeds"], [42])

    def test_nonfinite_or_over_budget_smoke_forbids_main(self):
        smoke = dict(self.pilot, training_seconds=float("nan"), cold_load_seconds=2.0, dev_evaluation_seconds=3.0, peak_reserved_gib=8.0)
        with self.assertRaisesRegex(ProtocolError, "non-finite"):
            choose_seed_count(smoke, self.config, 100)
        smoke.update(training_seconds=10.0, peak_reserved_gib=12.0)
        with self.assertRaisesRegex(ProtocolError, "forbidden"):
            choose_seed_count(smoke, self.config, 100)

    def test_pilot_envelope_rejects_missing_unknown_status_opened_seed_and_update_variants(self):
        variants = [
            dict(self.pilot, **{"test_opened": None}),
            {key: value for key, value in self.pilot.items() if key != "status"},
            dict(self.pilot, unexpected=True),
            dict(self.pilot, status="actual_model_pilot_failed"),
            dict(self.pilot, test_opened=True),
            dict(self.pilot, seed=314159),
            dict(self.pilot, updates=1),
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ProtocolError, "predeclared profile"):
                    choose_seed_count(variant, self.config, 1280)

    def test_pilot_envelope_rejects_exceeded_budgets(self):
        for field, value in (("training_seconds", 601.0), ("peak_reserved_gib", 11.500001)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ProtocolError, "forbidden"):
                    choose_seed_count(dict(self.pilot, **{field: value}), self.config, 1280)

    def test_checkpoint_selection_is_dev_only_and_deterministic(self):
        checkpoint = ROOT / "artifacts/experiment_i3/preflight/guard_probe/checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        (checkpoint / "training_state.pt").write_bytes(b"state")
        from training.laya_trace_i3.protocol import sha256
        weights_digest, state_digest = sha256(checkpoint / "model.safetensors"), sha256(checkpoint / "training_state.pt")
        digest = "a" * 64
        reload_evidence = {"strict_model_load": True, "model_state_before_sha256": digest, "model_state_after_sha256": digest, "training_state_file_sha256": state_digest, "optimizer_exact": True, "scheduler_exact": True, "scaler_exact": True, "stopping_state_exact": True}
        evidence_path = checkpoint / "reload.json"
        evidence_path.write_text(json.dumps(reload_evidence, sort_keys=True) + "\n")
        base = {"split": "dev", "checkpoint_path": "artifacts/experiment_i3/preflight/guard_probe/checkpoint", "weights_file_sha256": weights_digest, "model_state_sha256": digest, "training_state_file_sha256": state_digest, "labels": FIXED_LABELS, "reload_evidence": reload_evidence, "reload_evidence_path": "artifacts/experiment_i3/preflight/guard_probe/checkpoint/reload.json", "reload_evidence_sha256": sha256(evidence_path)}
        candidates = [dict(base, seed=314159, macro_f1=0.5, nll=0.8), dict(base, seed=42, macro_f1=0.5, nll=0.8)]
        self.assertEqual(select_dev_checkpoint(candidates, FIXED_LABELS)["selected_seed"], 42)
        bad = [dict(candidates[0], split="test")]
        with self.assertRaisesRegex(ProtocolError, "dev results"):
            select_dev_checkpoint(bad, FIXED_LABELS)
        bad = [dict(candidates[0], reload_evidence=dict(reload_evidence, optimizer_exact=False))]
        with self.assertRaisesRegex(ProtocolError, "does not match"):
            select_dev_checkpoint(bad, FIXED_LABELS)
        with self.assertRaisesRegex(ProtocolError, "checkpoint directory"):
            select_dev_checkpoint([dict(candidates[0], checkpoint_path="artifacts/experiment_i3/preflight/guard_probe/missing")], FIXED_LABELS)

    def test_lifecycle_mutations_and_runtime_budget_fail_closed(self):
        for section, key, value in (("budget", "iteration_max_elapsed_seconds", 1), ("evaluation", "final_test_policy", "disabled"), ("latency", "warm_definition", "unsynchronized")):
            mutated = copy.deepcopy(self.config)
            mutated[section][key] = value
            with self.assertRaises(ProtocolError):
                validate_config(mutated)
        with self.assertRaisesRegex(ProtocolError, "budget"):
            validate_runtime_budget(14401, 0, self.config)
        with self.assertRaisesRegex(ProtocolError, "budget"):
            validate_runtime_budget(0, 7201, self.config)

    def test_final_test_bytes_have_no_allow_test_bypass(self):
        with self.assertRaisesRegex(ProtocolError, "only through"):
            load_split(self.manifest, "test", FIXED_LABELS)

    def test_smoke_requires_review_and_never_cpu_falls_back(self):
        with self.assertRaisesRegex(ProtocolError, "fail-closed"):
            run_bounded_smoke(self.config, self.manifest)

    def test_authorized_smoke_builds_lock_owner_before_torch(self):
        captured = {}

        def stop_before_torch(lock_path, owner):
            captured["lock_path"] = lock_path
            captured["owner"] = owner
            raise RuntimeError("stopped before Torch/CUDA")

        with mock.patch("training.laya_trace_i3.experiment.GpuLock", side_effect=stop_before_torch):
            with self.assertRaisesRegex(RuntimeError, "before Torch/CUDA"):
                run_bounded_smoke(self.config, self.manifest, review_accepted=True)

        self.assertEqual(captured["lock_path"], ROOT / self.config["gpu_lock"])
        self.assertEqual(
            captured["owner"],
            {
                "command": "smoke",
                "config_sha256": sha256(CONFIG_PATH),
                "manifest_sha256": sha256(ROOT / self.config["data_manifest"]),
                "seed": self.config["budget"]["smoke_seed"],
            },
        )

    def test_actual_model_pilot_builds_local_lock_owner_before_torch(self):
        captured = {}

        def stop_before_torch(lock_path, owner):
            captured["lock_path"] = lock_path
            captured["owner"] = owner
            raise RuntimeError("stopped before actual Torch/model work")

        with mock.patch("training.laya_trace_i3.experiment._model_paths", return_value=(ROOT / "data/cache", ROOT / "data/cache/model.safetensors", ROOT / "data/laya")), mock.patch("training.laya_trace_i3.experiment.GpuLock", side_effect=stop_before_torch):
            with self.assertRaisesRegex(RuntimeError, "before actual Torch/model"):
                run_actual_model_pilot(self.config, self.manifest, review_accepted=True)

        self.assertEqual(captured["lock_path"], ROOT / self.config["gpu_lock"])
        self.assertEqual(captured["owner"]["command"], "actual-model-pilot")
        self.assertEqual(captured["owner"]["seed"], self.config["budget"]["smoke_seed"])
        self.assertFalse(captured["owner"] is None)

    def test_actual_model_paths_fail_closed_when_local_pin_is_missing(self):
        with mock.patch("training.laya_trace_i3.experiment.ROOT", ROOT / "does-not-exist"):
            with self.assertRaisesRegex(ProtocolError, "snapshot"):
                run_actual_model_pilot(self.config, self.manifest, review_accepted=True)


if __name__ == "__main__":
    unittest.main()
