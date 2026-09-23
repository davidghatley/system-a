from __future__ import annotations

import contextlib
import copy
import json
import math
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.laya_trace_i3 import experiment, protocol, runner
from training.laya_trace_i3.protocol import (
    AggregateBudget,
    ProtocolError,
    consume_final_test_attempt,
    create_pilot_decision,
    require_exact_seed_set,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "training/laya_trace_i3/config.json"
MANIFEST_PATH = ROOT / "artifacts/experiment_i3/preflight/data_manifest.v3.json"
SYNTHETIC_ROOT = ROOT / "artifacts/experiment_i3/preflight/synthetic"
LEDGER_ROOT = ROOT / "artifacts/review_60fd/test_ledgers"


class AggregateBudgetTests(unittest.TestCase):
    def ledger_path(self) -> Path:
        path = LEDGER_ROOT / f"{uuid.uuid4().hex}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_full_grant_rejects_4000_after_4000_and_exact_3200_runs(self) -> None:
        first = AggregateBudget(self.ledger_path(), {"case": "full-grant"})
        token = first.reserve(4000)
        first.finish(token, 4000)
        with self.assertRaisesRegex(ProtocolError, "full request"):
            first.reserve_up_to(4000)
        self.assertEqual(len(json.loads(first.path.read_text())["reservations"]), 1)

        # A separate ledger proves that exactly 3200 is admissible, rather
        # than relying on a partial 3200-second grant from the rejected request.
        exact = AggregateBudget(self.ledger_path(), {"case": "exact-remainder"})
        token, granted = exact.reserve_up_to(3200)
        self.assertEqual(granted, 3200)
        exact.finish(token, 3200)

    def test_actual_overrun_and_unresolved_reservation_are_fully_charged(self) -> None:
        overrun = AggregateBudget(self.ledger_path(), {"case": "overrun"})
        token = overrun.reserve(4000)
        with self.assertRaisesRegex(ProtocolError, "overrun"):
            overrun.finish(token, 4001)
        with self.assertRaisesRegex(ProtocolError, "full request"):
            overrun.reserve_up_to(3200)

        unresolved = AggregateBudget(self.ledger_path(), {"case": "unresolved"})
        unresolved.reserve(4000)  # deliberately never finalized
        token, grant = unresolved.reserve_up_to(3200)
        self.assertEqual(grant, 3200)
        unresolved.finish(token, 3200)
        with self.assertRaises(ProtocolError):
            unresolved.reserve_up_to(0.01)

    def test_malformed_identity_duplicate_and_invalid_inputs_fail_closed(self) -> None:
        invalid_json = self.ledger_path()
        invalid_json.write_text("{not-json\n", encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "malformed"):
            AggregateBudget(invalid_json, {"case": "invalid-json"}).reserve(1)

        wrong_shape = self.ledger_path()
        wrong_shape.write_text('{"schema":"broken"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "malformed"):
            AggregateBudget(wrong_shape, {"case": "shape"}).reserve(1)

        duplicate = self.ledger_path()
        duplicate.write_text(json.dumps({
            "schema": "i3-gpu-budget-v1",
            "identity": {"case": "duplicate"},
            "limit": 7200,
            "reservations": [
                {"token": "same", "reserved_seconds": 1, "state": "in_flight"},
                {"token": "same", "reserved_seconds": 1, "state": "in_flight"},
            ],
        }) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "duplicate"):
            AggregateBudget(duplicate, {"case": "duplicate"}).reserve(1)

        valid = AggregateBudget(self.ledger_path(), {"case": "identity"})
        for value in (0, -1, math.inf, math.nan, True):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                valid.reserve(value)
        token = valid.reserve(1)
        valid.finish(token, 1)
        with self.assertRaisesRegex(ProtocolError, "identity changed"):
            AggregateBudget(valid.path, {"case": "changed"}).reserve(1)

    def test_create_once_json_does_not_publish_partial_file_on_serialization_failure(self) -> None:
        path = LEDGER_ROOT / f"failed_publication_{uuid.uuid4().hex}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with patch("training.laya_trace_i3.protocol.json.dump", side_effect=RuntimeError("synthetic serialization failure")):
            with self.assertRaisesRegex(RuntimeError, "serialization failure"):
                protocol.create_once_json(path, {"incomplete": True}, exists_message="exists")
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_two_subprocesses_cannot_both_admit_4000_of_7200(self) -> None:
        path = self.ledger_path()
        code = """
import sys
from training.laya_trace_i3.protocol import AggregateBudget, ProtocolError
budget = AggregateBudget(__import__('pathlib').Path(sys.argv[1]), {'case': 'subprocess'}, 7200)
try:
    token = budget.reserve(4000)
except ProtocolError:
    print('REJECT')
else:
    print('GRANT:' + token)
"""
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", code, str(path)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        outputs = [process.communicate(timeout=30) for process in processes]
        for process, (stdout, stderr) in zip(processes, outputs):
            self.assertEqual(process.returncode, 0, stderr)
        decisions = [stdout.strip().split(":", 1)[0] for stdout, _ in outputs]
        self.assertCountEqual(decisions, ["GRANT", "REJECT"])
        ledger = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(ledger["reservations"]), 1)
        self.assertEqual(ledger["reservations"][0]["state"], "in_flight")


class RunnerOrchestrationTests(unittest.TestCase):
    def new_root(self) -> Path:
        root = SYNTHETIC_ROOT / f"orchestration_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        return root

    def make_config(
        self,
        root: Path | None = None,
        *,
        output_root: Path | None = None,
    ) -> tuple[Path, dict, Path]:
        root = root or self.new_root()
        config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        config["output_root"] = str((output_root or root).relative_to(ROOT))
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return config_path, config, root

    def make_decision(
        self,
        config: dict,
        config_path: Path,
        *,
        projected_seconds: float | None = None,
        training_seconds: float = 19,
    ) -> tuple[dict, Path, Path, Path]:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if projected_seconds is not None:
            updates = config["training"]["epochs"] * math.ceil(
                manifest["splits"]["train"]["rows"]
                / config["training"]["gradient_accumulation"]
            )
            training_seconds = (
                projected_seconds
                / config["budget"]["projection_multiplier"]
                * config["budget"]["smoke_updates"]
                / updates
            )

        def pilot_worker(*args, **kwargs):
            return {
                "status": "actual_model_pilot_measured",
                "seed": kwargs["seed"],
                "updates": config["budget"]["smoke_updates"],
                "training_seconds": training_seconds,
                "cold_load_seconds": 0.0,
                "dev_evaluation_seconds": 0.0,
                "peak_reserved_gib": 8.0,
                "finite": True,
                "test_opened": False,
            }

        # A pilot is never fabricated as a bare measurement file.  The runner
        # creates its admission, reserves the fixed pilot allowance, and the
        # synthetic worker returns only the measurement payload.
        with patch.object(experiment, "_runtime_worker", side_effect=pilot_worker):
            self.invoke("actual-model-pilot", config_path)
        pilot_path, decision_path = runner.future_run_paths(config)
        self.invoke("record-pilot-decision", config_path, review_accepted=False)
        record = json.loads(decision_path.read_text(encoding="utf-8"))
        if projected_seconds is not None:
            self.assertAlmostEqual(
                record["decision"]["projected_gpu_seconds_per_seed"],
                projected_seconds,
            )
        return manifest, pilot_path, decision_path, record["decision"]

    def invoke(
        self,
        command: str,
        config_path: Path,
        *extra: str,
        review_accepted: bool = True,
    ) -> None:
        argv = ["runner", command, "--config", str(config_path), *extra]
        if review_accepted:
            argv.append("--review-accepted")
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            runner.main()

    @staticmethod
    def completed_worker(calls: list[dict], command: str | None = None):
        def worker(*args, **kwargs):
            calls.append(kwargs)
            if command == "train":
                return {"status": "completed", "seed": kwargs["seed"]}
            if command == "actual-model-pilot":
                return {
                    "status": "actual_model_pilot_measured",
                    "seed": kwargs["seed"],
                    "updates": 2,
                    "training_seconds": 1,
                    "cold_load_seconds": 0,
                    "dev_evaluation_seconds": 0,
                    "peak_reserved_gib": 8,
                    "finite": True,
                    "test_opened": False,
                }
            return {"status": "base_eval_completed", "seed": kwargs["seed"]}
        return worker

    def install_completed_seed(self, config: dict, config_path: Path, root: Path, seed: int = 42) -> Path:
        """Create a genuine runner result plus a minimal reloadable CPU fixture."""
        def worker(*args, **kwargs):
            checkpoint = root / f"seed_{seed}" / "checkpoint"
            checkpoint.mkdir(parents=True, exist_ok=False)
            weights = checkpoint / "model.safetensors"
            state = checkpoint / "training_state.pt"
            weights.write_bytes(f"synthetic weights {seed}\n".encode())
            state.write_bytes(f"synthetic state {seed}\n".encode())
            model_digest = "a" * 64
            evidence = {
                "strict_model_load": True,
                "model_state_before_sha256": model_digest,
                "model_state_after_sha256": model_digest,
                "training_state_file_sha256": sha256(state),
                "optimizer_exact": True,
                "scheduler_exact": True,
                "scaler_exact": True,
                "stopping_state_exact": True,
            }
            evidence_path = checkpoint / "reload.json"
            evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
            return {
                "status": "completed",
                "seed": seed,
                "dev": {"macro_f1": 0.5, "nll": 0.7},
                "checkpoint": {
                    "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                    "weights_file_sha256": sha256(weights),
                    "model_state_sha256": model_digest,
                    "training_state_file_sha256": sha256(state),
                    "reload_evidence": evidence,
                    "reload_evidence_path": str(evidence_path.relative_to(ROOT)),
                    "reload_evidence_sha256": sha256(evidence_path),
                },
                "test_opened": False,
            }
        with patch.object(experiment, "_runtime_worker", side_effect=worker):
            self.invoke("train", config_path, "--seed", str(seed))
        return root / f"seed_{seed}" / "result.json"

    def test_pilot_admission_precedes_worker_and_repeat_never_recomputes(self) -> None:
        config_path, config, root = self.make_config()
        pilot_path, _ = runner.future_run_paths(config)
        admission_path = root / "future_actual_model_pilot.admission.json"
        calls: list[dict] = []

        def worker(*args, **kwargs):
            self.assertTrue(admission_path.is_file())
            self.assertFalse(pilot_path.exists())
            calls.append(kwargs)
            return {
                "status": "actual_model_pilot_measured",
                "seed": kwargs["seed"],
                "updates": 2,
                "training_seconds": 1,
                "cold_load_seconds": 0,
                "dev_evaluation_seconds": 0,
                "peak_reserved_gib": 8,
                "finite": True,
                "test_opened": False,
            }

        with patch.object(experiment, "_runtime_worker", side_effect=worker):
            self.invoke("actual-model-pilot", config_path)
        self.assertTrue(admission_path.is_file())
        self.assertTrue(pilot_path.is_file())
        original_pilot = pilot_path.read_bytes()
        original_admission = admission_path.read_bytes()
        with patch.object(experiment, "_runtime_worker", side_effect=worker):
            with self.assertRaisesRegex(ProtocolError, "already has immutable artifact"):
                self.invoke("actual-model-pilot", config_path)
        self.assertEqual(len(calls), 1)
        self.assertEqual(pilot_path.read_bytes(), original_pilot)
        self.assertEqual(admission_path.read_bytes(), original_admission)

    def test_preexisting_pilot_fails_before_worker(self) -> None:
        config_path, config, root = self.make_config()
        pilot_path, _ = runner.future_run_paths(config)
        pilot_path.write_text('{"synthetic":"preexisting"}\n', encoding="utf-8")
        worker = patch.object(runner, "run_actual_model_pilot")
        with worker as model_worker:
            with self.assertRaisesRegex(ProtocolError, "already has immutable artifact"):
                self.invoke("actual-model-pilot", config_path)
            model_worker.assert_not_called()
        self.assertFalse((root / "future_actual_model_pilot.admission.json").exists())

    def test_train_checks_existing_result_and_unresolved_admission_before_worker(self) -> None:
        config_path, config, root = self.make_config()
        self.make_decision(config, config_path, projected_seconds=3200)
        existing = root / "seed_42/result.json"
        existing.parent.mkdir()
        existing.write_text('{"seed":42,"status":"completed"}\n', encoding="utf-8")
        with patch.object(experiment, "_runtime_worker") as worker:
            with self.assertRaisesRegex(ProtocolError, "already has immutable artifact"):
                self.invoke("train", config_path, "--seed", "42")
            worker.assert_not_called()
        self.assertFalse((root / "seed_42/admission.json").exists())

        # A worker failure leaves a durable admission.  An automatic repeat is
        # denied even though no result was published.
        config_path, config, root = self.make_config()
        self.make_decision(config, config_path, projected_seconds=3200)
        calls: list[dict] = []
        with patch.object(experiment, "_runtime_worker", side_effect=RuntimeError("synthetic worker failure")) as worker:
            with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
                self.invoke("train", config_path, "--seed", "42")
            worker.assert_called_once()
            with self.assertRaisesRegex(ProtocolError, "admission already exists"):
                self.invoke("train", config_path, "--seed", "42")
            worker.assert_called_once()
        self.assertTrue((root / "seed_42/admission.json").is_file())
        with patch.object(runner, "run_recover_seed") as recovery_worker:
            with self.assertRaisesRegex(ProtocolError, "training context"):
                self.invoke("recover-seed", config_path, "--seed", "42")
            recovery_worker.assert_not_called()
        ledger = json.loads((root / "gpu_budget.json").read_text(encoding="utf-8"))
        self.assertTrue(any(item["state"] == "failed" for item in ledger["reservations"]))

    def test_consumed_marker_blocks_base_smoke_and_recovery_before_workers(self) -> None:
        for command, attribute, extra in (
            ("base-eval", "run_base_eval", ()),
            ("smoke", "run_bounded_smoke", ()),
            ("recover-seed", "run_recover_seed", ("--seed", "42")),
        ):
            with self.subTest(command=command):
                config_path, config, root = self.make_config()
                (root / "final_test.attempt.json").write_text(
                    '{"synthetic_consumed":true}\n', encoding="utf-8"
                )
                with patch.object(runner, attribute) as worker:
                    with self.assertRaisesRegex(ProtocolError, "final test was consumed"):
                        self.invoke(command, config_path, *extra)
                    worker.assert_not_called()
                self.assertFalse((root / f"{command}.admission.json").exists())

    def test_bounded_smoke_uses_shared_budget_with_synthetic_torch(self) -> None:
        config_path, config, root = self.make_config()
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.return_value = True
        torch_stub.cuda.max_memory_reserved.return_value = 1024 ** 3
        with patch.object(experiment, "validate_manifest"), \
             patch.object(experiment, "verify_split_files"), \
             patch.object(experiment, "GpuLock", return_value=contextlib.nullcontext()), \
             patch.dict(sys.modules, {"torch": torch_stub}):
            result = experiment.run_bounded_smoke(
                config,
                {},
                review_accepted=True,
                config_path=config_path,
            )
        self.assertEqual(result["status"], "smoke_measured")
        ledger = json.loads((root / "gpu_budget.json").read_text(encoding="utf-8"))
        self.assertEqual(len(ledger["reservations"]), 1)
        self.assertEqual(ledger["reservations"][0]["reserved_seconds"], 600.0)
        self.assertEqual(ledger["reservations"][0]["state"], "finished")
        self.assertEqual(ledger["identity"]["config_path"], str(config_path.relative_to(ROOT)))

    def test_consumed_final_marker_rejects_before_budget_or_test_loader(self) -> None:
        config_path, config, root = self.make_config()
        (root / "final_test.attempt.json").write_text(
            '{"synthetic_consumed":true}\n', encoding="utf-8"
        )
        with patch.object(experiment, "_run_final_test_worker") as worker, \
             patch.object(experiment, "_load_split_bytes") as loader:
            with self.assertRaisesRegex(ProtocolError, "consumed"):
                experiment.run_final_test(
                    config,
                    {},
                    "a" * 64,
                    review_accepted=True,
                    config_path=config_path,
                )
            worker.assert_not_called()
            loader.assert_not_called()
        self.assertFalse((root / "gpu_budget.json").exists())

    def test_base_smoke_admit_and_block_repeat(self) -> None:
        # Base evaluation composes the real runtime and ledger; only its worker is patched.
        config_path, config, root = self.make_config()
        calls: list[dict] = []
        with patch.object(experiment, "_runtime_worker", side_effect=self.completed_worker(calls)):
            self.invoke("base-eval", config_path)
        admission = root / "base_eval.admission.json"
        self.assertTrue(admission.is_file())
        self.assertTrue(protocol.claim_path_for_admission(admission).is_file())
        self.assertTrue((root / "base_eval.result.json").is_file())
        self.assertEqual(calls[0]["config_path"], config_path)
        with patch.object(experiment, "_runtime_worker", side_effect=self.completed_worker(calls)):
            with self.assertRaises(ProtocolError):
                self.invoke("base-eval", config_path)
        self.assertEqual(len(calls), 1)

        config_path, config, root = self.make_config()
        smoke_admission = root / "smoke.admission.json"
        route_calls: list[dict] = []

        def smoke_worker(*args, **kwargs):
            self.assertTrue(smoke_admission.is_file())
            route_calls.append(kwargs)
            return {"status": "smoke_measured", "seed": config["budget"]["smoke_seed"]}

        with patch.object(runner, "run_bounded_smoke", side_effect=smoke_worker):
            self.invoke("smoke", config_path)
            with self.assertRaisesRegex(ProtocolError, "immutable artifact|admission already exists"):
                self.invoke("smoke", config_path)
        self.assertEqual(len(route_calls), 1)

    def test_historical_root_is_permanently_blocked_but_final_route_keeps_legacy_entry(self) -> None:
        historical = ROOT / "artifacts/experiment_i3/preflight/runs"
        config_path, config, _ = self.make_config(output_root=historical)
        for command, attribute, extra in (
            ("actual-model-pilot", "run_actual_model_pilot", ()),
            ("train", "run_training_seed", ("--seed", "42")),
            ("base-eval", "run_base_eval", ()),
            ("smoke", "run_bounded_smoke", ()),
            ("recover-seed", "run_recover_seed", ("--seed", "42")),
        ):
            with self.subTest(command=command), patch.object(runner, attribute) as worker:
                with self.assertRaisesRegex(ProtocolError, "permanently blocked"):
                    self.invoke(command, config_path, *extra)
                worker.assert_not_called()

        # The final-test branch deliberately does not call the worker-route
        # guard.  This proves the legitimate first historical attempt can reach
        # the one-shot final evaluator; the synthetic marker test below proves
        # its exact-one semantics without touching historical or test bytes.
        with patch.object(runner, "run_final_test", return_value={"status": "mocked-final-entry"}) as final_worker:
            self.invoke(
                "final-test",
                config_path,
                "--freeze-sha256",
                "a" * 64,
            )
            final_worker.assert_called_once()
            self.assertEqual(final_worker.call_args.kwargs["config_path"], config_path)

    def test_final_attempt_marker_is_create_once_on_synthetic_root(self) -> None:
        root = self.new_root()
        freeze = root / "freeze.json"
        freeze.write_bytes(b'{"synthetic_freeze":true}\n')
        freeze_hash = sha256(freeze)
        provenance = {
            "budget_token": "synthetic-final-token",
            "budget_reservation": {
                "token": "synthetic-final-token",
                "reserved_seconds": 600.0,
                "state": "in_flight",
            },
        }
        marker = consume_final_test_attempt(root, freeze_hash, provenance)
        self.assertTrue(marker.is_file())
        first_bytes = marker.read_bytes()
        with self.assertRaisesRegex(ProtocolError, "already consumed"):
            consume_final_test_attempt(root, freeze_hash, provenance)
        self.assertEqual(marker.read_bytes(), first_bytes)

    def test_cumulative_selection_and_direct_unauthorized_seed(self) -> None:
        config_path, config, root = self.make_config()
        _, _, _, decision = self.make_decision(config, config_path, projected_seconds=3500)
        self.assertEqual(decision["selected_seeds"], [42])
        self.assertEqual(decision["charged_seconds_before_training"], 600.0)
        self.assertEqual(decision["aggregate_composition"]["two_seed_total_with_final_reserve_seconds"], 8200.0)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        calls: list[dict] = []
        with patch.object(experiment, "_runtime_worker", side_effect=self.completed_worker(calls, "train")):
            self.invoke("train", config_path, "--seed", "42")
            self.assertTrue((root / "seed_42/result.json").is_file())
            with self.assertRaisesRegex(ProtocolError, "not authorized"):
                runner.run_training_seed(
                    config,
                    manifest,
                    314159,
                    review_accepted=True,
                    reservation_seconds=3500,
                    config_path=config_path,
                )
        self.assertEqual(len(calls), 1)
        ledger = json.loads((root / "gpu_budget.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["reservations"][0]["reserved_seconds"], 600.0)
        self.assertEqual(ledger["reservations"][1]["reserved_seconds"], 600.0)
        self.assertEqual(ledger["reservations"][1]["state"], "held")
        self.assertEqual(ledger["reservations"][2]["reserved_seconds"], 3500.0)

    def test_runner_main_uses_actual_training_path_and_alternate_config_identity(self) -> None:
        config_path, config, root = self.make_config()
        _, _, _, decision = self.make_decision(config, config_path, projected_seconds=3200)
        self.assertEqual(decision["selected_seeds"], [42])
        calls: list[dict] = []
        with patch.object(experiment, "_runtime_worker", side_effect=self.completed_worker(calls, "train")):
            self.invoke("train", config_path, "--seed", "42")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["config_path"], config_path)
        ledger = json.loads((root / "gpu_budget.json").read_text(encoding="utf-8"))
        self.assertEqual(len(ledger["reservations"]), 3)
        self.assertEqual(ledger["reservations"][0]["reserved_seconds"], 600.0)
        self.assertEqual(ledger["reservations"][2]["reserved_seconds"], 3200.0)
        self.assertEqual(ledger["identity"]["config_sha256"], sha256(config_path))
        self.assertEqual(ledger["identity"]["config_path"], str(config_path.relative_to(ROOT)))
        self.assertNotEqual(ledger["identity"]["config_sha256"], sha256(CONFIG_PATH))
        admission = json.loads((root / "seed_42/admission.json").read_text(encoding="utf-8"))
        self.assertEqual(admission["config_sha256"], sha256(config_path))
        self.assertEqual(admission["run_classification"], "independent_future_run")
        self.assertIs(admission["continues_consumed_historical_experiment"], False)
        self.assertTrue((root / "seed_42/result.json").is_file())

    def test_malformed_ledger_fails_in_runner_before_worker(self) -> None:
        config_path, config, root = self.make_config()
        self.make_decision(config, config_path, projected_seconds=3200)
        (root / "gpu_budget.json").write_text("{malformed\n", encoding="utf-8")
        with patch.object(experiment, "_runtime_worker") as worker:
            with self.assertRaisesRegex(ProtocolError, "malformed aggregate budget ledger"):
                self.invoke("train", config_path, "--seed", "42")
            worker.assert_not_called()
        self.assertFalse((root / "seed_42/admission.json").exists())
        with patch.object(experiment, "_runtime_worker") as worker:
            with self.assertRaises(ProtocolError):
                self.invoke("train", config_path, "--seed", "42")
            worker.assert_not_called()

    def test_pair_and_single_seed_fallback_admission(self) -> None:
        pair_path, pair_config, pair_root = self.make_config()
        _, _, pair_decision_path, pair_decision = self.make_decision(pair_config, pair_path)
        self.assertEqual(pair_decision["selected_seeds"], [42, 314159])
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with patch.object(experiment, "GpuLock") as lock:
            with self.assertRaisesRegex(ProtocolError, "one-seed fallback"):
                experiment.run_recover_seed(
                    pair_config, manifest, 42, review_accepted=True,
                    config_path=pair_path,
                )
            lock.assert_not_called()
        pair_calls: list[dict] = []
        with patch.object(experiment, "_runtime_worker", side_effect=self.completed_worker(pair_calls, "train")):
            self.invoke("train", pair_path, "--seed", "42")
            self.invoke("train", pair_path, "--seed", "314159")
        self.assertEqual(len(pair_calls), 2)
        self.assertTrue((pair_root / "seed_42/result.json").is_file())
        self.assertTrue((pair_root / "seed_314159/result.json").is_file())
        self.assertTrue(pair_decision_path.is_file())

        fallback_path, fallback_config, fallback_root = self.make_config()
        _, _, _, fallback_decision = self.make_decision(
            fallback_config, fallback_path, training_seconds=100
        )
        self.assertEqual(fallback_decision["selected_seeds"], [42])
        fallback_calls: list[dict] = []
        with patch.object(experiment, "_runtime_worker", side_effect=self.completed_worker(fallback_calls, "train")):
            self.invoke("train", fallback_path, "--seed", "42")
            with self.assertRaisesRegex(ProtocolError, "not authorized"):
                self.invoke("train", fallback_path, "--seed", "314159")
        self.assertEqual(len(fallback_calls), 1)
        self.assertTrue((fallback_root / "seed_42/result.json").is_file())

    def test_dev_selection_requires_exact_selected_seed_set(self) -> None:
        config_path, config, root = self.make_config()
        _, _, _, decision = self.make_decision(config, config_path, training_seconds=100)
        self.assertEqual(decision["selected_seeds"], [42])
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with patch.object(runner, "validate_pilot_decision", return_value=decision):
            with self.assertRaisesRegex(ProtocolError, r"missing=\[42\]"):
                runner.load_dev_candidates(config, manifest, config_path)
            for seed in (42, 314159):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                (seed_dir / "result.json").write_text(
                    json.dumps({"seed": seed, "status": "synthetic"}) + "\n",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ProtocolError, r"unexpected=\[314159\]"):
                runner.load_dev_candidates(config, manifest, config_path)
        require_exact_seed_set([42, 314159], [42, 314159])

    def test_pilot_provenance_rejects_changed_pilot_admission_ledger_and_config(self) -> None:
        for changed in ("pilot", "admission", "ledger", "config"):
            with self.subTest(changed=changed):
                config_path, config, root = self.make_config()
                _, pilot_path, decision_path, _ = self.make_decision(config, config_path, training_seconds=100)
                if changed == "pilot":
                    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
                    pilot["training_seconds"] += 1.0
                    pilot_path.write_text(json.dumps(pilot, sort_keys=True) + "\n", encoding="utf-8")
                elif changed == "admission":
                    admission = root / "future_actual_model_pilot.admission.json"
                    record = json.loads(admission.read_text(encoding="utf-8"))
                    record["config_sha256"] = "0" * 64
                    admission.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
                elif changed == "ledger":
                    ledger = root / "gpu_budget.json"
                    record = json.loads(ledger.read_text(encoding="utf-8"))
                    record["reservations"][0]["actual_seconds"] += 1.0
                    ledger.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
                else:
                    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaises(ProtocolError):
                    protocol.validate_pilot_decision(
                        pilot_path,
                        decision_path,
                        config_path,
                        MANIFEST_PATH,
                        config,
                        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
                    )

    def test_changed_decision_code_and_manifest_reject(self) -> None:
        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        _, pilot_path, decision_path, _ = self.make_decision(config, config_path, training_seconds=100)
        record = json.loads(decision_path.read_text(encoding="utf-8"))
        record["decision"]["selected_seeds"] = [42, 314159]
        decision_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(ProtocolError):
            protocol.validate_pilot_decision(pilot_path, decision_path, config_path, MANIFEST_PATH, config, manifest)

        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        _, pilot_path, decision_path, _ = self.make_decision(config, config_path, training_seconds=100)
        with patch.object(protocol, "code_identity", return_value={"mutated.py": "0" * 64}):
            with self.assertRaises(ProtocolError):
                protocol.validate_pilot_decision(pilot_path, decision_path, config_path, MANIFEST_PATH, config, manifest)

        config_path, config, root = self.make_config()
        manifest_copy = root / "manifest.json"
        manifest_copy.write_text(MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        config["data_manifest"] = str(manifest_copy.relative_to(ROOT))
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = json.loads(manifest_copy.read_text(encoding="utf-8"))
        _, pilot_path, decision_path, _ = self.make_decision(config, config_path, training_seconds=100)
        manifest_copy.write_text(manifest_copy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(ProtocolError):
            protocol.validate_pilot_decision(pilot_path, decision_path, config_path, manifest_copy, config, manifest)

        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.make_decision(config, config_path, training_seconds=100)
        result_path = self.install_completed_seed(config, config_path, root)
        decision, candidates = runner.load_dev_candidates(config, manifest, config_path)
        self.assertEqual(decision["selected_seeds"], [42])
        self.assertEqual(candidates[0]["seed"], 42)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["admission_sha256"] = "0" * 64
        result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "admission SHA-256"):
            runner.load_dev_candidates(config, manifest, config_path)

        foreign_path, foreign_config, foreign_root = self.make_config()
        self.make_decision(foreign_config, foreign_path, training_seconds=100)
        foreign_result = self.install_completed_seed(foreign_config, foreign_path, foreign_root)
        foreign_result_path = root / "seed_42" / "result.json"
        foreign_result_path.write_bytes(foreign_result.read_bytes())
        with self.assertRaises(ProtocolError):
            runner.load_dev_candidates(config, manifest, config_path)

    def test_seed_result_rejects_changed_decision_code_config_and_manifest(self) -> None:
        for changed in ("decision", "code", "config", "manifest"):
            with self.subTest(changed=changed):
                config_path, config, root = self.make_config()
                if changed == "manifest":
                    manifest_path = root / "manifest.json"
                    manifest_path.write_text(MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
                    config["data_manifest"] = str(manifest_path.relative_to(ROOT))
                    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                manifest = json.loads((manifest_path if changed == "manifest" else MANIFEST_PATH).read_text(encoding="utf-8"))
                self.make_decision(config, config_path, training_seconds=100)
                self.install_completed_seed(config, config_path, root)
                if changed == "decision":
                    path = root / "pilot_decision.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["decision"]["selected_seeds"] = [42, 314159]
                    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
                elif changed == "code":
                    with patch.object(protocol, "code_identity", return_value={"mutated.py": "0" * 64}):
                        with self.assertRaises(ProtocolError):
                            runner.load_dev_candidates(config, manifest, config_path)
                    continue
                elif changed == "config":
                    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                else:
                    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaises(ProtocolError):
                    runner.load_dev_candidates(config, manifest, config_path)

        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.make_decision(config, config_path, training_seconds=100)
        self.install_completed_seed(config, config_path, root)
        self.invoke("dev-select", config_path, review_accepted=False)
        freeze = root / "freeze.json"
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["result_artifact_sha256"]["42"] = "0" * 64
        freeze.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(ProtocolError):
            experiment._final_selection(
                root,
                sha256(freeze),
                config["labels"],
                config,
                manifest,
                config_path,
            )

    def test_recovery_requires_decision_before_lock_or_model(self) -> None:
        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with patch.object(experiment, "GpuLock") as lock:
            with self.assertRaises(ProtocolError):
                experiment.run_recover_seed(config, manifest, 42, review_accepted=True, config_path=config_path)
            lock.assert_not_called()
        self.assertFalse((root / "seed_42" / "admission.json").exists())

    def test_recovery_rejects_crafted_reservation_and_requires_training_chain(self) -> None:
        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        _, pilot_path, decision_path, decision = self.make_decision(config, config_path, training_seconds=100)
        self.assertEqual(decision["selected_seeds"], [42])
        identity = protocol.budget_identity(config, config_path, MANIFEST_PATH)
        ledger = protocol.AggregateBudget(root / "gpu_budget.json", identity, 7200)
        metadata = {
            "command": "train",
            "seed": 42,
            "admission_sha256": "0" * 64,
            "pilot_decision_sha256": sha256(decision_path),
        }
        token = ledger.reserve(1, metadata=metadata)
        ledger.finish(token, 1)
        with self.assertRaisesRegex(ProtocolError, "admission is missing"):
            runner.admit_worker(
                config, config_path, MANIFEST_PATH, "recover-seed", seed=42,
                pilot_decision_path=decision_path,
            )
        self.assertFalse((root / "seed_42" / "recovery_admission.json").exists())

    def test_recovery_composes_from_real_training_publication_failure_and_binds_chain(self) -> None:
        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        _, _, decision_path, decision = self.make_decision(config, config_path, training_seconds=100)
        self.assertEqual(decision["selected_seeds"], [42])

        def training_worker(*args, **kwargs):
            checkpoint = root / "seed_42" / "checkpoint"
            checkpoint.mkdir(parents=True, exist_ok=False)
            weights = checkpoint / "model.safetensors"
            state_file = checkpoint / "training_state.pt"
            weights.write_bytes(b"synthetic recovery weights\n")
            state_file.write_bytes(b"synthetic recovery state\n")
            model_digest = "a" * 64
            evidence = {
                "strict_model_load": True,
                "model_state_before_sha256": model_digest,
                "model_state_after_sha256": model_digest,
                "training_state_file_sha256": sha256(state_file),
                "optimizer_exact": True,
                "scheduler_exact": True,
                "scaler_exact": True,
                "stopping_state_exact": True,
            }
            evidence_path = checkpoint / "reload.json"
            evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
            return {
                "status": "completed",
                "seed": 42,
                "dev": {"macro_f1": 0.5, "nll": 0.7},
                "checkpoint": {
                    "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                    "weights_file_sha256": sha256(weights),
                    "model_state_sha256": model_digest,
                    "training_state_file_sha256": sha256(state_file),
                    "reload_evidence": evidence,
                    "reload_evidence_path": str(evidence_path.relative_to(ROOT)),
                    "reload_evidence_sha256": sha256(evidence_path),
                },
                "test_opened": False,
            }

        with patch.object(experiment, "_runtime_worker", side_effect=training_worker), patch.object(
            runner, "persist_seed_result", side_effect=RuntimeError("synthetic publication failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "publication failure"):
                self.invoke("train", config_path, "--seed", "42")
        training_admission = root / "seed_42" / "admission.json"
        training_context = root / "seed_42" / "training_context.json"
        self.assertTrue(training_admission.is_file())
        self.assertTrue(training_context.is_file())
        self.assertFalse((root / "seed_42" / "result.json").exists())
        ledger = json.loads((root / "gpu_budget.json").read_text(encoding="utf-8"))
        training_reservation = next(item for item in ledger["reservations"] if item.get("metadata", {}).get("command") == "train")
        self.assertEqual(training_reservation["state"], "finished")

        def recovery_override(**kwargs):
            checkpoint = kwargs["checkpoint"]
            evidence = json.loads((checkpoint / "reload.json").read_text(encoding="utf-8"))
            return {
                "state": {"updates": 68, "microforwards": 4204, "epochs": 4},
                "checkpoint_hashes": {
                    "weights_file_sha256": sha256(checkpoint / "model.safetensors"),
                    "training_state_file_sha256": sha256(checkpoint / "training_state.pt"),
                    "reload_evidence_sha256": sha256(checkpoint / "reload.json"),
                },
                "result": {
                    "status": "recovered",
                    "recovery": {"authorized": True, "retraining": False, "test_opened": False},
                    "seed": 42,
                    "dev": {"macro_f1": 0.5, "nll": 0.7},
                    "checkpoint": {
                        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                        "weights_file_sha256": sha256(checkpoint / "model.safetensors"),
                        "training_state_file_sha256": sha256(checkpoint / "training_state.pt"),
                        "model_state_sha256": evidence["model_state_before_sha256"],
                        "reload_evidence": evidence,
                        "reload_evidence_path": str((checkpoint / "reload.json").relative_to(ROOT)),
                        "reload_evidence_sha256": sha256(checkpoint / "reload.json"),
                    },
                    "cold_load_seconds": "unknown",
                    "training_seconds": "unknown",
                    "dev_evaluation_seconds": "unknown",
                    "latency": "unknown",
                    "loss_curve": "unknown",
                    "updates": 68,
                    "microforwards": 4204,
                    "test_opened": False,
                },
            }

        def recovery_worker(**kwargs):
            outcome = recovery_override(**kwargs)
            return outcome["checkpoint_hashes"], outcome["result"]

        with patch.object(experiment, "GpuLock", return_value=contextlib.nullcontext()), patch.object(
            experiment, "_run_recover_model_worker", side_effect=recovery_worker
        ):
            recovered = experiment.run_recover_seed(
                config, manifest, 42, review_accepted=True,
                config_path=config_path, pilot_decision_path=decision_path,
            )
        self.assertEqual(recovered["status"], "recovered")
        self.assertEqual(recovered["training_admission_path"], str(training_admission.relative_to(ROOT)))
        self.assertEqual(recovered["training_budget_token"], training_reservation["token"])
        self.assertTrue((root / "seed_42" / "recovery_admission.json").is_file())
        self.assertTrue((root / "seed_42" / "result.json").is_file())
        with self.assertRaisesRegex(ProtocolError, "immutable artifact"):
            experiment.run_recover_seed(
                config, manifest, 42, review_accepted=True,
                config_path=config_path, pilot_decision_path=decision_path,
            )

    def test_direct_claims_are_one_shot_for_every_worker_route(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        config_path, config, root = self.make_config()
        pilot_admission = runner.admit_worker(
            config, config_path, MANIFEST_PATH, "actual-model-pilot",
            seed=42, requested_wall_seconds=600.0,
        )
        pilot_worker = self.completed_worker([], "actual-model-pilot")
        with patch.object(experiment, "_runtime_worker", side_effect=pilot_worker):
            experiment.run_actual_model_pilot(
                config, manifest, review_accepted=True, config_path=config_path,
                admission_path=pilot_admission,
            )
            with self.assertRaisesRegex(ProtocolError, "execution claim already exists"):
                experiment.run_actual_model_pilot(
                    config, manifest, review_accepted=True, config_path=config_path,
                    admission_path=pilot_admission,
                )

        config_path, config, root = self.make_config()
        base_admission = runner.admit_worker(
            config, config_path, MANIFEST_PATH, "base-eval", requested_wall_seconds=600.0,
        )
        base_worker = self.completed_worker([], "base-eval")
        with patch.object(experiment, "_runtime_worker", side_effect=base_worker):
            experiment.run_base_eval(
                config, manifest, review_accepted=True, config_path=config_path,
                admission_path=base_admission,
            )
            with self.assertRaisesRegex(ProtocolError, "execution claim already exists"):
                experiment.run_base_eval(
                    config, manifest, review_accepted=True, config_path=config_path,
                    admission_path=base_admission,
                )

        config_path, config, root = self.make_config()
        smoke_admission = runner.admit_worker(
            config, config_path, MANIFEST_PATH, "smoke", requested_wall_seconds=600.0,
        )
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.return_value = True
        torch_stub.cuda.max_memory_reserved.return_value = 1024 ** 3
        with patch.object(experiment, "validate_manifest"), patch.object(experiment, "verify_split_files"), patch.object(
            experiment, "GpuLock", return_value=contextlib.nullcontext()
        ), patch.dict(sys.modules, {"torch": torch_stub}):
            experiment.run_bounded_smoke(
                config, manifest, review_accepted=True, config_path=config_path,
                admission_path=smoke_admission,
            )
            with self.assertRaisesRegex(ProtocolError, "execution claim already exists"):
                experiment.run_bounded_smoke(
                    config, manifest, review_accepted=True, config_path=config_path,
                    admission_path=smoke_admission,
                )

        config_path, config, root = self.make_config()
        _, _, decision_path, decision = self.make_decision(config, config_path, training_seconds=100)
        train_admission = runner.admit_worker(
            config, config_path, MANIFEST_PATH, "train", seed=42,
            pilot_decision_path=decision_path, requested_wall_seconds=decision["projected_gpu_seconds_per_seed"],
        )
        train_worker = self.completed_worker([], "train")
        with patch.object(experiment, "_runtime_worker", side_effect=train_worker):
            experiment.run_training_seed(
                config, manifest, 42, review_accepted=True,
                config_path=config_path, admission_path=train_admission,
                pilot_decision_path=decision_path,
            )
            with self.assertRaisesRegex(ProtocolError, "execution claim already exists"):
                experiment.run_training_seed(
                    config, manifest, 42, review_accepted=True,
                    config_path=config_path, admission_path=train_admission,
                    pilot_decision_path=decision_path,
                )

    def test_final_hold_is_activated_and_finished_without_a_second_charge(self) -> None:
        config_path, config, root = self.make_config()
        manifest, _, decision_path, _ = self.make_decision(config, config_path, projected_seconds=3200)
        self.install_completed_seed(config, config_path, root, 42)
        self.invoke("dev-select", config_path, review_accepted=False)
        freeze_hash = sha256(root / "freeze.json")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        token = decision["final_budget_token"]
        observed = {}

        def worker(*args, **kwargs):
            observed.update(kwargs["final_reservation"])
            self.assertEqual(kwargs["final_reservation"]["state"], "in_flight")
            return {"status": "synthetic-final-worker"}

        with patch.object(experiment, "_run_final_test_worker", side_effect=worker):
            result = experiment.run_final_test(
                config, manifest, freeze_hash, review_accepted=True, config_path=config_path,
            )
        self.assertEqual(result["status"], "synthetic-final-worker")
        self.assertEqual(observed["token"], token)
        ledger = json.loads((root / "gpu_budget.json").read_text(encoding="utf-8"))
        final_items = [item for item in ledger["reservations"] if item.get("token") == token]
        self.assertEqual(len(final_items), 1)
        self.assertEqual(final_items[0]["state"], "finished")
        self.assertEqual(sum(item["reserved_seconds"] for item in ledger["reservations"]), 600 + 600 + 3200)
        with self.assertRaises(ProtocolError):
            experiment.run_final_test(
                config, manifest, freeze_hash, review_accepted=True, config_path=config_path,
            )

    def test_authenticated_safetensors_loader_uses_open_fd_and_rejects_mutation(self) -> None:
        root = self.new_root()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        weights = checkpoint / "model.safetensors"
        weights.write_bytes(b"authenticated synthetic weights\n")
        expected = sha256(weights)
        seen: list[str] = []

        def load_file(path):
            seen.append(path)
            return {"ok": True}

        self.assertEqual(
            experiment.load_authenticated_safetensors(weights, expected, load_file),
            {"ok": True},
        )
        self.assertTrue(seen[0].startswith("/proc/self/fd/"))
        weights.write_bytes(b"mutated bytes\n")
        seen.clear()
        with self.assertRaisesRegex(ProtocolError, "changed before authenticated load"):
            experiment.load_authenticated_safetensors(weights, expected, load_file)
        self.assertEqual(seen, [])

    def test_post_decision_headroom_is_closed_and_training_allowance_is_exact(self) -> None:
        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        _, _, decision_path, decision = self.make_decision(config, config_path, projected_seconds=3200)
        with self.assertRaisesRegex(ProtocolError, "immutable pilot decision"):
            experiment.run_base_eval(config, manifest, review_accepted=True, config_path=config_path)
        with self.assertRaisesRegex(ProtocolError, "immutable pilot decision"):
            experiment.run_bounded_smoke(config, manifest, review_accepted=True, config_path=config_path)
        self.assertFalse((root / "base_eval.admission.json").exists())
        self.assertFalse((root / "smoke.admission.json").exists())
        with self.assertRaisesRegex(ProtocolError, "must equal"):
            experiment.run_training_seed(
                config, manifest, 42, review_accepted=True,
                reservation_seconds=3201, config_path=config_path,
                pilot_decision_path=decision_path,
            )
        self.assertFalse((root / "seed_42" / "admission.json").exists())
        self.assertEqual(decision["projected_gpu_seconds_per_seed"], 3200.0)

    def test_runtime_exposes_internal_budget_token_to_runner(self) -> None:
        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        admission_path = runner.admit_worker(
            config, config_path, MANIFEST_PATH, "base-eval", requested_wall_seconds=600.0
        )
        worker = self.completed_worker([], "base-eval")
        with patch.object(experiment, "_runtime_worker", side_effect=worker):
            result = experiment._runtime(
                config,
                manifest,
                command="base-eval",
                seed=42,
                review_accepted=True,
                config_path=config_path,
                admission_path=admission_path,
                train_enabled=False,
            )
        self.assertIn("_budget_token", result)
        self.assertEqual(result["budget_token"], result["_budget_token"])
        self.assertNotIn("_budget_token", experiment._public_result(result))

    def test_direct_worker_guards_run_before_budget_or_gpu(self) -> None:
        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ProtocolError, "review authorization"):
            experiment.run_actual_model_pilot(config, manifest, review_accepted=False, config_path=config_path)
        self.assertFalse((root / "future_actual_model_pilot.admission.json").exists())
        (root / "final_test.attempt.json").write_text('{"consumed":true}\n', encoding="utf-8")
        with patch.object(experiment, "AggregateBudget") as budget, patch.object(experiment, "GpuLock") as lock:
            with self.assertRaisesRegex(ProtocolError, "final test was consumed"):
                experiment.run_bounded_smoke(config, manifest, review_accepted=True, config_path=config_path)
            with self.assertRaisesRegex(ProtocolError, "final test was consumed"):
                experiment.run_recover_seed(config, manifest, 42, review_accepted=True, config_path=config_path)
            with self.assertRaisesRegex(ProtocolError, "final test was consumed"):
                experiment._runtime(
                    config,
                    manifest,
                    command="base-eval",
                    seed=42,
                    review_accepted=True,
                    config_path=config_path,
                    admission_path=root / "missing.admission.json",
                )
            budget.assert_not_called()
            lock.assert_not_called()

    def test_budget_boundary_and_no_double_allocation(self) -> None:
        self.assertEqual(
            set(protocol.code_identity()),
            {"runner.py", "experiment.py", "protocol.py", "metrics.py", "b2_train.py"},
        )
        pilot = {
            "status": "actual_model_pilot_measured",
            "seed": 42,
            "updates": 2,
            "training_seconds": 0.0,
            "cold_load_seconds": 0.0,
            "dev_evaluation_seconds": 0.0,
            "peak_reserved_gib": 8.0,
            "finite": True,
            "test_opened": False,
        }
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        updates = 68
        pilot["training_seconds"] = 3000.0 / config["budget"]["projection_multiplier"] * config["budget"]["smoke_updates"] / updates
        exact = protocol.choose_seed_count(pilot, config, 1051, charged_seconds=600.0)
        self.assertEqual(exact["selected_seeds"], [42, 314159])
        pilot["training_seconds"] += 0.001
        over = protocol.choose_seed_count(pilot, config, 1051, charged_seconds=600.0)
        self.assertEqual(over["selected_seeds"], [42])
        self.assertEqual(exact["aggregate_composition"]["two_seed_total_with_final_reserve_seconds"], 7200.0)
        runner_config_path, runner_config, _ = self.make_config()
        _, _, _, runner_decision = self.make_decision(runner_config, runner_config_path, projected_seconds=3000)
        self.assertEqual(runner_decision["selected_seeds"], [42, 314159])
        self.assertEqual(runner_decision["aggregate_composition"]["two_seed_total_with_final_reserve_seconds"], 7200.0)

        config_path, config, root = self.make_config()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        pilot_path, decision_path = runner.future_run_paths(config)
        self.make_decision(config, config_path, training_seconds=19)
        self.assertTrue(pilot_path.is_file())
        decision_bytes = decision_path.read_bytes()
        with self.assertRaisesRegex(ProtocolError, "already exists"):
            create_pilot_decision(
                pilot_path, decision_path, config_path, MANIFEST_PATH, config, manifest
            )
        self.assertEqual(decision_path.read_bytes(), decision_bytes)

        result = root / "seed_42/result.json"
        result.parent.mkdir()
        result.write_text('{"seed":42,"status":"synthetic"}\n', encoding="utf-8")
        decision = {"selected_seeds": [42]}
        selection = {"selected_seed": 42}
        with patch.object(protocol, "fsync_directory", wraps=protocol.fsync_directory) as fsync:
            with patch.object(
                runner,
                "load_dev_candidates",
                return_value=(decision, []),
            ), patch.object(runner, "select_dev_checkpoint", return_value=selection):
                self.invoke("dev-select", config_path, review_accepted=False)
            fsync.assert_called_once_with(root)
        freeze = root / "freeze.json"
        self.assertTrue(freeze.is_file())
        freeze_bytes = freeze.read_bytes()
        with patch.object(
            runner,
            "load_dev_candidates",
            return_value=(decision, []),
        ), patch.object(runner, "select_dev_checkpoint", return_value=selection):
            with self.assertRaisesRegex(ProtocolError, "already exists"):
                self.invoke("dev-select", config_path, review_accepted=False)
        self.assertEqual(freeze.read_bytes(), freeze_bytes)
    def test_final_authenticated_load_failure_does_not_consume_marker(self) -> None:
        config_path, config, root = self.make_config()
        manifest, _, decision_path, _ = self.make_decision(config, config_path, projected_seconds=3200)
        self.install_completed_seed(config, config_path, root, 42)
        self.invoke("dev-select", config_path, review_accepted=False)
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        token = decision["final_budget_token"]
        fake_safetensors = MagicMock()
        fake_safetensors.torch.load_file = MagicMock(return_value={})
        with patch.object(experiment, "GpuLock", return_value=contextlib.nullcontext()), patch.object(
            experiment, "load_authenticated_safetensors", side_effect=ProtocolError("synthetic unauthenticated bytes")
        ), patch.object(experiment, "consume_final_test_attempt") as marker, patch.dict(
            sys.modules, {"safetensors": fake_safetensors, "safetensors.torch": fake_safetensors.torch}
        ):
            with self.assertRaisesRegex(ProtocolError, "unauthenticated bytes"):
                experiment.run_final_test(
                    config, manifest, sha256(root / "freeze.json"), review_accepted=True, config_path=config_path,
                )
            marker.assert_not_called()
        ledger = json.loads((root / "gpu_budget.json").read_text(encoding="utf-8"))
        item = next(value for value in ledger["reservations"] if value.get("token") == token)
        self.assertEqual(item["state"], "failed")
        self.assertFalse((root / "final_test.attempt.json").exists())

    def test_pilot_snapshot_prefix_and_loaded_code_identity_fail_closed(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        config_path, config, root = self.make_config()
        _, pilot_path, decision_path, _ = self.make_decision(config, config_path, training_seconds=100)
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        self.assertEqual(pilot["budget_ledger_sha256"], pilot["budget_snapshot"]["sha256"])
        ledger = protocol.AggregateBudget(
            root / "gpu_budget.json", protocol.budget_identity(config, config_path, MANIFEST_PATH), 7200,
        )
        later = ledger.reserve(1, metadata={"command": "synthetic-later", "seed": 42})
        ledger.finish(later, 1)
        self.assertEqual(
            protocol.validate_pilot_decision(
                pilot_path, decision_path, config_path, MANIFEST_PATH, config, manifest,
            )["selected_seeds"], [42],
        )

        config_path, config, root = self.make_config()
        _, pilot_path, decision_path, _ = self.make_decision(config, config_path, training_seconds=100)
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        pilot["budget_reservation"]["metadata"].pop("command")
        pilot_path.write_text(json.dumps(pilot, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(ProtocolError):
            protocol.validate_pilot_decision(
                pilot_path, decision_path, config_path, MANIFEST_PATH, config, manifest,
            )

        config_path, config, root = self.make_config()
        _, pilot_path, decision_path, _ = self.make_decision(config, config_path, training_seconds=100)
        with patch.object(protocol, "code_identity", return_value={"mutated.py": "0" * 64}):
            with self.assertRaises(ProtocolError):
                protocol.validate_pilot_decision(
                    pilot_path, decision_path, config_path, MANIFEST_PATH, config, manifest,
                )
        with patch.object(protocol, "_IMPORT_TIME_CODE_IDENTITY", {"mutated.py": "0" * 64}):
            with self.assertRaises(ProtocolError):
                protocol.validate_pilot_decision(
                    pilot_path, decision_path, config_path, MANIFEST_PATH, config, manifest,
                )
