import json
import math
import unittest
from pathlib import Path

from training.laya_local.b2_prepare import count_types, sha256
from training.laya_local.b2_train import (
    FROZEN_SUBSET_SHA256,
    checked_scaler_step,
    load_and_validate_training_state,
    nested_equal,
    proper_reward_reference,
    rlcd_objective,
    tree_to_cpu,
    validate_config,
    validate_model_structure,
    verify_frozen_subset,
)


ROOT = Path(__file__).resolve().parents[3]


class B2ConfigAndSubsetTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "training/laya_local/b2_config.json").read_text())

    def test_frozen_config_and_exact_schedule(self):
        validate_config(self.config)
        self.assertEqual(4 * 1280, 80 * 64)
        broken = dict(self.config, gradient_accumulation=63)
        with self.assertRaisesRegex(ValueError, "frozen config mismatch"):
            validate_config(broken)

    def test_independent_hashes_and_direct_parquet_equivalence(self):
        manifest = json.loads((ROOT / "artifacts/train_i2/b2_subset/manifest.json").read_text())
        verified_manifest, records = verify_frozen_subset(ROOT / "artifacts/train_i2/b2_subset", self.config)
        self.assertEqual(verified_manifest, manifest)
        self.assertEqual(FROZEN_SUBSET_SHA256, {
            "train": "d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656",
            "dev": "73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e",
            "manifest": "569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056",
        })
        for split, expected_rows, expected_questions in (("train", 256, 1280), ("dev", 100, 500)):
            rows = records[split]
            self.assertEqual(len(rows), expected_rows)
            self.assertEqual(sum(count_types(rows).values()), expected_questions)
        train_ids = manifest["splits"]["train"]["record_ids"]
        dev_ids = manifest["splits"]["dev"]["record_ids"]
        self.assertEqual(train_ids, [f"tr_agent_trace_observability_{i:06d}" for i in range(256)])
        self.assertEqual(dev_ids, [f"agent_trace_observability_{i:06d}" for i in range(100)])


class B2MathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_reward_log_floor_spherical_and_score_rps(self):
        torch = self.torch
        q = torch.tensor([[[0.0, 1.0]], [[0.75, 0.25]]])
        target = torch.tensor([[1.0, 0.0]])
        mask = torch.tensor([[True, True]])
        choice = proper_reward_reference(torch, q[:1], target, torch.tensor([0]), mask)
        self.assertAlmostEqual(float(choice), -9.21, places=5)
        score = proper_reward_reference(torch, q[1:], target, torch.tensor([1]), mask)
        expected = math.log(0.75) + 0.75 * (0.75 / math.sqrt(0.75 ** 2 + 0.25 ** 2)) - 0.25 ** 2
        self.assertAlmostEqual(float(score), expected, places=6)

    def test_rlcd_projection_advantage_and_action_zero_gradient(self):
        torch = self.torch
        torch.manual_seed(42)
        logits = torch.tensor([[0.2, -0.1, -1e4]], requires_grad=True)
        act = torch.tensor([[0.3, -0.2]], requires_grad=True)
        target = torch.tensor([[0.7, 0.3, 0.0]])
        mask = torch.tensor([[True, True, False]])
        reward = lambda *args, **kwargs: proper_reward_reference(torch, *args, **kwargs)
        terms = rlcd_objective(torch, logits, act, target, mask, torch.tensor([0]), reward, 0.4)
        self.assertTrue(torch.isfinite(terms["loss"]))
        self.assertTrue(torch.allclose((terms["eps"] * mask).sum(-1), torch.zeros(4, 1), atol=1e-7))
        self.assertAlmostEqual(float(terms["advantage"].mean(0)), 0.0, places=6)
        terms["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIsNotNone(act.grad)
        self.assertTrue(torch.equal(act.grad, torch.zeros_like(act.grad)))

    def test_checkpoint_tree_cpu_clone_is_exact_and_independent(self):
        torch = self.torch
        source = {"state": {1: {"step": torch.tensor(2.0)}}, "groups": [{"lr": 1e-4}]}
        cloned = tree_to_cpu(torch, source)
        self.assertTrue(nested_equal(torch, source, cloned))
        source["state"][1]["step"].add_(1)
        self.assertFalse(nested_equal(torch, source, cloned))

    def test_scale_64_accumulation_algebra(self):
        torch = self.torch
        value = torch.tensor(2.0, requires_grad=True)
        for _ in range(64):
            (64.0 * (value.square() / 64.0)).backward()
        unscaled = value.grad / 64.0
        self.assertEqual(float(unscaled), 4.0)

    def test_named_parameters_state_and_temperature_buffer_counts(self):
        class Sized:
            def __init__(self, size):
                self.size = size
            def numel(self):
                return self.size
        class Model:
            encoder_parameter = Sized(394781696)
            head_parameter = Sized(26512131)
            temperature = Sized(3)
            def named_parameters(self):
                return [("encoder.weight", self.encoder_parameter), ("head.weight", self.head_parameter)]
            def named_buffers(self):
                return [("temperature", self.temperature)]
            def state_dict(self):
                return dict(self.named_parameters() + self.named_buffers())
        named, encoder, nonencoder = validate_model_structure(Model())
        self.assertEqual([len(named), len(encoder), len(nonencoder)], [2, 1, 1])

    def test_live_optimizer_scheduler_scaler_round_trip_and_skip_guard(self):
        torch = self.torch
        source_parameter = torch.nn.Parameter(torch.tensor(1.0))
        source_optimizer = torch.optim.AdamW([source_parameter], lr=1e-3)
        source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(source_optimizer, T_max=80)
        source_parameter.grad = torch.tensor(2.0)
        source_optimizer.step()
        source_scheduler.step()
        class Scaler:
            def __init__(self, scale=64.0):
                self.scale = scale
            def state_dict(self):
                return {"scale": self.scale}
            def load_state_dict(self, state):
                self.scale = state["scale"]
            def get_scale(self):
                return self.scale
            def step(self, optimizer):
                return None
            def update(self):
                self.scale /= 2
        saved = tree_to_cpu(torch, {"optimizer": source_optimizer.state_dict(),
                                    "scheduler": source_scheduler.state_dict(), "scaler": {"scale": 64.0}})
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        scaler = Scaler(1.0)
        load_and_validate_training_state(torch, optimizer, scheduler, scaler, saved)
        self.assertTrue(nested_equal(torch, tree_to_cpu(torch, optimizer.state_dict()), saved["optimizer"]))
        class BadScaler(Scaler):
            def load_state_dict(self, state):
                self.scale = state["scale"] / 2
        with self.assertRaisesRegex(RuntimeError, "live scaler"):
            load_and_validate_training_state(torch, optimizer, scheduler, BadScaler(), saved)
        with self.assertRaisesRegex(RuntimeError, "overflow skipped"):
            checked_scaler_step(scaler, optimizer)


if __name__ == "__main__":
    unittest.main()
