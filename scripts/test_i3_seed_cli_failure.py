"""CPU-only composed CLI simulation of completed first seed then failed second."""
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

from training.laya_trace_i3 import runner
from training.laya_trace_i3.protocol import ROOT


class SeedCliFailureTest(unittest.TestCase):
    def test_second_seed_failure_retains_first_completed_result(self):
        fixture = ROOT / "artifacts/release_i3/seed_failure_fixtures" / uuid.uuid4().hex
        fixture.mkdir(parents=True)
        config = json.loads((ROOT / "training/laya_trace_i3/config.json").read_text())
        original_repo_path = runner.repo_path

        def redirect_output(value, *, artifact_output=False):
            if value == config["output_root"]:
                return fixture
            return original_repo_path(value, artifact_output=artifact_output)

        def simulated_seed(_config, _manifest, seed, *, review_accepted):
            self.assertTrue(review_accepted)
            if seed == 314159:
                raise RuntimeError("synthetic second-seed failure before publication")
            return {"seed": seed, "status": "completed", "test_opened": False}

        with mock.patch.object(runner, "require_compute_open"), mock.patch.object(runner, "run_training_seed", side_effect=simulated_seed), mock.patch.object(runner, "repo_path", side_effect=redirect_output), mock.patch("builtins.print"):
            with mock.patch.object(sys, "argv", ["runner.py", "train", "--seed", "42", "--review-accepted"]):
                runner.main()
            with mock.patch.object(sys, "argv", ["runner.py", "train", "--seed", "314159", "--review-accepted"]):
                with self.assertRaisesRegex(RuntimeError, "synthetic second-seed failure"):
                    runner.main()
        self.assertEqual(json.loads((fixture / "seed_42/result.json").read_text()), {"seed": 42, "status": "completed", "test_opened": False})
        self.assertFalse((fixture / "seed_314159/result.json").exists())


if __name__ == "__main__":
    unittest.main()
