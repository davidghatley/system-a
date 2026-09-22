import pathlib
import copy
import subprocess
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from reflex_replay.core import ReplayError, evaluate_prediction, load_predictions, load_suite, replay


def prediction(selected, probabilities):
    return {
        "model": {"id": "fixture", "revision": "1"},
        "probabilities": probabilities,
        "elapsed_seconds": 0.01,
    }


class EvaluationTests(unittest.TestCase):
    def test_successful_prediction(self):
        values = {"edit": 0.05, "execute": 0.7, "other_tool": 0.05, "read": 0.1, "respond_or_finish": 0.05, "search": 0.05}
        result = evaluate_prediction(prediction("execute", values), "execute")
        self.assertTrue(result["correct"])
        self.assertFalse(result["ambiguous_low_margin"])

    def test_wrong_prediction(self):
        values = {"edit": 0.05, "execute": 0.7, "other_tool": 0.05, "read": 0.1, "respond_or_finish": 0.05, "search": 0.05}
        self.assertFalse(evaluate_prediction(prediction("execute", values), "edit")["correct"])

    def test_ambiguous_prediction_uses_frozen_margin(self):
        values = {"edit": 0.05, "execute": 0.31, "other_tool": 0.05, "read": 0.30, "respond_or_finish": 0.14, "search": 0.15}
        result = evaluate_prediction(prediction("execute", values), "execute")
        self.assertTrue(result["ambiguous_low_margin"])
        self.assertAlmostEqual(result["top_two_margin"], 0.01)

    def test_invalid_distribution_is_error(self):
        with self.assertRaisesRegex(ReplayError, "exactly"):
            evaluate_prediction(prediction("edit", {"edit": 1.0}), "edit")


class CliTests(unittest.TestCase):
    def test_named_case_runs_with_default_suite_predictions(self):
        run = subprocess.run(
            [sys.executable, "-m", "reflex_replay", "--case", "majority-success"],
            cwd=HERE,
            text=True,
            capture_output=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("EVALUATION: CORRECT", run.stdout)
        self.assertNotIn("CASE: accepted-integration", run.stdout)

    def test_malformed_stored_result_exits_two(self):
        run = subprocess.run(
            [sys.executable, "-m", "reflex_replay", "--predictions", str(HERE / "tests" / "invalid_predictions.jsonl")],
            cwd=HERE,
            text=True,
            capture_output=True,
        )
        self.assertEqual(run.returncode, 2)
        self.assertIn("error:", run.stderr)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_suite(HERE.parents[1], HERE / "examples" / "frozen_suite.json")
        self.stored = load_predictions(HERE / "examples" / "stored_predictions.jsonl")

    def test_tampering_bound_prediction_fields_fails_closed(self):
        for field in ("probabilities", "model", "reference", "elapsed_seconds"):
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.stored[0])
                if field == "probabilities":
                    tampered[field]["search"] += 0.001
                elif field == "model":
                    tampered[field]["revision"] = "tampered"
                elif field == "reference":
                    tampered[field] = "execute"
                else:
                    tampered[field] = 99.0
                with self.assertRaisesRegex(ReplayError, "integrity|bound reference"):
                    replay(self.cases, [tampered])

    def test_tampering_compact_source_hash_fails_closed(self):
        tampered = copy.deepcopy(self.stored[0])
        tampered["integrity"]["artifact"]["compact_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ReplayError, "artifact identity"):
            replay(self.cases, [tampered])


if __name__ == "__main__":
    unittest.main()
