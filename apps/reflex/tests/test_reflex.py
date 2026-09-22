import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from reflex import Reflex, ReflexInputError, ReflexModelError, validate_record
from reflex.core import _probabilities, _validate_laya_source_integrity, resolve_checkpoint
from shared.typed_decisions import parse_record


class FakeAgent:
    device = "cpu"

    def system_one(self, state, questions):
        return {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "inspect",
                    "probabilities": {"inspect": 0.75, "act": 0.25},
                    "confidence": 0.42,
                },
                "allow": {"type": "noul", "noul": 0.2, "confidence": 0.8},
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }


class ScoreAgent:
    device = "cpu"

    def system_one(self, state, questions):
        return {"answers": {"risk": {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "score": 1.6, "confidence": 0.7}}}


class RecordingAgent(FakeAgent):
    def system_one(self, state, questions):
        self.seen = (state, questions)
        return super().system_one(state, questions)


class MalformedAgent:
    device = "cpu"

    def __init__(self, response):
        self.response = response

    def system_one(self, state, questions):
        return self.response


def valid_record():
    return {
        "state": {"status": "signature unchecked"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose the next mode.",
                "criteria": {"inspect": "verify", "act": "change"},
            },
            "allow": {"type": "noul", "instructions": "Publishing is allowed."},
        },
        "gold": {
            "route": {"type": "choice", "label": "inspect", "probabilities": {"inspect": 1.0, "act": 0.0}},
            "allow": {"type": "noul", "label": "false", "probabilities": {"false": 1.0, "true": 0.0}},
        },
    }


class ValidationTests(unittest.TestCase):
    def test_accepts_native_contract(self):
        record = valid_record()
        self.assertEqual(validate_record(record), record)

    def test_rejects_bad_gold_sum(self):
        record = valid_record()
        record["gold"]["allow"]["probabilities"] = {"false": 0.8, "true": 0.3}
        with self.assertRaisesRegex(ReflexInputError, "sum to 1"):
            validate_record(record)

    def test_rejects_bad_question(self):
        record = valid_record()
        record["questions"]["route"]["criteria"] = {"only": "one"}
        with self.assertRaisesRegex(ReflexInputError, "at least two"):
            validate_record(record)

    def test_rejects_legacy_bare_gold(self):
        record = valid_record()
        record["gold"]["route"] = {"inspect": 1.0, "act": 0.0}
        with self.assertRaisesRegex(ReflexInputError, "native"):
            validate_record(record)

    def test_native_gold_and_trace2decision_metadata_are_preserved(self):
        record = {
            "id": "trace-7-step-2", "trajectory_id": "trace-7", "task_group": "security",
            "source_step": 2, "metadata": {"split": "test"},
            "state": {"observation": "signature missing"},
            "questions": {"route": {"type": "choice", "instructions": "Choose.", "criteria": {"inspect": "check", "act": "change"}}},
            "gold": {"route": {"type": "choice", "label": "inspect", "probabilities": {"inspect": 1.0, "act": 0.0}}},
        }
        parsed = parse_record(record)
        self.assertEqual(parsed["trajectory_id"], "trace-7")
        self.assertEqual(parsed["metadata"], {"split": "test"})
        self.assertEqual(validate_record(record)["gold"]["route"]["label"], "inspect")

    def test_gold_is_optional_for_inference(self):
        record = valid_record()
        record.pop("gold")
        parsed = parse_record(record)
        self.assertNotIn("gold", parsed)
        agent = RecordingAgent()
        self.assertNotIn("reference_evaluation", Reflex(agent).decide(record))

    def test_goldless_malformed_questions_are_rejected(self):
        for field, value in (("type", "unsupported"), ("instructions", "   "), ("criteria", {"only": "one"})):
            with self.subTest(field=field):
                record = valid_record()
                record.pop("gold")
                record["questions"]["route"][field] = value
                agent = RecordingAgent()
                with self.assertRaises(ReflexInputError):
                    Reflex(agent).decide(record)
                self.assertFalse(hasattr(agent, "seen"))

    def test_trace2decision_record_reaches_model_without_metadata(self):
        record = {
            "id": "r1", "trajectory_id": "t1", "task_group": "agent-trace",
            "source_step": 4, "metadata": {"annotator": "fixture"},
            "state": {"observation": "check"},
            "questions": {"route": {"type": "choice", "instructions": "Choose.", "criteria": {"inspect": "check", "act": "change"}}, "allow": {"type": "noul", "instructions": "Allow."}},
            "gold": {"route": {"type": "choice", "label": "inspect", "probabilities": {"inspect": 1.0, "act": 0.0}}, "allow": {"type": "noul", "label": "false", "probabilities": {"false": 1.0, "true": 0.0}}},
        }
        agent = RecordingAgent()
        result = Reflex(agent).decide(record)
        self.assertEqual(result["reference_evaluation"]["route"]["gold_argmax"], "inspect")
        self.assertNotIn("metadata", agent.seen[0])
        self.assertEqual(agent.seen[0], record["state"])


class SourceIntegrityTests(unittest.TestCase):
    def test_ignored_python_bytecode_cache_is_allowed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            _validate_laya_source_integrity(
                pathlib.Path(directory), "!! laya/__pycache__/agent.cpython-314.pyc"
            )
            _validate_laya_source_integrity(pathlib.Path(directory), "!! laya/__pycache__/")

    def test_tracked_python_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            source = pathlib.Path(directory)
            (source / "laya").mkdir()
            (source / "laya" / "agent.py").write_text("# modified")
            with self.assertRaisesRegex(ReflexModelError, "changed import payload"):
                _validate_laya_source_integrity(source, " M laya/agent.py")

    def test_untracked_python_and_ignored_native_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            source = pathlib.Path(directory)
            (source / "laya").mkdir()
            (source / "laya" / "injected.py").write_text("payload")
            with self.assertRaisesRegex(ReflexModelError, "injected.py"):
                _validate_laya_source_integrity(source, "?? laya/injected.py")
            with self.assertRaisesRegex(ReflexModelError, "native.cpython-314-x86_64-linux-gnu.so"):
                _validate_laya_source_integrity(
                    source, "!! laya/native.cpython-314-x86_64-linux-gnu.so"
                )


class OutputTests(unittest.TestCase):
    def test_emits_complete_probabilities(self):
        result = Reflex(FakeAgent()).decide(valid_record())
        self.assertEqual(result["answers"]["route"]["selected"], "inspect")
        self.assertEqual(result["answers"]["allow"]["probabilities"], {"false": 0.8, "true": 0.2})
        self.assertTrue(result["reference_evaluation"]["route"]["argmax_agrees"])
        self.assertIn("not a universal policy", result["warning"])

    def test_score_emits_distribution_and_expected_score(self):
        record = {"state": {}, "questions": {"risk": {"type": "score", "instructions": "Rate risk.", "criteria": ["low", "medium", "high"]}}}
        answer = Reflex(ScoreAgent()).decide(record)["answers"]["risk"]
        self.assertEqual(answer["expected_score"], 1.6)
        self.assertEqual(answer["probabilities"], {"0": 0.1, "1": 0.2, "2": 0.7})

    def test_malformed_model_responses_are_actionable(self):
        cases = [{}, {"answers": {"route": {"type": "choice", "confidence": 0.5}}},
                 {"answers": {"route": {"type": "choice", "probabilities": {"inspect": float("nan"), "act": 0.5}, "choice": "inspect", "confidence": 0.5}}}]
        for response in cases:
            with self.subTest(response=response), self.assertRaisesRegex(ReflexModelError, "invalid response"):
                Reflex(MalformedAgent(response)).decide(valid_record())

    def test_noul_displayed_probabilities_sum_to_one(self):
        response = {"answers": {"allow": {"type": "noul", "noul": 0.12345, "confidence": 0.8}, "route": {"type": "choice", "choice": "inspect", "probabilities": {"inspect": 0.75, "act": 0.25}, "confidence": 0.42}}}
        answer = Reflex(MalformedAgent(response)).decide(valid_record())["answers"]["allow"]
        self.assertEqual(sum(answer["probabilities"].values()), 1.0)
        self.assertEqual(answer["probabilities"], {"false": 0.8765, "true": 0.1235})

    def test_normalizes_model_rounding_error_only(self):
        names = [str(index) for index in range(6)]
        rounded = {name: value for name, value in zip(names, [0.1666] * 5 + [0.1668])}
        normalized = _probabilities(rounded, names, "route")
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            _probabilities({name: 0.16 for name in names}, names, "route")

    def test_invalid_probability_range_fails_cleanly(self):
        response = {"answers": {"route": {"type": "choice", "choice": "inspect", "probabilities": {"inspect": 1.2, "act": -0.2}, "confidence": 0.5}, "allow": {"type": "noul", "noul": 0.2, "confidence": 0.8}}}
        with self.assertRaisesRegex(ReflexModelError, "probabilities.inspect"):
            Reflex(MalformedAgent(response)).decide(valid_record())

    def test_repeated_output_is_stable(self):
        reflex = Reflex(FakeAgent())
        self.assertEqual(reflex.decide(valid_record()), reflex.decide(valid_record()))

    def test_unavailable_source_is_actionable(self):
        with self.assertRaisesRegex(ReflexModelError, "source is unavailable"):
            Reflex.from_checkpoint(laya_source=HERE / "does-not-exist", offline=True)

    def test_unavailable_checkpoint_is_actionable(self):
        with mock.patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError("not cached")):
            with self.assertRaisesRegex(ReflexModelError, "Run once without --offline"):
                resolve_checkpoint(HERE / "cache", offline=True)


class CliTests(unittest.TestCase):
    def test_malformed_json_is_actionable_without_loading_model(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = pathlib.Path(directory) / "bad.json"
            path.write_text("{")
            run = subprocess.run(
                [sys.executable, "-m", "reflex", str(path), "--offline"],
                cwd=HERE,
                text=True,
                capture_output=True,
            )
        self.assertEqual(run.returncode, 2)
        self.assertIn("invalid JSON", run.stderr)


if __name__ == "__main__":
    unittest.main()
