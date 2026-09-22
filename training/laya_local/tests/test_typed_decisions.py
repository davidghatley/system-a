import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.typed_decisions import ValidationError, parse_record
from shared.typed_decisions import load_jsonl
from training.laya_local.metrics import reconstruct_choice_counts
from training.laya_local.positive_control import answer_distribution, parse_public_record
from training.laya_local.data import to_laya_items


class SchemaTest(unittest.TestCase):
    def record(self):
        return {
            "id": "one",
            "state": json.dumps({"message": "hello"}),
            "questions": json.dumps({
                "route": {"type": "choice", "instructions": "Choose.", "criteria": {"a": "A", "b": "B"}},
                "safe": {"type": "noul", "instructions": "Safe?", "criteria": {"false": "no", "true": "yes"}},
                "level": {"type": "score", "instructions": "Level?", "criteria": ["low", "high"]},
            }),
            "gold": json.dumps({
                "route": {"type": "choice", "label": "b", "probabilities": {"a": 0.2, "b": 0.8}},
                "safe": {"type": "noul", "label": "true", "probabilities": {"false": 0.1, "true": 0.9}},
                "level": {"type": "score", "label": "0", "probabilities": {"0": 0.75, "1": 0.25}},
            }),
        }

    def test_decodes_native_hub_strings(self):
        parsed = parse_record(self.record())
        self.assertEqual(parsed["state"]["message"], "hello")
        self.assertEqual(parsed["gold"]["route"]["probabilities"]["b"], 0.8)

    def test_public_json_strings_preserve_id_workflow_and_types(self):
        row = self.record()
        row["workflow"] = "agent_trace_observability"
        parsed = parse_public_record(row)
        self.assertEqual(parsed["id"], "one")
        self.assertEqual(parsed["workflow"], "agent_trace_observability")
        self.assertEqual(list(parsed["questions"]), ["route", "safe", "level"])

    def test_score_and_noul_returned_argmax_semantics(self):
        noul = answer_distribution({"noul": 0.7}, {"type": "noul"})
        score = answer_distribution({"probabilities": {"0": 0.2, "1": 0.8}}, {"type": "score"})
        self.assertEqual(max(noul, key=noul.get), "true")
        self.assertEqual(max(score, key=score.get), "1")

    def test_trace_metadata_identity_stays_out_of_model_sequence(self):
        record = {
            "metadata": {"id": "trace-row", "trajectory_id": "trajectory"},
            "state": "state",
            "questions": {
                "route": {"type": "choice", "instructions": "Choose.", "criteria": {"a": "A", "b": "B"}},
            },
            "gold": {
                "route": {"type": "choice", "label": "b", "probabilities": {"a": 0.2, "b": 0.8}},
            },
        }
        common = types.ModuleType("laya.common")
        common.QTYPES = {"choice": 0}
        common.build_sequence = lambda tokenizer, state, question, max_len, head_max_len, truncate_left: ([1, 2], [1, 2])
        common.render_options = lambda question: ["a", "b"]
        laya = types.ModuleType("laya")
        with patch.dict(sys.modules, {"laya": laya, "laya.common": common}):
            item = to_laya_items(record, object(), max_len=512, head_max_len=192)[0]
        self.assertEqual(item["record_id"], "trace-row")
        self.assertEqual(item["metadata"], record["metadata"])
        self.assertEqual(item["ids"], [1, 2])
        self.assertEqual(item["markers"], [1, 2])

    def test_rejects_probability_key_reordering(self):
        record = self.record()
        gold = json.loads(record["gold"])
        gold["route"]["probabilities"] = {"b": 0.8, "a": 0.2}
        record["gold"] = json.dumps(gold)
        with self.assertRaisesRegex(ValidationError, "keys/order mismatch"):
            parse_record(record)

    def test_rejects_non_distribution(self):
        record = self.record()
        gold = json.loads(record["gold"])
        gold["safe"]["probabilities"]["true"] = 0.8
        record["gold"] = json.dumps(gold)
        with self.assertRaisesRegex(ValidationError, "sum to 1"):
            parse_record(record)

    def test_frozen_dev_reconstructs_native_counts_including_tie(self):
        root = Path(__file__).resolve().parents[3]
        dev = load_jsonl(root / "artifacts/train_i1/subsets/dev.jsonl")
        baseline = __import__("json").loads((root / "artifacts/train_i1/run/baseline.json").read_text())
        post = __import__("json").loads((root / "artifacts/train_i1/run/post_training.json").read_text())
        self.assertEqual(reconstruct_choice_counts(dev, baseline["predictions"]), {"choice_correct": 19, "choice_total": 40})
        self.assertEqual(reconstruct_choice_counts(dev, post["predictions"]), {"choice_correct": 11, "choice_total": 40})
        tie = next(row for row in baseline["predictions"] if row["record_id"] == "agent_trace_observability_000003" and row["question_id"] == "action")
        self.assertEqual(tie["gold_label"], 1)


if __name__ == "__main__":
    unittest.main()
