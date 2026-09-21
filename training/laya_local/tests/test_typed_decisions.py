import json
import unittest
from pathlib import Path

from shared.typed_decisions import ValidationError, parse_record
from shared.typed_decisions import load_jsonl
from training.laya_local.metrics import reconstruct_choice_counts


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
        with self.assertRaisesRegex(ValidationError, "sum to one"):
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
