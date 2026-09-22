import json
import unittest
from pathlib import Path

from i2_convert import LayaQ, MAX_LEN, HEAD_MAX_LEN, TOKENIZER_DEFAULT, AutoTokenizer, build_sequence, state_for


class I2Tests(unittest.TestCase):
    def test_pinned_empty_room_and_markers(self):
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DEFAULT, local_files_only=True)
        ids, markers = build_sequence(tok, "", LayaQ, MAX_LEN, HEAD_MAX_LEN)
        self.assertEqual(len(ids), 97)
        self.assertEqual(markers, [13, 23, 36, 47, 61, 77])
        self.assertEqual(MAX_LEN - len(ids), 415)

    def test_every_output_row_has_metadata_outside_state(self):
        root = Path(__file__).parents[2] / "artifacts/trace2decision_i2/output"
        rows = 0
        for split in ("train", "dev", "test"):
            for line in (root / f"{split}.jsonl").read_text().splitlines():
                row = json.loads(line); meta = row["metadata"]; rows += 1
                self.assertTrue({"id", "trajectory_id", "task_group", "source_step"} <= set(meta))
                self.assertNotIn(meta["id"], row["state"])
                self.assertLessEqual(meta["laya_sequence_length"], 512)
        self.assertEqual(rows, 1544)

    def test_newest_fit_is_rendered_in_source_order(self):
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DEFAULT, local_files_only=True)
        prefix = [
            {"role": "system", "content": "constraints"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "old assistant marker"},
            {"role": "tool", "tool_call_id": "old", "content": "old result"},
            {"role": "assistant", "content": "new assistant marker"},
            {"role": "tool", "tool_call_id": "new", "content": "new result"},
        ]
        state, info = state_for(prefix, tok, 415)
        self.assertEqual(info["history_source_indices"], sorted(info["history_source_indices"]))
        self.assertLess(state.index("old assistant marker"), state.index("new assistant marker"))

    def test_pre_policy_audit_is_distinct_and_has_threshold_counts(self):
        report = json.loads((Path(__file__).parents[2] / "artifacts/trace2decision_i2/output/report.json").read_text())
        audit = report["token_audit"]
        self.assertGreater(audit["pre_policy"]["mean"], audit["post_policy"]["768"]["mean"])
        self.assertGreater(audit["pre_policy"]["exceeding"]["512"]["count"], 0)
        self.assertGreater(audit["pre_policy"]["exceeding"]["768"]["count"], 0)
        self.assertGreater(audit["pre_policy"]["exceeding"]["1024"]["count"], 0)
        self.assertEqual(audit["pre_policy"]["exceeding"]["512"]["percentage"],
                         round(100 * audit["pre_policy"]["exceeding"]["512"]["count"] / 1544, 2))


if __name__ == "__main__":
    unittest.main()
