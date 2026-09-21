import json
import unittest
from pathlib import Path

from convert import ONTOLOGY, action_for_tool, convert, render_state


def row(task: str, trajectory: str, target: dict, step: int = 1) -> dict:
    messages = [
        {"role": "system", "content": "constraints", "reasoning_content": "", "tool_calls": [], "tool_call_id": "", "name": ""},
        {"role": "user", "content": task, "reasoning_content": "", "tool_calls": [], "tool_call_id": "", "name": ""},
        target,
    ]
    return {
        "task": task,
        "source_trajectory_sha256": trajectory,
        "split": "train",
        "assistant_step": step,
        "assistant_steps": step,
        "target_message_index": 2,
        "n_messages": 3,
        "messages": messages,
    }


class ConverterTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(action_for_tool("read"), "read")
        self.assertEqual(action_for_tool("grep"), "search")
        self.assertEqual(action_for_tool("apply_patch"), "edit")
        self.assertEqual(action_for_tool("bash"), "execute")
        self.assertEqual(action_for_tool("delegate"), "other_tool")

    def test_target_is_not_rendered(self):
        prefix = [{"role": "user", "content": "task"}]
        before, _ = render_state(prefix)
        target = {"role": "assistant", "content": "TARGET_CANARY", "tool_calls": []}
        after, _ = render_state(prefix)
        self.assertEqual(before, after)
        self.assertNotIn(target["content"], after)

    def test_contract_and_group_split(self):
        call = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "SECRET_REASONING",
            "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": "{}"}}],
            "tool_call_id": "",
            "name": "",
        }
        response = {"role": "assistant", "content": "done", "reasoning_content": "SECRET", "tool_calls": [], "tool_call_id": "", "name": ""}
        root = Path(__file__).parents[2] / "artifacts" / "trace2decision_i1" / "test_output"
        root.mkdir(parents=True, exist_ok=True)
        source = root / "fixture.jsonl"
        source.write_text(json.dumps(row("same task", "a" * 64, call)) + "\n" + json.dumps(row("same task", "b" * 64, response)) + "\n")
        report = convert(source, root)
        self.assertEqual(report["split"]["task_group_crossings"], 0)
        self.assertEqual(report["conversion"]["target_mutation_failures"], 0)
        records = []
        for split in ("train", "dev", "test"):
            records.extend(json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines())
        self.assertEqual(len(records), 1)  # identical rendered pre-target state deduplicates
        self.assertEqual(set(records[0]), {"state", "questions", "gold"})
        self.assertEqual(set(records[0]["gold"]["next_action"]["probabilities"]), set(ONTOLOGY))
        self.assertAlmostEqual(sum(records[0]["gold"]["next_action"]["probabilities"].values()), 1.0)
        self.assertNotIn("SECRET", records[0]["state"])


if __name__ == "__main__":
    unittest.main()
