import copy
import itertools
import json
import unittest
from pathlib import Path

from i3_convert import ONTOLOGY, TOKENIZER_DEFAULT, AutoTokenizer, classify_target, state_for


def row(calls, *, content="", step=1, steps=2):
    return {
        "assistant_step": step,
        "assistant_steps": steps,
        "target_message_index": 2,
        "messages": [
            {"role": "system", "content": "constraints"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": content, "tool_calls": calls},
        ],
    }


def call(identifier, name, arguments="{}"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


class I3Tests(unittest.TestCase):
    def test_one_and_same_category_multi_call_are_retained(self):
        self.assertEqual(classify_target(row([call("a", "read")]))["label"], "read")
        result = classify_target(row([call("a", "read"), call("b", "read_file")]))
        self.assertTrue(result["retain"])
        self.assertEqual(result["label"], "read")

    def test_mixed_category_exclusion_is_order_invariant(self):
        calls = [call("a", "read"), call("b", "grep"), call("c", "bash")]
        decisions = []
        for permutation in itertools.permutations(calls):
            result = classify_target(row(list(permutation)))
            decisions.append((result["retain"], result["reason"], tuple(result["categories"])))
        self.assertEqual(set(decisions), {(False, "mixed_target_categories", ("execute", "read", "search"))})

    def test_no_call_requires_completed_response_evidence(self):
        fixtures = [
            row([], content="reasoning only", step=2, steps=2),
            row([], content="partial answer", step=1, steps=2),
            row([], content="missing finish", step=2, steps=2),
            row([], content="", step=2, steps=2),
        ]
        fixtures[0]["messages"][-1]["reasoning_content"] = "unfinished reasoning"
        for fixture in fixtures:
            result = classify_target(fixture)
            self.assertFalse(result["retain"])
            self.assertEqual(result["reason"], "no_call_without_positive_completion")
        self.assertNotIn("label", classify_target(fixtures[0]))

    def test_malformed_call_is_excluded(self):
        malformed = call("a", "read")
        del malformed["function"]["arguments"]
        self.assertEqual(classify_target(row([malformed]))["reason"], "malformed_target_tool_call")

    def test_ambiguous_structure_is_explicitly_excluded(self):
        malformed = row([call("a", "read")])
        malformed["messages"][1] = "not a message"
        self.assertEqual(classify_target(malformed)["reason"], "malformed_message_structure")
        missing_calls = row(None)
        self.assertEqual(classify_target(missing_calls)["reason"], "target_tool_calls_not_list")

    def test_target_mutation_cannot_change_prefix_state(self):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DEFAULT, local_files_only=True)
        fixture = row([call("TARGET_ID", "bash", "TARGET_ARGUMENT_CANARY")])
        prefix = fixture["messages"][:-1]
        state, _ = state_for(prefix, tokenizer, 415)
        mutated = copy.deepcopy(fixture)
        mutated["messages"][-1] = {"role": "assistant", "content": "TARGET_CONTENT_CANARY", "tool_calls": []}
        mutated_state, _ = state_for(mutated["messages"][:-1], tokenizer, 415)
        self.assertEqual(state, mutated_state)
        self.assertNotIn("TARGET_ARGUMENT_CANARY", state)
        self.assertNotIn("TARGET_CONTENT_CANARY", state)

    def test_generated_counts_and_sample_strata(self):
        output = Path(__file__).parents[2] / "artifacts/trace2decision_i3/output"
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["conversion"]["retained_rows"], 1313)
        self.assertEqual(report["conversion"]["excluded_rows"], 508)
        self.assertEqual(report["conversion"]["mixed_category_turns"]["total"], 76)
        self.assertEqual(report["conversion"]["exclusion_reasons"]["no_call_without_positive_completion"], 208)
        self.assertEqual(report["ontology"]["absent_classes"], ["other_tool", "respond_or_finish"])
        self.assertEqual(report["split"]["v2_assignment_changes"], [])
        samples = [json.loads(line) for line in (output / "sample_review.jsonl").read_text().splitlines()]
        expected = {
            (split, label)
            for split in ("train", "dev")
            for label in ONTOLOGY
            if report["conversion"]["retained_by_split_category"][split].get(label, 0)
        }
        self.assertEqual({(sample["split"], sample["category"]) for sample in samples}, expected)


if __name__ == "__main__":
    unittest.main()
