from __future__ import annotations

import unittest

from baselines.trace_action_i3.core import (
    LABELS,
    SparseLogistic,
    bootstrap_difference,
    evaluate_predictions,
    fit_logistic,
    majority_model,
    previous_action_from_state,
    transition_model,
    transition_predict,
)


def row(ident: str, trajectory: str, step: int, label: str, tool: str | None = None) -> dict:
    observation = "[NO OBSERVATION]" if tool is None else f"[TOOL]\nTOOL_RESULT id=x\nname={tool}"
    state = f"TASK\nsynthetic {label}\n\nLATEST RELEVANT OBSERVATION\n{observation}\n\nRECENT HISTORY\n[NO ADDITIONAL HISTORY]"
    return {"id": ident, "trajectory_id": trajectory, "step": step, "label": label, "state": state}


class BaselineTests(unittest.TestCase):
    def test_majority_tie_uses_fixed_label_order_and_smoothing(self) -> None:
        model = majority_model([row("a", "t", 1, "read"), row("b", "u", 1, "search")], LABELS, 1.0)
        self.assertEqual(model["prediction"], "read")
        self.assertTrue(all(value > 0 for value in model["probabilities"]))
        self.assertAlmostEqual(sum(model["probabilities"]), 1.0)

    def test_transition_fits_training_sequence_but_predicts_from_state(self) -> None:
        train = [row("a", "t", 1, "read"), row("b", "t", 2, "execute"),
                 row("c", "u", 1, "read"), row("d", "u", 2, "execute")]
        fallback = majority_model(train, LABELS, 0.5)["probabilities"]
        model = transition_model(train, LABELS, 0.5, fallback)
        inferred = transition_predict(model, [row("x", "v", 9, "search", "read")["state"]])[0]
        self.assertEqual(LABELS[max(range(len(LABELS)), key=lambda i: inferred[i])], "execute")
        self.assertEqual(previous_action_from_state(row("x", "v", 1, "read", None)["state"]), None)
        self.assertEqual(transition_predict(model, [row("x", "v", 1, "read", None)["state"]])[0], fallback)
        history_state = ("TASK\nname=write must not count\n\nLATEST RELEVANT OBSERVATION\n[NO OBSERVATION]"
                         "\n\nRECENT HISTORY\n[ASSISTANT]\nTOOL_CALL {\"name\":\"grep\"}")
        self.assertEqual(previous_action_from_state(history_state), "search")

    def test_transition_does_not_bridge_filtered_step_gap(self) -> None:
        train = [row("a", "t", 1, "read"), row("b", "t", 3, "execute")]
        fallback = majority_model(train, LABELS, 0.5)["probabilities"]
        model = transition_model(train, LABELS, 0.5, fallback)
        self.assertEqual(model["fit_transitions"], 0)
        self.assertEqual(model["skipped_step_gaps"], 1)

    def test_metrics_keep_missing_classes_in_macro_average(self) -> None:
        probabilities = [[0.9, 0.02, 0.02, 0.02, 0.02, 0.02], [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]]
        metrics = evaluate_predictions(["read", "search"], probabilities)
        self.assertAlmostEqual(metrics["macro_f1"], (2.0 / 3.0) / 6.0)
        self.assertEqual(metrics["per_class"]["edit"]["support"], 0)
        self.assertEqual(len(metrics["confusion"]["matrix"]), 6)

    def test_sparse_logistic_is_deterministic(self) -> None:
        texts = ["alpha alpha inspect", "alpha inspect", "omega run", "omega omega run"] * 3
        targets = ["read", "read", "execute", "execute"] * 3
        config = {"id": "tiny", "ngram_range": [1, 1], "min_df": 1, "max_features": 20,
                  "C": 1.0, "epochs": 8, "learning_rate": 0.2}
        first = fit_logistic(texts, targets, LABELS, config, 7)
        second = fit_logistic(texts, targets, LABELS, config, 7)
        self.assertEqual(first.serializable(), second.serializable())
        restored = SparseLogistic.from_serializable(first.serializable())
        self.assertEqual(first.predict_proba(texts), restored.predict_proba(texts))
        predictions = first.predict_proba(["alpha inspect", "omega run"])
        self.assertGreater(predictions[0][0], predictions[0][3])
        self.assertGreater(predictions[1][3], predictions[1][0])

    def test_bootstrap_resamples_whole_trajectories_deterministically(self) -> None:
        gold = ["read", "read", "search"]
        good = [[0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.02, 0.9, 0.02, 0.02, 0.02, 0.02]]
        bad = [[0.02, 0.9, 0.02, 0.02, 0.02, 0.02]] * 3
        first = bootstrap_difference(gold, good, bad, ["a", "a", "b"], replicates=20, seed=4)
        second = bootstrap_difference(gold, good, bad, ["a", "a", "b"], replicates=20, seed=4)
        self.assertEqual(first, second)
        self.assertEqual(first["independent_groups"], 2)


if __name__ == "__main__":
    unittest.main()
