#!/usr/bin/env python3
"""CPU/offline inference parity against saved dev predictions; never opens test."""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release/i3"))
from inference import LABELS, LocalLayaPredictor  # noqa: E402


def main():
    dev = ROOT / "artifacts/trace2decision_i3/output/dev.jsonl"
    rows = [json.loads(line) for line in dev.read_text().splitlines()]
    saved = json.loads((ROOT / "artifacts/experiment_i3/seed42_recovery.json").read_text())["dev"]["predictions"]
    by_id = {row["record_id"]: row for row in saved}
    predictor = LocalLayaPredictor(ROOT / "release/i3/bundle")
    comparisons = []
    for index in (0, 1, 2, 3, 4, 70, 139):
        row = rows[index]
        record_id = row["metadata"]["id"]
        # The public runtime accepts only the two input fields; gold and metadata
        # must never reach rendering, tokenization, or the network.
        public_input = {"state": row["state"], "questions": row["questions"]}
        actual, calibrated = predictor.predict_pair(public_input)
        reference = by_id[record_id]["probabilities"]
        scaled_logits = [math.log(max(reference[label], 1e-300)) / predictor.temperature for label in LABELS]
        shift = max(scaled_logits)
        exp_values = [math.exp(value - shift) for value in scaled_logits]
        expected_calibrated = dict(zip(LABELS, (value / sum(exp_values) for value in exp_values)))
        comparisons.append({
            "record_id": record_id,
            "max_absolute_probability_difference": max(abs(actual[label] - reference[label]) for label in LABELS),
            "predicted_label": max(actual, key=actual.get),
            "saved_predicted_label": max(reference, key=reference.get),
            "calibrated_max_absolute_difference_from_saved_raw_scaling": max(abs(calibrated[label] - expected_calibrated[label]) for label in LABELS),
            "actual_probabilities": actual,
            "saved_probabilities": reference,
        })
    digest = hashlib.sha256()
    with (ROOT / "release/i3/bundle/model/model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    out = {
        "scope": "seven existing dev records; CPU bundle inference versus saved seed42 dev predictions; no test access",
        "bundle_weights_sha256": digest.hexdigest(),
        "comparisons": comparisons,
        "max_absolute_probability_difference": max(x["max_absolute_probability_difference"] for x in comparisons),
        "calibrated_max_absolute_difference_from_saved_raw_scaling": max(x["calibrated_max_absolute_difference_from_saved_raw_scaling"] for x in comparisons),
        "argmax_matches": all(x["predicted_label"] == x["saved_predicted_label"] for x in comparisons),
    }
    assert out["max_absolute_probability_difference"] <= 1e-6
    assert out["calibrated_max_absolute_difference_from_saved_raw_scaling"] <= 1e-6
    assert out["argmax_matches"]
    path = ROOT / "artifacts/release_i3/package_parity.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
