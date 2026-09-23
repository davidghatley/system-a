#!/usr/bin/env python3
"""CPU-only, batch-one end-to-end bundle inference timing (no test data)."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release/i3"))
from inference import LocalLayaPredictor  # noqa: E402

SAMPLE = {
    "state": "A failing unit test points at the parser. Inspect the implementation before changing it.",
    "questions": {"next_action": {
        "type": "choice",
        "instructions": "Choose the next observable agent action type.",
        "criteria": {
            "edit": "Create or modify files or structured content.",
            "execute": "Execute a shell command, program, or test.",
            "other_tool": "Call another tool not covered by the named action types.",
            "read": "Read a known file or resource.",
            "respond_or_finish": "Respond without an executable tool call or finish the task.",
            "search": "Search or list files, symbols, or resources.",
        },
    }},
}


def main():
    start = time.perf_counter()
    predictor = LocalLayaPredictor(ROOT / "release/i3/bundle", device="cpu")
    cold_load = time.perf_counter() - start
    for _ in range(3):
        predictor.predict_pair(SAMPLE)
    values = []
    for _ in range(10):
        start = time.perf_counter()
        raw, calibrated = predictor.predict_pair(SAMPLE)
        values.append(time.perf_counter() - start)
        assert len(raw) == len(calibrated) == 6
    ordered = sorted(values)
    result = {
        "hardware": "local CPU; CUDA_VISIBLE_DEVICES empty; OMP/MKL/OPENBLAS threads 4",
        "sample": "synthetic fixed-schema next-action record; batch size one; no test data",
        "timed_boundary": "validate input, Laya sequence build/tokenize, collate CPU tensors, inference-mode forward, raw and dev-temperature softmax, CPU dictionary output; no device transfer on CPU",
        "cold_load_seconds": cold_load,
        "warmups": 3,
        "measured_samples": values,
        "median_seconds": statistics.median(values),
        "p95_seconds_nearest_rank": ordered[9],
        "p95_definition": "nearest-rank ceil(0.95*n) with n=10",
        "gpu_worker_seconds": 0,
    }
    path = ROOT / "artifacts/release_i3/cpu_latency.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "measured_samples"}, indent=2))


if __name__ == "__main__":
    main()
