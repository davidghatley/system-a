#!/usr/bin/env python3
"""Offline audit of I3 checkpoint evidence and dev-only predictions."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/experiment_i3/preflight/runs"
MANIFEST = ROOT / "artifacts/experiment_i3/preflight/data_manifest.v3.json"
CLASSES = ("edit", "execute", "other_tool", "read", "respond_or_finish", "search")
FLOOR = 1e-12
TOL = 1e-10


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def dev_metrics(rows):
    y = [r["gold_label"] for r in rows]
    pred = [max(r["probabilities"], key=r["probabilities"].get) for r in rows]
    n = len(rows)
    accuracy = sum(a == b for a, b in zip(y, pred)) / n
    f1s = []
    for c in CLASSES:
        tp = sum(a == c and b == c for a, b in zip(y, pred))
        fp = sum(a != c and b == c for a, b in zip(y, pred))
        fn = sum(a == c and b != c for a, b in zip(y, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    nll = -sum(math.log(max(FLOOR, r["probabilities"][r["gold_label"]])) for r in rows) / n
    brier = sum(sum((r["probabilities"][c] - (c == r["gold_label"])) ** 2 for c in CLASSES) for r in rows) / n
    return {"accuracy": accuracy, "macro_f1": statistics.mean(f1s), "nll": nll, "brier": brier}


def assert_metrics_close(actual, reported):
    for key, value in actual.items():
        assert math.isclose(value, reported[key], rel_tol=0.0, abs_tol=TOL), (key, value, reported[key])


def audit_seed(seed, dev_rows):
    d = RUNS / f"seed_{seed}"
    result = read(d / "result.json")
    reload_meta = read(d / "checkpoint/reload.json")
    checkpoint = result["checkpoint"]
    weights = d / "checkpoint/model.safetensors"
    state = d / "checkpoint/training_state.pt"
    checks = {
        "weights_digest_matches_result": sha(weights) == checkpoint["weights_file_sha256"],
        "state_digest_matches_result": sha(state) == checkpoint["training_state_file_sha256"],
        "reload_digest_matches_result": sha(d / "checkpoint/reload.json") == checkpoint["reload_evidence_sha256"],
        "checkpoint_model_state_matches_reload": reload_meta["model_state_after_sha256"] == checkpoint["model_state_sha256"],
        "state_digest_matches_reload": sha(state) == reload_meta["training_state_file_sha256"],
        "reload_before_after_equal": reload_meta["model_state_before_sha256"] == reload_meta["model_state_after_sha256"],
        "reload_equals_result_model": reload_meta["model_state_after_sha256"] == checkpoint["model_state_sha256"],
        # These are booleans written by the runner, not independent live-object proof.
        "reload_exact_flags_reported_true": all(reload_meta.get(k) is True for k in ("strict_model_load", "optimizer_exact", "scaler_exact", "scheduler_exact", "stopping_state_exact")),
    }
    dev = result["dev"]
    rows = dev["predictions"]
    y = [r["gold_label"] for r in rows]
    ids = [r["record_id"] for r in rows]
    gold = [r["gold_label"] for r in rows]
    probability_valid = all(set(r["probabilities"]) == set(CLASSES)
                            and all(math.isfinite(v) and v >= 0 for v in r["probabilities"].values())
                            and math.isclose(sum(r["probabilities"].values()), 1.0, abs_tol=1e-5)
                            for r in rows)
    membership = ids == [r["metadata"]["id"] for r in dev_rows] and gold == [r["gold"]["next_action"]["label"] for r in dev_rows]
    metrics = dev_metrics(rows)
    assert_metrics_close(metrics, dev)
    checks.update({"exact_dev_id_and_gold_order_matches_split": membership,
                   "prediction_count_matches_dev_split": len(rows) == len(dev_rows),
                   "probability_vectors_valid": probability_valid,
                   "recomputed_metrics_match_reported": True})
    return {"seed": seed, "checks": checks, "checkpoints": {"weights_sha256": sha(weights), "training_state_sha256": sha(state), "reload_json_sha256": sha(d / "checkpoint/reload.json")},
            "dev": {"records": len(rows), **metrics, "nll_probability_floor": FLOOR, "metric_tolerance": TOL,
                    "ids_sha256_sorted": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                    "labels_sha256_ordered": hashlib.sha256("\n".join(gold).encode()).hexdigest()}}


def audit_baseline(dev_rows):
    p = ROOT / "artifacts/baseline_i3/preflight/corrected_v3/dev_predictions.jsonl"
    baseline = [json.loads(line) for line in p.read_text().splitlines()]
    assert [(r["id"], r["gold"]) for r in baseline] == [(r["metadata"]["id"], r["gold"]["next_action"]["label"]) for r in dev_rows]
    probs = [r["methods"]["tfidf_logistic"]["probabilities"] for r in baseline]
    rows = [{"gold_label": r["gold"], "probabilities": r["methods"]["tfidf_logistic"]["probabilities"]} for r in baseline]
    # Validate the baseline distributions using the same named class set.
    assert all(set(p) == set(CLASSES) and all(math.isfinite(v) and v >= 0 for v in p.values())
               and math.isclose(sum(p.values()), 1.0, abs_tol=1e-5) for p in probs)
    metrics = dev_metrics(rows)
    baseline_result = read(ROOT / "artifacts/baseline_i3/preflight/corrected_v3/result.json")
    reported = baseline_result["methods"]["tfidf_logistic"]["metrics"]
    assert_metrics_close(metrics, reported)
    return {"source": str(p.relative_to(ROOT)), "sha256": sha(p), "method": "tfidf_logistic",
            "same_dev_id_gold_order": True, "reported_metrics_match": True,
            "records": len(rows), **metrics, "nll_probability_floor": FLOOR}


def main():
    manifest = read(MANIFEST)
    split = manifest["splits"]["dev"]
    dev_path = ROOT / split["path"]
    assert sha(dev_path) == split["sha256"], "accepted dev split SHA mismatch"
    # Read only the accepted dev split; never parse/open test records.
    dev_rows = [json.loads(line) for line in dev_path.read_text().splitlines()]
    assert len(dev_rows) == split["rows"]
    out = {"audit": "i3_evidence_audit", "scope": "checkpoint bytes and dev records only; does not open test split or final-test result",
           "manifest_sha256": sha(MANIFEST), "split_manifest": {k: {"sha256": v["sha256"], "rows": v["rows"], "trajectories": v["trajectories"]} for k, v in manifest["splits"].items()},
           "dev_split_sha256_verified": True, "dev_split_path": split["path"],
           "baseline": audit_baseline(dev_rows),
           "seeds": [audit_seed(s, dev_rows) for s in (42, 314159)]}
    outpath = ROOT / "artifacts/release_i3/evidence_audit.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    assert all(all(s["checks"].values()) for s in out["seeds"]), "checkpoint evidence mismatch"
    assert all(all(s["checks"].values()) for s in out["seeds"]), "dev/checkpoint audit failed"
    print(outpath.relative_to(ROOT))


if __name__ == "__main__":
    main()
