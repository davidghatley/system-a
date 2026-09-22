#!/usr/bin/env python3
"""Run the bounded public typed-decisions B1 positive-control evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/train_i1/source_test.parquet"
SOURCE_REV = "ea9306458d6e9563628369a3d1e72e362fb381d2"
PREDECLARATION = ROOT / "artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md"
MODELS = {
    "base": ("convaiinnovations/laya", "1c5edc17a7acd8701df6fc341c0d179f1c62c982"),
    "specialist": ("convaiinnovations/laya-typed-decisions", "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2"),
}
EXPECTED_SHA = "833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_public_record(row: dict[str, Any]) -> dict[str, Any]:
    """Decode Hub parquet JSON columns without changing public IDs or order."""
    out = dict(row)
    for key in ("state", "questions", "gold"):
        if isinstance(out[key], str):
            out[key] = json.loads(out[key])
    return out


def answer_distribution(answer: dict[str, Any], question: dict[str, Any]) -> dict[str, float]:
    """Return the native option distribution for all three Laya answer types."""
    if question["type"] == "noul":
        p_true = float(answer["noul"])
        return {"false": 1.0 - p_true, "true": p_true}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def evaluate(agent: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = []
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    nll = 0.0
    for row_index, record in enumerate(records):
        result = agent.predict(record["state"], record["questions"])
        for qid, question in record["questions"].items():
            answer = result["answers"][qid]
            gold = record["gold"][qid]["label"]
            probs = answer_distribution(answer, question)
            prediction = max(probs, key=probs.get)
            probability = max(probs[gold], 1e-12)
            correct = prediction == gold
            qtype = question["type"]
            workflow = record["workflow"]
            groups[(workflow, qtype)].append(correct)
            nll -= math.log(probability)
            predictions.append({
                "record_id": record["id"], "workflow": workflow, "question_id": qid,
                "type": qtype, "gold_label": gold, "prediction": prediction,
                "probabilities": probs, "gold_probability": probability, "correct": correct,
            })
    def summary(rows: list[bool]) -> dict[str, Any]:
        return {"questions": len(rows), "correct": sum(rows),
                "accuracy": sum(rows) / len(rows) if rows else None}
    by_workflow_type = {f"{w}/{t}": summary(v) for (w, t), v in sorted(groups.items())}
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_workflow: dict[str, list[bool]] = defaultdict(list)
    for (w, t), values in groups.items():
        by_type[t].extend(values)
        by_workflow[w].extend(values)
    total = len(predictions)
    return {
        "questions": total, "correct": sum(p["correct"] for p in predictions),
        "all_question_hard_accuracy": sum(p["correct"] for p in predictions) / total,
        "choice": summary(by_type["choice"]), "score": summary(by_type["score"]),
        "noul": summary(by_type["noul"]),
        "by_workflow": {w: summary(v) for w, v in sorted(by_workflow.items())},
        "by_workflow_type": by_workflow_type,
        "gold_nll_mean": nll / total, "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "artifacts/train_i2/positive_control_public.json")
    parser.add_argument("--cache", type=pathlib.Path, default=ROOT / "data/cache")
    parser.add_argument("--laya-source", type=pathlib.Path, default=ROOT / "data/laya")
    args = parser.parse_args()
    if sha256(DATA) != EXPECTED_SHA:
        raise RuntimeError("frozen public source SHA-256 mismatch")
    if not PREDECLARATION.exists():
        raise RuntimeError("public predeclaration must exist before model loading")
    predecl_sha = sha256(PREDECLARATION)
    os.environ.update({"HF_HOME": str(args.cache / "huggingface"), "HF_HUB_CACHE": str(args.cache / "huggingface/hub"),
                       "TRANSFORMERS_CACHE": str(args.cache / "huggingface/transformers"), "TORCH_HOME": str(args.cache / "torch"),
                       "XDG_CACHE_HOME": str(args.cache / "xdg"), "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
    import pyarrow.parquet as pq
    records = [parse_public_record(r) for r in pq.read_table(DATA).to_pylist()]
    if len(records) != 100 or any(r["workflow"] != "agent_trace_observability" for r in records):
        raise RuntimeError("unexpected frozen public subset")
    if args.laya_source.joinpath(".git").exists():
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.laya_source, text=True).strip()
        if rev != "d113dca2512fb3eaca313534bc54c7162d87c1d4":
            raise RuntimeError(f"Laya source revision mismatch: {rev}")
    sys.path.insert(0, str(args.laya_source))
    from huggingface_hub import snapshot_download
    import laya
    common = {"max_len": 512, "head_max_len": 192}
    result = {"status": "started", "subset": {"path": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA), "rows": len(records)},
              "source": {"dataset": "LocalLLaMA/typed-decisions", "config": "agent_trace_observability", "split": "test", "revision": SOURCE_REV},
              "predeclaration_sha256": predecl_sha, "laya_source_revision": "d113dca2512fb3eaca313534bc54c7162d87c1d4",
              "construction": common, "models": {}, "commands": [".venv/bin/python -m unittest discover -s training/laya_local/tests -v",
              "env TMPDIR=$PWD/data/cache/tmp HF_HOME=$PWD/data/cache/huggingface HF_HUB_CACHE=$PWD/data/cache/huggingface/hub TRANSFORMERS_CACHE=$PWD/data/cache/huggingface/transformers TORCH_HOME=$PWD/data/cache/torch XDG_CACHE_HOME=$PWD/data/cache/xdg .venv/bin/python training/laya_local/positive_control.py"]}
    for name, (model_id, revision) in MODELS.items():
        started = time.perf_counter()
        model_dir = snapshot_download(model_id, revision=revision, cache_dir=args.cache / "huggingface/hub")
        agent = laya.load(model_dir, device="cuda")
        agent.cfg.update(common)
        metrics = evaluate(agent, records)
        result["models"][name] = {"id": model_id, "revision": revision, "wall_seconds": time.perf_counter() - started, **metrics}
        del agent
        import torch
        torch.cuda.empty_cache()
    result["status"] = "completed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: {m: v[m] for m in ("choice", "all_question_hard_accuracy", "gold_nll_mean", "wall_seconds")} for k, v in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
