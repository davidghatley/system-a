#!/usr/bin/env python3
"""Verify one unchanged Trace2Decision v2 row across Iteration 2 tracks."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/trace2decision_i2/output/dev.jsonl"
OUTPUT = ROOT / "artifacts/iteration_2/integration_result.json"
LAYA = ROOT / "data/laya"
BASE_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
B2_CHECKPOINT = ROOT / "artifacts/train_i2/b2_run/checkpoint/model.safetensors"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "data/cache/huggingface/hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "data/cache/huggingface/transformers"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data/cache/torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "data/cache/xdg"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "apps/reflex"))
    sys.path.insert(0, str(LAYA))

    from shared.typed_decisions import load_jsonl, parse_record
    from training.laya_local.data import to_laya_items

    raw_line = SOURCE.open("rb").readline()
    raw_record = json.loads(raw_line)
    raw_before = canonical(raw_record)
    parsed = parse_record(raw_record)
    loaded = load_jsonl(SOURCE)[0]
    if parsed != loaded or canonical(raw_record) != raw_before:
        raise RuntimeError("shared validation/loading changed the source row")

    metadata = parsed.get("metadata")
    required_metadata = {"id", "trajectory_id", "task_group", "source_step"}
    if not isinstance(metadata, dict) or not required_metadata.issubset(metadata):
        raise RuntimeError("v2 stable metadata is missing")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model, collate_items

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; integration verification has no CPU fallback")
    device = torch.device("cuda:0")
    snapshot = ROOT / "data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots" / BASE_REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    items = to_laya_items(parsed, tokenizer, max_len=512, head_max_len=192)
    if len(items) != 1 or items[0]["record_id"] != metadata["id"] or items[0]["metadata"] != metadata:
        raise RuntimeError("trainer failed to retain stable v2 provenance out of band")

    stripped = {key: parsed[key] for key in ("state", "questions", "gold")}
    stripped_item = to_laya_items(stripped, tokenizer, max_len=512, head_max_len=192)[0]
    model_fields = ("ids", "markers", "qtype", "target", "label", "native_label", "gold_argmax", "option_keys")
    if any(items[0][field] != stripped_item[field] for field in model_fields):
        raise RuntimeError("metadata changed trainer model-facing construction")

    batch = collate_items([[items[0]]], tokenizer.pad_token_id)
    tensor_keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    if any(key not in batch for key in tensor_keys) or "metadata" in batch:
        raise RuntimeError("unexpected trainer tensor boundary")
    if batch["meta"][0]["metadata"] != metadata:
        raise RuntimeError("collator did not retain out-of-band metadata")

    model_cfg = json.loads((snapshot / "rl_agent_config.json").read_text())
    model_cfg.update(max_len=512, head_max_len=192)
    model = build_model(model_cfg, encoder_dir=str(snapshot / "encoder"))
    model.load_state_dict(load_file(B2_CHECKPOINT), strict=True)
    model.to(device).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits, _ = model(*(batch[key].to(device) for key in tensor_keys))
    option_count = len(items[0]["option_keys"])
    probabilities = torch.softmax(logits[0, :option_count].float(), -1).cpu().tolist()
    prediction_index = max(range(option_count), key=probabilities.__getitem__)
    b2_prediction = items[0]["option_keys"][prediction_index]
    b2_gold = items[0]["option_keys"][items[0]["native_label"]]
    b2_nll = float(-torch.log_softmax(logits[0, :option_count].float(), -1)[items[0]["native_label"]])
    del model
    torch.cuda.empty_cache()

    from reflex.core import Reflex

    reflex_started = time.perf_counter()
    reflex = Reflex.from_checkpoint(
        cache_dir=ROOT / "data/cache/huggingface/hub",
        laya_source=LAYA,
        device="cuda:0",
        offline=True,
        checkpoint="specialist",
    )
    reflex_result = reflex.decide(raw_record)
    reflex_seconds = time.perf_counter() - reflex_started
    if canonical(raw_record) != raw_before:
        raise RuntimeError("cross-track execution mutated the source row")
    answer = reflex_result["answers"]["next_action"]

    result = {
        "status": "passed",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_line": 1,
        "source_line_sha256": sha256_bytes(raw_line),
        "canonical_row_sha256_before_after": sha256_bytes(raw_before),
        "row_unchanged": True,
        "stable_metadata": {key: metadata[key] for key in sorted(required_metadata)},
        "shared": {"parse_equals_load": parsed == loaded, "metadata_preserved": parsed["metadata"] == metadata},
        "trainer": {
            "record_id": items[0]["record_id"],
            "input_tokens": len(items[0]["ids"]),
            "option_markers": len(items[0]["markers"]),
            "metadata_excluded_from_model_fields": True,
            "metadata_retained_in_collator_meta": True,
            "b2_checkpoint": str(B2_CHECKPOINT.relative_to(ROOT)),
            "prediction": b2_prediction,
            "gold_label": b2_gold,
            "correct": b2_prediction == b2_gold,
            "gold_nll": b2_nll,
            "probabilities": dict(zip(items[0]["option_keys"], probabilities)),
        },
        "reflex": {
            "checkpoint": reflex_result["checkpoint"],
            "selected": answer["selected"],
            "gold_label": parsed["gold"]["next_action"]["label"],
            "correct": answer["selected"] == parsed["gold"]["next_action"]["label"],
            "probabilities": answer["probabilities"],
            "reference_evaluation": reflex_result.get("reference_evaluation"),
            "load_and_inference_seconds": reflex_seconds,
        },
        "model_tensor_keys": list(tensor_keys),
        "limitations": [
            "One deterministic v2 row is an interoperability proof, not an accuracy estimate.",
            "The source trace's action label and task success were not independently authenticated.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
