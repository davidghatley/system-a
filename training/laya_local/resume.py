#!/usr/bin/env python3
"""Bounded continuation: restore a checkpoint and perform one optimizer update."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.laya_local.train import ROOT as TRAIN_ROOT, parameter_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import Adafactor, AutoTokenizer
    sys.path.insert(0, str(ROOT / "data/laya"))
    from laya.common import build_model, collate_items
    from shared.typed_decisions import load_jsonl
    from training.laya_local.data import to_laya_items

    if not torch.cuda.is_available():
        raise RuntimeError("resume smoke requires the authorized GPU")
    state = torch.load(args.checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    required = ("optimizer", "scaler", "step", "data_cursor", "data_order", "data_rng_state",
                "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state")
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"checkpoint is not resume-capable; missing {missing}")
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    cache = ROOT / "data/cache"
    snapshot = cache / "huggingface/hub/models--convaiinnovations--laya/snapshots" / cfg["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    model_cfg = json.loads((snapshot / "rl_agent_config.json").read_text())
    model_cfg.update(max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])
    items = [item for record in load_jsonl(args.train) for item in to_laya_items(record, tokenizer, max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])]
    model = build_model(model_cfg, encoder_dir=str(snapshot / "encoder"))
    model.load_state_dict(load_file(args.checkpoint / "model.safetensors"), strict=True)
    model.encoder.config.reference_compile = False
    model.to("cuda")
    named = list(model.named_parameters())
    encoder = [(n, p) for n, p in named if n.startswith("encoder.")]
    scorer = [(n, p) for n, p in named if not n.startswith("encoder.") and not n.startswith("act_head.")]
    action = [(n, p) for n, p in named if n.startswith("act_head.")]
    for _, p in action: p.requires_grad_(False)
    before = {"optimized": parameter_sha256(encoder + scorer), "action_head": parameter_sha256(action)}
    optimizer = Adafactor([{"params": [p for _, p in encoder], "lr": cfg["encoder_learning_rate"]}, {"params": [p for _, p in scorer], "lr": cfg["scorer_learning_rate"]}], scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=cfg["fp16_init_scale"])
    optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
    order, cursor = state["data_order"], state["data_cursor"]
    item = items[order[cursor % len(order)]]
    batch = collate_items([[item]], tokenizer.pad_token_id)
    tensors = [batch[key].to("cuda") for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = model(*tensors)
    target, mask = batch["target"].to("cuda"), batch["marker_mask"].to("cuda")
    loss = -(target * torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    gradients = [p.grad for _, p in encoder + scorer if p.grad is not None]
    finite = bool(torch.isfinite(torch.stack([g.detach().float().norm() for g in gradients])).all())
    if not finite: raise RuntimeError("non-finite resume gradient")
    scaler.step(optimizer); scaler.update()
    after = {"optimized": parameter_sha256(encoder + scorer), "action_head": parameter_sha256(action)}
    result = {"status": "verified", "source_step": state["step"], "resumed_step": state["step"] + 1, "data_cursor": cursor + 1, "finite_gradients": finite, "intended_parameter_changed": before["optimized"] != after["optimized"], "action_head_unchanged": before["action_head"] == after["action_head"], "loss": float(loss.detach())}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
