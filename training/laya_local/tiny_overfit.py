#!/usr/bin/env python3
"""Bounded one-example overfit diagnostic; not acceptance evidence."""

import json
import os
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    cache = ROOT / "data/cache"
    for name, suffix in {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub", "TRANSFORMERS_CACHE": "huggingface/transformers",
        "TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg",
    }.items():
        os.environ[name] = str(cache / suffix)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import Adafactor, AutoTokenizer

    cfg = json.loads((ROOT / "training/laya_local/config.json").read_text())
    source = ROOT / "data/laya"
    sys.path.insert(0, str(source))
    from laya.common import build_model, collate_items
    from shared.typed_decisions import load_jsonl
    from training.laya_local.data import to_laya_items

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda:0")
    snapshot = cache / "huggingface/hub/models--convaiinnovations--laya/snapshots" / cfg["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    model_cfg = json.loads((snapshot / "rl_agent_config.json").read_text())
    model_cfg.update(max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])
    model = build_model(model_cfg, encoder_dir=str(snapshot / "encoder"))
    model.load_state_dict(load_file(snapshot / "model.safetensors"), strict=True)
    model.encoder.config.reference_compile = False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    record = load_jsonl(ROOT / "artifacts/train_i1/subsets/train.jsonl")[0]
    item = next(item for item in to_laya_items(record, tokenizer, max_len=cfg["max_len"], head_max_len=cfg["head_max_len"]) if item["question_type"] == "choice")
    batch = collate_items([[item]], tokenizer.pad_token_id)
    inputs = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    target = batch["target"].to(device)
    mask = batch["marker_mask"].to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if not name.startswith("act_head.")]
    optimizer = Adafactor([parameter for _, parameter in trainable], lr=cfg["scorer_learning_rate"], scale_parameter=False, relative_step=False, warmup_init=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0)

    def measure():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(*inputs)
        probabilities = torch.softmax(logits[0, :len(item["target"])].float(), -1)
        loss = float(-(target[0] * probabilities.log()).sum())
        return {"loss": loss, "probabilities": probabilities.cpu().tolist(), "prediction": int(probabilities.argmax()), "gold_argmax": item["label"]}

    before = measure()
    curve = []
    finite = True
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1, 17):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(*inputs)
        loss = -(target * torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for _, parameter in trainable if parameter.grad is not None]
        finite = finite and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        if not finite:
            raise RuntimeError("non-finite tiny-overfit gradient")
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
        scaler.step(optimizer)
        scaler.update()
        curve.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    result = {
        "purpose": "diagnostic only; cannot satisfy primary acceptance",
        "record_id": item["record_id"], "question_id": item["question_id"],
        "steps": 16, "before": before, "after": measure(), "training_loss": curve,
        "all_gradients_finite": finite, "fp16_scale": float(scaler.get_scale()),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "wall_seconds": time.perf_counter() - started,
    }
    output = ROOT / "artifacts/train_i1/tiny_overfit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
