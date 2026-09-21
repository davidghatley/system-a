#!/usr/bin/env python3
"""Run the frozen, bounded single-GPU Laya learning proof."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/laya_local/config.json")
    parser.add_argument("--train", type=Path, default=ROOT / "artifacts/train_i1/subsets/train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "artifacts/train_i1/subsets/dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/train_i1/run")
    return parser.parse_args()


def command_output(command, cwd=ROOT):
    try:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def parameter_sha256(named_parameters):
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["micro_batch"] != 1 or cfg["max_steps"] > 32 or cfg["gradient_accumulation"] > 2:
        raise ValueError("bounded-run guard requires microbatch=1, max_steps<=32, accumulation<=2")
    if cfg["amp_dtype"] != "fp16" or cfg["fp16_init_scale"] != 1.0:
        raise ValueError("this proof requires FP16 with initial scale 1.0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data/cache"
    for name, suffix in {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub", "HF_DATASETS_CACHE": "huggingface/datasets",
        "TRANSFORMERS_CACHE": "huggingface/transformers", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg",
    }.items():
        os.environ[name] = str(cache / suffix)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import Adafactor, AutoTokenizer

    source_dir = ROOT / "data/laya"
    source_revision = command_output(["git", "rev-parse", "HEAD"], cwd=source_dir)
    if source_revision != cfg["laya_source_revision"]:
        raise RuntimeError(f"Laya source revision mismatch: {source_revision}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU training fallback is forbidden")
    sys.path.insert(0, str(source_dir))
    from laya.common import build_model, collate_items
    from shared.typed_decisions import load_jsonl
    from training.laya_local.data import to_laya_items

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    snapshot = cache / "huggingface/hub/models--convaiinnovations--laya/snapshots" / cfg["model_revision"]
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned local model snapshot missing: {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    model_cfg = json.loads((snapshot / "rl_agent_config.json").read_text())
    model_cfg.update(max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])

    train_records = load_jsonl(args.train)
    dev_records = load_jsonl(args.dev)
    train_items = [item for record in train_records for item in to_laya_items(
        record, tokenizer, max_len=cfg["max_len"], head_max_len=cfg["head_max_len"]
    )]
    dev_items = [item for record in dev_records for item in to_laya_items(
        record, tokenizer, max_len=cfg["max_len"], head_max_len=cfg["head_max_len"]
    )]

    def make_model():
        candidate = build_model(model_cfg, encoder_dir=str(snapshot / "encoder"))
        candidate.load_state_dict(load_file(snapshot / "model.safetensors"), strict=True)
        candidate.encoder.config.reference_compile = False
        if cfg["gradient_checkpointing"]:
            candidate.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return candidate

    model = make_model()
    named = list(model.named_parameters())
    encoder = [(n, p) for n, p in named if n.startswith("encoder.")]
    action = [(n, p) for n, p in named if n.startswith("act_head.")]
    scorer = [(n, p) for n, p in named if not n.startswith("encoder.") and not n.startswith("act_head.")]
    optimized_ids = {id(p) for _, p in encoder + scorer}
    if len(optimized_ids) != len(encoder) + len(scorer) or any(id(p) in optimized_ids for _, p in action):
        raise RuntimeError("optimizer parameter partition overlaps the intentionally unused action head")
    for _, parameter in action:
        parameter.requires_grad_(False)
    hashes_before = {
        "encoder": parameter_sha256(encoder), "scorer": parameter_sha256(scorer), "action_head": parameter_sha256(action)
    }
    optimizer = Adafactor([
        {"params": [p for _, p in encoder], "lr": cfg["encoder_learning_rate"]},
        {"params": [p for _, p in scorer], "lr": cfg["scorer_learning_rate"]},
    ], scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=cfg["fp16_init_scale"])
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def evaluate(items):
        model.eval()
        choice_correct = choice_total = 0
        nll_sum = 0.0
        rows = []
        with torch.no_grad():
            for item in items:
                batch = collate_items([[item]], tokenizer.pad_token_id)
                tensors = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(*tensors)
                k = len(item["target"])
                log_probs = torch.log_softmax(logits[0, :k].float(), -1)
                target = torch.tensor(item["target"], device=device)
                nll = float(-(target * log_probs).sum())
                prediction = int(log_probs.argmax())
                if item["question_type"] == "choice":
                    choice_total += 1
                    choice_correct += prediction == item["label"]
                nll_sum += nll
                rows.append({"record_id": item["record_id"], "question_id": item["question_id"], "type": item["question_type"], "prediction": prediction, "gold_label": item["native_label"], "gold_argmax": item["gold_argmax"], "nll": nll})
        return {"choice_correct": choice_correct, "choice_total": choice_total, "choice_accuracy": choice_correct / choice_total, "gold_nll": nll_sum / len(items), "predictions": rows}

    baseline = evaluate(dev_items)
    (args.output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    training_started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    generator = random.Random(seed)
    order = list(range(len(train_items)))
    generator.shuffle(order)
    cursor = 0
    loss_curve = []
    all_gradients_finite = True
    for step in range(1, cfg["max_steps"] + 1):
        losses = []
        for _ in range(cfg["gradient_accumulation"]):
            if cursor == len(order):
                generator.shuffle(order)
                cursor = 0
            item = train_items[order[cursor]]
            cursor += 1
            batch = collate_items([[item]], tokenizer.pad_token_id)
            tensors = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(*tensors)
            target = batch["target"].to(device)
            mask = batch["marker_mask"].to(device)
            loss = -(target * torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at update {step}")
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        gradients = [p.grad for _, p in encoder + scorer if p.grad is not None]
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        all_gradients_finite &= finite
        if not finite:
            raise RuntimeError(f"non-finite gradient at update {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in encoder + scorer], cfg["max_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        loss_curve.append({"step": step, "loss": sum(losses) / len(losses), "pre_clip_grad_norm": grad_norm, "fp16_scale": float(scaler.get_scale())})
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - training_started
    hashes_after = {
        "encoder": parameter_sha256(encoder), "scorer": parameter_sha256(scorer), "action_head": parameter_sha256(action)
    }
    post = evaluate(dev_items)
    (args.output_dir / "post_training.json").write_text(json.dumps(post, indent=2, sort_keys=True) + "\n")

    checkpoint = args.output_dir / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, checkpoint / "model.safetensors")
    torch.save({
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": cfg["max_steps"],
        "data_cursor": cursor, "data_order": order, "data_rng_state": generator.getstate(),
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }, checkpoint / "training_state.pt")
    (checkpoint / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    model = make_model()
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    model.to(device)
    resume_optimizer = Adafactor([
        {"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": cfg["encoder_learning_rate"]},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.") and not n.startswith("act_head.")], "lr": cfg["scorer_learning_rate"]},
    ], scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=cfg["weight_decay"])
    resume_scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=cfg["fp16_init_scale"])
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    resume_optimizer.load_state_dict(state["optimizer"])
    resume_scaler.load_state_dict(state["scaler"])
    reloaded = evaluate(dev_items)
    reload_matches = reloaded["predictions"] == post["predictions"]
    result = {
        "status": "completed" if reload_matches else "failed_reload_mismatch",
        "config": cfg,
        "source_revisions": {"laya": source_revision, "model": cfg["model_revision"], "dataset": cfg["dataset_revision"]},
        "environment": {"python": platform.python_version(), "packages": package_versions(["torch", "transformers", "safetensors", "numpy", "pyarrow"]), "gpu": torch.cuda.get_device_name(device), "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])},
        "counts": {"train_records": len(train_records), "train_items": len(train_items), "dev_records": len(dev_records), "dev_items": len(dev_items)},
        "baseline": {key: value for key, value in baseline.items() if key != "predictions"},
        "post_training": {key: value for key, value in post.items() if key != "predictions"},
        "improvement_absolute": post["choice_accuracy"] - baseline["choice_accuracy"],
        "loss_curve": loss_curve,
        "gradients_all_finite": all_gradients_finite,
        "parameter_groups": {"before": hashes_before, "after": hashes_after, "encoder_changed": hashes_before["encoder"] != hashes_after["encoder"], "scorer_changed": hashes_before["scorer"] != hashes_after["scorer"], "action_head_unchanged": hashes_before["action_head"] == hashes_after["action_head"]},
        "checkpoint": {"step": state["step"], "optimizer_state_loaded": True, "scaler_state_loaded": True, "reload_predictions_exact": reload_matches},
        "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3},
        "timing": {"train_seconds": train_seconds, "total_seconds": time.perf_counter() - started},
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
