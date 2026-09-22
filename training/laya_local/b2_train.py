#!/usr/bin/env python3
"""Run the frozen B2 single-GPU notebook-faithful RLCD retry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "training/laya_local/b2_config.json"
SUBSET = ROOT / "artifacts/train_i2/b2_subset"
PREDECLARATION = ROOT / "artifacts/train_i2/B2_PREDECLARATION_REPAIR_1.md"
PREDECLARATION_HASH = ROOT / "artifacts/train_i2/B2_PREDECLARATION_REPAIR_1.sha256"
NOTEBOOK = ROOT / "data/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb"
LAYA_COMMON = ROOT / "data/laya/laya/common.py"
B2_CONFIG_SHA256 = "a42fd2fd176ea6bc0e68e371be81eeb8be624caa021f096f594e8bfda7689cb4"
FROZEN_SUBSET_SHA256 = {
    "train": "d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656",
    "dev": "73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e",
    "manifest": "569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(cfg: dict[str, Any]) -> None:
    exact = {
        "model_id": "convaiinnovations/laya",
        "model_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "model_weights_sha256": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        "model_config_sha256": "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
        "laya_source_revision": "d113dca2512fb3eaca313534bc54c7162d87c1d4",
        "laya_common_sha256": "f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2",
        "notebook_sha256": "2c37036054dfb92073b518e994cb66c8a9009fb8d15b060d9bdfc6d4f94b1d39",
        "dataset_id": "LocalLLaMA/typed-decisions",
        "dataset_revision": "ea9306458d6e9563628369a3d1e72e362fb381d2",
        "dataset_config": "agent_trace_observability",
        "train_source_sha256": "e7b2f78fe539517c6a66a90c68ed671c6133194a88d27dbff5d84713af46b901",
        "dev_source_sha256": "833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d",
        "train_rows": 256, "dev_rows": 100, "expected_items_per_train_epoch": 1280,
        "expected_dev_items": 500, "seed": 42, "max_len": 512, "head_max_len": 192,
        "truncate_left": False, "epochs": 4, "micro_batch_sequences": 1,
        "gradient_accumulation": 64, "updates_per_epoch": 20, "total_updates": 80,
        "group_size": 4, "sigmas_by_epoch": [0.4, 0.3, 0.2, 0.1], "optimizer": "AdamW",
        "encoder_learning_rate": 2.5e-5, "nonencoder_learning_rate": 1e-4,
        "weight_decay": 0.01, "scheduler": "CosineAnnealingLR", "scheduler_t_max": 80,
        "scheduler_eta_min": 1e-6, "amp_dtype": "fp16", "fp16_init_scale": 64.0,
        "fp16_growth_interval": 1000,
        "max_grad_norm": 1.0, "reward_log_floor": -9.21,
        "reward_spherical_weight": 0.75, "reward_rps_weight": 1.0, "ce_weight": 1.0,
        "b1_baseline_choice_correct_crosscheck": 71, "dev_choice_total": 200,
        "required_choice_correct_improvement": 10,
        "max_peak_reserved_gib": 11.5, "gradient_checkpointing": True,
    }
    mismatches = {key: (cfg.get(key), value) for key, value in exact.items() if cfg.get(key) != value}
    if mismatches:
        raise ValueError(f"B2 frozen config mismatch: {mismatches}")
    if cfg["epochs"] * cfg["expected_items_per_train_epoch"] != cfg["total_updates"] * cfg["gradient_accumulation"]:
        raise ValueError("B2 schedule does not produce exactly 80 full updates")
    if cfg["calibration"] != "none; evaluate raw logits":
        raise ValueError("calibration fitting/application is forbidden")
    if cfg["fp16_growth_interval"] <= cfg["total_updates"]:
        raise ValueError("FP16 scale growth must remain disabled throughout the 80-update run")


def ordered_record_bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode()


def verify_frozen_subset(subset_dir: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Verify immutable bytes and reconstruct every record before Torch is imported."""
    manifest_path = subset_dir / "manifest.json"
    if sha256(manifest_path) != FROZEN_SUBSET_SHA256["manifest"]:
        raise RuntimeError("frozen B2 manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    files = {split: subset_dir / f"{split}.jsonl" for split in ("train", "dev")}
    records = {}
    for split, path in files.items():
        if sha256(path) != FROZEN_SUBSET_SHA256[split]:
            raise RuntimeError(f"frozen {split} JSONL SHA-256 mismatch")
        records[split] = [json.loads(line) for line in path.read_text().splitlines()]

    import pyarrow.parquet as pq
    sys.path.insert(0, str(ROOT))
    from shared.typed_decisions import parse_record

    expected_ids = {
        "train": [f"tr_agent_trace_observability_{i:06d}" for i in range(256)],
        "dev": [f"agent_trace_observability_{i:06d}" for i in range(100)],
    }
    sources = {
        "train": ROOT / "artifacts/train_i1/source_train.parquet",
        "dev": ROOT / "artifacts/train_i1/source_test.parquet",
    }
    counts = {"train": cfg["train_rows"], "dev": cfg["dev_rows"]}
    for split in ("train", "dev"):
        source = sources[split]
        expected_source_hash = cfg[f"{split}_source_sha256"]
        entry = manifest["splits"][split]
        if sha256(source) != expected_source_hash or entry["source_sha256"] != expected_source_hash:
            raise RuntimeError(f"frozen {split} Parquet SHA-256 mismatch")
        direct = [parse_record(row) for row in pq.read_table(source).slice(0, counts[split]).to_pylist()]
        loaded_ids = [record.get("id") for record in records[split]]
        if loaded_ids != expected_ids[split] or entry["record_ids"] != expected_ids[split]:
            raise RuntimeError(f"frozen {split} record IDs/order mismatch")
        if len(direct) != counts[split] or len(records[split]) != counts[split]:
            raise RuntimeError(f"frozen {split} row count mismatch")
        if [ordered_record_bytes(record) for record in records[split]] != [ordered_record_bytes(record) for record in direct]:
            raise RuntimeError(f"frozen {split} content/order differs from pinned Parquet")
    return manifest, records


def validate_model_structure(model: Any) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]], list[tuple[str, Any]]]:
    named = list(model.named_parameters())
    encoder = [(name, parameter) for name, parameter in named if "encoder." in name]
    nonencoder = [(name, parameter) for name, parameter in named if "encoder." not in name]
    if len({id(p) for _, p in encoder + nonencoder}) != len(named):
        raise RuntimeError("optimizer groups do not partition every named parameter exactly once")
    counts = tuple(sum(p.numel() for _, p in group) for group in (named, encoder, nonencoder))
    if counts != (421293827, 394781696, 26512131):
        raise RuntimeError("pinned model named-parameter counts changed")
    state = model.state_dict()
    if sum(value.numel() for value in state.values()) != 421293830:
        raise RuntimeError("pinned model state_dict element count changed")
    buffers = dict(model.named_buffers())
    if "temperature" in dict(named) or "temperature" not in buffers or buffers["temperature"].numel() != 3:
        raise RuntimeError("temperature must be a three-element registered buffer, not a named parameter")
    return named, encoder, nonencoder


def proper_reward_reference(torch: Any, q: Any, target: Any, qtype: Any, mask: Any,
                            *, w_sph: float = 0.75, w_rps: float = 1.0,
                            log_floor: float = -9.21) -> Any:
    """Independent transcription of pinned laya.common.proper_reward for CPU tests."""
    q = q * mask
    log_score = (target * torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)).sum(-1)
    spherical = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    reward = log_score + w_sph * spherical
    is_score = (qtype == 1).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
        reward = reward - w_rps * rps * is_score
    return reward


def rlcd_objective(torch: Any, logits: Any, act: Any, target: Any, mask: Any, qtype: Any,
                   proper_reward: Any, sigma: float, *, group_size: int = 4,
                   w_sph: float = 0.75, w_rps: float = 1.0,
                   log_floor: float = -9.21, ce_weight: float = 1.0) -> dict[str, Any]:
    """Exact notebook equations before gradient-accumulation scaling."""
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph,
                               w_rps=w_rps, log_floor=log_floor)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(advantage * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = loss_rl + ce_weight * loss_ce + 0.0 * act.sum()
    return {"loss": loss, "loss_rl": loss_rl, "loss_ce": loss_ce,
            "reward": reward, "advantage": advantage, "eps": eps}


def nested_equal(torch: Any, left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(nested_equal(torch, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(nested_equal(torch, a, b) for a, b in zip(left, right))
    return left == right


def tree_to_cpu(torch: Any, value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_to_cpu(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(torch, item) for item in value)
    return value


def load_and_validate_training_state(torch: Any, optimizer: Any, scheduler: Any, scaler: Any,
                                     saved: dict[str, Any]) -> None:
    objects = {"optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
    for key, obj in objects.items():
        obj.load_state_dict(saved[key])
        if not nested_equal(torch, tree_to_cpu(torch, obj.state_dict()), saved[key]):
            raise RuntimeError(f"live {key} checkpoint reload is not exact")


def checked_scaler_step(scaler: Any, optimizer: Any) -> float:
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scale_after < scale_before:
        raise RuntimeError("FP16 overflow skipped the optimizer step; no retry")
    return scale_after


def state_digest(torch: Any, state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--subset-dir", type=Path, default=SUBSET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/train_i2/b2_run")
    args = parser.parse_args()
    allowed_output = (ROOT / "artifacts/train_i2").resolve()
    output_dir = args.output_dir.resolve()
    if output_dir != allowed_output and allowed_output not in output_dir.parents:
        raise ValueError("B2 output must remain under artifacts/train_i2")
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    if sha256(args.config) != B2_CONFIG_SHA256:
        raise RuntimeError("frozen B2 config full-file SHA-256 mismatch")

    manifest_path = args.subset_dir / "manifest.json"
    manifest, frozen_records = verify_frozen_subset(args.subset_dir, cfg)
    if sha256(NOTEBOOK) != cfg["notebook_sha256"] or sha256(LAYA_COMMON) != cfg["laya_common_sha256"]:
        raise RuntimeError("pinned notebook or Laya formula source SHA-256 mismatch")
    if not PREDECLARATION.exists() or not PREDECLARATION_HASH.exists():
        raise RuntimeError("B2 predeclaration and separate full-file hash must exist before model loading")
    declared_hash = PREDECLARATION_HASH.read_text().split()[0]
    if sha256(PREDECLARATION) != declared_hash:
        raise RuntimeError("B2 predeclaration full-file SHA-256 mismatch")
    source_dir = ROOT / "data/laya"
    if command_output(["git", "rev-parse", "HEAD"], source_dir) != cfg["laya_source_revision"]:
        raise RuntimeError("pinned Laya source revision mismatch")

    cache = ROOT / "data/cache"
    os.environ.update({
        "HF_HOME": str(cache / "huggingface"), "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface/transformers"), "TORCH_HOME": str(cache / "torch"),
        "XDG_CACHE_HOME": str(cache / "xdg"), "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; B2 has no CPU fallback")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)

    snapshot = cache / "huggingface/hub/models--convaiinnovations--laya/snapshots" / cfg["model_revision"]
    weights_path = snapshot / "model.safetensors"
    model_config_path = snapshot / "rl_agent_config.json"
    if not snapshot.is_dir() or sha256(weights_path) != cfg["model_weights_sha256"] or sha256(model_config_path) != cfg["model_config_sha256"]:
        raise RuntimeError("pinned local model snapshot missing or SHA-256 mismatch")
    sys.path.insert(0, str(source_dir))
    from laya.common import QTYPES, build_model, build_sequence, collate_items, proper_reward, render_options

    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    model_cfg = json.loads(model_config_path.read_text())
    model_cfg.update(max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])

    def to_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for record in records:
            for question_id, question in record["questions"].items():
                qtype = question["type"]
                criteria = question.get("criteria", {})
                internal = {"t": qtype, "ins": question["instructions"], "crit": criteria}
                ids, markers = build_sequence(tokenizer, record["state"], internal, cfg["max_len"], cfg["head_max_len"], truncate_left=False)
                keys = list(criteria) if qtype == "choice" else ([str(i) for i in range(len(criteria))] if qtype == "score" else ["false", "true"])
                target = [float(record["gold"][question_id]["probabilities"].get(key, 0.0)) for key in keys]
                total = sum(target)
                target = [value / total for value in target] if total > 0 else [1.0 / len(target)] * len(target)
                if len(markers) != len(render_options(internal)):
                    raise RuntimeError(f"question markers truncated: {record.get('id')}/{question_id}")
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES[qtype], "target": target,
                              "label": target.index(max(target)), "native_label": keys.index(record["gold"][question_id]["label"]),
                              "record_id": record.get("id"), "question_id": question_id,
                              "question_type": qtype, "workflow": record.get("workflow"), "option_keys": keys})
        return items

    train_records, dev_records = frozen_records["train"], frozen_records["dev"]
    train_items, dev_items = to_items(train_records), to_items(dev_records)
    if len(train_records) != 256 or len(train_items) != 1280 or len(dev_records) != 100 or len(dev_items) != 500:
        raise RuntimeError("frozen B2 row/item count mismatch")

    def make_model() -> Any:
        candidate = build_model(model_cfg, encoder_dir=str(snapshot / "encoder"))
        candidate.load_state_dict(load_file(weights_path), strict=True)
        candidate.encoder.config.reference_compile = False
        candidate.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return candidate

    model = make_model()
    named, encoder, nonencoder = validate_model_structure(model)
    action_names = [name for name, _ in nonencoder if name.startswith("act_head.")]
    if not action_names or not all(parameter.requires_grad for _, parameter in named):
        raise RuntimeError("notebook action-head/all-trainable behavior changed")

    optimizer = torch.optim.AdamW([
        {"params": [p for _, p in encoder], "lr": cfg["encoder_learning_rate"]},
        {"params": [p for _, p in nonencoder], "lr": cfg["nonencoder_learning_rate"]},
    ], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"], eta_min=cfg["scheduler_eta_min"])
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=cfg["fp16_init_scale"],
                                  growth_interval=cfg["fp16_growth_interval"])
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    max_reserved = int(cfg["max_peak_reserved_gib"] * 1024 ** 3)

    def guard_memory() -> None:
        if torch.cuda.max_memory_reserved(device) > max_reserved:
            raise RuntimeError("peak CUDA reserved memory exceeded frozen 11.5 GiB limit; no retry")

    def evaluate(items: list[dict[str, Any]]) -> dict[str, Any]:
        model.eval()
        groups: dict[str, list[bool]] = defaultdict(list)
        workflow_groups: dict[str, list[bool]] = defaultdict(list)
        predictions, gold_nll_sum, soft_nll_sum = [], 0.0, 0.0
        with torch.no_grad():
            for item in items:
                batch = collate_items([[item]], tokenizer.pad_token_id)
                tensors = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(*tensors)
                k = len(item["target"])
                log_probs = torch.log_softmax(logits[0, :k].float(), -1)
                target = torch.tensor(item["target"], dtype=torch.float32, device=device)
                gold_nll = float(-log_probs[item["native_label"]])
                soft_nll = float(-(target * log_probs).sum())
                prediction = int(log_probs.argmax())
                correct = prediction == item["native_label"]
                groups[item["question_type"]].append(correct)
                workflow_groups[item["workflow"]].append(correct)
                gold_nll_sum += gold_nll
                soft_nll_sum += soft_nll
                predictions.append({"record_id": item["record_id"], "workflow": item["workflow"],
                                    "question_id": item["question_id"], "type": item["question_type"],
                                    "prediction": item["option_keys"][prediction],
                                    "gold_label": item["option_keys"][item["native_label"]], "correct": correct,
                                    "gold_nll": gold_nll, "soft_target_nll": soft_nll})
                guard_memory()
        def summary(values: list[bool]) -> dict[str, Any]:
            return {"correct": sum(values), "questions": len(values), "accuracy": sum(values) / len(values)}
        all_values = [value for values in groups.values() for value in values]
        return {"choice": summary(groups["choice"]), "score": summary(groups["score"]), "noul": summary(groups["noul"]),
                "all": summary(all_values), "by_workflow": {key: summary(value) for key, value in sorted(workflow_groups.items())},
                "gold_nll_mean": gold_nll_sum / len(items), "soft_target_nll_mean": soft_nll_sum / len(items),
                "predictions": predictions}

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = evaluate(dev_items)
    if baseline["choice"]["questions"] != cfg["dev_choice_total"] or not all(
        math.isfinite(baseline[key]) for key in ("gold_nll_mean", "soft_target_nll_mean")
    ):
        raise RuntimeError("B2 baseline count or finite-metric guard failed; training forbidden")
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update = microforward = 0
    loss_curve = []
    order = list(range(len(train_items)))
    training_started = time.perf_counter()
    for epoch, sigma in enumerate(cfg["sigmas_by_epoch"]):
        random.Random(cfg["seed"] + epoch).shuffle(order)
        for item_index in order:
            item = train_items[item_index]
            batch = collate_items([[item]], tokenizer.pad_token_id)
            tensors = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = model(*tensors)
            terms = rlcd_objective(torch, logits.float(), act, batch["target"].to(device), batch["marker_mask"].to(device),
                                   batch["qtype"].to(device), proper_reward, sigma, group_size=cfg["group_size"],
                                   w_sph=cfg["reward_spherical_weight"], w_rps=cfg["reward_rps_weight"],
                                   log_floor=cfg["reward_log_floor"], ce_weight=cfg["ce_weight"])
            if not all(bool(torch.isfinite(terms[key]).all()) for key in ("loss", "loss_rl", "loss_ce", "reward", "advantage")):
                raise RuntimeError(f"non-finite B2 objective at microforward {microforward + 1}; no retry")
            scaler.scale(terms["loss"] / cfg["gradient_accumulation"]).backward()
            microforward += 1
            if microforward % cfg["gradient_accumulation"] == 0:
                scaler.unscale_(optimizer)
                gradients = [parameter.grad for _, parameter in named]
                if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise RuntimeError(f"missing/non-finite gradient before update {update + 1}; no retry")
                grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], cfg["max_grad_norm"])
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError(f"non-finite gradient norm before update {update + 1}; no retry")
                current_scale = checked_scaler_step(scaler, optimizer)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                loss_curve.append({"update": update, "epoch": epoch + 1, "sigma": sigma,
                                   "loss": float(terms["loss"].detach()), "loss_rl": float(terms["loss_rl"].detach()),
                                   "loss_ce": float(terms["loss_ce"].detach()), "reward_mean": float(terms["reward"].mean()),
                                    "pre_clip_grad_norm": float(grad_norm), "fp16_scale": current_scale,
                                   "learning_rates": scheduler.get_last_lr()})
            guard_memory()
    if microforward != 5120 or update != cfg["total_updates"]:
        raise RuntimeError(f"exact stopping guard failed: {microforward} microforwards, {update} updates")
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - training_started

    checkpoint = output_dir / "checkpoint"
    checkpoint.mkdir(exist_ok=False)
    cpu_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    before_digest = state_digest(torch, cpu_state)
    save_file(cpu_state, checkpoint / "model.safetensors")
    training_state = tree_to_cpu(torch, {
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "updates": update, "microforwards": microforward, "epochs": cfg["epochs"],
    })
    torch.save(training_state, checkpoint / "training_state.pt")
    (checkpoint / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    del cpu_state, model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    model = make_model()
    reloaded_state = load_file(checkpoint / "model.safetensors")
    model.load_state_dict(reloaded_state, strict=True)
    after_digest = state_digest(torch, model.state_dict())
    if before_digest != after_digest:
        raise RuntimeError("checkpoint model reload is not bit-exact")
    model.to(device)
    named_reload = list(model.named_parameters())
    encoder_reload = [p for name, p in named_reload if "encoder." in name]
    nonencoder_reload = [p for name, p in named_reload if "encoder." not in name]
    optimizer = torch.optim.AdamW([{"params": encoder_reload, "lr": cfg["encoder_learning_rate"]},
                                   {"params": nonencoder_reload, "lr": cfg["nonencoder_learning_rate"]}], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"], eta_min=cfg["scheduler_eta_min"])
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=cfg["fp16_init_scale"],
                                  growth_interval=cfg["fp16_growth_interval"])
    loaded_training_state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    if not nested_equal(torch, training_state, loaded_training_state):
        raise RuntimeError("serialized training-state checkpoint payload changed")
    load_and_validate_training_state(torch, optimizer, scheduler, scaler, loaded_training_state)
    if loaded_training_state["updates"] != 80 or loaded_training_state["microforwards"] != 5120 or loaded_training_state["epochs"] != 4:
        raise RuntimeError("checkpoint stopping state mismatch")
    guard_memory()

    final = evaluate(dev_items)
    (output_dir / "final.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    peak_reserved = torch.cuda.max_memory_reserved(device)
    result = {
        "status": "passed" if final["choice"]["correct"] >= baseline["choice"]["correct"] + cfg["required_choice_correct_improvement"] else "failed_primary_gate",
        "config": cfg, "predeclaration_sha256": declared_hash,
        "subset_manifest_sha256": sha256(manifest_path),
        "source_revisions": {"laya": cfg["laya_source_revision"], "model": cfg["model_revision"], "dataset": cfg["dataset_revision"]},
        "counts": {"train_rows": len(train_records), "train_items": len(train_items), "dev_rows": len(dev_records), "dev_items": len(dev_items),
                   "microforwards": microforward, "updates": update},
        "baseline": {key: value for key, value in baseline.items() if key != "predictions"},
        "final": {key: value for key, value in final.items() if key != "predictions"},
        "primary_gate": {"b1_crosscheck": "71/200", "b2_measured_baseline_correct": baseline["choice"]["correct"],
                         "required_improvement": cfg["required_choice_correct_improvement"],
                         "required_final_correct": baseline["choice"]["correct"] + cfg["required_choice_correct_improvement"],
                         "actual_correct": final["choice"]["correct"], "choice_total": cfg["dev_choice_total"]},
        "loss_curve": loss_curve,
        "checkpoint": {"model_state_sha256": before_digest, "model_reload_exact": True,
                       "training_state_reload_exact": True, "weights_file_sha256": sha256(checkpoint / "model.safetensors")},
        "memory": {"peak_reserved_bytes": peak_reserved, "peak_reserved_gib": peak_reserved / 1024 ** 3,
                   "limit_gib": cfg["max_peak_reserved_gib"]},
        "timing": {"training_seconds": train_seconds, "total_seconds": time.perf_counter() - started},
        "environment": {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
                        "transformers": importlib.metadata.version("transformers"), "gpu": torch.cuda.get_device_name(device)},
        "evaluations": ["baseline", "final_after_reload"], "calibration_fitted": False,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        required = baseline["choice"]["correct"] + cfg["required_choice_correct_improvement"]
        raise RuntimeError(f"B2 primary gate failed: {final['choice']['correct']}/200 < {required}/200")


if __name__ == "__main__":
    main()
