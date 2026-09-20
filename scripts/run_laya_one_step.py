#!/usr/bin/env python3
"""Run exactly one forward/backward/AdamW step on the pinned Laya model."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_TOTAL = 421_293_827
EXPECTED_ENCODER = 394_781_696
EXPECTED_HEAD = 26_512_131
EXPECTED_STATE_ELEMENTS = 421_293_830


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=pathlib.Path, default=ROOT / "experiments/laya_smoke_test/config.json")
  parser.add_argument("--output", type=pathlib.Path, default=ROOT / "artifacts/laya_one_step.json")
  parser.add_argument("--cache-dir", type=pathlib.Path, default=ROOT / "data/cache")
  parser.add_argument("--laya-source", type=pathlib.Path, default=ROOT / "data/laya")
  parser.add_argument("--dry-run", action="store_true", help="Validate guards without importing ML packages or downloading files")
  parser.add_argument("--diagnostic-fp16-scale1", action="store_true", help="Run one FP16 forward/backward at scale 1.0; optimizer steps are prohibited")
  parser.add_argument("--fp16-scale1-optimizer-step", action="store_true", help="Run exactly one guarded FP16 optimizer step at scale 1.0")
  return parser.parse_args()


def contained(path):
  resolved = path.resolve()
  if resolved != ROOT and ROOT not in resolved.parents:
    raise ValueError(f"Path escapes repository boundary: {resolved}")
  return resolved


def command_output(command, cwd=ROOT):
  try:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
  except (FileNotFoundError, subprocess.CalledProcessError):
    return None


def write_result(path, result):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def file_sha256(path):
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


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
  config_path = contained(args.config)
  output_path = contained(args.output)
  cache_dir = contained(args.cache_dir)
  source_dir = contained(args.laya_source)
  cfg = json.loads(config_path.read_text())
  if args.diagnostic_fp16_scale1 and args.fp16_scale1_optimizer_step:
    raise ValueError("Safety guard: diagnostic and optimizer-step modes are mutually exclusive")
  if cfg.get("max_steps") != 1:
    raise ValueError("Safety guard: max_steps must equal 1")
  if cfg.get("micro_batch") != 1 or cfg.get("gradient_accumulation") != 1:
    raise ValueError("Safety guard: this run requires one sequence and one forward/backward")
  if cfg.get("max_len") != 512 or cfg.get("head_max_len") != 192:
    raise ValueError("Safety guard: token budgets must be exactly 512/192")
  if cfg.get("group_size") != 2:
    raise ValueError("Safety guard: perturbation group size must equal 2")

  gpu_query = command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"])
  result = {
    "status": "dry_run" if args.dry_run else "started",
    "safety": {"max_steps": 1, "forwards": 0, "backwards": 0, "optimizer_steps_attempted": 0, "optimizer_steps_completed": 0},
    "paths": {"root": str(ROOT), "config": str(config_path), "cache": str(cache_dir), "source": str(source_dir), "output": str(output_path)},
    "pins": {"laya_source": cfg["laya_source_revision"], "model": cfg["model_revision"], "dataset_unused": cfg["dataset_revision"]},
    "resolved_config": cfg,
    "hardware_preflight": {"nvidia_smi": gpu_query},
  }
  if args.dry_run:
    print(json.dumps(result, indent=2, sort_keys=True))
    return

  # Set all relevant caches before importing packages that can initialize them.
  cache_dir.mkdir(parents=True, exist_ok=True)
  for name, suffix in {
    "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub", "HF_DATASETS_CACHE": "huggingface/datasets",
    "TRANSFORMERS_CACHE": "huggingface/transformers", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg",
  }.items():
    os.environ[name] = str(cache_dir / suffix)
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

  try:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
      raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    source_revision = command_output(["git", "rev-parse", "HEAD"], cwd=source_dir)
    if source_revision != cfg["laya_source_revision"]:
      raise RuntimeError(f"Laya source revision mismatch: {source_revision!r}")
    sys.path.insert(0, str(source_dir))
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_model, build_sequence, collate_items, proper_reward

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    result["environment"] = {
      "python": platform.python_version(),
      "packages": package_versions(["torch", "transformers", "huggingface-hub", "safetensors", "numpy"]),
      "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu_name": props.name,
      "gpu_total_bytes": props.total_memory, "compute_capability": list(torch.cuda.get_device_capability(device)),
      "driver_version": gpu_query.split(",")[1].strip() if gpu_query else None,
    }

    model_dir = pathlib.Path(snapshot_download(cfg["model_id"], revision=cfg["model_revision"], cache_dir=cache_dir / "huggingface/hub"))
    if contained(model_dir) != model_dir.resolve():
      raise RuntimeError("Resolved snapshot is outside repository")
    _fix_tokenizer_config(str(model_dir))
    model_cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    model_cfg.update(max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)

    state = "Task: choose the safest next operation.\nRecent history: input was validated and no tool has run."
    question = {"t": "choice", "ins": "Select exactly one next operation.", "crit": {"inspect": "Inspect state", "execute": "Execute a tool"}}
    ids, markers = build_sequence(tokenizer, state, question, cfg["max_len"], cfg["head_max_len"], truncate_left=True)
    if len(markers) != 2:
      raise RuntimeError("Expected both deterministic options to survive tokenization")
    item = {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": [1.0, 0.0], "label": 0}
    batch = collate_items([[item]], tokenizer.pad_token_id)
    padded_ids = torch.full((1, cfg["max_len"]), tokenizer.pad_token_id, dtype=torch.long)
    padded_attention = torch.zeros((1, cfg["max_len"]), dtype=torch.long)
    padded_ids[:, :len(ids)] = batch["input_ids"]
    padded_attention[:, :len(ids)] = batch["attention_mask"]
    batch["input_ids"] = padded_ids
    batch["attention_mask"] = padded_attention
    result["batch"] = {"sequences": 1, "tensor_shape": list(batch["input_ids"].shape), "actual_tokens": len(ids), "option_counts": [len(markers)]}

    model = build_model(model_cfg, encoder_dir=str(model_dir / "encoder"))
    weights_path = model_dir / "model.safetensors"
    model.load_state_dict(load_file(weights_path), strict=True)
    model.encoder.config.reference_compile = False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    named = list(model.named_parameters())
    encoder = [(n, p) for n, p in named if n.startswith("encoder.")]
    action = [(n, p) for n, p in named if n.startswith("act_head.")]
    trainable_head = [(n, p) for n, p in named if not n.startswith("encoder.") and not n.startswith("act_head.")]
    counts = {"total": sum(p.numel() for _, p in named), "encoder": sum(p.numel() for _, p in encoder), "non_encoder": sum(p.numel() for n, p in named if not n.startswith("encoder.")), "optimized_head": sum(p.numel() for _, p in trainable_head), "excluded_action": sum(p.numel() for _, p in action), "state_elements_including_buffers": sum(value.numel() for value in model.state_dict().values())}
    if (counts["total"], counts["encoder"], counts["non_encoder"]) != (EXPECTED_TOTAL, EXPECTED_ENCODER, EXPECTED_HEAD):
      raise RuntimeError(f"Unexpected parameter counts: {counts}")
    if counts["state_elements_including_buffers"] != EXPECTED_STATE_ELEMENTS:
      raise RuntimeError(f"Unexpected state element count: {counts}")
    if any(p.dtype != torch.float32 for _, p in named):
      raise RuntimeError("Expected FP32 master parameters")
    optimized_ids = [id(p) for _, p in encoder + trainable_head]
    if len(optimized_ids) != len(set(optimized_ids)) or any(id(p) in optimized_ids for _, p in action):
      raise RuntimeError("Optimizer partition is invalid")
    result["parameters"] = {**counts, "master_dtype": "float32", "action_optimized": False}

    model.to(device).train()
    optimizer = torch.optim.AdamW([
      {"params": [p for _, p in encoder], "lr": cfg["encoder_learning_rate"]},
      {"params": [p for _, p in trainable_head], "lr": cfg["head_learning_rate"]},
    ], weight_decay=cfg["weight_decay"], foreach=False)
    scale1 = args.diagnostic_fp16_scale1 or args.fp16_scale1_optimizer_step
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1.0 if scale1 else 65536.0)
    optimizer.zero_grad(set_to_none=True)
    action_before = {n: p.detach().clone() for n, p in action}
    parameter_hash_before = parameter_sha256(named) if args.diagnostic_fp16_scale1 else None
    group_hashes_before = None
    if args.fp16_scale1_optimizer_step:
      group_hashes_before = {
        "encoder": parameter_sha256(encoder),
        "scorer": parameter_sha256(trainable_head),
        "action_head": parameter_sha256(action),
      }
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    forward_started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
      logits, _ = model(*(batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    result["safety"]["forwards"] = 1
    logits = logits.float()
    mask = batch["marker_mask"].to(device)
    target = batch["target"].to(device)
    qtype = batch["qtype"].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn((2,) + logits.shape, device=device, generator=generator) * 0.4 * mask
    eps = (eps - eps.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)) * mask
    noisy = logits.detach().unsqueeze(0) + eps
    probabilities = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
      reward = proper_reward(probabilities, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
      advantage = reward - reward.mean(0, keepdim=True)
      advantage = advantage / (advantage.std() + 1e-6)
    log_probability = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * 0.4 ** 2)
    loss_rl = -(advantage * log_probability).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = loss_rl + loss_ce
    result["loss"] = {"combined": float(loss), "cross_entropy": float(loss_ce), "rl": float(loss_rl), "reward_mean": float(reward.mean()), "reward_std": float(reward.std()), "finite_before_backward": {"combined": bool(torch.isfinite(loss)), "cross_entropy": bool(torch.isfinite(loss_ce)), "rl": bool(torch.isfinite(loss_rl)), "reward_all": bool(torch.isfinite(reward).all())}}
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(reward).all()):
      raise RuntimeError("Non-finite loss or reward; backward refused")
    backward_started = time.perf_counter()
    scaler.scale(loss).backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    result["safety"]["backwards"] = 1
    optimized = encoder + trainable_head
    gradients = [(name, parameter.grad) for name, parameter in optimized if parameter.grad is not None]
    pre_nonfinite = [name for name, gradient in gradients if not bool(torch.isfinite(gradient).all())]
    pre_norm = float(torch.nn.utils.get_total_norm([gradient for _, gradient in gradients]))
    unscale_started = time.perf_counter()
    scaler.unscale_(optimizer)
    torch.cuda.synchronize(device)
    unscale_seconds = time.perf_counter() - unscale_started
    nonfinite_gradient_parameters = [name for name, parameter in optimized if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    grad_norm = float(torch.nn.utils.get_total_norm([p.grad for _, p in optimized if p.grad is not None]))
    result["gradients"] = {
      "pre_unscale": {"finite": not pre_nonfinite, "global_norm": pre_norm, "first_offending": pre_nonfinite[0] if pre_nonfinite else None, "all_offending": pre_nonfinite},
      "post_unscale": {"finite": not nonfinite_gradient_parameters, "global_norm": grad_norm, "first_offending": nonfinite_gradient_parameters[0] if nonfinite_gradient_parameters else None, "all_offending": nonfinite_gradient_parameters},
      "finite": not nonfinite_gradient_parameters,
      "nonfinite_parameter_count": len(nonfinite_gradient_parameters),
      "first_nonfinite_parameters": nonfinite_gradient_parameters[:10],
    }
    if args.diagnostic_fp16_scale1:
      parameter_hash_after = parameter_sha256(named)
      parameters_unchanged = parameter_hash_before == parameter_hash_after
      if result["safety"]["optimizer_steps_attempted"] != 0 or result["safety"]["optimizer_steps_completed"] != 0:
        raise RuntimeError("Diagnostic invariant failed: an optimizer step was attempted")
      if not parameters_unchanged:
        raise RuntimeError("Diagnostic invariant failed: parameters changed")
      peak_reserved = torch.cuda.max_memory_reserved(device)
      result.update({
        "status": "completed_diagnostic",
        "diagnostic": {"mode": "fp16_scale1_no_step", "grad_scaler_init_scale": 1.0, "grad_scaler_scale_at_backward": float(scaler.get_scale()), "optimizer_step_statically_prohibited": True},
        "parameter_verification": {"sha256_before": parameter_hash_before, "sha256_after": parameter_hash_after, "unchanged": parameters_unchanged},
        "timings": {"forward_seconds": forward_seconds, "backward_seconds": backward_seconds, "unscale_seconds": unscale_seconds, "total_seconds": time.perf_counter() - started},
        "memory": {"baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": peak_reserved, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3, "peak_reserved_gib": peak_reserved / 1024**3},
        "files": {"model_safetensors_sha256": file_sha256(weights_path), "resolved_snapshot": str(model_dir)},
      })
      write_result(output_path, result)
      print(json.dumps({"status": result["status"], "output": str(output_path), "optimizer_steps_attempted": 0, "optimizer_steps_completed": 0}, indent=2))
      return
    if nonfinite_gradient_parameters:
      raise RuntimeError("Non-finite gradients; optimizer step refused")
    tracked = {}
    for group_name, params in (("encoder", encoder), ("head", trainable_head)):
      candidates = [(n, p) for n, p in params if p.grad is not None and bool((p.grad != 0).any())]
      if not candidates:
        raise RuntimeError(f"No nonzero {group_name} gradient")
      name, parameter = candidates[0]
      index = int(parameter.grad.abs().argmax())
      tracked[group_name] = (name, parameter, index, float(parameter.detach().view(-1)[index]))
    torch.nn.utils.clip_grad_norm_([p for _, p in optimized], 1.0)
    result["safety"]["optimizer_steps_attempted"] = 1
    optimizer_step_started = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    optimizer_step_seconds = time.perf_counter() - optimizer_step_started
    result["safety"]["optimizer_steps_completed"] = 1

    updates = {}
    for group_name, (name, parameter, index, before) in tracked.items():
      after = float(parameter.detach().view(-1)[index])
      updates[group_name] = {"parameter": name, "before": before, "after": after, "delta": after - before, "updated": after != before}
    action_unchanged = all(torch.equal(before, dict(action)[name].detach()) for name, before in action_before.items())
    group_hashes_after = None
    if args.fp16_scale1_optimizer_step:
      group_hashes_after = {
        "encoder": parameter_sha256(encoder),
        "scorer": parameter_sha256(trainable_head),
        "action_head": parameter_sha256(action),
      }
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if not all(row["updated"] for row in updates.values()) or not action_unchanged:
      raise RuntimeError(f"Update invariant failed: updates={updates}, action_unchanged={action_unchanged}")
    if args.fp16_scale1_optimizer_step and (group_hashes_before["encoder"] == group_hashes_after["encoder"] or group_hashes_before["scorer"] == group_hashes_after["scorer"] or group_hashes_before["action_head"] != group_hashes_after["action_head"]):
      raise RuntimeError(f"Parameter hash invariant failed: before={group_hashes_before}, after={group_hashes_after}")
    result.update({
      "status": "completed", "step_seconds": time.perf_counter() - started,
      "gradients": {**result["gradients"], "pre_clip_global_norm": grad_norm},
      "updates": {**updates, "action_head_unchanged": action_unchanged},
      "memory": {"baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": peak_reserved, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3, "peak_reserved_gib": peak_reserved / 1024**3, "within_11_5_gib": peak_reserved <= 11.5 * 1024**3},
      "files": {"model_safetensors_sha256": file_sha256(weights_path), "resolved_snapshot": str(model_dir)},
    })
    if args.fp16_scale1_optimizer_step:
      result["optimizer_step"] = {"mode": "fp16_scale1", "grad_scaler_init_scale": 1.0, "seconds": optimizer_step_seconds}
      result["parameter_verification"] = {
        "before": group_hashes_before,
        "after": group_hashes_after,
        "encoder_changed": group_hashes_before["encoder"] != group_hashes_after["encoder"],
        "scorer_changed": group_hashes_before["scorer"] != group_hashes_after["scorer"],
        "action_head_unchanged": group_hashes_before["action_head"] == group_hashes_after["action_head"],
      }
    if not result["memory"]["within_11_5_gib"]:
      result["status"] = "failed_memory_limit"
  except Exception as exc:
    result["status"] = "blocked" if result["safety"]["optimizer_steps_completed"] == 0 else "failed_after_step"
    result["blocker"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
      torch = sys.modules["torch"]
      result["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write_result(output_path, result)
    raise
  write_result(output_path, result)
  print(json.dumps({"status": result["status"], "output": str(output_path), "optimizer_steps_completed": 1}, indent=2))


if __name__ == "__main__":
  main()
