#!/usr/bin/env python3
"""Dependency-free Laya/GPU preflight. Downloads only small JSON metadata."""

import json
import pathlib
import subprocess
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
cfg = json.loads((ROOT / "config.json").read_text())


def get_json(url):
  request = urllib.request.Request(url, headers={"User-Agent": "laya-feasibility-smoke/1"})
  with urllib.request.urlopen(request, timeout=30) as response:
    return json.load(response)


model = get_json(f"https://huggingface.co/api/models/{cfg['model_id']}")
dataset = get_json(f"https://huggingface.co/api/datasets/{cfg['dataset_id']}")
assert model["sha"] == cfg["model_revision"], (model["sha"], cfg["model_revision"])
assert dataset["sha"] == cfg["dataset_revision"], (dataset["sha"], cfg["dataset_revision"])

params = int(model["safetensors"]["total"])
fp16_gib = params * 2 / 1024**3
adam_fp32_lower_gib = params * 16 / 1024**3
print(f"Pinned model revision:   {model['sha']}")
print(f"Pinned dataset revision: {dataset['sha']}")
print(f"Parameters:              {params:,}")
print(f"FP16 weights:            {fp16_gib:.2f} GiB")
print(f"FP32 params+grad+Adam:   {adam_fp32_lower_gib:.2f} GiB lower bound")

try:
  gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  print(f"GPU:                     {gpu}")
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
  print(f"GPU query unavailable:   {exc}")

try:
  import torch
except ImportError:
  print("PyTorch:                 not installed; reward smoke skipped")
else:
  print(f"PyTorch:                 {torch.__version__}")
  print(f"CUDA available:          {torch.cuda.is_available()}")

print("Preflight passed. This does not prove that a backward pass fits in VRAM.")
