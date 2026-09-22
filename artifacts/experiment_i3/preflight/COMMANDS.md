# Iteration 3 experiment-engineer commands

All Python commands were run from repository root on 2026-09-22 with CUDA
hidden, offline flags set, bytecode disabled, and CPU math capped at four
threads. Train/dev were schema/hash checked; no test records, model, prediction,
training loop, or CUDA runtime was opened.

```text
git status --short --branch && git rev-parse HEAD && git log -1 --oneline 2d4408d
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py validate-config
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py dry-run --report artifacts/experiment_i3/preflight/dry_run.json
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py validate-accepted-manifest --manifest-sha256 158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py repair-preflight --report artifacts/experiment_i3/preflight/repair_preflight.json
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py base-eval
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m py_compile training/laya_trace_i3/__init__.py training/laya_trace_i3/metrics.py training/laya_trace_i3/protocol.py training/laya_trace_i3/runner.py training/laya_trace_i3/test_i3.py
git diff --check
```

Repair-1 observed results: 15 unit tests passed in 0.337 seconds; config
validation returned `config_valid`; accepted-manifest validation verified 1,051
train and 140 dev rows; repair preflight proved all six mappings/objectives and
163/20 bootstrap groups with Torch/model/test all false; and `base-eval`
returned `blocked_before_model_or_cuda`. The exact repair evidence is
`repair_preflight.json`.

The `py_compile` invocation was CPU-only and loaded no project modules, Torch,
model, or data. It may create ignored bytecode under the owned source directory;
that bytecode is not evidence and is not required for reproduction.
