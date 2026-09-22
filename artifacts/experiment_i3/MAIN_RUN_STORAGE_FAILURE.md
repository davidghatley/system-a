# Main Run Storage Failure

Date: 2026-09-22

## Verified

- The independently authorized two-seed command started with the final test closed.
- Seed 42 completed its frozen four-epoch, 68-update optimization loop.
- The process failed while serializing `seed_42/checkpoint/model.safetensors` with
  `No space left on device (os error 28)`.
- This was a filesystem-capacity failure, not a CUDA out-of-memory failure.
- No complete checkpoint, seed result, aggregate main-run report, dev selection,
  or final-test result was produced.
- `artifacts/experiment_i3/preflight/runs/seed_42/checkpoint/` is empty.
- After failure, the filesystem reported 1.3 GiB available and the GPU was idle
  with 11,322 MiB free.

## Command

```text
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -B training/laya_trace_i3/runner.py train --review-accepted --report artifacts/experiment_i3/main_run.json
```

## Repository-Local Capacity Evidence

```text
Filesystem /dev/mapper/root: 464G total, 461G used, 1.3G available, 100% use
artifacts/train_i1: 3.2G
artifacts/train_i2: 4.8G
artifacts/experiment_i3: 168K after the failed serialization
data/cache: 3.3G (contains the pinned local model required for execution)
.venv: 7.0G (required runtime environment)
```

No cleanup or retry was performed. Project policy requires preserving evidence
and forbids destructive cleanup without explicit user authorization.
