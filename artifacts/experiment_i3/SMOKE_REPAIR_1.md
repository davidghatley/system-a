# Authorized Smoke Startup Repair 1

Date: 2026-09-22

Scope was limited to `training/laya_trace_i3/` and `artifacts/experiment_i3/`.
Prior review reports were not edited.

## Reported Failure

The authorized smoke startup command was reported as:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py smoke --review-accepted
```

It failed before CUDA with `NameError: ROOT is not defined` at
`training/laya_trace_i3/experiment.py:137`, while constructing the GPU-lock
owner record. This failure is user-reported; it was not rerun with CUDA.

## Fix

- Imported the existing canonical `ROOT` from `training.laya_trace_i3.protocol`
  into `experiment.py`.
- Added a CPU-only regression test that calls
  `run_bounded_smoke(..., review_accepted=True)`, verifies the complete lock
  owner payload and lock path, then stops at the mocked `GpuLock` constructor
  before the Torch import or any CUDA work.
- The existing default `review_accepted=False` test remains unchanged, so the
  smoke path remains fail-closed without explicit review authorization.

## Verified Facts

- CPU test command passed: 19 tests in 0.300 seconds.
- `git diff --check` passed.
- No CUDA initialization, Torch/model work, smoke execution, or test-record
  access was performed.
- The regression test passed through manifest validation, split-file checks,
  config and manifest hashing, and lock-owner construction before stopping.
- Prior reports `PRE_RUN_REVIEW_3.md` and `REPAIR_2.md` were not modified.

## Reproduction Commands

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
git diff --check
```

## SHA-256

Hashes were captured after the repair and before this report was added:

```text
f4f50a80aef10ce48aab17eba78059dd56ef6950abb02e1e09dc5363870d9c67  training/laya_trace_i3/experiment.py
2735135b6d3208403e89d9f82277e104e80dea9ecd9a6d33a7787408879e5f64  training/laya_trace_i3/test_i3.py
fb52e9e2cc3915c1b3d58a20f9fdc29101db97859eb0071dd222344801e9fdc0  training/laya_trace_i3/protocol.py
748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae  training/laya_trace_i3/config.json
158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2  artifacts/experiment_i3/preflight/data_manifest.v3.json
f5be8a9b92cd082d03b0e53079018b122a664b5f6e757ff16c9ef9a6782212a1  artifacts/experiment_i3/PRE_RUN_REVIEW_3.md
ef086b62a5bfa62a31bf2e7d8b63e85404204e4d00c77aa94919e5892bda5260  artifacts/experiment_i3/REPAIR_2.md
```

Readiness: ready for independent retry review. No claim of successful CUDA or
model execution is made.
