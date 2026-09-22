# Iteration 3 Executor Implementation

Date: 2026-09-22

## Scope

The actual-model Iteration 3 executor is implemented only in
`training/laya_trace_i3/`. This file records implementation evidence; it is
not an acceptance or GPU-run report. No CUDA, model load, inference, training,
or test-record access was performed during this implementation.

## Implementation

- `actual-model-pilot` validates the accepted train/dev manifest and local model
  pins, acquires the configured non-blocking GPU lock, then imports Torch. It
  performs a representative real Laya forward and exactly two real optimizer
  updates on accepted train records. It records cold load, dev evaluation,
  training elapsed time, peak reserved VRAM, finite-value status, and a
  conservative projection input. CPU fallback, OOM retry, non-finite output,
  and budget overflow fail closed.
- `base-eval` loads the strictly pinned local Laya model and evaluates all
  accepted dev records with raw six-label probabilities.
- `train` runs the frozen four-epoch schedule per declared seed, with the
  Iteration 2 RLCD+CE objective, AdamW encoder/non-encoder groups, FP16 scaler,
  cosine scheduler, partial final accumulation, gradient and VRAM guards, and
  no test access.
- Each seed writes `model.safetensors`, `training_state.pt`, and hash-bound
  reload evidence. Reload requires strict model loading, bit-exact model-state
  digest equality, exact optimizer/scheduler/scaler state, and exact stopping
  state before dev scoring.
- Dev scoring uses the fixed six-label macro-F1, accuracy, per-class,
  confusion, clipped NLL, and Brier metrics. It also emits deterministic
  task-group bootstrap intervals. `dev-select` accepts only complete dev
  candidates and writes a dev-only freeze artifact.
- Warm latency measures synchronized tokenize/build through CPU probability
  copy for three warmups and ten records. `final-test` remains the existing
  one-shot marker-before-open path and cannot be bypassed through `load_split`.

The frozen data, labels, objective, optimizer recipe, model revision, source
revision, no-OOM-retry rule, and 14,400 elapsed / 7,200 aggregate GPU-second
guards were not changed. No config re-freeze was necessary; the existing
execution gate remains closed by default and every actual-model command
requires explicit `--review-accepted` authorization. This is intentionally
ready for independent CPU review, not self-acceptance.

## CPU Evidence

Command:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
```

Result: 21/21 tests passed. The added tests verify actual-pilot lock-owner
construction before Torch/model work and missing local model fail-closed
behavior. `git diff --check` and Python compilation passed.

Unauthorized guard command:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py actual-model-pilot
```

Result: rejected with `ProtocolError` requiring explicit independent review
authorization, before model access. No GPU work was attempted.

## Proposed Actual Pilot Invocation

This is the exact bounded invocation proposed for an independently authorized
GPU run. It is recorded, not executed here:

```text
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py actual-model-pilot --review-accepted --report artifacts/experiment_i3/actual_model_pilot.json
```

The report must be inspected for finite values, `test_opened=false`, peak
reserved VRAM at or below 11.5 GiB, and elapsed/aggregate GPU time within the
frozen limits before using it for two-seed projection. A failed, OOM, or
non-finite pilot forbids main training; it is not retried.

## Reproducibility Hashes

Compute hashes immediately before an independently authorized run with:

```text
sha256sum training/laya_trace_i3/config.json training/laya_trace_i3/experiment.py training/laya_trace_i3/protocol.py training/laya_trace_i3/runner.py training/laya_trace_i3/test_i3.py artifacts/experiment_i3/preflight/data_manifest.v3.json
```

The executor verifies the pinned local model weights/config and Laya source
hashes against `config.json` at runtime. The accepted manifest hash is recorded
in the lock owner and every runtime result. External downloads are forbidden by
the proposed offline environment and the runtime uses only
`data/cache/huggingface` and `data/laya`.

Hashes at this implementation checkpoint:

```text
748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae  training/laya_trace_i3/config.json
831c5911c447426c39071ccef03330685f11c5dba5fe127cfc4ad96df24cd00a  training/laya_trace_i3/experiment.py
fb52e9e2cc3915c1b3d58a20f9fdc29101db97859eb0071dd222344801e9fdc0  training/laya_trace_i3/protocol.py
64a11e9c22755a9492008158a04dce3f138aba629665ba51c320b1d87b050f66  training/laya_trace_i3/runner.py
a8af6e8333e566044e41d770da4d92cf176ea0f56be07656750f507b7ae289ba  training/laya_trace_i3/test_i3.py
158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2  artifacts/experiment_i3/preflight/data_manifest.v3.json
```

## Known Limitations

- Actual CUDA compatibility, load/training/dev timing, VRAM, metrics, strict
  reload, and seed variation remain unknown until an independent authorized
  GPU run.
- The CPU tests mock lifecycle boundaries; they do not substitute for CUDA or
  model evidence.
- The bootstrap interval is a deterministic task-group sampling-stability
  summary, not a universal generalization guarantee.
- Final-test remains exploratory and one-shot. No test rows or labels were
  opened by this implementation or CPU validation.
