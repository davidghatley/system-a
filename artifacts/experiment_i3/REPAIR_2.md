# Iteration 3 pre-run repair 2

Date: 2026-09-22. Scope was limited to `training/laya_trace_i3/`,
`artifacts/experiment_i3/`, and `research/28_iteration_3_experiment.md`.
`PRE_RUN_REVIEW_2.md` was not edited. CPU/offline only: no CUDA initialization,
Torch/model loading, smoke, training, inference, or final-test labels/results
were inspected.

## Decision

**Ready for independent re-review; not self-accepted for GPU work.** The
execution gate remains fail-closed until independent review explicitly opens
it.

## Verified changes

- `run_bounded_smoke` is an operational bounded path, but requires explicit
  review authorization and the opened gate. It acquires `GpuLock` before the
  in-function Torch import, rejects CPU fallback, performs two bounded updates
  over six logits using RLCD plus unit CE, synchronizes CUDA boundaries, and
  applies peak-memory, smoke-time, iteration, and aggregate GPU-second guards.
- `select_dev_checkpoint` rejects nonexistent/non-directory checkpoints,
  recomputes `model.safetensors` and `training_state.pt` hashes, and requires a
  hash-bound reload-evidence artifact matching strict model, optimizer,
  scheduler, scaler, and stopping-state evidence.
- `load_split` has no test bypass. `final_test_once` is the sole composed path:
  it atomically consumes the dev-freeze marker and only then opens test bytes;
  a second attempt is rejected.
- Config validation rejects mutations to the frozen 14,400-second iteration
  ceiling, 7,200 aggregate GPU-second ceiling, final-test policy, and latency
  definitions. Latency evidence must identify runtime monotonic timing,
  synchronized boundaries, all frozen phases, and the exact event count.
- Corrected data, accepted manifest bytes, six fixed labels, and the retained
  whole-target-turn estimand were not changed.

## Exact CPU/offline commands and results

All Python commands used `CUDA_VISIBLE_DEVICES=''`, four-thread CPU limits,
`PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`.

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
# 18 tests passed in 0.278s; OK

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py validate-config
# config_valid; config_sha256=748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae; torch_imported=false; model_loaded=false

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m py_compile training/laya_trace_i3/protocol.py training/laya_trace_i3/experiment.py training/laya_trace_i3/runner.py training/laya_trace_i3/test_i3.py
# passed without output

git diff --check
# passed without output
```

## Hashes

Verified before this report was written:

```text
fb52e9e2cc3915c1b3d58a20f9fdc29101db97859eb0071dd222344801e9fdc0  training/laya_trace_i3/protocol.py
bf3ec546372ce0d1bd67a98b44b9413abbe7428c463f3ca6d567c88be3e6211d  training/laya_trace_i3/experiment.py
314679ec15e02394f65a56b398c37905283dd17fef483e046586b2b8479c9551  training/laya_trace_i3/runner.py
7da655da1aab7c8a7c4515b408e1de1aaff0b6adc3c8de8fbc3d41be862c4acd  training/laya_trace_i3/test_i3.py
748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae  training/laya_trace_i3/config.json
4a8f9d0e91b425dc44bfe1cdfee504b3d57e367819b786862dcab127ed278986  artifacts/experiment_i3/PRE_RUN_REVIEW_2.md
```

## Inferred

The composed APIs and reject-by-default tests prevent the four reviewed
bypasses under their stated contracts. This is static/CPU contract evidence,
not evidence of successful CUDA execution or model compatibility.

## Unknown

Actual model compatibility, strict tensor reload, CUDA synchronization,
latency, VRAM, smoke duration, aggregate runtime, checkpoint production, and
final-test labels/results remain unknown. No final-test bytes were opened.
