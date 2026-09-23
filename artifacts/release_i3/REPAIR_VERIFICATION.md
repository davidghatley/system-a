# Iteration 3 repair verification

Date: 2026-09-23. Scope: CPU-only audit of the uncommitted Iteration 3
executor repairs in `training/laya_trace_i3/{experiment.py,protocol.py,runner.py,test_i3.py}`.
No CUDA device/model execution, training, held-out test record access, or test
marker modification was performed.

## Verified

- Final CPU suite result is recorded below; it includes 40 tests.
- Existing tests exercise the six-choice objective/mapping, shuffled
  accumulation-tail scaling, pilot envelope and seed selection, checkpoint
  selection/reload-evidence schema, lock-before-Torch guards, recovery
  checkpoint hashes/stopping counters and exclusive publication, and final-test
  selection/lock/marker/decode ordering with an intercepted test loader.
- The final-test ordering test replaces the test decoder before any held-out
  bytes are read. This verification did not open the manifest's test path.
- Source audit: training writes the CPU checkpoint/state before post-dev
  evaluation, deletes the original model/optimizer/scheduler/scaler, then
  constructs a fresh model and reloads checkpoint weights and training state.
  The comparison verifies model-state digests and compares serialized CPU
- Repair implemented: after strict checkpoint loading, training-state loading,
  and `load_state_dict()` for the fresh optimizer/scheduler/scaler, runtime now
  structurally compares each restored object's live `state_dict()` with its
  pre-save CPU snapshot. Tensor leaves require matching type, dtype, shape, and
  identical bytes from contiguous CPU tensor buffers; nested containers and
  scalar values are checked recursively. This distinguishes +0.0 from -0.0.
  Any mismatch fails closed before reload evidence can be written.
- Added CPU regression fixtures for equal/mismatched restored nested state, a
  actual-Torch CPU signed-zero/mismatch fixture, a partial two-seed publication
  that must block selection without deleting the first seed, and checkpoint-byte
  checksum mismatch rejection. The two newly added filesystem scenarios use
  unique durable paths under `artifacts/experiment_i3/preflight/guard_probe/`
  and intentionally retain their evidence; they do not use auto-cleanup
  temporary directories or overwrite prior fixture paths. Existing tests
  already cover declared two-seed/fallback decision branches and mutated pilot
  checksum rejection.
- Source audit: original model and optimizer references are deleted before the
  second model is moved to CUDA; however GPU allocator lifetime/peak-memory
  behavior cannot be established by this CPU-only review.

## Exact test command/result

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
```

Result: **40/40 passed**, 22.652 seconds.

## Inferred / unknown

- INFERRED: based on source ordering, deleting the original model and optimizer
  should release their Python references before the reloaded model occupies the
  device. CUDA allocator and other live references can retain storage; this is
  not a measured guarantee.
- UNKNOWN: actual CUDA peak memory and lifetime, hardware/model compatibility,
  end-to-end training, and final-test results. Resolving GPU-specific unknowns
  requires explicit parent authorization; none was attempted here.
