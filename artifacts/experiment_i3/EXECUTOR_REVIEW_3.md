# Executor Review 3

Date: 2026-09-22

Scope: independent CPU/offline review of `EXECUTOR_REPAIR_1.md` and the current
`training/laya_trace_i3/` executor. I modified only this report. No CUDA device,
model weights, training, inference, test rows, or final-test result was opened.

## Commands and Evidence

All Python checks used:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B
```

The full CPU suite passed: **24/24 tests** in `0.458s`.

`validate-config` and `repair-preflight` both passed with `torch_imported=false`,
`model_loaded=false`, and `test_opened=false`. Current implementation hashes
are:

```text
training/laya_trace_i3/config.json 748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae
training/laya_trace_i3/experiment.py f37891af485b30f94e3c507fa97a8d08866a113cef45c02a93ccffdec078f92b
training/laya_trace_i3/protocol.py fb52e9e2cc3915c1b3d58a20f9fdc29101db97859eb0071dd222344801e9fdc0
training/laya_trace_i3/runner.py a92ae3a9937d2cbbd4be5cfb8b6fe376bd51d7911b6574afa34d471acfe5f0bb
training/laya_trace_i3/test_i3.py 869ecfd62e30a9b7a71806ccd93ff1f8f01d1ff54f272c5a7651f1a537f631bd
```

## Verified Passes

### Accumulation and schedule

- `accumulation_windows()` uses positional windows, not record identity, so
  shuffled epoch tails are handled correctly.
- For 1,051 accepted training rows and accumulation 64, each epoch has 17
  windows: 16 divisors of 64 and a final divisor of 27. Four epochs therefore
  produce exactly 68 optimizer updates and 4,204 microforwards.
- The training loop scales each loss and calls backward immediately. The window
  metadata contains positions, indices, and an integer divisor only; it does not
  retain tensors or autograd graphs.
- Ordering is verified in source and tests: AMP scale/backward, unscale,
  finite-gradient check, clipping, checked scaler step, scheduler step, then
  gradient reset. Scheduler `T_max` is the exact 68-update schedule.

### Final-test lifecycle

- CPU freeze provenance, manifest/config hashes, dev-only selection, checkpoint
  bytes, reload-evidence bytes, strict-load evidence, and exact optimizer,
  scheduler, scaler, and stopping-state flags are validated before the marker.
- The final command checks for an existing result before entering the lock,
  acquires the nonblocking GPU lock before Torch/model imports, then atomically
  creates the one-shot marker before decoding test rows.
- The selected checkpoint is loaded with `strict=True`. Test metrics use fixed
  labels and the declared NLL floor; bootstrap is task-group based with fixed
  seed 20260920 and 1,000 replicates. Latency uses the declared three warmups,
  ten synchronized measurements, and fixed included phases.
- Final output is written through a temporary file and `os.replace`, and an
  existing result or consumed marker raises a no-retry error.
- Adversarial mocks verified selection rejection before lock/test access and
  lock -> marker -> test-decode ordering. A marker failure stopped before test
  decode (`lock-init`, `lock`, `marker`, `unlock`). Direct test loading is
  rejected; there is no test bypass argument.

### Budgets and preserved pilot

- Configuration requires 11.5 GiB peak reserved memory, 14,400 seconds elapsed,
  7,200 aggregate GPU-seconds, and `no_oom_retry=true`. Runtime guards reject
  non-finite values and over-budget elapsed or aggregate GPU time with no retry.
- The preserved pilot artifact hash is
  `2323e509f35e9d2e0e2fba64bd12dcb28a488f6086715c0ac4e115226a65d511`.
- Using its seven measured fields, the declared formula reproduces
  `831.0023072487093` GPU-seconds per seed and
  `1662.0046144974185` for both seeds. The projection is below 7,200, so the
  conservative rule selects `[42, 314159]`, not the fallback single seed.

## Blocker

The persisted pilot artifact is not consumable by the executor's own
`choose_seed_count()` validator. `actual_model_pilot.json` contains the extra
keys `status` and `test_opened`; the validator requires an exact seven-key
schema and raises `ProtocolError: smoke evidence does not match the predeclared
profile` when passed the artifact as written. The arithmetic only succeeds
after manually projecting a reduced dictionary containing the seven required
fields. This means the recorded pilot cannot directly authorize the two-seed
decision, despite the reproduced value being about 1662 GPU-seconds.

This is a concrete readiness blocker. It must be resolved by making the
persisted pilot schema and `choose_seed_count()` contract agree, with a CPU test
that passes the actual artifact through the validator. No implementation change
was made in this review.

## Unknowns

Actual CUDA compatibility, model loading, GPU memory, synchronization timing,
training completion, checkpoint creation, and final-test metrics remain
unmeasured by design. These are not treated as failures under the requested
CPU/offline boundary.

CHANGES_REQUIRED
