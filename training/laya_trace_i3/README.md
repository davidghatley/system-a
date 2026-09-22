# Laya Trace Iteration 3 preflight

This separate path prepares the accepted Iteration 2 Laya recipe for a bounded
observed-next-action comparison. It does not modify or import the B2 runner.
The corrected 1,051/140/122 v3 manifest is structurally accepted, but the
execution gate remains blocked pending independent acceptance of this repair;
no command loads a model, initializes CUDA, inspects test records, runs
inference, or trains.

## Current commands

Run from the repository root with CPU/GPU visibility and math threads bounded:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py validate-config
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py dry-run
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
```

The accepted-shape manifest can be checked before model work. The pre-model
command requires its immutable hash out of band:

```text
.venv/bin/python -B training/laya_trace_i3/runner.py validate-accepted-manifest --manifest-sha256 <sha256>
```

That command verifies only train/dev bytes and counts. It does not open test,
import Torch, or load weights. GPU execution remains blocked until independent
re-review accepts the repaired protocol.

The full gated path is exposed as `base-eval`, `train`, `dev-select`,
`final-test`, and `smoke`; each currently emits a blocked CPU plan.

## Execution contract prepared for the next gate

- One row is one whole target assistant turn and exactly one `next_action`
  question. Metrics cannot silently score multiple question-level items.
- Labels and their order are fixed in `config.json`; absent classes contribute
  zero to macro-F1 and remain in the confusion matrix.
- The six output logits follow renderer criteria order, not reporting order.
  `config.json` freezes the permutation and a nonzero choice-logit RLCD+CE
  objective; the unrelated binary `act_head` is excluded rather than multiplied
  by zero.
- Raw-softmax metrics are macro-F1, accuracy, support/recall by class, confusion,
  NLL with declared numerical clipping, and multiclass Brier score.
- Two seeds (`42`, `314159`) are the default. Only a successful measured smoke
  may select seed `42` alone, and only if the conservative two-seed projection
  exceeds 7,200 aggregate GPU-seconds. Failure, OOM, or non-finite smoke output
  forbids the main run rather than triggering a fallback.
- Every future GPU process must use the implemented single non-blocking `fcntl` lock at the
  configured local lock path before importing Torch. The lock owner record must
  include command, PID, start time, config hash, manifest hash, and seed.
- Checkpoints retain file hashes and a deterministic tensor-state digest. The
  dev selector rejects candidates unless runtime evidence proves strict,
  bit-exact model reload and exact optimizer/scheduler/scaler/stopping state.
- Candidate selection uses dev macro-F1 only, then dev NLL, then declared seed.
  A freeze artifact records all candidates and hashes. Final test starts by
  atomically creating a single-use attempt marker before opening test bytes;
  failed attempts are not retried. The previously inspected test is described
  only as exploratory held-out evidence.
- Warm latency is synchronized tokenization-through-probability-copy time for
  one whole record at batch size one. Cold load and training timers have the
  separate definitions frozen in `config.json`.

CPU tests exercise lock contention and reject malformed reload/latency evidence;
actual CUDA synchronization, model reload, smoke, checkpoint freeze, and test
execution remain intentionally inactive pending independent re-review.
