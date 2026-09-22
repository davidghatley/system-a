# Iteration 3 Laya experiment preflight

Date: 2026-09-22
Starting commit: `2d4408de85587a046c3f14fec1835dd9e69940c9`

## Decision

**REPAIRED CONTRACT; BLOCKED BEFORE MODEL LOAD.** The accepted Iteration 2
B2 trainer is suitable as the source of the pinned Laya model, sequence
construction, RLCD plus CE recipe, optimizer partition, FP16 adaptation, finite
guards, memory ceiling, and exact checkpoint reload checks. Its public-data
subset assumptions, five-question item accounting, accuracy gate, hard-coded
paths, and B2 output contract are not safe to reuse directly for one-label
Trace2Decision records.

The separate Iteration 3 preflight fixes a record-level six-class metric
contract and an accepted 1,051/140/122 manifest interface. Execution remains
blocked pending independent re-review of the repair.

## Frozen before results

- Base: `convaiinnovations/laya` revision
  `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, with the accepted B2 weights,
  config, Laya source, `common.py`, and notebook hashes.
- Labels in order: `read`, `search`, `edit`, `execute`, `other_tool`,
  `respond_or_finish`.
- Unit: one retained record and one `next_action` question represent the entire
  observed target assistant turn. Gold and metadata are evaluation-only.
- Recipe: four epochs, accumulation 64, sigma 0.4/0.3/0.2/0.1, AdamW encoder
  LR 2.5e-5 and non-encoder LR 1e-4, cosine schedule, explicit six-choice RLCD
  plus full CE, FP16
  scale 64, clipping 1.0, and no OOM retry. A short accumulation window is
  normalized and flushed at each epoch boundary before sigma changes. Exact
  updates are derived from the accepted train count rather than guessed now.
- Primary metric: fixed-label macro-F1, with five absolute points over the
  dev-selected inexpensive baseline as the practical added-value target.
  Accuracy, per-class support/recall, confusion, raw-softmax NLL, and multiclass
  Brier are secondary metrics. Missing classes remain in the macro average.
- Selection: complete declared seed checkpoints are ranked by dev macro-F1,
  then dev NLL, then seed. Test cannot select a checkpoint or setting.
- Seeds: 42 and 314159. Seed 42 alone is allowed only when a successful bounded
  smoke yields a conservative two-seed projection over 7,200 aggregate
  GPU-seconds. Non-finite, over-memory, over-smoke-time, or failed smoke forbids
  main training instead of authorizing fallback.
- Test: only after a dev-only freeze. The future executor must atomically consume
  the sole attempt before opening test bytes. There is no retry and no untouched
  confirmatory-test claim.
- Latency: synchronized batch-one whole-record time from tokenize/build through
  probability copy, separately from cold load and training time.

## Verified versus pending

**Verified by CPU implementation/tests:** fixed metric arithmetic and finite/input
guards; missing-class behavior; pending-manifest execution block; pinned config
validation; count-derived update arithmetic; two-seed default and sole measured
fallback rule; deterministic dev-only checkpoint selection; repository-local
output boundaries; all six reporting labels map to renderer choice logits and
produce a finite, nonzero-gradient RLCD+CE reference objective; train/dev expose
1,051/140 rows and 163/20 valid task-group bootstrap IDs; lock contention is
rejected; and malformed reload/latency evidence is rejected.

**Prepared contract, not yet GPU-runtime verified:** exact model/training-state
reload and checkpoint hashes, CUDA synchronization, warm/cold timing, peak VRAM,
and final test consumption. These require an independently accepted CUDA runner.

**Unknown:** semantic/family leakage, measured smoke cost, whether two seeds fit
the two-GPU-hour ceiling, observed metrics, seed variation, and whether Laya
beats base or inexpensive baselines.

No new train/dev/test predictions were inspected. No model was loaded, CUDA was
hidden, and no training or inference was run.
