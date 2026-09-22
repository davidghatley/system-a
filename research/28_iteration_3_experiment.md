# Iteration 3 experiment integration / spec freeze

Date: 2026-09-22. Status: **actual-model executor implemented; GPU work remains
blocked pending independent authorization**.

## VERIFIED

- Corrected v3 structural verification and the accepted manifest agree on 1,313
  rows: train/dev/test 1,051/140/122 and task groups 163/20/23. The earlier
  1,184/156/139 chronology is superseded, not an alternate estimand. Output
  hashes are recorded in `artifacts/experiment_i3/preflight/data_manifest.v3.json`.
- The accepted-shape manifest, fixed six-label one-question contract, finite
  guards, group-aware protocol fields, non-blocking GPU lock, deterministic
  dev selection, strict-reload contract, timings/VRAM fields, and atomic
  test-once lifecycle are implemented. The runner has `base-eval`, `train`,
  `dev-select`, `final-test`, and `smoke` entry points.
- The source-level `artifacts/trace2decision_i3/provenance.json` preserves the
  inherited i2 source/split lineage; it is not the v3 renderer attestation. The
  accepted v3 renderer identity and rerun hashes are instead in the manifest and
  `artifacts/trace2decision_i3/verification.json`.
- The original frozen specification and hash are retained as version 1. The
  repaired objective required a transparent pre-GPU version 2 re-freeze recorded
  by `PREDECLARATION.sha256`.

## IMPLEMENTATION

CPU execution is intentionally model-free while `execution_gate` remains
`blocked_until_independent_data_verification_and_pre_run_review`. Plan commands
verify train/dev bytes and schema, load no test bytes, import no Torch, and emit
`blocked_before_model_or_cuda`. The config reuses the pinned Iteration 2 base
and explicit six-choice RLCD+CE recipe, seeds `[42, 314159]`, four epochs,
macro-F1 selection and
the +0.05 added-value threshold. The existing metric implementation provides
fixed-label macro-F1, accuracy, per-class support/recall, confusion, clipped
NLL, Brier, duplicate-ID and non-finite guards.

Reproducibility inputs: starting commit `2d4408de85587a046c3f14fec1835dd9e69940c9`;
source provenance `artifacts/trace2decision_i3/provenance.json`; structural
verification `artifacts/trace2decision_i3/verification.json`; renderer
`pipelines/trace2decision/i3_convert.py`; commands below.

## UNKNOWN / BLOCKER

No-call completion remains absence-based and excluded. The estimand is therefore
retained whole target turns with at least one recognized tool call and exactly
one action category; the frozen sixth label may be absent. Broader family
independence and semantic leakage remain unknown.
Therefore no model execution, smoke timing, VRAM, base/fine-tuned metrics,
checkpoint reload, or test result is claimed. Test-once remains exploratory,
not sealed or confirmatory. This is an explicit pre-run block, not a failed
training result.

## Checks

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py validate-config
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py base-eval
```

The first command is the bounded CPU unit suite; the latter two validate the
frozen config and the blocked full-path plan. No command opens test records.

Observed repair-1 checks: `unittest` **15/15 passed**; `validate-config` returned
`config_valid`; accepted-manifest verification passed for 1,051 train and 140
dev rows; `repair-preflight` proved all six choice mappings had finite nonzero
RLCD losses and gradients and proved 163/20 train/dev task-group IDs; `base-eval`
returned `blocked_before_model_or_cuda` with 68 planned updates/seed and
`test_opened=false`. The re-freeze hash and whitespace checks passed. All
commands used offline mode, hidden CUDA, and at most four CPU math threads.

## Repair 2 status

Repair 2 addresses the four blocking findings without changing the accepted
manifest, corrected data, whole-turn estimand, or six-label order. The GPU
worker is now an actual lazy bounded path: it requires an explicit independent
review authorization, acquires the nonblocking lock, and only then imports
Torch; it performs bounded CUDA-only six-logit RLCD+CE updates and records
elapsed time, aggregate GPU seconds, peak reserved memory, and finite smoke
evidence. The current frozen execution gate remains closed, so no GPU work was
performed here.

Checkpoint selection now requires existing `model.safetensors` and
`training_state.pt` bytes, recomputed hashes, and a hash-bound JSON reload
evidence artifact. Test bytes can only be reached through `final_test_once`,
which consumes the dev-freeze marker before opening the test split and rejects
retry or direct `allow_test` access. Lifecycle validation freezes the 14,400
second iteration ceiling, 7,200 aggregate GPU-second ceiling, final-test
policy, and synchronized latency definitions. The CPU suite adds adversarial
probes for lifecycle mutation, checkpoint absence, test bypass, and CPU smoke
fallback.

This section records implementation evidence only. CUDA/model loading,
smoke/training/inference, and final-test labels/results remain unknown and were
not inspected.

## Actual-model executor implementation

The model-free placeholders have been replaced by a lazy, local-only executor
in `training/laya_trace_i3/experiment.py` and CLI routes in `runner.py`. The
executor reuses the pinned Iteration 2 Laya construction and RLCD+CE objective,
but acquires the non-blocking GPU lock before any Torch import. It now provides
an explicitly authorized two-update actual-model pilot, strict base dev
evaluation, per-seed four-epoch training, hash-bound strict model and complete
training-state reload evidence, synchronized latency, task-group bootstrap
summaries, dev-only selection, and the preserved one-shot final-test path.

The config/data/objective/labels and no-OOM-retry contract remain unchanged;
no re-freeze was required. The default execution gate is still closed and
actual-model commands require `--review-accepted`. CPU evidence, the exact
proposed pilot invocation, hashes, and limitations are recorded in
`artifacts/experiment_i3/EXECUTOR_IMPLEMENTATION.md`.

This implementation is ready for independent CPU review, not self-acceptance.
Actual model compatibility, CUDA timings, VRAM, metrics, reload behavior, and
test results remain unknown because no model or CUDA work was run here.

## Executor repair 1

The actual Iteration 3 executor now performs true immediate microbatch
backward with positional partial-window normalization. It retains no list of
loss graphs, preserves the frozen 68-update schedule and all AMP/clipping/
RLCD+CE/no-retry guards, and has CPU coverage for shuffled final tails.
The final-test route now validates the supplied dev freeze and selected
checkpoint/reload evidence before taking the nonblocking GPU lock and
atomically consuming the one-shot marker. It then opens test once, strictly
loads the selected checkpoint, evaluates fixed metrics/predictions and grouped
bootstrap, records synchronized latency, and atomically writes the result;
failures cannot retry. CPU tests and evidence are recorded in
`artifacts/experiment_i3/EXECUTOR_REPAIR_1.md`. No CUDA/model work or actual
final marker was used. The measured pilot artifact remains intact; its
conservative two-seed projection is 1662.004614497419 GPU seconds.
