# Executor Review 4

Date: 2026-09-22

Scope: independent recheck of only the pilot-schema blocker recorded in
`EXECUTOR_REVIEW_3.md`, after `PILOT_SCHEMA_REPAIR.md`. CPU/offline only. No
CUDA device, model weights, training, test rows, final-test execution, or
implementation edits were used or made.

## Commands and Evidence

All Python checks used:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B
```

The direct check loaded the unchanged persisted artifact
`artifacts/experiment_i3/preflight/actual_model_pilot.json` and passed it
directly to `choose_seed_count()` with the accepted training-row count of
1,051. It returned:

```text
expected_updates_per_seed=68
projected_gpu_seconds_two_seeds=1662.0046144974185
selected_seeds=[42, 314159]
fallback_applied=False
```

The artifact SHA-256 before and after the call was unchanged:

```text
2323e509f35e9d2e0e2fba64bd12dcb28a488f6086715c0ac4e115226a65d511
```

Seven exact-envelope adversarial variants were each rejected with
`ProtocolError`: missing `status`, an unknown key, wrong `status`,
`test_opened=true`, wrong seed, wrong update count, and `test_opened=null`.

Focused command:

```text
.venv/bin/python -B -m unittest training.laya_trace_i3.test_i3.ProtocolTest.test_two_seed_default_and_only_declared_fallback training.laya_trace_i3.test_i3.ProtocolTest.test_nonfinite_or_over_budget_smoke_forbids_main training.laya_trace_i3.test_i3.ProtocolTest.test_pilot_envelope_rejects_missing_unknown_status_opened_seed_and_update_variants training.laya_trace_i3.test_i3.ProtocolTest.test_pilot_envelope_rejects_exceeded_budgets
```

Result: **4/4 tests passed** in `0.001s`.

## Pilot-Schema Finding

The blocker is resolved. The persisted pilot envelope is accepted directly by
`choose_seed_count()`, retains the declared finite-measurement and budget
guards, selects the declared two-seed plan under the conservative projection,
and rejects the exact-envelope adversarial mutations. No pilot artifact or
implementation file was changed during this recheck.

## Carried-Forward Verified Findings

- Accumulation remains positional. For 1,051 accepted training rows and
  accumulation 64, each epoch has 17 windows, four epochs have 68 optimizer
  updates and 4,204 microforwards, and the short tail is scaled by its divisor.
- The training-loop ordering remains AMP scale/backward, unscale,
  finite-gradient check, clipping, checked scaler step, scheduler step, and
  gradient reset. The schedule remains `T_max=68`.
- Final-test lifecycle guards remain verified: frozen provenance and selection
  checks precede the nonblocking GPU lock; the marker is atomically consumed
  before test decoding; strict checkpoint reload and exact state evidence are
  required; test metrics, grouped bootstrap, and synchronized latency use the
  declared fixed protocol; output is atomic and retries are rejected.
- Final-test adversarial mocks previously verified selection rejection before
  lock/test access and lock -> marker -> test-decode ordering. Direct test
  loading remains rejected and there is no test bypass argument.

## Unknowns

Actual CUDA compatibility, model loading, GPU memory, synchronization timing,
training completion, checkpoint creation, and final-test metrics remain
unmeasured by design. These are outside this pilot-schema-only CPU/offline
recheck.

ACCEPTED_FOR_MAIN_RUN
