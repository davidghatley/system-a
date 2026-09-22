# Pilot Schema Repair

Date: 2026-09-22

Scope: repair only the `choose_seed_count()` pilot-evidence schema blocker from
`EXECUTOR_REVIEW_3.md`. No measured pilot artifact, CUDA/model work, or
final-test code or artifact was changed.

## Change

`choose_seed_count()` now accepts the exact persisted
`artifacts/experiment_i3/preflight/actual_model_pilot.json` envelope. The exact
key set is the seven existing measurement fields plus `status` and
`test_opened`. It requires `status == "actual_model_pilot_measured"` and
`test_opened is False`, while retaining the existing checks for finite numeric
measurements, the declared seed and update count, and memory/time budgets.

Focused tests use the real persisted JSON shape and reject missing fields,
unknown fields, wrong status, `test_opened == true`, wrong seed, wrong update
count, non-finite values, and exceeded time or memory budgets.

## Verification

Command:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3
```

Result: **26/26 tests passed** in `0.462s`.

The focused schema subset also passed: **4/4 tests passed**.

## SHA-256

Measured with:

```text
sha256sum training/laya_trace_i3/protocol.py training/laya_trace_i3/test_i3.py artifacts/experiment_i3/preflight/actual_model_pilot.json
```

```text
24921d8d8b249f05e1e7b5ad23922cf9b8db7aca988a8283ba54b96213491681  training/laya_trace_i3/protocol.py
dc7e9f0b6ec57fe0e55f8d2a291a73980c8aca2d5389de5adc684ccb788da497  training/laya_trace_i3/test_i3.py
2323e509f35e9d2e0e2fba64bd12dcb28a488f6086715c0ac4e115226a65d511  artifacts/experiment_i3/preflight/actual_model_pilot.json
```

The pilot artifact hash matches the preserved hash recorded by the independent
review. The test run used CPU/offline settings and did not open CUDA, model
weights, test rows, or final-test execution.
