# Iteration 3 re-review repair: atomic plan, claims, recovery, and authenticated final bytes

Date: 2026-09-23. This repair is limited to `training/laya_trace_i3/runner.py`, `experiment.py`, `protocol.py`, `test_orchestration.py`, and this report. No real model, GPU, training, recovery checkpoint evaluation, or real held-out test bytes were executed. No external source revisions were used; all checks were local and offline. Checks preceded the local review commit; nothing was pushed. Historical experiment artifacts were not edited.

## Implementation

### 1. Atomic cumulative budget planning

**VERIFIED by CPU synthetic tests and ledger inspection.** `protocol.BUDGET_EPSILON` is the single boundary tolerance used by decision composition, admissions, training reservations, and the aggregate ledger. Decision creation now performs a compare-and-swap reservation under the ledger flock: it rechecks the pilot-bound prefix, appends one immutable `state: held` 600-second final reservation, durably writes it, and rechecks the complete projected composition. The decision binds the pre-hold `planning_snapshot`, the post-hold `budget_snapshot`, the held token, exact reservation, reservation hash, and post-hold ledger hash in the immutable `i3-future-pilot-decision-v2` record. The main decision budget snapshot/hash therefore describe the ledger after the hold; the separate planning snapshot preserves the charge used for the training projection.

A held charge is included by `AggregateBudget._charged_seconds()` and therefore consumes the 7200-second aggregate limit. Once a decision or hold exists, the shared non-final guard rejects pilot/base/smoke and other headroom-consuming routes. The non-final ledger reservation also rechecks the decision/hold while holding the ledger lock, closing the check-to-reservation race. Training requests must equal the immutable per-seed projection within the shared epsilon; an over-request is rejected before a worker or budget reservation.

The final route calls `activate_hold()` and changes the existing reservation to `in_flight`; it never appends a second 600-second charge. The same token is later finished. The exact-boundary composition `600 pilot + 2 x 3000 training + 600 final = 7200` is covered, as are fallback composition, post-decision closure, and the no-double-allocation case.

### 2. One-shot execution claims and lock-race recheck

**VERIFIED by CPU direct/runner synthetic tests.** Each pilot, base, smoke, seed-training, and recovery admission now has a create-once execution claim. The claim is bound to the admission path and SHA, command, seed, actual config/manifest paths and hashes, full code identity, and (for seed routes) the immutable decision. Claims are validated and atomically hard-link published before any budget reservation or worker entry. Reusing a pre-created admission rejects on the existing claim before another reservation or worker; an automatically created admission also rejects on its existing admission/artifact.

Every non-final worker rechecks `require_non_final_worker_open()` immediately after acquiring `GpuLock` and before importing Torch/model code. Recovery and the final route have the same post-lock race check. Pilot, base, smoke, training, recovery, and their result/context artifacts carry the claim path and SHA. The tests cover direct reuse for each route, runner admission order, and the consumed-marker race probe.

### 3. Pilot ledger provenance

**VERIFIED by CPU adversarial tests.** The pilot result now contains the post-pilot `budget_snapshot`, including the exact reservation list, charged seconds, identity, limit, and ledger SHA. Its `budget_ledger_sha256` must equal that attached snapshot SHA. The pilot reservation metadata must be exactly `command=actual-model-pilot`, the configured smoke seed, and the pilot admission SHA. Decision validation checks the attached snapshot as an exact prefix of the current locked ledger, allowing only later appends. Earlier reservation edits, metadata changes, ledger rewrites, or a mismatched recorded hash fail closed.

### 4. Genuine recovery lifecycle

**VERIFIED by CPU runner-composed synthetic tests.** Training uses `seed_N/admission.json`; recovery uses the separate `seed_N/recovery_admission.json`. Recovery admission is not accepted unless it binds a real create-once training admission, its claim, a training context, the exact finished training budget token/reservation, and the exact checkpoint/reload bytes. The training context is durably written before the runner publishes `result.json`, so a deliberate result-publication failure leaves the evidence needed for a later recovery. Recovery results bind the training admission path/SHA and training token, and dev validation enforces the same chain. A pair decision cannot enter recovery, and a fabricated generic reservation is rejected. The orchestration test now composes an actual synthetic runner training worker, fails publication after its checkpoint/reservation are durable, and then recovers through the dedicated path.

### 5. Authenticated final evaluator

**VERIFIED by CPU synthetic loader/marker tests; real CUDA execution remains untested.** Before the final marker or test bytes, the held final token is activated. The selected `model.safetensors` file is opened once with `O_NOFOLLOW`, read fully into an immutable `bytes` buffer, hashed, and passed to `safetensors.torch.load`. The pathname or descriptor is never reopened for deserialization, so concurrent in-place writes after the read cannot alter the loaded state. A changed/unauthenticated file causes failure before marker consumption.

The final marker is create-once and carries freeze, decision, config, manifest, full code identity, selected checkpoint hash, budget ledger path/hash, token, reserved seconds, and the exact in-flight reservation. The final result is create-once and carries the same identities and selection, the marker path/hash, and the finished reservation. Repeated final invocation is rejected. The former two-argument marker helper and `final_test_once()` compatibility probe are retired and cannot bypass authenticated admission.

### 6. Loaded-code identity

**VERIFIED by CPU patch tests and source-identity inspection.** `protocol.py` captures a five-file disk identity when the module loads. Admission, claim, decision, seed result, freeze, and final-result validation compare current disk identity with that snapshot and reject provenance/publication when they differ. This detects many in-process edits but does not prove the bytes of a dynamic import performed between checks; therefore a fresh process launched from an immutable complete Git tree remains an operational precondition. No runtime test is claimed to make concurrent source mutation safe.

## Verification commands and results

All Python commands ran from `/home/davidhatley/Projects/research/system_a` with the repository `.venv`, CUDA hidden, offline model/transformer settings, one CPU thread per BLAS backend, and bytecode writes disabled. No model/GPU/training/held-out route was invoked.

### Orchestration suite, run 1

```text
$ CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_orchestration -v
Ran 34 tests in 6.160s
OK
```

### Orchestration suite, run 2

```text
$ CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_orchestration -v
Ran 34 tests in 7.139s
OK
```

### Integrated `test_i3` and full CPU suites

The obsolete final-test fixture was updated to assert that a selection lacking decision/hold provenance is rejected before lock, marker, or test decoding. The complete combined suite then passed twice:

```text
$ CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 training.laya_trace_i3.test_orchestration scripts.test_i3_dev_diagnostics release.i3.test_inference
Ran 87 tests in 6.029s
OK

$ CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 training.laya_trace_i3.test_orchestration scripts.test_i3_dev_diagnostics release.i3.test_inference
Ran 87 tests in 6.593s
OK
```

The 87 cases comprise 41 `test_i3`, 34 orchestration, one calibration regression, and 11 release import/input-contract cases.

### Static checks

```text
$ PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m py_compile training/laya_trace_i3/runner.py training/laya_trace_i3/experiment.py training/laya_trace_i3/protocol.py training/laya_trace_i3/test_orchestration.py
(exit 0; no output)

$ git diff --check
(exit 0; no output)

$ CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m training.laya_trace_i3.runner validate-config
{
  "config_sha256": "748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae",
  "model_loaded": false,
  "status": "config_valid",
  "torch_imported": false
}
```

## Source and historical-artifact hashes

Source hashes at the verification point (the report itself is not part of the five-file code identity):

```text
959572beaebe6eff880fe3c44155c8b94902ebbeda4ee0c601c838d55640bbff  training/laya_trace_i3/runner.py
11793123cf15f9ab339949b9c90fcdd658e8451b2db71aa73950fb165c1b964d  training/laya_trace_i3/experiment.py
16e97ab18a71f7c249717f45376e54267bbb48f27acbded479d9f3ac4cd8bb94  training/laya_trace_i3/protocol.py
080303bf41331c1dfc60c2e2882e9823442f71fb9d8fc10d386b43fb989e6b19  training/laya_trace_i3/test_orchestration.py
```

The following historical artifact hashes were rechecked and remain unchanged:

```text
2323e509f35e9d2e0e2fba64bd12dcb28a488f6086715c0ac4e115226a65d511  artifacts/experiment_i3/preflight/actual_model_pilot.json
7791622622b29499348ac9cf1adc0cd09e412191507e66d746b1a16c4472e671  artifacts/experiment_i3/preflight/runs/final_test.attempt.json
d5eea426d0dc67ef449446b582058830322a6c2124338e35829d6557ad2aba2b  artifacts/experiment_i3/preflight/runs/final_test.result.json
c18059b0409fd79db6e7d13a046453112caa2d74efc373981898355ffc320832  artifacts/experiment_i3/preflight/runs/freeze.json
ff7e2ca8beb914c09b9429e365759668be2a63a7da0ba122873fa3cfc2ae518e  artifacts/experiment_i3/seed42_recovery.json
85ab9e9cd205d72558e42033932afb91b8499d5bfa089f5486d711ca12ca4321  artifacts/experiment_i3/seed314159_run.json
```

Synthetic CPU ledgers, claims, contexts, and run roots remain under `artifacts/review_60fd/test_ledgers/` and `artifacts/experiment_i3/preflight/synthetic/`; they are not historical evidence.

## INFERRED

- The hard-link create-once publication, directory fsync, descriptor authentication, and local flock ordering are intended to provide the stated single-host lifecycle guarantees on this filesystem. The CPU suite does not emulate power loss.
- A held final reservation is conservative accounting, not a measured CUDA duration. A worker that overruns its reservation is durably charged and raises.
- A separate future output root remains an independent namespace; it is not a continuation or retroactive authorization of the consumed historical run.
- The import-time disk identity check detects many source edits and blocks mismatched publication, but it cannot prove bytes of a dynamic import between checks. A fresh process from an immutable complete Git tree is required.

## UNKNOWN / remaining limitations

- No real CUDA synchronization, model load, pilot, base evaluation, training, recovery evaluation, or final-test execution was performed. Actual wall-time distributions, deadline overshoot, and final latency remain unknown.
- `AggregateBudget` remains conservative pre-admission wall-time accounting rather than a hard real-time kill switch.
- No true multi-host filesystem, crash/power-cut injection, or adversarial process-death test was performed.
- The full 87-test CPU suite is green twice; no stale final-test fixture remains.
- Historical final-test evidence remains exploratory and read-only. This repair did not reopen held-out data or convert the historical attempt into a new experiment.
