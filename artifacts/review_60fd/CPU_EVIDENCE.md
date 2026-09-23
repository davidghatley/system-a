# Review 60fd CPU evidence

Date: 2026-09-23. Scope: CPU-only source tests, dev-only diagnostics, release import isolation, and synthetic orchestration. No historical held-out file was opened, no GPU workload or training was run, and nothing was pushed.

## Local checkpoints

- Implementation commits: `6a07e6f` (`fix: harden System-A release execution`) and `23542a6` (`fix: authenticate checkpoint bytes before loading`).
- Reproducible dev-evidence commit: `c72676e` (`data: add reproducible i3 dev evidence`).
- Commit-derived acceptance used exact commit `23542a625582d45f5f2f774796422bc1da585f9c`.

## Integrated CPU checks

Supported combined command:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 training.laya_trace_i3.test_orchestration scripts.test_i3_dev_diagnostics release.i3.test_inference
```

The final source passed twice: **87 tests, OK** in 6.062 s and **87 tests, OK** in 6.129 s. The cases comprise 41 protocol/recovery tests, 34 provenance/cumulative-budget orchestration tests, one grouped-temperature alignment regression, and 11 release import/input-contract tests. The orchestration suite alone passed twice: 34/34 in 7.467 s and 34/34 in 7.465 s.

`py_compile` passed for release, runner/experiment/protocol, tests, and diagnostic scripts. `validate-config` reports Torch not imported and no model loaded. Git whitespace checks pass.

## Hermetic tests and dev-only diagnostics

- `training/laya_trace_i3/test_i3.py` creates durable synthetic markers, pilot/decision/result identities, checkpoint bytes, metadata, and concurrency fixtures. It contains no `TemporaryDirectory`, historical checkpoint digest, historical pilot dependency, or dependency on `artifacts/experiment_i3/preflight/runs/seed_42`.
- Recovery fixtures compute hashes from their own bytes and reject byte tampering and incorrect stopping state.
- The grouped-CV regression independently reconstructs interleaved fold assignments and original row alignment, then compares pooled scaled metrics; concatenation in fold order is observably different.
- `scripts/i3_dev_diagnostics.py` reproduces seed 42, seed 314159, and tracked TF-IDF from committed dev-only extracts. Review output `artifacts/review_60fd/diagnostics/metrics.json` has SHA-256 `0cfba0dbb9b621319fef37c0731c5fc0093348a003315d4a112672c4398cd563` and is value-identical to historical `artifacts/release_i3/metrics.json`.
- `scripts/i3_evidence_audit.py --dev-only` validates saved prediction IDs, gold labels, and metadata against the accepted dev split, recomputes seed and TF-IDF metrics, never inspects checkpoints/test records, and writes `artifacts/review_60fd/diagnostics/dev_only_evidence_audit.json` (SHA-256 `bbc3dc1b09f35d1f887fb39666fd853358bee9d92e1fb8e152defb2032677e0d`).

Compact dev-only inputs contain 140 rows each:

- seed 42 extract: `87f6125a66c4cb2c6aa098cae331cdfa3107667755e45ce4a920fe6284409a7a`; untracked source run JSON: `ff7e2ca8beb914c09b9429e365759668be2a63a7da0ba122873fa3cfc2ae518e`;
- seed 314159 extract: `247a4abee00da4292436461e398519498e1176c8a0049f48de12d7c48decab3e`; untracked source run JSON: `85ab9e9cd205d72558e42033932afb91b8499d5bfa089f5486d711ca12ca4321`.

Exact dev-only extraction serialization, reproduced in memory and byte-compared with both committed files:

```python
rows = (result.get("dev") or result["reports"][0]["dev"])["predictions"]
data = "".join(
    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    for row in rows
).encode("utf-8")
```

Both committed extracts are byte-identical to that serialization of their named source JSON. A clean checkout does not require the untracked source JSONs.

## Release importer and real CPU package parity

`release/i3/bundle_integrity.py` verifies manifest SHA-256 `10abd35118b1e5caa67896c8ba7487f670b2e788be3ae96f518777b22ce55d13`. Relative to `60fd5b3`, only `inference.py` and `test_inference.py` hashes differ. Selected weights remain `9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947`.

Importer tests cover exact private namespace collisions, foreign public `laya` modules, manifest-authenticated relative imports, post-verification source/helper tampering, repeated loads, same-path source-set changes, two bundles, temporary finder cleanup, unchanged `sys.path`, and process-private state restoration. The suite passed twice.

With local weights present and CUDA hidden, `scripts/i3_verify_package.py` completed a real seven-record CPU dev parity check without held-out access. Output SHA-256 is `1c1491b0caa91106ff20908cb0d68346e613b209cb9f57ef8cc41980e7200a3c`; maximum raw difference is `5.066394805908203e-07`, calibrated difference is `1.176670855196349e-07`, and all argmax labels match. The artifact records Python `3.14.7`, Torch `2.11.0+cu128`, NumPy `2.5.3`, safetensors `0.8.0`, Transformers `5.17.0`, Torch threads 6/6, no OMP/MKL/OpenBLAS override, and `1e-6` raw/calibrated tolerances. A one-thread diagnostic changed one raw probability by `1.6987323760986328e-06` while preserving argmax; that thread-capped invocation is not the supported default-runtime parity check and the sensitivity is recorded.

## Commit-derived isolated acceptance

Exact command:

```sh
scripts/run_review_cpu_suite.sh 23542a625582d45f5f2f774796422bc1da585f9c commit_23542a6
```

Inventory records `object_type=commit` and source commit `23542a625582d45f5f2f774796422bc1da585f9c`. The retained source is `data/review_60fd/isolated_tree_commit_23542a6/`; it explicitly omits:

- `artifacts/trace2decision_i3/output/test.jsonl`
- `artifacts/experiment_i3/preflight/runs/`
- `release/i3/bundle/model/model.safetensors`
- `data/cache/`

The 87-test suite passed twice in that commit-derived tree: 6.093 s and 9.165 s. Dev-only diagnostics and audit completed with hashes `0cfba0db...cd563` and `bbc3dc1b...67e0d`. Package parity was not counted as a pass: with weights absent it failed exactly with `PACKAGE PARITY unavailable; missing local prerequisites: release/i3/bundle/model/model.safetensors`.

Evidence hashes:

- inventory: `d94bde065c7391d5153f95ed939eaf9fe12ed50b225c1f2342ec397b69d7ec2a`
- run 1: `debc25dad026851126190b1f49b7e42ee475b3f134f2b9835e7337e10cba1321`
- run 2: `b3d9969c88aff690fa60affb9020a0fc3cca77e5847124877adef87672063440`
- diagnostics log: `19e20dceebdd7e3e6f6dabf507b65bf2efa990aa9b9dbd29cc9867806559fd6b`
- dev audit log: `dc4f05057025c84e001624bbdab43aa2b97d00aecace0ea14f7e61da091efc73`
- missing-weight log: `31d4df83a761608b88d6c1e2a53e67be93111c2b51dbcd4c886dd302ea707660`

Historical checkpoint/result reconstruction remains unavailable and is not represented as verified.
