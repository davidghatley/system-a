# I3 evidence audit (dev membership audit update)

## Reproduction and scope

Run at repository root:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/i3_evidence_audit.py
```

The CPU/offline script verifies the accepted manifest's dev JSONL SHA before
reading it; it reads **only** this dev JSONL, checkpoint files, the two seed
result/reload metadata files, and saved baseline dev predictions/result. It
does not read test JSONL or the historical final-test result. It compares
ordered `(ID, gold)` pairs and row count from the dev split to both seed
prediction lists, checks probability keys/nonnegative finite values/sums,
recomputes metrics with NLL probability floor `1e-12`, and checks those against
reported run metrics to absolute tolerance `1e-10`. It also checks exact ID/gold
order in the saved TF-IDF baseline predictions and recomputes its metrics.
Machine-readable compact evidence: `evidence_audit.json`.

## Verified results

Accepted dev split hash:
`ade8262625a46eef78dcb2b11b42eaced444e1604bab1af5423412a67ae92783` (140
rows). Both seed prediction files match every dev ID and gold label **in the
same order**, and all predictions have valid six-class probability vectors.
Recomputed metrics agree with the respective result report at tolerance
`1e-10`:

| Run | Accuracy | Macro-F1 | NLL | Brier |
|---|---:|---:|---:|---:|
| Seed 42 | 0.67857 | 0.44526 | 3.40557 | 0.53269 |
| Seed 314159 | 0.71429 | 0.44194 | 3.74809 | 0.55254 |
| TF-IDF logistic baseline | 0.60714 | 0.29274 | 0.95189 | 0.51498 |

TF-IDF baseline file: `artifacts/baseline_i3/preflight/corrected_v3/dev_predictions.jsonl`;
SHA-256 and ID/gold check are in JSON. (Its NLL is substantially lower even
though accuracy is lower; these are distinct reported metrics.)

Both seeds' checkpoint file digests match their result metadata. Reload JSON
digests match result metadata; before/after model-state hash strings match and
agree with the result's model-state hash. **Interpretation limit:** strict-load,
optimizer/scaler/scheduler/stopping `*_exact` booleans are recorded metadata
claims (`reload_exact_flags_reported_true`), not an independent live-object
proof performed by this audit.

## Recovery vs. disk exhaustion; classification

- **REPORTED (failure):** `MAIN_RUN_STORAGE_FAILURE.md` records the two-seed
  command failing while serializing seed-42 `model.safetensors` with
  `No space left on device` and 1.3 GiB available. Recorded command:
  `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false .venv/bin/python -B
  training/laya_trace_i3/runner.py train --review-accepted --report
  artifacts/experiment_i3/main_run.json`.
- **REPORTED (recovery procedure):** `SEED_ORCHESTRATION_REPAIR.md` gives exact
  recovery command:
  `env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 .venv/bin/python -B
  training/laya_trace_i3/runner.py recover-seed --seed 42 --review-accepted`.
  It describes authorized recovery as lock-before-Torch, validating checkpoint
  byte hashes/reload/stopping evidence, CPU-loading and evaluating dev only,
  then atomically publishing `seed_42/result.json` with status `recovered`.
- **VERIFIED (current evidence only):** seed-42 checkpoint/result now exists;
  file hashes and dev membership/metrics pass this audit. The historical
  `seed42_recovery.json` record is a saved prediction artifact, not an audit
  trail of invocation. The repair document proves the intended command and
  procedure, not execution of that command on a particular occasion.
- **UNKNOWN:** whether the exact documented recovery command was the command
  actually executed, its execution time and authorization event, and how the
  filesystem capacity condition was resolved. Current bytes cannot establish
  those historical circumstances. Do not treat the reported failed save as
  conflicting with the now-present, separately validated recovery checkpoint.

## Test access, provenance, and mistake disclosure

This continuation did not read held-out test records or reopen the historical
test result. In the previous audit attempt, I inadvertently opened
`final_test.result.json` while bulk-reading run evidence; it was not used in
analysis or conclusions. The previous audit report incorrectly said I had not
opened it; this disclosure supersedes that wording. No test record content is
quoted or used here. Test-exposure authorization/history remains **UNKNOWN**
from this dev-only audit.

Manifest SHA-256: `158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2`.
Data/preprocessing/tokenizer source revisions recorded there are respectively
`1371ed38f8890d0520a53bc7ad850308eb4d7a22`,
`d113dca2512fb3eaca313534bc54c7162d87c1d4`, and
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`. These identify input lineage,
not necessarily exact dirty training/recovery code. Workspace had concurrent
modified training files; exact executed training/recovery source revision
remains **UNKNOWN**. No GPU/model execution performed.
