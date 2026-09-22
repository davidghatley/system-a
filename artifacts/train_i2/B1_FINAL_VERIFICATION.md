# Trainer B1 Final Verification

Date: 2026-09-21
Role: independent Trainer B1 gate verifier

## Decision

**B1 evidence: ACCEPTED.**

**B2: EXPLICITLY AUTHORIZED under the frozen Iteration 2 criteria.** This is
authorization only. No B2 work, model inference, or training was performed by
this verification. Before B2 starts, its immutable subsets, seed, optimizer,
rates, context, updates/epochs, batching, objective, primary metric, and
stopping rule must still be predeclared as required by
`research/27_iteration_2_acceptance.md`.

The specialist achieves `136/200 = 0.680` primary native-label choice accuracy,
versus `71/200 = 0.355` for the base: `+0.325` absolute and `+65` correct
answers. This exceeds the frozen gate of at least `+0.05` and at least `+8`
additional correct choice answers.

## Basis

### Verified

- The frozen source file SHA-256 is
  `833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d`.
  It contains 100 rows in exact ID order
  `agent_trace_observability_000000` through
  `agent_trace_observability_000099`, all in workflow
  `agent_trace_observability`.
- Independent Parquet decoding finds 500 questions: 200 `choice`, 200 `score`,
  and 100 `noul`. The original predeclaration's “400 questions total” is a
  descriptive error. The erratum correctly records 500; the primary
  200-choice subset and gate are unaffected.
- The as-run predeclaration remains byte-for-byte preserved at full-file
  SHA-256
  `5bb9eeb19c033265a91b0f20c131aeca6307f0566e158cef2cde1d31bf26a7c2`.
  Removing only line 17 produces
  `bdf907ff22833aaa183f9e0f548809c089938a3eca9390d1a121c200033f321c`,
  not its originally recorded `a8d824...` value. Applying only the documented
  400-to-500 correction and then removing line 17 produces the independently
  reviewed expected value
  `f94bd744fc667ad37b8891c33321ff5e1374cc9887440a990b53807d8c710e29`.
  The erratum states all three facts accurately and does not rewrite the
  as-run object.
- The output SHA-256 is
  `6bfed0f71e395d8b1f29b3e9f8a462d942c7900a45b305517ede6a81be9d29dc`.
  It records status `completed`, the correct source/predeclaration hashes,
  source revision, Laya revision, model revisions, and common
  `max_len=512`, `head_max_len=192` settings.
- Both models have exactly 500 prediction records. Their ordered
  `(record_id, question_id, type, gold_label, workflow)` tuples exactly match
  the ordered source questions. Recomputed probability argmax, correctness,
  per-type totals, all-question totals, and gold NLL agree with the output:

| model | choice | score | noul | all hard | mean gold NLL |
|---|---:|---:|---:|---:|---:|
| base | 71/200 = 0.355 | 65/200 = 0.325 | 58/100 = 0.580 | 194/500 = 0.388 | 1.3373310093 |
| specialist | 136/200 = 0.680 | 148/200 = 0.740 | 84/100 = 0.840 | 368/500 = 0.736 | 0.8008518995 |

- `positive_control.py` uses the same source, construction, prediction path,
  and metric implementation for both pinned checkpoints. `choice` and `score`
  consume returned probability maps; `noul` maps returned `p_true` to
  `false=1-p_true, true=p_true`; hard labels are argmax; gold NLL uses the
  returned gold-label probability. The common 512-token context is disclosed
  as a fair comparison setting, not a reproduction of the specialist card's
  1024-token benchmark.
- The local Laya checkout is exactly
  `d113dca2512fb3eaca313534bc54c7162d87c1d4`. Local snapshot directories exist
  for base revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` and
  specialist revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`; those same
  revisions are recorded in the completed output.
- The repaired trainer suite passes all six tests. In-memory compilation of
  all nine `training/laya_local/*.py` and trainer test files passes without
  writing bytecode.

### Inferred

- The predeclaration's mtime (`16:33:12 -0600`) precedes the completed output's
  mtime (`16:34:01 -0600`), and the output embeds the current predeclaration's
  full-file hash. Together with the declared chronology and evaluator behavior,
  this supports that the rows, metrics, and gate were frozen before the public
  predictions. Filesystem mtimes and a self-recorded output are supporting
  evidence, not independent cryptographic timestamping.
- Accepting the immutable as-run file plus explicit erratum is preferable to
  retroactively rewriting the purported pre-run object. The erratum fully
  repairs the documentation defects without changing rows, predictions,
  settings, pins, metrics, or threshold.

### Unknown

- Cryptographic proof that the predeclaration existed in exactly its current
  form before either model was loaded or any prediction was made is
  unavailable. The original self-exclusion hash is wrong, and no trusted
  pre-run timestamp, signed record, or pre-run commit is present. This remains
  unknown rather than being upgraded to verified by the erratum.
- The missing cryptographic proof does not contradict the frozen criteria or
  the supported chronology, and it does not affect the independently
  reproducible source/output alignment or the large gate margin. It is
  therefore recorded as an evidence limitation, not treated as a B1 blocker.
- No claim is made about workflows outside the bounded public
  `agent_trace_observability/test` subset or about reproduction of the model
  card's aggregate benchmark.

## Commands And Results

All commands ran from the repository root unless noted. No model inference or
training was run.

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 6 tests ran; 6 passed.

```text
sha256sum artifacts/train_i1/source_test.parquet artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md artifacts/train_i2/positive_control_public.json
```

Result, in order:

```text
833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d
5bb9eeb19c033265a91b0f20c131aeca6307f0566e158cef2cde1d31bf26a7c2
6bfed0f71e395d8b1f29b3e9f8a462d942c7900a45b305517ede6a81be9d29dc
```

```text
(from data/laya) git rev-parse HEAD
```

Result: `d113dca2512fb3eaca313534bc54c7162d87c1d4`.

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from pathlib import Path; files=[*Path("training/laya_local").glob("*.py"),Path("training/laya_local/tests/test_typed_decisions.py")]; [compile(p.read_bytes(),str(p),"exec") for p in files]; print(f"in-memory compile passed: {len(files)} files")'
```

Result: `in-memory compile passed: 9 files`.

A repository-local inline Python audit used `hashlib`, `json`, `math`,
`collections.Counter`, and `pyarrow.parquet` to assert the hashes; reproduce
the line-17 and corrected hashes; decode and check row IDs/workflow/type counts;
check output metadata and pins; align every prediction to every source
question; and independently recompute argmax, correctness, totals, and NLL.
Its final output was:

```text
base choice [71, 200] score [65, 200] noul [58, 100] all [194, 500] nll 1.3373310093 max_probability_sum_error 1.0e-04
specialist choice [136, 200] score [148, 200] noul [84, 100] all [368, 500] nll 0.8008518995 max_probability_sum_error 1.0e-04
choice_delta 0.32500000000000007 additional_correct 65
deterministic artifact audit: PASS
```

An initial version of that audit stopped because it imposed an unneeded
`1e-9` probability-sum assertion on four-decimal returned probabilities. The
largest observed sum deviation was `1e-4`. The corrected audit reports that
deviation and verifies metrics from the returned values exactly; it does not
silently normalize or alter predictions.

## Boundary Confirmation

This verification wrote only
`artifacts/train_i2/B1_FINAL_VERIFICATION.md`. It did not edit implementation,
predictions, source data, predeclarations, or prior reports. It performed no
commit and no B2 work.
