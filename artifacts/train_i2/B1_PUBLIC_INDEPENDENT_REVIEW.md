# B1 Public Positive-Control Independent Review

Date: 2026-09-21

## Determination

**REPAIR.** The corrected public evaluation itself is valid and passes the
predeclared B1 gate, but the predeclaration contains a total-count error and
its recorded content hash does not verify under its stated method. B2 remains
prohibited until this documentation repair is made and rechecked.

## Verified evidence

- `artifacts/train_i1/source_test.parquet` hashes to
  `833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d`.
  Parquet inspection gives exactly 100 rows, all workflow
  `agent_trace_observability`, in ordered IDs `...000000` through `...000099`.
- Independent decoding of the public `questions` columns gives exactly 500
  questions: choice 200, score 200, noul 100. The 400 in
  `POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md`, line 42, is therefore wrong;
  it must say 500. The 400-case/2,000-decision model-benchmark wording in the
  historical review is a different aggregate claim and is not this subset.
- `positive_control.py` uses the frozen source hash, checks 100 rows and the
  workflow, preserves Parquet order/IDs, and evaluates both pinned checkpoints
  through the same `predict(state, questions)` path with `max_len=512` and
  `head_max_len=192`. This is a fair common evaluator. The common 512 context
  is explicitly disclosed as an evaluation choice, not reproduction of the
  specialist card's published 1024 context.
- The evaluator's probability semantics are type-correct: choice and score
  use returned `answer["probabilities"]`; noul maps `p_true` to
  `false=1-p_true, true=p_true`; hard prediction is probability argmax; NLL is
  `-log(max(returned gold probability, 1e-12))`. The shared schema confirms
  score keys are integer level strings and noul keys are false/true
  (`shared/typed_decisions/schema.py`).
- Independent recomputation from all 500 prediction records in
  `artifacts/train_i2/positive_control_public.json` matches every reported
  count, accuracy, and mean NLL:

  | model | choice | score | noul | all hard | mean gold NLL |
  |---|---:|---:|---:|---:|---:|
  | base | 71/200 = .355 | 65/200 = .325 | 58/100 = .580 | 194/500 = .388 | 1.3373310093 |
  | specialist | 136/200 = .680 | 148/200 = .740 | 84/100 = .840 | 368/500 = .736 | 0.8008518995 |

  The primary specialist gain is 65 additional choice answers, +0.325
  absolute accuracy, exceeding the frozen +8 / +0.05 threshold.
- Pins and context disclosure match the predeclaration and public review:
  dataset `LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2`,
  base `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`,
  specialist `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`,
  source `d113dca2512fb3eaca313534bc54c7162d87c1d4`.

## Chronology and hash review

The recorded full predeclaration SHA-256 `5bb9eeb19c033265a91b0f20c131aeca6307f0566e158cef2cde1d31bf26a7c2`
matches the current file. Repository mtimes do **not** support the claimed
order: predeclaration is 16:33:12, implementation 16:32:08, and output
16:34:01. Mtime is not cryptographic proof of model-load order, and here it
also leaves chronology inconsistent with the claim.
However, removing the self-reference line as described and hashing the
remaining current content produces `bdf907ff22833aaa183f9e0f548809c089938a3eca9390d1a121c200033f321c`,
not the recorded `a8d824...`. Thus the content-hash claim is not reproducible.

## Checks run

```text
.venv/bin/python -c '<repository-local Parquet/hash and JSON recomputation>'
.venv/bin/python -m py_compile training/laya_local/positive_control.py shared/typed_decisions/schema.py
.venv/bin/python -m unittest training.laya_local.tests.test_typed_decisions -v
```

The static compilation passed. The bounded suite ran six tests: five passed;
the one pre-existing `test_rejects_non_distribution` failure expects “sum to
one” while the validator emits “sum to 1”. No implementation was modified,
no training or model evaluation was run, and no commit was made.

## Exact repair required

In `POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md`, change line 42 from
`(400 questions total)` to `(500 questions total)`. Then recompute and record
the predeclaration content hash by removing the complete self-reference line
(line 17) before hashing; after this text correction the expected content hash
is `f94bd744fc667ad37b8891c33321ff5e1374cc9887440a990b53807d8c710e29`.
Do not change rows, pins, settings, threshold, predictions, or implementation.

## Evidence limits

**VERIFIED:** public row/type counts, evaluator semantics, fairness settings,
prediction-derived metrics, threshold arithmetic, and the 400-versus-500
document inconsistency. **INFERRED:** filesystem mtimes support chronology.
**UNKNOWN:** cryptographic proof of the exact model-load time; no new inference
was performed by this review.
