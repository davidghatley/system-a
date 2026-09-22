# B1 Repair Cycle 2 Report

Date: 2026-09-21

## Work completed

- Preserved the as-run
  `artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md` exactly.
- Added `artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION_ERRATA.md`.
  It records the original full-file hash, corrects the descriptive total to
  500 (200 choice + 200 score + 100 noul), and records the reproducible
  post-run hash procedure and value without claiming pre-run cryptographic
  proof.
- Updated `artifacts/train_i2/PUBLIC_POSITIVE_CONTROL_REVIEW.md` to reference
  the erratum; results and evaluation claims were not changed.
- Corrected only the stale test assertion from `sum to one` to `sum to 1` in
  `training/laya_local/tests/test_typed_decisions.py`.

No inference or training was rerun. Predictions, rows, settings, pins,
threshold, evaluator implementation, and historical invalid derivative evidence
were not changed.

## Verification

Commands run from the repository root:

```text
sha256sum artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md
.venv/bin/python -m unittest discover -s training/laya_local/tests -v
.venv/bin/python -m py_compile training/laya_local/*.py training/laya_local/tests/test_typed_decisions.py
```

Results:

- The preserved predeclaration hashes to
  `5bb9eeb19c033265a91b0f20c131aeca6307f0566e158cef2cde1d31bf26a7c2`.
- The trainer test suite passes: 6 tests run, 6 passed.
- Trainer compilation passes.

## Evidence status and boundaries

**VERIFIED:** the frozen file's full hash; the corrected 500-question
composition; the post-run content-hash procedure/value documented in the
erratum; the review reference; the single assertion change; and the test and
compile results above.

**INFERRED:** none added by this repair.

**UNKNOWN:** cryptographic proof that the predeclaration was immutable before
model loading or prediction remains unavailable; this repair does not claim it.

Only `training/laya_local/**` and `artifacts/train_i2/**` were written for this
repair. No commit was created and no B2 work was performed.
