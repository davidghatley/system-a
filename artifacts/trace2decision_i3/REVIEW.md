# Independent review: Trace2Decision i3

Date: 2026-09-22

## Decision

**CHANGES_REQUIRED.** The mixed-turn repair is accepted for the observed
tool-bearing population, but the no-call completion rule is not established
strongly enough to accept the dataset for training/evaluation.

## Verified

- `i3_tests.py`: **7/7 passed** (CPU, offline, four-thread limits). This
  covers homogeneous multi-call retention, mixed-category order invariance,
  malformed structures, no-call fixture behavior, target mutation, counts,
  and sample-stratum presence.
- Independent source audit: 1,821 rows; 76 mixed-category target turns. All
  permutations of every mixed row produced the same exclusion decision.
  The output has 1,479 retained and 342 excluded rows, with exclusions
  76 mixed, 265 unresolved-prefix, and 1 non-final no-call.
- The apparent reviewer discrepancy is reconciled: **65** is the number of
  mixed rows that were in retained v2 (51/3/11 train/dev/test); **76** is the
  number of mixed rows in the full 1,821-row source. The remaining 11 mixed
  rows were not v2-retained. Thus 76 converter exclusions is correct, not a
  count contradiction.
- `verification.json` reports pass, zero verifier failures, 1,479 stable
  IDs, 206 trajectories/groups, zero cross-split duplicate rendered states,
  and zero duplicate exact states. Retained v3 rows have no v2 assignment
  changes (`[]`). Output/rerun report and all five output hashes agree.
- Split counts are train/dev/test **1184/156/139** rows and **163/20/23**
  trajectories/groups. `sample_review.jsonl` covers every populated category
  in both train and dev (10 strata). This verifies structural stratification;
  it is not, by itself, an independent qualitative adjudication of each
  label.
- Actual rendered Laya sequences are **187–512 tokens** (mean 474.1), with
  zero over 512. The verifier checked exact prefix reconstruction, target-only
  strings/arguments, and provenance metadata boundary for 2,210 unique target
  strings. These are converter-level checks; semantic command duplication and
  broader future-derived leakage remain unproven.
- `checksums.sha256`, `provenance.json`, and `commands.log` identify the
  pinned source/tokenizer, offline CPU-only process, and deterministic rerun.

## Severity findings

### P0 — no-call completion evidence remains insufficient (CHANGES_REQUIRED)

The converter treats nonempty content plus
`assistant_step == assistant_steps` as completion. The source README verifies
that rows are cumulative prefixes and that the final assistant message is the
target, but it does not provide a finish reason or stream-completion field.
The source audit found 208 no-call targets: 207 final-step, 208 nonempty
content, and 16 with reasoning content. One non-final row is excluded, but
the 207 final/no-call rows are still accepted by an absence-based rule rather
than positive completion evidence. A final step can also be a partial or
reasoning-only event under the stated concern.

**Exact repair:** either (a) identify and validate a source-supported positive
completion marker for every retained no-call event, with an audit ledger for
final/non-final, reasoning-bearing, empty/malformed, and ambiguous cases; or
(b) exclude all no-call events and narrow the declared estimand to completed
tool-bearing turns. Add regression fixtures for reasoning-only, partial,
missing-finish, and positively completed no-call events, then rerun the
converter and verifier and record new hashes/counts before training.

### P1 — sample review and leakage evidence are structural, not full semantic review

The checks establish strict-prefix rendering and exact target-string/argument
non-leakage, and metadata is outside the rendered state. They do not establish
that semantically equivalent command templates or future-derived information
cannot occur in task/observation text. The grouping check is exact normalized
task/trajectory grouping only; near-duplicate repository/recipe families
across splits remain **UNKNOWN**. Do not claim broad family independence or
semantic leakage absence.

**Required follow-up:** inspect the 10 train/dev strata against source target
turns, add semantic command-template canaries, and audit near-duplicate task
families before making a generalization claim. This is not a reason to alter
the already verified hashes unless leakage is found.

## Acceptance scope

The whole-turn category rule, order invariance, explicit mixed exclusions,
unchanged retained split assignment, duplicate/task exact grouping, token
limit, metadata boundary, and deterministic counts/hashes are **VERIFIED**.
Completion semantics for `respond_or_finish`, broader family independence,
semantic leakage absence, and qualitative sample correctness are **UNKNOWN**
or only **INFERRED** from structure. No downstream predictions were inspected.

## Reproduction commands/evidence

Evidence: `artifacts/trace2decision_i3/{commands.log,checksums.sha256,provenance.json,verification.json}` and `artifacts/trace2decision_i3/output/{report.json,exclusions.jsonl,sample_review.jsonl}`.

Independent commands run:

```text
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=pipelines/trace2decision .venv/bin/python -B -m unittest pipelines/trace2decision/i3_tests.py -v
```

An offline Python audit independently recomputed source mixed/no-call counts,
all mixed-call permutations, v2-retained mixed rows, output row counts, and
SHA-256 hashes; it reported 76 mixed, 208 no-call, 65 v2-retained mixed, and
the hashes recorded above. No implementation or dataset file was modified.
