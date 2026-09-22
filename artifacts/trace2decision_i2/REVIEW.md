# Trace2Decision v2 Independent Review

Date: 2026-09-21

## Verdict

**NOT ACCEPTED pending targeted repairs.** One P1 finding makes the claimed
pre-policy context audit non-falsifiable, and one P2 finding reverses retained
conversation history. The emitted 1,544-row derivative otherwise passed the
bounded checks below. No implementation, test, report, or output file was
modified by this review.

## Findings

### P1 — 768/1024 audit is measured after 512-budget compression

`state_for()` first reduces protected fields and omits history until the state
fits the 415-token room left by the 512-token Laya envelope. Conversion then
calls `build_sequence()` on that already-compressed state for 768 and 1024.
Consequently the reported 768 and 1024 distributions are exactly the 512
distribution (`min=187`, `mean=473.7`, `max=512` in both cases), not
pre-policy measurements. This fails the checklist requirement for meaningful
pre-policy truncation statistics and hides how much source context the policy
discarded before construction. The report's `history_omitted` counts are useful
but do not repair this missing token audit.

**Repair brief:** add a side-effect-free pre-policy rendering/audit path that
constructs the full bounded-per-message/task/system/observation candidate and
all complete history candidates before the 512 selection loop. Tokenize that
candidate with the pinned tokenizer and report count distributions at 512,
768, and 1024 (including rows exceeding each budget, and separately the
post-policy/final distributions). Keep the model-facing 512 policy unchanged;
rerun the verifier and update the report/evidence hashes.

### P2 — retained history is reversed chronologically

The selection loop iterates candidates in reverse and appends each accepted
item to `history`; the final state therefore emits retained history newest to
oldest. The source candidates themselves are chronological. This is
deterministic, but it is not a conversational history order and can present a
later tool result before the assistant/tool messages that explain it.

**Repair brief:** select newest-fitting messages as now, then restore their
original source order before rendering (`history = sorted(selected,
key=source_index)` or equivalent carrying indexes). Preserve the exact
selection rule, record the resulting order policy, regenerate output, and
rerun tests/verifier/hash comparison.

## Falsification results

### VERIFIED passes

- **v1 preservation:** v1 output files remain present and their recorded
  hashes match the accepted evidence: train
  `d1441e6184d5ef843192f19ed10adaf6c2065fb4971ace478de50d4084240818`, dev
  `202338e20b5a02cc98938b13332da3484d34137c32a4ac5ecf8f20f72b297e8f`, test
  `616d97e562b04723e8566028cb1a9117868580c94680db31d1f270ac334aa5b4`.
  The v2 source is the same pinned source SHA
  `4e6b11f5b60976368f2a76c252808eb766af2ddcf5958d93f25160e45737d3aa`.
- **Rows, IDs, metadata, split:** existing rerun evidence reports 1,544 rows,
  1,544 unique IDs, no trajectory/task-group split crossings, and no shared
  validator failures. The output has required `id`, `trajectory_id`,
  `task_group`, and `source_step` metadata outside the native model-facing
  fields.
- **Leakage and labels:** direct source/output recomputation over all 1,544
  rows found zero label mismatches against the target tool mapping. The
  converter uses `messages[:-1]`; target arguments, target reasoning, and
  source metadata are not rendered by its state renderer. The existing canary
  and identifier checks passed. This verifies converter-level exclusion, not
  publisher authenticity or causal correctness.
- **Pinned Laya construction and 512 fit:** the converter imports the local
  `laya.common` builder, and `data/laya` is exactly commit
  `d113dca2512fb3eaca313534bc54c7162d87c1d4`; the pinned tokenizer revision is
  `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Existing evidence reports the
  expected empty envelope of 97 tokens, 415-token state room, marker sequence
  `[13, 23, 36, 47, 61, 77]`, and every final sequence at most 512. The
  equality assertion against the actual builder output makes final fit
  construction-based rather than an output-length claim.
- **Protected survival:** every retained state contains the explicit TASK,
  SYSTEM CONSTRAINTS, AVAILABLE ACTIONS, and LATEST RELEVANT OBSERVATION
  sections. Latest tool observation selection is deterministic by source
  index; no-tool prefixes use the latest non-system/non-task prefix message.
  Whether truncating a protected field loses task-critical content remains
  UNKNOWN.
- **Rerun hashes:** `rerun_verification.json` reports `pass=true`, equal
  canonical reports, and equal train/dev/test/sample hashes.

### INFERRED

- A latest tool result is a reasonable proxy for the latest relevant
  observation in this cumulative trace format; it is not independently
  authenticated.
- Normalized initial task grouping is a useful leakage boundary, but does not
  prove repository-family independence.

### UNKNOWN / residual risks

- Physical tool execution, teacher identity, and rights for embedded content
  were not authenticated.
- No downstream training, learnability, action optimality, or task success was
  established.
- The shared validator and verifier do not prove that every possible metadata
  value is absent from state; they check required identifiers and explicit
  canaries. Structural renderer inspection supports the exclusion claim.
- The full verifier was not rerun during this review because its CLI writes a
  rerun directory and evidence file, which would violate the instruction to
  write only this review. Existing checksum-backed rerun evidence was read,
  and read-only checks plus the bounded unit tests were run.

## Commands and evidence inspected

All commands were run from the repository root with offline/local assets:

```text
read AGENTS.md, research/27_iteration_2_acceptance.md
read artifacts/trace2decision_i1/{REVIEW.md,VERIFICATION.md,output/report.json}
read pipelines/trace2decision/{i2_convert.py,i2_verify.py,i2_tests.py}
read artifacts/trace2decision_i2/{IMPLEMENTATION_REPORT.md,audit.md,commands.log,output/report.json,rerun_verification.json}
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=pipelines/trace2decision .venv/bin/python -B -m unittest pipelines/trace2decision/i2_tests.py -v
git -C data/laya rev-parse HEAD && git -C data/laya status --short && git -C data/laya diff --stat
read-only Python scan of all v2 rows against source target labels and five deterministic source/output samples
git status --short
```

The unit command passed 2/2. The read-only scan reported 1,544 rows and zero
source-label mismatches. The final status check is recorded below after this
file was written.

## Review boundary check

Only `artifacts/trace2decision_i2/REVIEW.md` was written for this review; no
commit was made and no training or broad search was performed.
