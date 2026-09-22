# Trace2Decision I2 Final Verification

Date: 2026-09-21

## Status

**ACCEPTED.** The repaired Trace2Decision I2 derivative passes the requested
deterministic tests, verifier rerun, metadata/split/leakage checks, chronology
check, protected-section survival check, and context-budget checks. No
implementation was modified and no training was run.

## Inputs and revisions inspected

- Source: `artifacts/trace2decision_i1/source/traces.jsonl`, dataset
  `11-47/glm-5.2-coding-and-debugging-traces`, revision
  `1371ed38f8890d0520a53bc7ad850308eb4d7a22`, SHA-256
  `4e6b11f5b60976368f2a76c252808eb766af2ddcf5958d93f25160e45737d3aa`.
- I2 output: `artifacts/trace2decision_i2/output/`.
- Laya source: local commit `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
- Tokenizer: local snapshot `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Read before verification: `research/27_iteration_2_acceptance.md`,
  `REVIEW.md`, `REPAIR_1.md`, `IMPLEMENTATION_REPORT.md`, `audit.md`,
  `dataset_card.md`, `pipelines/trace2decision/i2_convert.py`,
  `i2_tests.py`, `i2_verify.py`, output report, and prior rerun evidence.

Initial worktree inspection showed unrelated pre-existing modifications and
untracked workstream artifacts. They were preserved. This verification writes
only this report.

## Commands and results

All commands ran from the repository root with local/offline assets:

```text
git status --short && git log --oneline -10
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=pipelines/trace2decision .venv/bin/python -B -m unittest pipelines/trace2decision/i2_tests.py -v
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=pipelines/trace2decision .venv/bin/python -B pipelines/trace2decision/i2_verify.py --source artifacts/trace2decision_i1/source/traces.jsonl --expected-dir artifacts/trace2decision_i2/output --rerun-dir artifacts/trace2decision_i2/rerun --evidence artifacts/trace2decision_i2/rerun_verification.json
```

The first verifier invocation exceeded the 120-second tool timeout while
processing the deterministic rerun; it did not produce a result. The same
verifier command was rerun with a 300-second timeout and completed successfully
(one non-fatal tokenizer warning for an unselected 8241-token candidate was
emitted).

**VERIFIED:** all 4/4 tests passed:
`test_every_output_row_has_metadata_outside_state`,
`test_newest_fit_is_rendered_in_source_order`,
`test_pinned_empty_room_and_markers`, and
`test_pre_policy_audit_is_distinct_and_has_threshold_counts`.

**VERIFIED:** verifier `pass=true`, expected and rerun validation both valid,
both with 1,544 rows and 1,544 stable IDs; no verifier failures; canonical
reports equal; and all four rerun file hashes equal.

## Independent checks

The following read-only Python check independently loaded every I2 train/dev/test
row, called `shared.typed_decisions.validate_record`, computed hashes, and
checked metadata, state text, split membership, chronology, and section
survival.

**VERIFIED counts:**

- Retained rows: **1,544** (train 1,235; dev 159; test 150).
- Stable metadata IDs: **1,544 unique**, duplicate IDs **0**.
- Shared validation failures: **0**.
- Trajectory split crossings: **0**; task-group split crossings: **0**.
- Leakage indicators: ID **0**, trajectory ID **0**, task-group ID **0**,
  `TARGET_CANARY_9a31` **0**.
- Rows with retained history: **415**; chronological-order violations: **0**.
- TASK survival: **1,544/1,544**; SYSTEM CONSTRAINTS: **1,544/1,544**;
  AVAILABLE ACTIONS: **1,544/1,544**; LATEST RELEVANT OBSERVATION:
  **1,544/1,544**.
- Final post-policy sequence lengths: min **187**, mean **473.7**, max
  **512**; rows over selected 512-token limit: **0**.

## Context audit

**VERIFIED pre-policy full-candidate measurements** (before optional history
selection): min **187**, mean **4,244.48**, max **16,656** tokens.

| Threshold | Rows exceeding | Percentage |
|---:|---:|---:|
| 512 | 1,504 | 97.41% |
| 768 | 1,402 | 90.80% |
| 1024 | 1,285 | 83.23% |

**VERIFIED post-policy distributions:** for each of 512, 768, and 1024
descriptive builder measurements, min **187**, mean **473.7**, max **512**,
with zero rows exceeding 512, 768, or 1024. The model-facing policy remains
the 512-token construction; 768/1024 are not training or model runs.

The pinned empty Laya envelope is **97** tokens, leaving **415** state tokens;
the marker sequence is `[13, 23, 36, 47, 61, 77]`. The four protected sections
survived every retained state as shown above.

## v1 preservation and I2 hashes

**VERIFIED:** independently recomputed v1 output hashes remain:

| File | SHA-256 |
|---|---|
| train.jsonl | `d1441e6184d5ef843192f19ed10adaf6c2065fb4971ace478de50d4084240818` |
| dev.jsonl | `202338e20b5a02cc98938b13332da3484d34137c32a4ac5ecf8f20f72b297e8f` |
| test.jsonl | `616d97e562b04723e8566028cb1a9117868580c94680db31d1f270ac334aa5b4` |
| samples.jsonl | `7f4bf7528993c68dd34a3f68dccb9a5e7afed07a97a96d146f1122ad6de97af1` |

I2 output hashes, also confirmed equal by the deterministic verifier, are:

- train `79255bd82a26c663845ff8c82a97e3d912b0e19a7231cc8cec2e480e837b5e27`
- dev `eed09d564f1ae17cabb2ef47bec8ae94aaa08c41ccd6da692149f14b0b47a540`
- test `01272aa8bb7f1b95b2c82092ae55ebcb037b3e960e70141d7ab1538c34abd9ba`
- samples `ae359f131b984697f850f1972b3b8be7e45be0d714f06a306b8df1d1ceb64dea`

## Limitations

**INFERRED:** latest tool output is used as the latest relevant observation in
this cumulative trace format; it is not independently authenticated.

**UNKNOWN:** physical tool execution, teacher identity, publisher rights, and
causal task success were not authenticated. The checks establish converter and
contract properties, not action optimality, downstream learnability, or
training performance. Protected text is bounded head/tail content, so whether
an omitted middle detail is task-critical remains unknown. The 768/1024 values
are descriptive measurements only.
