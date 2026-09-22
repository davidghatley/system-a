# B1 Independent Review

Date: 2026-09-21

## Determination

**INVALID — wrong instrument/data, not a valid B1 positive-control result.**
The recorded 26/150 versus 30/150 result is a reproducible comparison of the
two pinned checkpoints on the Iteration 1 `Trace2Decision` derivative. It is
not the required comparison on a fixed subset of the public
`LocalLLaMA/typed-decisions` dataset. Therefore the reported B1 BLOCK cannot
be accepted as the B1 decision, and it must not authorize B2. This is not a
compute or access block: the required public files are already present under
`artifacts/train_i1/`.

## Findings (severity ordered)

### 1. BLOCKER — evaluator input is a Trace2Decision derivative, not the public dataset
**VERIFIED.** `training/laya_local/positive_control.py` sets `DATA` to
`artifacts/trace2decision_i1/rerun2/test.jsonl`, records
`trace2decision_source=1371ed38...`, and verifies that file's SHA-256 as
`616d97e...`. The dataset card for that file identifies it as a deterministic
derivative of `11-47/glm-5.2-coding-and-debugging-traces`, with coding/tool-use
states and observed next actions.

The public source is instead
`LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2`.
The already-frozen `artifacts/train_i1/source_test.parquet` has SHA-256
`833ad136...`, 100 rows, and public columns including `workflow`, `state`,
`questions`, `gold`, and per-question convenience columns. Its first state is
an agent-observability case; the first B1 state is a coding bug report. They
are not alternate serializations of the same examples.

The B1 file's claimed workflow does not cure this. The Trace2Decision rows
have no `id` or `workflow`, exactly one `next_action` question, and 150
choice-only rows. Static inspection gives 150 choices and no `score`/`noul`
questions. The public `agent_trace_observability` test split has 100 cases,
five questions per case, and (in the frozen local copy) 200 `choice`, 100
`score`, and 100 `noul` questions. The 150-row B1 input is therefore neither a
fixed public subset nor the public workflow's test cases.

### 2. HIGH — the predeclaration freezes the wrong object and falsely excludes public workflows
**VERIFIED.** `POSITIVE_CONTROL_PREDECLARATION.md` says the Iteration 1
derivative has no other typed workflows and declares that derivative as the
entire B1 subset. The public dataset card at the pinned revision declares four
configs (`agent_trace_observability`, `customer_service`, `invoice_processing`,
`security_incidents`) plus `all`; each per-workflow test split has 100 cases,
and `all` has the aggregate benchmark. Thus the predeclaration is not a
predeclaration of the required public data. Omitting optional aggregate
workflows would be acceptable if explicitly bounded, but substituting a new
Trace2Decision derivative is not.

### 3. HIGH — public-schema all-question semantics are not implemented by this evaluator
**VERIFIED.** The loop obtains `answer["probabilities"]` and the native gold
probability, which is appropriate for NLL when keys are preserved. It only
sets a hard `pred` from `answer["choice"]` for `choice` questions; for
`score` and `noul` it sets `pred=None` and still counts that item in
`all_question_native_accuracy`. On the current choice-only derivative this
happens to equal choice accuracy. On the actual public schema it would count
all `score` and `noul` answers as incorrect, rather than using Laya's native
score/noul output semantics (`expected_score` and normalized
`false`/`true` probabilities). A corrected public run must either define a
type-correct all-question metric or predeclare choice-only evaluation and
report the other types separately; it must not reuse this accidental equality.

### 4. MEDIUM — checkpoint fairness is mostly sound but the context change must be disclosed
**VERIFIED / INFERRED.** Both models are loaded at the pinned revisions
`convaiinnovations/laya@1c5edc17...` and
`convaiinnovations/laya-typed-decisions@f9ab0b...`, then receive the same
`max_len=512`, `head_max_len=192`, record order, and `Agent.predict` loop.
This is a fair head-to-head construction. It is not the specialist card's
published 1024-token context (the base card is 512), so the corrected report
must state that common 512-token context is an evaluation choice and should
not present the result as reproduction of the card's 1024-token benchmark.

### 5. LOW — chronology evidence is plausible but not a sufficient correction
**VERIFIED.** File mtimes show the predeclaration at 16:11:35, the modified
positive-control implementation at 16:15:34, and predictions at 16:16:12.
This supports the claimed ordering, and the predeclaration explicitly says it
preceded model loading. However these artifacts are uncommitted and there is
no independent prediction-load timestamp or immutable predeclaration hash.
Chronology is therefore supported, not cryptographically proven. More
importantly, even perfect chronology cannot validate the wrong data source.

## Model-card and output comparison

**VERIFIED.** The pinned specialist card says it was fine-tuned on the public
benchmark's 1,200 training cases / 6,000 decisions and reports 400 test cases /
2,000 decisions, including an `agent-trace observability` workflow result.
The public card's schema is a request-shaped `{state, questions, gold}` record;
questions can be `noul`, `choice`, or `score`, and gold contains a declared
label plus a probability distribution. The B1 input instead contains only
Trace2Decision's six-action `next_action` choice ontology and one-hot gold.
Consequently the specialist's published domain match cannot be used to
justify this B1 input, and the B1 numbers cannot validate the card's
workflow-specific or aggregate claims.

**VERIFIED.** The native-label choice comparison itself is correctly formed
for the derivative: `gold[qid]["label"]` is used, the returned choice string
is compared to it, and both checkpoints use the same evaluator. The measured
specialist gain (+4/150, +2.667 points) is therefore a valid *derivative*
measurement, just not evidence for the requested B1 gate.

## Exact repair brief

1. Write a new predeclaration before loading either checkpoint or inspecting
   predictions. Mandatory primary subset: the 100 rows of the pinned public
   `agent_trace_observability/test` Parquet already recorded as
   `artifacts/train_i1/source_test.parquet` (SHA-256
   `833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d`).
   Preserve the public `id`, `workflow`, JSON-string/native fields, all five
   questions, and file order. Optionally predeclare the similarly pinned
   `all/test` aggregate (400 cases / 2,000 decisions) and report each workflow
   plus aggregate; do not search for or derive another dataset.
2. Use the same pinned base and specialist revisions and the same common
   context/config and inference order for both models. Parse public JSON
   strings without converting them into Trace2Decision records. Record source
   revision, source and selected-subset hashes, model revisions, context, and
   the predeclaration hash before inference.
3. Repair the evaluator's metric contract for the public types: primary
   native-label choice accuracy over the declared choice subset; type-correct
   native-label metrics for `noul` and `score` if all-question metrics are
   claimed; and gold NLL from the returned probability assigned to each native
   gold label. Never count `pred=None` as the public non-choice prediction.
   Report per-workflow/per-type and aggregate results. Keep the existing
   +5-point / +8-choice threshold and do not relabel the present derivative
   result as a pass.
4. Re-run only bounded static tests and the corrected inference evaluation.
   Do not start B2 unless the corrected public-data B1 satisfies the already
   frozen gate. Do not train or modify Trace2Decision data as a repair.

## Checks run

**VERIFIED:** no training, B2 work, network download, or implementation edit
was performed. I ran:

```text
.venv/bin/python -m unittest discover -s training/laya_local/tests -v
.venv/bin/python -m py_compile training/laya_local/positive_control.py training/laya_local/metrics.py training/laya_local/data.py
```

Compilation passed. The unit suite ran four tests and had one pre-existing
failure: `test_rejects_non_distribution` expects `"sum to one"`, while the
validator emits `"sum to 1"`; the other three tests passed. I also performed
bounded Parquet/JSON schema, row-count, workflow/type-count, and SHA-256
checks using the repository-local frozen artifacts. No implementation files
were changed.

## Evidence limits

- **UNKNOWN:** Whether either checkpoint would pass the required threshold on
  the corrected public workflow or aggregate; no such evaluation was run.
- **UNKNOWN:** Whether the specialist card's published 0.766 result is
  independently reproducible here; the card is a pinned external claim, not
  evidence for this invalid run.
- **INFERRED:** The derivative's lower accuracy gain may reflect domain and
  ontology mismatch, but this review does not attribute causality.
