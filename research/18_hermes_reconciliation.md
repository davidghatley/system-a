# Hermes Audit Reconciliation And Minimum Remediation Experiment

Reconciliation date: 2026-09-20

## Decision

**RECOMMENDED: GO for one bounded, local-only preprocessing prototype. NO-GO for main training, protocol refreeze, or official test scoring.**

The original conditional-GO was directionally right that Hermes contains usable serialized observations, but premature about conversion readiness. The adversarial NO-GO was correct for training. Existing data nevertheless supports a narrower and coherent preprocessing experiment: use only the largest exact six-tool regime, require a complete immediately adjacent prior call/result exchange, reject all trajectories with malformed reasoning structure, reject invalid labels and over-budget states, then exact-deduplicate tokenized states and split deterministic task-template proxy groups.

This prototype retains **28,220 unique 512-token-representable targets from 5,300 trajectories and 640 task-template proxy groups**. It therefore has ample dataset-internal volume for testing preprocessing mechanics. It does not establish repository/site-disjoint generalization, successful trajectories, row-attested teacher or harness identity, authenticated physical execution, or embedded-content redistribution rights.

Machine-readable measurements are in `artifacts/dataset_candidates/hermes/reconciled_profile.json`. The deterministic local-only measurement is:

```bash
.venv/bin/python scripts/reconcile_hermes_candidate.py
```

The recorded invocation is in `artifacts/dataset_candidates/hermes/reconciliation_commands.log`. No data was downloaded, no task was replayed, no model was run, and no training occurred.

## Exact Prototype Subset

- **VERIFIED:** Source is the pinned Kimi Parquet at revision `b92885e4f0161d4b2536512710e004d4892cac6e`, SHA-256 `d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02`.
- **VERIFIED:** Select rows whose complete row-level tool definition set is exactly `patch`, `process`, `read_file`, `search_files`, `terminal`, and `write_file`. This is one invariant toolset containing 6,071 source trajectories. Categories are not used as a selection or model-input field.
- **VERIFIED:** Create a candidate only at an assistant turn immediately preceded by an assistant `<tool_call>` message and its adjacent tool `<tool_response>` message. Require equal nonzero payload counts, valid JSON, and exact ordered call/response name agreement. Retain the complete pair and response IDs as serialized. There were zero linkage exceptions among candidate pairs in this subset.
- **VERIFIED:** Structurally remove canonical, non-nested `<think>...</think>` spans. Conservatively reject the entire trajectory if any message has unmatched or nested tags. This rejects 16 trajectories containing 18 malformed messages; no boundary is guessed.
- **VERIFIED:** Parse the target only after reasoning removal. Exclude a target if any call is malformed or outside the six active names; 31 targets are excluded. Parser failure never becomes `RESPOND_OR_FINISH`.
- **VERIFIED:** Render only task text, the six fixed availability names, the complete latest assistant call, the complete latest tool response, and six frozen binary question strings. The renderer has no target-content argument and includes no top-level ID, category, subcategory, teacher, split, or provenance metadata.
- **VERIFIED:** Tokenize with the pinned local Laya tokenizer revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, including special tokens and question text. Do not truncate payloads. Reject states longer than 512 tokens.
- **VERIFIED:** Exact-deduplicate the resulting token-ID sequence globally, retaining the deterministic first `(trajectory_id, message_index)` after sorting. This removes 637 duplicate states.
- **VERIFIED:** Group by lowercased whitespace-normalized task after replacing numeric and quoted literals. Assign the complete group using SHA-256 of UTF-8 bytes `20260920\n<group>`, first 64 bits modulo 10,000: train `[0,7000)`, development `[7000,8000)`, calibration `[8000,9000)`, test `[9000,10000)`.

The last group key is explicitly a **template proxy**, not an authenticated repository, website, environment, or semantic task-family identifier.

## Measured Funnel

| Stage | Result |
|---|---:|
| Exact six-tool source trajectories | 6,071 |
| Observation-conditioned candidates after conservative trajectory reasoning rejection | 63,239 |
| Invalid/out-of-toolset target-call exclusions | 31 |
| Prior call/result linkage exceptions | 0 |
| Valid states rejected above 512 tokens | 34,351 |
| Representable states before exact deduplication | 28,857 |
| Exact tokenized-state duplicates removed | 637 |
| Final targets | 28,220 |
| Final trajectories | 5,300 |
| Final task-template proxy groups | 640 |

**VERIFIED:** The retained token lengths include special tokens and all six questions: minimum 174, median 283, mean 309.32, p95 474, and maximum 512. The prototype obtains representability by exclusion, not lossy truncation. Of the 63,208 candidates remaining after invalid-target exclusion, 34,351 (54.35%) are over budget. This materially changes the eligible population and must be treated as a selection effect.

**VERIFIED:** Exact-state deduplication removes 2.21% of representable states. Final state hashes are unique; no task-template proxy group or trajectory crosses a split.

| Split | Groups | Trajectories | Targets |
|---|---:|---:|---:|
| Train | 453 | 3,825 | 20,304 |
| Development | 63 | 515 | 2,744 |
| Calibration | 55 | 454 | 2,481 |
| Test | 69 | 506 | 2,691 |

**VERIFIED:** Every tool has positives in every partition. Positive-trajectory support in development/calibration/test respectively is: `patch` 43/35/38, `process` 12/16/16, `read_file` 102/113/107, `search_files` 164/141/134, `terminal` 455/398/420, and `write_file` 254/213/281. `process` is sparse in held-out partitions and cannot support strong per-family inferential claims without a separately frozen minimum-support rule.

## Coherent Typed Target

The prototype uses six `noul` questions, one for each fixed available tool, with exact options `no | yes`. The primary label is the resulting six-bit set:

- Empty set means `RESPOND_OR_FINISH`.
- A nonempty set means the exact set of tool names called by the recorded assistant turn.
- Same-name multiplicity is collapsed and must be reported separately if later analyzed.
- For mixed non-reasoning text plus calls, the tool set is the label; the text is not assigned a second semantic class.
- Unknown or malformed calls are excluded and never interpreted as an empty set.
- `OTHER_TOOL` is reserved but unreachable because both row availability and eligible labels are fixed to the same six names. If the subset changes, this decision must be reopened.
- Control mode is derived from set emptiness; there is no redundant independently predicted mode question.
- A future scorer must decode the highest-scoring member of the 64 valid six-bit bundles. It must not emit a separate contradictory mode.

This resolves target coherence for the bounded prototype. It does not freeze calibration, confidence construction, thresholds, baselines, or the full revised Experiment 001 protocol.

## Nine Mandatory Conditions

1. **Protocol claim and manifests: PARTIAL.** The defensible claim is publisher-attributed, within-corpus recorded-action imitation, not successful-trace imitation. The script freezes a deterministic four-way allocation algorithm and reports counts, but does not emit training manifests or amend protocol `0.2.0`. Prototype work may proceed; training may not.
2. **Typed bundle and availability: PASS FOR THIS SUBSET.** Six invariant available tools, six binary questions, empty-set mode semantics, malformed-call exclusion, mixed-text semantics, and a coherent 64-bundle decoder are specified. A global 26-tool taxonomy is rejected.
3. **Structural reasoning and canaries: PASS FOR PROFILING, PARTIAL FOR PRODUCTION.** Structural removal rejects every unresolved malformed selected trajectory and the renderer structurally accepts no target content. A production converter must additionally emit rows and run per-row target/reasoning sentinel mutation tests before writing manifests.
4. **Call/response linkage and exclusion ledger: PASS FOR AGGREGATES, PARTIAL FOR PRODUCTION.** Every retained prior pair has equal count and ordered name agreement, and response payloads/IDs remain intact. Calls have no call-side IDs and execution is not authenticated. The profiler reports exclusion counts but does not yet publish the required trajectory/turn ledger.
5. **Deduplication and task-family isolation: PARTIAL.** Exact tokenized states are globally unique and deterministic task-template proxy groups do not cross splits. Repository/site/environment identifiers are absent, so stronger family isolation cannot be verified and is not invented. This blocks a task-family-disjoint training claim.
6. **Tokenizer, renderer, and 512-token representation: PASS FOR RETAINED ROWS.** The exact local tokenizer, question strings, special-token accounting, renderer, and no-truncation policy are frozen in the script. Every retained state preserves task, availability, and a complete latest call/result pair at no more than 512 tokens. The 54.35% candidate exclusion requires bias tables before training.
7. **Tool/system regimes: PASS FOR THIS SUBSET.** The prototype uses only the largest exact six-tool regime and excludes all browser and 15-tool rows. Regime availability therefore cannot serve as a between-regime shortcut within the candidate.
8. **Provenance and license: DOCUMENTED BUT UNRESOLVED.** Kimi-K2.5, Hermes Agent, and real execution remain publisher-level claims. Dataset-internal adjacency and payload consistency are verified; physical execution is not. Embedded-content licenses and rights remain unknown. This is acceptable for private profiling only, not an affirmative provenance finding or redistribution clearance.
9. **Post-filter volume, balance, and precision: PARTIAL.** Eligible rows, trajectories, groups, split balance, duplicate rate, token lengths, and family support are recomputed. The volume is sufficient for a bounded converter test and exceeds old row minima. No blinded power/precision analysis or prespecified independent-trajectory/per-family gate exists, and `process` held-out support is low. Main training remains blocked.

## Evidence Boundary

**VERIFIED dataset-internal facts:** exact source bytes; row tool definitions; structural tags; parsed call/result payloads; immediate adjacency; ordered name agreement; response IDs; token counts under the pinned tokenizer; state hashes; deterministic proxy-group allocation; retained counts and label support.

**INFERRED:** Detailed outputs, perfect retained adjacency, and ordered name agreement make these observations authentic enough for a narrow study of the dataset's recorded interaction serialization. They do not authenticate external physical execution.

**UNKNOWN:** physical tool execution; fidelity of payloads to external environments; row-level teacher and harness identity; terminal task success; repository/site/environment identity; semantic task-family independence; embedded-content licenses, privacy status, and redistribution rights; bias induced by the 512-token exclusion; statistical power for frozen trajectory-level claims.

## Smallest Remediation Experiment

Run only the deterministic preprocessing prototype represented by `scripts/reconcile_hermes_candidate.py`, then extend it minimally to emit an exclusion ledger and split manifests without altering the subset, renderer, labels, tokenizer, budget, or split algorithm. Before writing any model-ready data, require these assertions:

1. Source checksum equals the recorded SHA-256.
2. Every row has exactly the six frozen tools.
3. Every retained prefix has structurally balanced reasoning removal and zero residual reasoning markers.
4. Every retained prior call/result pair is complete, adjacent, parseable, count-equal, and ordered-name-equal.
5. Every target label is a subset of the six names; malformed input is excluded, not relabeled.
6. Mutating every target scalar and excluded metadata scalar leaves rendered token IDs unchanged.
7. Every rendered state is at most 512 tokens and contains the complete task, availability line, call, result, and six questions.
8. Tokenized state hashes are globally unique and task-template proxy groups and trajectories have zero split overlap.
9. Exclusion and support tables are emitted by split, tool, source category, trajectory, and proxy group.

This experiment is preprocessing only. It should produce no model outputs and should not access an official test prediction path.

## Stop Gates

- **STOP** if any assertion above fails or any generated row contains target/reasoning canaries, incomplete payloads, an unavailable label, or more than 512 tokens.
- **STOP before training** unless a revised protocol explicitly narrows generalization to deterministic template-proxy-disjoint within-corpus imitation, or auditable repository/site/task-family identities are established. Do not call the current split semantically task-family-disjoint.
- **STOP before training** until minimum independent trajectory and positive-trajectory support per family, the treatment of sparse `process`, power/precision, trajectory-macro estimands, baselines, calibration, and threshold selection are frozen.
- **STOP before training** if 512-token exclusions show material split-, group-, category-, or label-dependent distortion under prespecified tolerances.
- **STOP before redistribution or derivative release** until an embedded-content provenance/license policy is explicitly accepted.
- **STOP any claim** that observations are externally authenticated, trajectories are successful, or teacher/harness identity is row-attested.

## Reconciled Verdict

Hermes should not be rejected as unusable, because one coherent regime retains thousands of independent trajectories and tens of thousands of exact, observation-conditioned, fully representable target states after conservative filtering. It should also not replace Kimi K3 in Experiment 001 yet, because the missing identity/provenance fields and the large representation exclusion cannot be repaired by assertion.

The smallest evidence-based next step is therefore **GO for the frozen six-tool preprocessing prototype and its ledgers; NO-GO for all training and official evaluation**. If the stop gates cannot be satisfied without expanding scope, changing the estimand after inspection, or inventing task-family/provenance trust, reject Hermes for Experiment 001.
