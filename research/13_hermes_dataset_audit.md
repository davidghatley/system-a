# Hermes Dataset Audit For Experiment 001

Audit date: 2026-09-20

## Verdict

**RECOMMENDED: GO, with a protocol amendment.** Use only the pinned `kimi` config of `lambda/hermes-agent-reasoning-traces` as the replacement source for Experiment 001. It is viable for task-group-disjoint, observation-conditioned next-control imitation: the complete config has 7,646 trajectories, 87,667 reconstructable assistant targets, explicit tool definitions, 106,218 serialized real tool responses, a named Kimi-K2.5 teacher, one Hermes harness, and Apache-2.0 metadata.

**VERIFIED:** This corpus fixes the fatal Kimi K3 defect. Of 87,667 assistant targets, 80,021 (91.28%) have at least one earlier authentic serialized tool response. The pinned Kimi K3 artifact had zero identifiable authentic result messages.

**VERIFIED:** Hermes cannot retain the old claim that all demonstrations are successful or support outcome-aware filtering. It has no trajectory reward, verifier verdict, completion status, or terminal task-success field. Tool-local `exit_code`, `error`, `status`, and `success` values describe individual calls, not end-task success.

**INFERRED:** Hermes Kimi is the simplest currently audited coherent replacement, not literally the smallest public observation-bearing candidate. Exgentic advertises only 1,781 sessions, but it is cross-harness, reconstruction-heavy, and likewise lacks a documented task reward. Calling Hermes "smallest" is defensible only as "smallest audited single-teacher/single-harness source meeting Experiment 001's row and split needs," not by raw trajectory count.

## Scope And Pin

- **VERIFIED:** Repository: `lambda/hermes-agent-reasoning-traces`.
- **VERIFIED:** Revision: `b92885e4f0161d4b2536512710e004d4892cac6e`.
- **VERIFIED:** Audited artifact: `data/kimi/train.parquet`, 508,262,558 bytes, SHA-256 `d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02`.
- **VERIFIED:** Full Kimi config audited: 7,646/7,646 trajectories. The 599.6 MB GLM shard and redundant combined Kimi files were not downloaded.
- **VERIFIED:** Teacher: `moonshotai/Kimi-K2.5`; inference used vLLM's `kimi_k2` tool and reasoning parsers according to the dataset card.
- **VERIFIED:** Harness: NousResearch Hermes Agent through `hermes-agent-generator`, according to the card.
- **VERIFIED:** Repository metadata license: Apache-2.0.
- **UNKNOWN:** The audit did not independently resolve licenses of every cloned repository, website, or task payload represented inside trajectories. Apache-2.0 repository metadata does not prove that every embedded third-party excerpt is relicensable.

The profile is `artifacts/dataset_candidates/hermes/profile.json`; source provenance and checksum are in `artifacts/dataset_candidates/hermes/source_manifest.json`; commands are in `artifacts/dataset_candidates/hermes/commands.log`. The reusable audit command is:

```bash
.venv/bin/python scripts/audit_hermes_candidate.py
```

No training, model run, or task execution was performed.

## Schema And Volume

**VERIFIED:** The top-level schema is six fields: `id`, `conversations`, `tools`, `category`, `subcategory`, and `task`. `conversations` is a list of `{from, value}` strings using `system`, `human`, `gpt`, and `tool` roles. There is one source split, `train`.

**VERIFIED:** The measured 186,083 turns comprise 87,667 `gpt`, 81,569 `tool`, 9,201 `human`, and 7,646 `system` turns. There are 87,667 candidate assistant-turn targets, including 81,570 tool-calling targets and 6,097 response-only targets.

**VERIFIED:** Trajectories have 24.34 turns on average (p95 54, max 54), and 13.89 calls on average (p95 32, max 78). The nine source categories range from 36 Conversational trajectories to 2,010 Terminal & Coding trajectories.

## Authentic Observation Linkage

**VERIFIED:** Calls are serialized as `<tool_call>` payloads only in `gpt` messages; results are serialized as `<tool_response>` payloads only in `tool` messages. Restricting parsing by role prevents XML examples in system prompts from being counted as execution records.

**VERIFIED:** The audit found 106,220 call spans and 106,218 response spans. It matched 106,218 responses immediately after calls. Two call payloads fail JSON parsing; the two-count discrepancy is therefore localized rather than evidence of broad missing observations.

**VERIFIED:** Tool responses retain linkage/provenance structures such as `tool_call_id`, tool name, content/output, and commonly call-local outcome data. Measured result payloads include 46,993 `exit_code`, 50,602 `error`, 1,155 `status`, and 14,039 `success` key occurrences.

**INFERRED:** The dataset card's "real tool execution" claim is consistent with the measured role structure and payloads. The audit did not independently replay executions, so physical authenticity beyond the serialized provenance remains a publisher claim.

## Tools And Candidate Targets

**VERIFIED:** Six tool-definition sets expose 26 names. The stable minimal target is `RESPOND_OR_FINISH` plus exact membership for these published tool names: `clarify`, `delegate_task`, `execute_code`, `memory`, `patch`, `process`, `read_file`, `search_files`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `todo`, `write_file`, and the eleven `browser_*` actions listed in the profile.

**VERIFIED:** Tool-use is highly imbalanced: `terminal` has 46,923 parsed calls, `write_file` 21,628, `read_file` 11,051, and `search_files` 10,616. Browser actions form a separate 1,048-trajectory tool regime.

**VERIFIED:** Forty-two parsed calls use 29 names absent from the published taxonomy, generally malformed or hallucinated one-off names; two additional call payloads are unparseable. These 44 calls must be excluded before target freeze, not silently mapped to a valid family.

**RECOMMENDED:** Do not invent a coarse universal taxonomy. Preserve the 26 published Hermes names for the primary target, preserve parallel calls as multi-label membership, collapse duplicate same-name calls as in the existing protocol, and report support per name. A sensitivity analysis may merge the eleven browser actions into a `browser` family, but it must be frozen separately.

## Duplicates And Splits

**VERIFIED:** UUIDs are unique, and no complete reasoning-stripped trajectory is byte-identical. Task prompts are heavily reused: only 1,020 unique exact/normalized task texts exist among 7,646 trajectories; 764 task groups are duplicated, with 6,626 excess rows and a largest group of 122 trajectories.

**VERIFIED:** The source provides no held-out split. A row- or trajectory-ID-random split would leak repeated prompts and likely task templates.

**RECOMMENDED:** After invalid-call filtering, assign all trajectories sharing normalized task text to one partition. Deterministically hash that group key with seed `20260920` into train/calibration/test, then assert zero overlap by normalized task, complete trajectory hash, and rendered prefix hash. Stratify or verify category/tool support after grouping; do not break groups to repair balance.

**INFERRED:** With 1,020 task groups and more than 87,000 candidate targets before any optional length policy, a task-group 70/15/15 split should comfortably exceed the protocol minima of 1,500 train and 300 test rows. Exact post-filter partition counts remain **UNKNOWN** until the split hash and malformed-tag policy are frozen.

## Reasoning Exclusion And Lengths

**VERIFIED:** Complete `<think>...</think>` spans occur in 87,668 messages. Removing them before rendering produced zero residual complete spans in all reconstructed prefixes. The target assistant message is excluded by construction.

**UNKNOWN:** The current audit does not structurally classify malformed or unclosed `<think>` tags. This must be an exclusion check, not repaired by guessing boundaries.

**VERIFIED:** Even after complete reasoning-span removal, prefixes are long: mean 47,441 characters/3,948 whitespace tokens; p95 142,639/10,046; max 495,422/30,795. These are tokenizer-independent measurements. Exact Laya tokenizer lengths are **UNKNOWN** because no tokenizer is frozen in the current protocol.

**INFERRED:** The existing 512-token "task plus most recent observable history" policy will truncate nearly every row and may omit the action-relevant result if implemented naively. Tool definitions embedded in each system prompt also consume substantial context.

## Exact Minimum Preprocessing

**RECOMMENDED:** The minimum honest conversion for Experiment 001 is exactly:

1. Pin revision `b92885e4f0161d4b2536512710e004d4892cac6e` and verify the recorded Parquet SHA-256.
2. Use only config `kimi`; do not mix GLM teachers or the browser/non-browser tool regimes without explicit conditioning.
3. Parse actions only from `<tool_call>` in `gpt` messages and observations only from `<tool_response>` in `tool` messages; retain role, order, and full result payload.
4. Drop the 44 malformed/undefined call instances and any target turn containing one. Record dropped trajectory IDs and reasons.
5. Create one target per remaining `gpt` turn. Input is the task plus all prior observable messages; target is `RESPOND_OR_FINISH` or the multi-label set of published tool names. Exclude arguments from targets.
6. Remove complete `<think>...</think>` spans from every input message; reject rows with any residual, malformed, or unclosed reasoning tag. Assert the target message is absent.
7. Remove repeated tool definitions from historical turns, retain one canonical definition block for the active row-level toolset, and preserve the latest complete call/result exchange during truncation.
8. Exact-deduplicate rendered states within normalized-task groups, then split by normalized task text with seed `20260920`; assert no task, trajectory, or rendered-state overlap.
9. Freeze the Laya tokenizer and report pre/post-truncation token distributions plus the fraction retaining the latest complete observation. Do not proceed if truncation loses that observation.
10. Amend the hypothesis and protocol from "successful frontier-agent traces" to "recorded frontier-agent traces," set source outcome labels to unavailable, and remove any outcome-conditioned interpretation.

No additional framework, replay system, outcome model, or full GLM download is required for the pilot decision.

## Criteria Comparison

| Criterion | Kimi K3 pinned release | Hermes Kimi pinned config | Exgentic public description |
|---|---:|---:|---:|
| Named teacher/harness | **VERIFIED** | **VERIFIED** | **VERIFIED**, multiple |
| Authentic serialized observations | 0 | 106,218 response spans | **VERIFIED** reconstructable |
| Coherent single harness | Yes | Yes | No, cross-harness |
| Published tools | Four actions | 26 definitions | Per-session definitions |
| Terminal task outcome | Acceptance metadata, but no usable observations | None | Not documented |
| Source-held-out split | Yes | No | Not established here |
| License metadata | CC-BY-4.0 | Apache-2.0 | See separate audit; not reverified here |
| Raw trajectories | 582 | 7,646 | 1,781 sessions |
| Reconstruction burden | Low but invalid for observations | Low/moderate XML parsing | High OpenTelemetry reconstruction |

## Final Decision

**RECOMMENDED:** Replace Kimi K3 with the pinned Hermes Kimi config for the narrow offline Experiment 001 after applying the ten preprocessing steps and amending the success claim. Do not train under protocol `0.2.0` unchanged.

**RECOMMENDED:** Do not describe Hermes as globally the smallest observation-bearing dataset. Describe it as the simplest audited single-teacher/single-harness candidate that clears the observation and sample-size gates. Exgentic remains smaller by session count and should retain that factual qualifier.

**UNKNOWN:** Whether a 512-token Laya input can preserve enough task and recent observation content for useful prediction. That is the remaining data-representation gate; it requires tokenizer profiling, not training.
