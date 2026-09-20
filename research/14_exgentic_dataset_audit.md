# Exgentic Dataset Audit For Experiment 001

Audit date: 2026-09-20

## Decision

**RECOMMENDED - NO-GO as the sole, drop-in replacement for Kimi K3.** Exgentic is genuinely observation-bearing and repairs Kimi's decisive defect, but it does not preserve task IDs, task outcomes, rewards, or benchmark scores. Session-disjoint splitting is possible; task/template-disjoint splitting and success-conditioned interpretation are not. Its five harnesses and at least 1,017 sampled raw tool names also change Experiment 001 from a coherent single-harness pilot into a substantial normalization experiment.

**RECOMMENDED - CONDITIONAL GO only after changing the protocol.** Exgentic can support a smaller benchmark-contaminated, session-held-out study of observation-conditioned next-action imitation. It should not support claims about successful control, task generalization, or evaluation on any of its six source benchmarks. For the current objective, Hermes remains the lower-burden primary replacement because the published Hermes representation is already turn-oriented, has 14,701 trajectories and 174,550 calls, and uses one agent framework, although Hermes also lacks terminal rewards and still needs its own pinned raw audit.

## Pin, Scope, And Reproduction

- **VERIFIED:** Dataset `Exgentic/agent-llm-traces`, Hub commit `70036b93a04e61b0ea2706a68b962f4f26774587`.
- **VERIFIED:** Hub metadata and the pinned README declare CDLA-Permissive-2.0. Applicability of benchmark and underlying repository terms is **UNKNOWN**; the dataset-level license does not erase upstream restrictions.
- **VERIFIED:** The source has 39 Parquet shards, 1,781 advertised sessions, and 983,592,848 Parquet bytes.
- **VERIFIED:** The bounded sample is 10 shards plus README, 231,491,639 bytes (23.5% of Parquet bytes), 457 sessions, and 9,249 spans. It covers every advertised benchmark and harness and several models in each major domain where available. Sampling stopped once viability relative to Kimi/Hermes was decidable.
- **VERIFIED:** File sizes and SHA-256 values are in `artifacts/dataset_candidates/exgentic/source_manifest.json`; measurements are in `artifacts/dataset_candidates/exgentic/profile.json`; commands are in `artifacts/dataset_candidates/exgentic/commands.log`.
- **VERIFIED:** Rerunning the profiler without downloading produced profile SHA-256 `9e277f7de824f7a42d57eae10e7ef6f86345bfd33358e2e84eb0b2bb9054ccf6` and manifest SHA-256 `6a5d1f0438c536d4b3c4b8fc627c88230d3befddec99fd582c2e4a4945dba8db`.

```bash
.venv/bin/python scripts/audit_exgentic_dataset.py --download
.venv/bin/python scripts/audit_exgentic_dataset.py
```

## Population And Provenance

- **VERIFIED:** The pinned card advertises six benchmarks: AppWorld 406 sessions, BrowseCompPlus 133, SWE-bench 391, tau2-airline 196, tau2-retail 469, and tau2-telecom 186. These sum to 1,781.
- **VERIFIED:** The card advertises five harnesses: `claude_code`, `openai_solo`, `smolagents_code`, `tool_calling`, and `tool_calling_with_shortlisting`. All five occur in the sample.
- **VERIFIED:** All six benchmarks occur in the sample. Coverage is not balanced: the selected shards were chosen for structural diversity, not population estimates.
- **VERIFIED:** The card names DeepSeek-V3.2, Kimi-K2.5, Claude Opus 4.5, Gemini 3 Pro Preview, and GPT-5.2. Their card counts sum to only 1,217, not 1,781.
- **VERIFIED:** Sample rows also identify GPT-4.1 in tau2 traces. Therefore the card's “5 models” statement is incomplete, and exact full-corpus teacher/model counts are **UNKNOWN** without scanning all shards.
- **INFERRED:** `models` can list both the primary model and another model used in a session. It is not safe to equate list entries with one teacher per session without per-span attribution.

## Observation And Action Reconstruction

- **VERIFIED:** Serialized message parts in the sample include `text`, `thinking`, `tool_call`, and `tool_call_response`. Tool responses contain authentic result-channel provenance rather than assistant-authored summaries.
- **VERIFIED:** The sample contains 10,043 output tool-call parts and cumulative inputs containing 149,151 tool-response parts. The latter is not a unique-transition count because each later request repeats history.
- **VERIFIED:** Of 6,748 action-bearing spans having a later LLM span, 6,226 (92.27%) have every non-empty output call ID represented by a response in the immediately following span's input. At call level, 8,831 of 9,702 IDs (91.02%) match.
- **VERIFIED:** The remaining 522 spans/871 call IDs are not deterministically reconstructable by the simple adjacent-span rule. They may reflect intervening calls, serialization differences, failed calls, parallelism, or missing results; this bounded audit does not assign causes.
- **VERIFIED:** 504 spans have non-empty input but unparseable or absent output-message JSON under the common parser. They must not become targets without harness-specific review.
- **INFERRED:** Action/result reconstruction is practical at useful scale if extraction keeps only ID-matched transitions and records drop reasons. Unlike Kimi, no observation-viability stop is triggered.
- **UNKNOWN:** Full-corpus reconstructability rates and whether each of the five harnesses individually clears an acceptable threshold.

## Tools And Candidate Actions

- **VERIFIED:** Tool definitions are present in 7,988 of 9,249 sampled spans. Among spans with definitions, the count ranges from 1 to 487 (median 30).
- **VERIFIED:** The sample exposes 1,017 exact tool names. Names include ordinary coding tools, benchmark APIs, MCP-prefixed aliases, planning/delegation tools, terminal control, and explicit `finish` operations.
- **INFERRED:** Tool definitions are candidate-action sets only for function-calling harnesses. They do not prove availability, permission, or semantic equivalence, and hundreds of definitions make flat classification unsuitable.
- **RECOMMENDED:** Preserve exact name, arguments, definition, harness, and benchmark. Add only this candidate typed target taxonomy:

| Type | Membership rule | Notes |
|---|---|---|
| `CALL_ENV_READ` | side-effect-free lookup/search/show/read/status tools | Derive from schema/name; uncertain cases stay raw-only |
| `CALL_ENV_WRITE` | create/update/delete/send/pay/execute or other state-changing APIs | Preserve an irreversible/sensitive flag |
| `CALL_COMPUTE` | shell, code execution, notebook, test, or process-control call | Do not merge command arguments |
| `CALL_FILE` | file read/write/edit/glob/grep operations | Preserve operation subtype |
| `CALL_WEB` | browser/search/fetch/navigation operations | Keep site/domain where exposed |
| `CALL_PLAN_OR_DELEGATE` | todo, plan, task/subagent, scheduling, memory operations | Often harness-internal rather than environment control |
| `FINISH_OR_ESCALATE` | finish, transfer-to-human, explicit refusal/abstention | Separate success claims from mere stopping |
| `RESPOND_TEXT` | assistant output with no executable call | Keep multi-call bundles distinct from text-only turns |

**RECOMMENDED:** This taxonomy is proposed, not frozen. Never train only the coarse type; use it as an auxiliary label while retaining the executable raw target.

## Outcomes, Duplicates, And Splits

- **VERIFIED:** The eight-column schema has no task ID, prompt ID, reward, grader score, pass/fail, patch verdict, or terminal outcome. Span status is an API-operation status, not benchmark success.
- **UNKNOWN:** Which sessions succeeded, whether a final `finish` was correct, and whether source selection retained failures.
- **VERIFIED:** No duplicate `session_id` occurs in the 457-session sample. There are 349 excess exact serialized outputs, but these include common/empty responses and are not evidence of duplicate trajectories.
- **UNKNOWN:** Exact task, template, repository/issue, and full-trajectory duplication cannot be measured from the published top-level fields alone.
- **VERIFIED:** Group splitting by `session_id` is straightforward. Stratification by benchmark/harness/model is feasible.
- **UNKNOWN:** Task-disjoint, template-disjoint, repository-disjoint, and source-entity-disjoint splitting is impossible to verify from explicit metadata. Prompt-text fingerprinting may approximate it but cannot establish it.
- **RECOMMENDED:** If used, split whole sessions first and group near-duplicate initial task text before producing decision rows. Report this as session/text-group held-out, not task-disjoint, unless authoritative benchmark IDs are recovered.

## Reasoning, Context, And Contamination

- **VERIFIED:** The sample contains 7,831 `thinking` parts. Reasoning exposure varies by provider and harness and is not necessary to reconstruct calls/results.
- **RECOMMENDED:** Exclude all `thinking` parts recursively from model inputs and targets. Retaining them creates provider-style shortcuts and chain-of-thought governance concerns.
- **VERIFIED:** Serialized input length in the sample is median 24,519 characters, p95 313,437, and maximum 673,140. Output length is median 372, p95 1,793, and maximum 12,638 characters. These are characters, not tokenizer counts.
- **INFERRED:** Naively materializing every cumulative prefix would multiply storage and over-weight long sessions. Delta extraction with bounded observation windows is mandatory for a roughly 400M model.
- **VERIFIED:** Every session comes from AppWorld, BrowseCompPlus, SWE-bench, or tau2. These are evaluation suites, not incidental production tasks.
- **RECOMMENDED:** Treat all six benchmarks as contaminated after training. Do not evaluate Experiment 001, tune thresholds, or make headline generalization claims on overlapping benchmark tasks or repositories.

## Exact Preprocessing Burden

**RECOMMENDED:** A valid Exgentic conversion requires all of the following; omitting any item changes the claim or leaks information.

1. Pin all used shard bytes and record hashes, source revision, dataset license, and upstream benchmark terms.
2. Validate JSON separately for input messages, output messages, and tool definitions; quarantine malformed targets and retain a drop-reason table.
3. Group by `session_id`; order spans deterministically by timestamp plus span ID; reject ambiguous orderings rather than infer causality.
4. Select LLM chat spans only. Do not interpret OpenTelemetry status as task reward.
5. Parse output parts into text and one-or-more raw calls. Preserve multi-call bundles, call IDs, exact arguments, model, harness, benchmark, and tool definitions.
6. Locate each call result by ID in a later input, preferring the immediate next LLM input. Keep only fully matched bundles for the primary dataset; separately flag partial, terminal, failed, and unmatched bundles.
7. Deduplicate cumulative history into one transition per unique session/call ID. Do not count repeated historical responses as new observations.
8. Remove `thinking` recursively. Build state from observable text/call/result history only and assert that target outputs are absent.
9. Normalize raw tools to the proposed typed taxonomy while retaining raw executable labels. Maintain harness-specific adapters for function calls, code agents, Claude Code, and shortlist semantics.
10. Recover initial task text, normalize it, near-duplicate-group it, and split groups before row expansion. Mark task identity as inferred unless authoritative IDs are joined.
11. Apply a tokenizer-pinned context policy: retain task, tool candidates needed at the decision, recent matched observations, and bounded history; report truncation by benchmark/harness/model.
12. Produce per-harness counts, reconstruction/drop rates, target frequencies, duplicate groups, context quantiles, and split-overlap checks before freezing.

**INFERRED:** This is materially more work than Hermes conversion: at least five harness adapters, shortlist/candidate handling, cumulative-history deduplication, malformed-output quarantine, 1,017-name raw taxonomy handling, and inferred task grouping. It is not a simple replacement of the Kimi loader.

## Comparative Verdict

| Criterion | Kimi K3 pinned release | Exgentic pinned release | Hermes published candidate |
|---|---|---|---|
| Authentic observations | **VERIFIED:** none | **VERIFIED:** present and mostly adjacent-ID reconstructable in sample | **INFERRED from prior card review:** explicit tool responses; raw pin still needed |
| Coherent harness | One | Five | One framework |
| Scale | 582 trajectories/3,956 rows | 1,781 advertised sessions | 14,701 advertised trajectories |
| Outcomes/rewards | Acceptance metadata, but no observations | None in schema | None in published schema |
| Group split | Source trajectory split | Session split only; task identity absent | Trajectory split feasible; task grouping still to audit |
| Contamination | Coding-task generation risk | Directly composed of six benchmarks | Generated/repository/web task risk |
| Conversion burden | Low but unusable | High | Medium, pending verification |

**VERIFIED:** Exgentic is a valid answer to “does the artifact contain actions and observations?” It is a much better substrate than pinned Kimi for observation-conditioned imitation.

**RECOMMENDED:** It does not replace Kimi under the current Experiment 001 contract. Prefer a pinned Hermes audit as the next primary-corpus decision. Retain Exgentic as a cross-harness stress-test or auxiliary corpus only after the 12-step conversion, and freeze a revised claim that explicitly drops reward/success and task-disjoint guarantees.
