# Adversarial Review Of The Hermes Replacement Audit

Review date: 2026-09-20

## Decision

**RECOMMENDED: NO-GO for freezing or training on Hermes as the Experiment 001 replacement now.** The audit's conditional-GO is not sustained by its own required gates. Hermes is observation-bearing and remains a plausible candidate, but no frozen, leakage-audited conversion exists; the current proposal does not establish reasoning exclusion, state deduplication, task-family isolation, a coherent Laya typed target, or 512-token representability.

This is a readiness NO-GO, not a claim that Hermes can never be used. Reconsideration requires every mandatory condition below and a newly frozen protocol before training or test scoring.

Machine-readable results are in `artifacts/reviews/hermes_review.json`. The independent local-only check is reproducible with:

```bash
.venv/bin/python scripts/review_hermes_audit.py
```

No data was downloaded, no task was replayed, and no model was trained or run. The check read only the persistent pinned Parquet and local pinned tokenizer.

## Scope And Source Integrity

- **VERIFIED:** The reviewed file is `data/hermes_audit/kimi-b92885e4f0161d4b2536512710e004d4892cac6e.parquet`, 508,262,558 bytes, SHA-256 `d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02`, with 7,646 rows. These values agree with the source manifest.
- **VERIFIED:** The review independently scanned all conversations, tool definitions, tasks, calls, responses, and reasoning markers.
- **UNKNOWN:** The manifest's `hub_lfs_sha256` is null. Local agreement with the earlier checksum pins the reviewed bytes, but the recorded manifest does not independently bind that checksum to Hub LFS metadata.

## Findings

### 1. Reasoning exclusion is not feasible under the proposed regex

- **VERIFIED:** The data contains 87,684 opening `<think>` tags, 87,714 closing tags, and 87,672 complete regex spans. Thirty-two messages have unbalanced or noncanonical structure.
- **VERIFIED:** After applying the audit's complete-span substitution, 24 messages still contain opening or closing reasoning markers. The earlier assertion of zero residual complete spans tested the same complete-span regex after substitution and therefore could not detect unmatched markers.
- **INFERRED:** Some residual markers may be examples or malformed serialization rather than hidden reasoning, but they still falsify the claim that complete-span removal alone establishes structural reasoning exclusion.
- **RECOMMENDED:** Parse reasoning tags structurally, classify every malformed case, reject affected target rows unless boundaries are unambiguous, and run a target/reasoning canary through rendering and tokenization.

### 2. Tool-response linkage is strong serialization evidence, not authenticated execution

- **VERIFIED:** There are 106,220 call spans and 106,218 response spans. All 106,218 responses parse as JSON. Two calls do not parse.
- **VERIFIED:** For 81,568 parseable assistant call turns, the immediately following tool message has the same response count and the ordered response names exactly match ordered call names. There are no parsed name mismatches; one call turn is unparseable for this check.
- **VERIFIED:** All 106,218 parsed calls omit an explicit call ID. Responses carry publisher-generated IDs; 106,212 match the pattern `functions.<name>:<index>`, and response IDs are unique within every trajectory. Because calls have no IDs, those response IDs cannot be independently joined to an ID stored on the call.
- **VERIFIED:** No execution was replayed. The bytes contain no signature, external execution-log reference, or independently checkable execution identity.
- **INFERRED:** Adjacency, ordered name agreement, detailed output payloads, and unique response IDs make the serialization internally coherent. They do not prove that a physical tool produced each payload. “Authentic” or “real execution” remains a publisher attribution, not a verified measurement.
- **RECOMMENDED:** Describe these as serialized publisher-attributed tool responses. Preserve ordered names and response IDs and reject all linkage exceptions.

### 3. The headline target count is not the eligible target count

- **VERIFIED:** There are 87,667 `gpt` turns: 81,570 with at least one complete call span and 6,097 with none. This reproduces the headline count but not eligibility.
- **VERIFIED:** Only 81,532 call-bearing targets contain exclusively parseable calls whose names belong to that row's own tool definitions. Thirty-six targets contain 42 calls outside their row toolset, and two targets contain unparseable calls.
- **VERIFIED:** 23,617 call-bearing targets also contain non-reasoning text outside call tags. A binary `CALL_TOOL` label discards this mixed behavior; calling all no-call turns `RESPOND_OR_FINISH` also merges ordinary responses and terminal completion without a terminal-status field.
- **VERIFIED:** 80,021 targets have an earlier tool response, but this includes targets later invalidated by malformed calls, reasoning structure, duplicates, or representation policy. The final eligible count is therefore **UNKNOWN**.
- **RECOMMENDED:** Publish one exclusion ledger keyed by trajectory and turn, then recompute counts after all filters and exact rendered-state deduplication.

### 4. The proposed target is not yet a coherent Laya typed target

- **VERIFIED:** Hermes has six row-level toolsets, not one uniform 26-tool action space. The browser regime covers 1,048 trajectories. Exact repeated task groups cross row toolsets in 64 cases.
- **VERIFIED:** `RESPOND_OR_FINISH` plus a set of tool names is a label vocabulary, not a complete specification of Laya `choice`/`noul` questions, option text, valid-bundle decoding, absent-tool handling, or confidence construction.
- **VERIFIED:** Protocol 0.2.0 separately predicts `control_mode` and per-family membership, although control mode is implied by a nonempty family set. The Hermes audit does not resolve that previously documented redundancy or impossible output bundles.
- **INFERRED:** Asking all 26 family questions makes unavailable tools deterministic negatives and exposes browser/non-browser regime. Asking only row-available families changes bundle dimensionality and can suppress false positives for unavailable tools. Either choice needs a frozen estimand and decoder.
- **UNKNOWN:** Whether the 26 exact tool names are scientifically preferable to a train-only family map. The current audit selected a global taxonomy after scanning the complete corpus, contrary to the prior design requirement to freeze taxonomy from training data and map test-only names to `OTHER_TOOL`.
- **RECOMMENDED:** Define one coherent bundle before splitting labels: row-level availability input, train-only taxonomy decisions, `OTHER_TOOL`, malformed/unknown behavior, option text, valid decoder, mixed text-plus-call semantics, and trajectory-macro scoring. Do not infer `RESPOND_OR_FINISH` from parser failure.

### 5. Complete-trajectory deduplication missed target-state duplication

- **VERIFIED:** The original statement that all 7,646 reasoning-stripped complete trajectories are unique is true but irrelevant to the row-level learning unit.
- **VERIFIED:** Across the 87,667 reconstructed reasoning-stripped prefixes, 1,411 exact duplicate-prefix groups contain 5,719 excess target rows. This directly contradicts any implication that complete-trajectory uniqueness establishes state uniqueness.
- **VERIFIED:** There are only 1,020 exact normalized task strings, 764 duplicated groups, and a largest group of 122 trajectories. A simple heuristic replacing numeric and quoted literals reduces the task keys further to 985; this is diagnostic, not a validated semantic clustering method.
- **VERIFIED:** One exact task group crosses categories and 64 cross toolsets. Therefore normalized task text is not a stable proxy for category, environment, or available action space.
- **UNKNOWN:** Repository identity, website identity, generated template family, and near-semantic task overlap are not measured in the audit.
- **INFERRED:** Exact normalized-task grouping prevents literal prompt leakage, but does not establish the protocol's claimed task-disjoint generalization. It may also place large correlated groups unevenly across partitions.
- **RECOMMENDED:** Deduplicate the actual rendered and tokenized states before splitting. Group by exact task plus audited template/repository/site families; publish deterministic train/development/calibration/test manifests, category/tool support, trajectory counts, and post-filter rows. The phrase “hash with seed” is insufficient without canonical bytes, hash function, bucket boundaries, and allocation rules.

### 6. The 512-token gate fails for a material fraction under even a minimal state

- **VERIFIED:** Using the pinned local Laya tokenizer at model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, task plus latest complete tool message has median 166 tokens, p95 3,472, maximum 118,109, and exceeds 512 tokens for 23,359 of 80,021 observation-bearing targets (29.19%).
- **VERIFIED:** These counts exclude Laya question text, options, separators, and special-token overhead, so the actual state budget is below 512 and the affected fraction can only increase.
- **VERIFIED:** Serialized row tool definitions have median 2,219 tokens, minimum 1,312, and exceed 512 in all 7,646 trajectories. The audit's requirement to retain a canonical active tool-definition block cannot coexist literally with a 512-token sequence.
- **INFERRED:** Message-boundary truncation can retain a compact latest observation for many rows, but 29.19% cannot retain even task plus the complete latest tool message in the nominal total budget. Blind left truncation would split payloads or discard task/tool availability.
- **UNKNOWN:** Whether a compressed, leakage-safe representation of tool availability and large observations preserves sufficient action-relevant information. That requires a frozen renderer and profiling, not training.
- **RECOMMENDED:** Do not call 512-token feasibility an implementation detail. Freeze question overhead and state budget, preserve task plus active-tool availability and complete latest call/result, reject or separately classify over-budget rows, and report exclusions by tool, category, task group, label, and split. If support becomes inadequate or biased, increase context under a separately measured protocol or reject Hermes.

### 7. Teacher, harness, outcome, and license claims remain weaker than stated

- **VERIFIED:** Rows have no teacher or harness provenance field. None of the six distinct system prompts mentions Kimi or Moonshot. Teacher and harness identities come from repository-level documentation.
- **VERIFIED:** Six system-prompt/toolset variants exist, including a 1,048-trajectory browser regime. This is compatible with one software project but does not establish one invariant interaction contract.
- **VERIFIED:** There is no terminal trajectory-success field. Tool responses include both failure-like and success-like payloads; these are call-local signals and cannot recover terminal task outcome.
- **INFERRED:** “One teacher/one harness” is a reasonable publisher-level corpus description, but not row-attested provenance. “Successful frontier-agent traces” is unsupported and must be removed.
- **VERIFIED:** Repository metadata says Apache-2.0.
- **UNKNOWN:** Licenses, terms, privacy status, and redistribution rights for embedded repositories, websites, browser content, and task payloads. Apache-2.0 metadata does not automatically relicense third-party content.
- **RECOMMENDED:** Treat teacher/harness/execution as publisher claims, preserve the six regimes in reporting, and adopt an explicit embedded-content provenance policy before redistribution or release of derivatives.

## What Survived Challenge

- **VERIFIED:** Hermes genuinely contains explicit serialized tool-response channels, unlike the pinned Kimi K3 release.
- **VERIFIED:** Ordered adjacency and tool-name agreement are nearly complete and materially stronger than mere tag counting.
- **VERIFIED:** The source has ample raw volume and explicit per-row tool definitions.
- **VERIFIED:** Exact normalized-task grouping is possible and preferable to row-random splitting.
- **INFERRED:** After a substantial conversion and protocol refreeze, Hermes may support a narrow experiment in publisher-attributed, within-corpus recorded-action imitation.

These strengths justify remediation work, not a current GO.

## Mandatory Conditions To Reopen

1. **RECOMMENDED:** Refreeze Experiment 001 for recorded traces, not successful traces; remove outcome-aware interpretation and replace source train/validation assumptions with explicit group manifests.
2. **RECOMMENDED:** Freeze a coherent Laya typed bundle, row-level availability representation, train-only taxonomy policy, `OTHER_TOOL`, malformed/unknown behavior, mixed-response semantics, and valid-bundle decoder.
3. **RECOMMENDED:** Structurally parse reasoning and reject every unresolved malformed case; pass target and reasoning canaries after tokenization.
4. **RECOMMENDED:** Validate ordered call/response count and name linkage, retain response IDs, and publish every excluded trajectory/turn and reason.
5. **RECOMMENDED:** Deduplicate rendered/tokenized prefixes and establish task-template/repository/site-disjoint manifests, not only normalized-text groups. Demonstrate minimum trajectory and per-family support after filtering.
6. **RECOMMENDED:** Freeze the Laya tokenizer, exact questions, state renderer, and message-boundary truncation. Demonstrate that retained rows preserve task, active-tool availability, and a complete latest call/result inside the actual post-question state budget.
7. **RECOMMENDED:** Resolve the six tool/system regimes in the estimand and report regime-specific support and metrics; do not let browser availability become an unreported shortcut.
8. **RECOMMENDED:** Document embedded-content license/provenance policy and qualify teacher, harness, and real-execution statements as publisher attribution absent stronger evidence.
9. **RECOMMENDED:** Recompute eligible rows, trajectories, independent task groups, split balance, family support, duplicate rates, and power/precision only after conditions 1-8. Freeze the revised protocol before model outputs or official test access.

## Final Verdict

**RECOMMENDED: NO-GO.** The original conditional-GO depends on future preprocessing that is not merely mechanical: it changes the eligible population, split unit, action space, state representation, and scientific claim. Three measured failures are already material: malformed reasoning survives the proposed removal, 5,719 exact duplicate target prefixes were missed, and at least 23,359 observation-bearing targets cannot fit task plus their latest complete tool message in 512 tokens before Laya overhead. Until all mandatory conditions pass in durable artifacts, Hermes is a candidate requiring protocol redesign, not an approved replacement dataset.
