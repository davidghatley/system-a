# Trace2Decision Iteration 1 Dataset Audit

Audit date: 2026-09-21. Candidate count: two. Search stopped when the primary
clearly met the selection requirements. No paid API was used.

## Decision

**PRIMARY, SELECTED:** `11-47/glm-5.2-coding-and-debugging-traces` at commit
`1371ed38f8890d0520a53bc7ad850308eb4d7a22`.

**FALLBACK, NOT USED:** `lambda/hermes-agent-reasoning-traces`, Kimi config, at
commit `b92885e4f0161d4b2536512710e004d4892cac6e`.

## Current Audit

| Requirement | GLM-5.2 primary | Hermes Kimi fallback |
|---|---|---|
| Recent compact source | VERIFIED: created 2026-09-16; 35.2 MB canonical JSONL and 1.9 MB Parquet | VERIFIED: 508.3 MB Kimi Parquet |
| Named strong teacher | VERIFIED at publisher/row level: `teacher_model=glm-5.2`, provider `z.ai` | VERIFIED at publisher level: `moonshotai/Kimi-K2.5` |
| Calls and observations | VERIFIED: structured calls and 1,668 tool-result messages linked to prior call IDs | VERIFIED by prior full audit: 106,218 response spans |
| Trajectory boundaries | VERIFIED: 207 distinct `source_trajectory_sha256` values | VERIFIED: 7,646 UUID trajectories |
| Enough examples | VERIFIED: 1,821 source decisions; 1,544 clean converted decisions over 207 trajectories | VERIFIED: much larger than needed |
| License/provenance | VERIFIED metadata: CC-BY-4.0; moonshiner harness and Codex acceptance review documented | VERIFIED metadata: Apache-2.0; Hermes harness documented |
| Conversion burden | Low: native cumulative messages and call IDs | Moderate: XML parsing, reasoning removal, about 508 MB |

Live Hugging Face metadata, README, schema preview, manifest, and repository
files were inspected. The exact primary source bytes, source README, and source
manifest are retained under `source/`. The fallback HEAD was rechecked live and
still matched its prior audited revision.

## Primary Caveat

The publisher says calls, results, and corrections are retained, but direct
measurement found 2,063 calls and 1,668 ID-linked tool results in each
trajectory's longest cumulative row. There were zero orphan results, but 395
call IDs had no matched result. These are not silently accepted: 277 decision
rows whose prefixes contain any unresolved prior call ID are excluded. Every
retained prefix therefore has complete ID linkage for every prior call.

The remaining 1,544 decisions include 1,337 with at least one prior genuine
tool-result message. Initial decisions legitimately have no prior observation.
The result payloads are authentic serialized result-channel data; physical
execution was not independently replayed.

## Stop Rationale

The primary is compact, names a strong teacher, contains genuine structured
calls and results, has explicit trajectory boundaries, exceeds 500 decisions
and 100 trajectories after conservative filtering, has usable public license
metadata, and is locally manageable. Per the frozen checklist, no third
candidate was audited.
