# Trace2Decision Iteration 1 Independent Review

Date: 2026-09-21

## Status

**ACCEPTED.** No acceptance-blocking repair is required. No training or
Iteration 2 work was performed.

## Checks performed

VERIFIED:

- Read `AGENTS.md`, the frozen acceptance checklist, implementation report,
  audit/card, manifests, source metadata, converter, verifier, tests, reports,
  and deterministic samples.
- `python3 -m unittest discover -s pipelines/trace2decision -p 'test_*.py' -v`:
  3/3 passed.
- `python3 pipelines/trace2decision/verify.py ... --rerun-dir
  artifacts/trace2decision_i1/rerun2 --evidence
  artifacts/trace2decision_i1/rerun_verification2.json`: PASS; 1,544/1,544
  records passed the shared validator and independent checks; report and all
  four compared file hashes matched.
- `sha256sum -c artifacts/trace2decision_i1/checksums.sha256`: all 12 listed
  files passed.
- An additional source scan found zero malformed target boundaries, duplicate
  call IDs, orphan/forward tool results, or forward result references. The
  converter's unresolved-prefix exclusion is therefore supported by actual
  prior-call ordering, not merely role labels.
- Recomputed grouped splitting from the pinned source and converter rules:
  zero trajectory crossings and zero normalized-task split crossings. The
  generated report also records zero state-hash crossings and zero duplicate
  state removals.
- Target mutation checks are structural and report zero failures. Target
  messages are excluded via `messages[:-1]`; reasoning content, future-derived
  `tools_used`, source metadata, and target arguments are not rendered into
  state. Target action names observed in the source are `bash`, `edit`, `find`,
  `grep`, `ls`, `read`, and `write`; their mappings agree with the documented
  ontology. Parallel targets are handled by the explicitly documented first
  serialized call rule.
- Targeted samples were coherent: pre-target task/history, linked tool
  results, and one-hot observed labels were consistent. The source manifest,
  pinned revision, source hash, retained README, and license metadata are
  present and checksum-backed.

## Findings by requested risk

### Falsification result: target-turn leakage — PASS (VERIFIED)

No canary leakage was found in conversion, output validation, or the direct
source/output inspection. This establishes converter-level exclusion, not
independent authenticity of the publisher's traces.

### Falsification result: unresolved calls/results — PASS with explicit loss (VERIFIED)

All retained prefixes have complete ID linkage under the converter rule. The
report excludes 277 rows with unresolved prior calls and records 395 missing
results in the full trajectory accounting. This is conservative and honest;
the physical cause of the missing results remains UNKNOWN.

### Falsification result: action-map correctness — PASS (VERIFIED)

All observed target tool names map to the documented read/search/edit/execute
families. `other_tool` has no observed positives and is retained as a
catch-all. Labels are observed behavior, not optimality (VERIFIED).

### Falsification result: trajectory/task/state split crossings — PASS (VERIFIED)

Independent recomputation found zero trajectory and normalized-task crossings;
the report records zero state-hash crossings and duplicate groups. Grouping by
normalized initial task is an inferred leakage boundary, not an authenticated
repository-family clustering guarantee (INFERRED).

### Falsification result: deterministic rerun — PASS (VERIFIED)

The fresh bounded rerun in `rerun2` reproduced report statistics and train,
dev, test, and sample hashes exactly.

### Falsification result: shared-validator compatibility — PASS (VERIFIED)

The concrete `shared.typed_decisions` validator accepted every emitted row in
both expected and rerun datasets.

### Falsification result: provenance/license claims — ACCEPTABLE with limits

VERIFIED: the source revision, bytes, README, manifest, teacher/provider
metadata, harness description, and CC-BY-4.0 declaration are retained and
pinned. INFERRED: serialized result-channel content is consistent with actual
harness observations. UNKNOWN: physical execution, teacher identity
attestation, and rights for every embedded task/repository snippet/model
output. The card already states these limitations, so they do not block this
derivative's acceptance.

## Non-blocking observation

**P2 documentation:** the retained source README's suggested attribution URL
uses `greghavens/...`, while the selected dataset identifier is
`11-47/...`. This is a provenance presentation inconsistency, not a failed
revision or license check because the selected repository, immutable revision,
source bytes, and source README are all recorded. Optional later repair brief:
add an explicit attribution note reconciling the publisher URL and selected
repository ID; do not alter data or hashes for Iteration 1.

## Unknowns retained

- The two Nemotron timeout attempts are not evidence about this converter and
  were not used to relax any criterion.
- No physical tool execution replay or cryptographic teacher attestation was
  performed.
- No downstream learning, calibration, action optimality, or transfer claim is
  established.
