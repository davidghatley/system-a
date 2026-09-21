# System-A Project State

Updated: 2026-09-21

## Current Iteration

Iteration 1 is complete. Reflex and Trace2Decision are accepted. Local Trainer is blocked on the frozen learning criterion, although its bounded RTX 3060 training and resume mechanics are verified. Iteration 2 is not authorized.

## Active Workstreams

| Workstream | Acceptance | Result |
|---|---|---|
| Reflex | ACCEPTED | Local typed-decision API/CLI, realistic example, offline inference, tests, and measured latency |
| Local Trainer | BLOCKED | Technical workflow verified; fixed-dev native-label choice accuracy regressed from 47.5% to 27.5% |
| Trace2Decision | ACCEPTED | 1,544 GLM-5.2-derived decisions across 207 clean trajectory groups |

The frozen criteria are `research/26_iteration_1_acceptance.md`. Reviews and final verification records are under each `artifacts/*_i1/` directory.

## Confirmed Findings

- Reflex runs the pinned public checkpoint locally and offline after model availability. Final RTX 3060 tokenization-plus-inference latency was 39.689 ms median and 40.527 ms p95 over 10 measured samples after 3 warmups.
- Reflex checkpoint outputs disagreed with both realistic example references. This supports its documented use as advisory observability telemetry, not as an authorization or safety policy.
- A bounded 32-update Laya run completed without OOM at 5.508 GiB peak reserved VRAM with finite gradients, expected encoder/scorer changes, unchanged action head, checkpoint reload, and a verified one-step resume.
- The Trainer did not establish useful fixed-dev learning: native-label exact choice accuracy changed from 19/40 to 11/40. Gold-distribution NLL improved from 1.4952 to 1.2102, but this remains diagnostic and cannot replace the failed frozen criterion.
- `11-47/glm-5.2-coding-and-debugging-traces` at revision `1371ed38f8890d0520a53bc7ad850308eb4d7a22` yielded 1,544 clean decisions from 207 trajectories after excluding 277 prefixes with unresolved prior calls.
- Trace2Decision splits contain 1,235 train, 159 dev, and 150 test decisions with zero trajectory, normalized-task, or rendered-state crossings. All records pass the shared validator and deterministic rerun hashes matched.
- The previous Kimi and Hermes/raw-trace findings remain valid historical evidence but are not prerequisites for practical-state specialization.

## Blockers

- Local Trainer Iteration 1 failed its predeclared +5-point choice-accuracy learning threshold. No large hyperparameter search was run.
- Trace source authenticity and teacher identity are publisher-attributed rather than cryptographically attested; embedded-content rights are not independently resolved.
- The first Trace2Decision ontology has no observed `other_tool` positives and is specific to one coding harness.

## Decisions

- Preserve `state + questions + gold distributions` as the only shared contract.
- Treat Reflex scores as advisory and expose limitations prominently.
- Treat trace labels as behavioral cloning, not action optimality.
- Keep generated model checkpoints and downloaded source data local; track compact evidence and the shareable derivative dataset.
- Use Luna for time-sensitive implementation/review tasks. Nemotron Ultra is available at zero reported cost but timed out twice on the bounded trace audit, so it is not the default for this workflow.
- Do not begin any Iteration 2 work before user feedback.

## Next Proposed Actions

1. Share/test Reflex as the immediate application artifact.
2. Choose whether to authorize a separately predeclared Trainer retry focused on converting NLL improvement into hard-label accuracy.
3. If Trainer learning is established, authorize Trace2Decision Iteration 2 training against simple baselines.
