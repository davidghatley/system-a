# System-A Project State

Updated: 2026-09-21

## Current Iteration

Iteration 2 is accepted. The trace-to-training-to-evaluation pipeline is connected and parent-verified; historical Iteration 1 artifacts remain unchanged. Iteration 3 is not authorized.

## Active Workstreams

| Workstream | Acceptance | Result |
|---|---|---|
| Reflex | ACCEPTED | Pinned specialist, shared native contract, offline execution, tests, warnings, and RTX 3060 benchmark verified |
| Local Trainer | ACCEPTED | B1 validated the evaluator; the sole predeclared B2 run improved native-label choice accuracy from 36.0% to 70.5% |
| Trace2Decision | ACCEPTED | Separate deterministic v2 derivative has stable metadata, protected 512-token construction, grouped splits, and 1,544 rows |
| Cross-track integration | ACCEPTED | One unchanged v2 row passed shared loading, trainer construction, accepted B2 inference, and Reflex inference with metadata excluded from model tensors |

Iteration 2 criteria are frozen in `research/27_iteration_2_acceptance.md`.

## Confirmed Findings

- Reflex runs the pinned public checkpoint locally and offline after model availability. Final RTX 3060 tokenization-plus-inference latency was 39.689 ms median and 40.527 ms p95 over 10 measured samples after 3 warmups.
- Reflex checkpoint outputs disagreed with both realistic example references. This supports its documented use as advisory observability telemetry, not as an authorization or safety policy.
- A bounded 32-update Laya run completed without OOM at 5.508 GiB peak reserved VRAM with finite gradients, expected encoder/scorer changes, unchanged action head, checkpoint reload, and a verified one-step resume.
- The Trainer did not establish useful fixed-dev learning: native-label exact choice accuracy changed from 19/40 to 11/40. Gold-distribution NLL improved from 1.4952 to 1.2102, but this remains diagnostic and cannot replace the failed frozen criterion.
- `11-47/glm-5.2-coding-and-debugging-traces` at revision `1371ed38f8890d0520a53bc7ad850308eb4d7a22` yielded 1,544 clean decisions from 207 trajectories after excluding 277 prefixes with unresolved prior calls.
- Trace2Decision splits contain 1,235 train, 159 dev, and 150 test decisions with zero trajectory, normalized-task, or rendered-state crossings. All records pass the shared validator and deterministic rerun hashes matched.
- The previous Kimi and Hermes/raw-trace findings remain valid historical evidence but are not prerequisites for practical-state specialization.
- Live Hub inspection identifies `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` as the intended public specialist. Its published metrics are claims to be independently checked by the Iteration 2 positive control.
- B1 independently reproduced a clear public positive control: specialist choice accuracy 68.0% versus base 35.5%, a +32.5-point gain.
- The sole B2 run completed 5,120 microforwards and 80 updates in 772.8 s total at 8.670 GiB peak reserved VRAM. Fixed-dev native-label choice accuracy improved from 72/200 to 141/200 (+34.5 points); all-question accuracy improved from 38.8% to 73.0% and native-gold NLL fell from 1.5310 to 0.6894.
- Trace2Decision v2 retains 1,544 rows (1,235/159/150), stable IDs and grouped splits, all required protected sections, zero post-policy sequences over 512 tokens, and deterministic rerun hashes.
- Parent integration verified source row `t2d-i2-4bba28e1b18498c2d153a95a223fc411` unchanged through shared loading, trainer/B2 inference, and Reflex specialist inference. Metadata remained out of model tensors. Both models missed this one row's label; this is not an accuracy claim.

## Blockers

- Local Trainer Iteration 1 remains a historical failed run; Iteration 2 B2 separately passed its frozen learning criterion without a hyperparameter search.
- Trace source authenticity and teacher identity are publisher-attributed rather than cryptographically attested; embedded-content rights are not independently resolved.
- The first Trace2Decision ontology has no observed `other_tool` positives and is specific to one coding harness.

## Decisions

- Preserve `state + questions + gold distributions` as the only shared contract.
- Treat Reflex scores as advisory and expose limitations prominently.
- Treat trace labels as behavioral cloning, not action optimality.
- Keep generated model checkpoints and downloaded source data local; track compact evidence and the shareable derivative dataset.
- Use Luna for time-sensitive implementation/review tasks. Nemotron Ultra is available at zero reported cost but timed out twice on the bounded trace audit, so it is not the default for this workflow.
- Do not train on Trace2Decision or begin Iteration 3 before user feedback.

## Next Proposed Actions

1. Review accepted Iteration 2 deliverables and evidence in commit `7d2caf6`.
2. Keep Trace2Decision training and Iteration 3 blocked until explicit user authorization.
