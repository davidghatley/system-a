# Reflex Iteration 2 Evidence

## VERIFIED

- Default specialist: `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`; explicit base: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`; source: `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
- Shared parser supports optional native `{type,label,probabilities}` gold, preserves metadata, and Reflex passes only state/questions to Laya. The unchanged Iteration 1 example remains compatible.
- 18 no-weight tests and compileall pass. Trace2Decision compatibility and metadata exclusion are tested.
- Unchanged example succeeds offline for both base and specialist; full outputs are `base_output.json` and `specialist_output.json`.
- Specialist RTX 3060 benchmark succeeds offline: 3 warmups, 10 samples, 40.150/41.093/41.893/41.893 ms min/median/p95/max, excluding model load.

## INFERRED

The specialist is practically source-compatible in this environment because it loads through the same pinned `laya.agent.Agent` path. Fixed-environment repeatability is expected but cross-hardware bitwise stability is unverified.

## UNKNOWN

The general release example is not a specialist accuracy evaluation: both checkpoints selected `publish` and `true`. Calibration, representative accuracy, CPU latency, and long-state behavior remain unmeasured.

Commands, exact output hashes, provenance, and full stdout are in `COMMANDS_RESULTS.md` and the adjacent JSON files.
