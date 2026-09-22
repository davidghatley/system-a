# Iteration 2 Acceptance Checklist

Date frozen: 2026-09-21

## Shared

- [x] No paid APIs, Trace2Decision training, broad dataset search, or Iteration 3 work.
- [x] Exact model/source/data revisions, commands, measurements, and limitations are recorded.
- [x] `shared.typed_decisions` is the common validation contract and preserves optional non-model metadata.
- [x] Parent cross-track verification moves unchanged v2 rows through validation, loader, Laya construction, and inference/evaluation while excluding metadata from model state.

## Reflex

- [x] Pin `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` and verify source compatibility.
- [x] Remove or minimize private schema duplication; accept the shared native gold shape and metadata.
- [x] A Trace2Decision-format record runs without reshaping and gold is never model input.
- [x] Base and specialist run on the unchanged Iteration 1 example; both outputs are reported without tuning the example.
- [x] Specialist offline inference, tests, warnings, README, and reproducible RTX 3060 benchmark pass.

## Trainer B1 Positive Control

- [x] Freeze subset and native-label metrics before model predictions.
- [x] Evaluate pinned base and specialist with identical code and data.
- [x] Record exact revisions and per-workflow plus aggregate results where bounded.
- [x] B1 passes only if specialist clearly outperforms base on the fixed evaluator. If it does not, stop before B2.

## Trainer B2 Retry

- [x] Run only after B1 passes.
- [x] Predeclare immutable subsets, seed, optimizer, rates, context, updates/epochs, batching, objective, primary metric, and stopping rule.
- [x] Use a bounded hardware-adapted version of the inspected public recipe and no open-ended search.
- [x] One main run completes with finite gradients, reloadable checkpoint, metrics, loss, VRAM, and wall time.
- [x] Native-label choice accuracy improves by at least 5 absolute points on fixed held-out development data.
- [x] Gold NLL and all-question metrics are recorded but do not replace the primary criterion.

## Trace2Decision V2

- [x] Preserve Iteration 1 files and hashes; emit a separate v2 derivative from the same pinned source.
- [x] Every row has stable `id`, `trajectory_id`, `task_group`, and `source_step` metadata outside model-facing state.
- [x] Actual pinned Laya tokenization reports length/truncation and survival at 512, with descriptive 768/1024 values if useful.
- [x] Deterministic policy reserves task, critical system constraints, available actions, and latest relevant observation, then fills remaining budget with newest history.
- [x] At least 1,544 rows remain unless a correctness exclusion is explicitly justified.
- [x] Task and latest relevant observation survive every retained model input and final input fits the selected context by construction.
- [x] Splits remain trajectory/task grouped with zero target leakage and deterministic rerun hashes.

## Review

Each workstream receives implementer, independent reviewer, at most two targeted repairs, and verifier. The parent alone accepts cross-track interoperability. One cross-track repair cycle is allowed. Iteration 3 remains prohibited.

Final status: **ACCEPTED** on 2026-09-21. Parent cross-track evidence is in
`artifacts/iteration_2/INTEGRATION_VERIFICATION.md`.
