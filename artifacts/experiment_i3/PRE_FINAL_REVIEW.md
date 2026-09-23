# Iteration 3 Pre-Final Review

Date: 2026-09-22

## Scope and evidence boundary

This review was performed CPU/offline only. I inspected the recovered seed-42
result, completed seed-314159 result, both reload records, both checkpoint
manifests, the actual-model pilot, the selected baseline result, `dev_selection.json`,
`freeze.json`, the command record, and the final-test binding code. I did not
open test records, labels, or test results; did not run model/CUDA work; and did
not consume the final marker. No implementation or prior report was edited.

## Verified measurements

- The fixed label order is exactly `read, search, edit, execute, other_tool,
  respond_or_finish` in both neural results and the baseline. Both neural dev
  results contain 140 whole-turn records; absent `other_tool` and
  `respond_or_finish` classes remain in the six-class macro average.
- Seed 42 recovered dev: accuracy `0.6785714286`, macro-F1 `0.4452647890`,
  NLL `3.4055659592`, Brier `0.5326921745`. Recovery is explicitly marked
  authorized, non-retraining, and test-closed.
- Seed 314159 completed dev: accuracy `0.7142857143`, macro-F1 `0.4419447311`,
  NLL `3.7480882207`, Brier `0.5525369932`. Its base result is accuracy
  `0.1`, macro-F1 `0.0779505582`, NLL `2.3605153683`, Brier `1.0378210468`.
  Fine-tuning therefore improves base macro-F1 by `0.3639941729` absolute.
- The selected inexpensive baseline is dev-only TF-IDF logistic `word12-c2`:
  macro-F1 `0.2927361252`, accuracy `0.6071428571`, NLL `0.9518856046`, and
  Brier `0.5149799499`. Selected seed 42 adds `0.1525286638` macro-F1 points,
  exceeding the predeclared `0.05` absolute added-value threshold.
- Seed variation is present but small on macro-F1: seed 42 is higher by
  `0.0033200579`; it also has lower NLL by `0.3425226316`. The deterministic
  rule therefore selects seed 42 by macro-F1, not by an undisclosed test
  criterion.

## Checkpoints, stopping, and hashes

- Both required seeds are represented exactly once: `{42, 314159}`. Each run
  has exactly 68 updates and 4 epochs; each has 4,204 microforwards. Seed 42's
  timing and loss curve are honestly `unknown` after recovery.
- Reload evidence for both seeds says strict model load and exact optimizer,
  scaler, scheduler, and stopping state. Before/after model-state digests are
  equal for each seed.
- Independently checked on-disk hashes match the recorded evidence:
  - Seed 42 weights `9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947`;
    training state `f32c1032708e191079ca5c56072d87a08e88a21e205e2229424432dff48706cb`;
    reload evidence `b47b091270f3983f1d56ab7e92c8f2196d61298e884c02f3cf698f796ed1e084`.
  - Seed 314159 weights `6bdf67f74a4e933d5f49afadd557661858abc792e5242a9827bbb385a18f0873`;
    training state `3bfd2237ce2f1b17fdbb75913cbff03d7432c0cf53136a502499c0e89614d5b5`;
    reload evidence `f4ac4eb100b62467db7ac847d58b5d7024a4328a2166119d6832d34faac58e70`.
  - Config hash is `748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae`;
    accepted manifest hash is `158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2`.
  - The freeze hash is exactly
    `c18059b0409fd79db6e7d13a046453112caa2d74efc373981898355ffc320832`.

## Resource and budget evidence

- Actual-model pilot evidence is finite, test-closed, seed 42, and records 2
  bounded updates, 19.157471 seconds training, 7.365299 seconds dev
  evaluation, 6.082518 seconds cold load, and 8.767578 GiB peak reserved.
- The completed seed-314159 result records 767.102596 seconds training,
  7.443807 seconds dev evaluation, 6.359177 seconds cold load, and 9.517578
  GiB peak reserved. Seed-42 full-run timing, loss curve, latency, and peak
  VRAM remain unknown and are not inferred from the pilot.
- The recorded conservative projection is 831.002307 GPU-seconds per seed and
  1662.004614 GPU-seconds for both seeds, below the 7200 aggregate threshold.
  Thus no single-seed fallback was applied (`fallback_applied: false`).

## Freeze and final-test binding

- `dev_selection.json` and `freeze.json` agree on candidate count 2, selected
  seed 42, selected checkpoint
  `artifacts/experiment_i3/preflight/runs/seed_42/checkpoint`, and selected
  weights hash `9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947`.
  Selection is explicitly `dev`-only and uses highest macro-F1, then lowest
  NLL, then lowest declared seed. Both artifacts say `test_opened: false`.
- The final-test entry point requires the freeze SHA, revalidates the frozen
  selection and checkpoint, binds the owner record to the selected seed, and
  resolves the selected checkpoint before execution. It acquires the GPU lock,
  atomically creates the one-shot attempt marker, and only then opens test
  bytes. The command was not executed.

## Limitations and blockers

The metrics are dev-only and subject to dev-selection optimism. Seed 42's
recovered timing, peak VRAM, latency, and loss curve are unknown. The pilot and
completed-run resource values are recorded evidence, not rerun here under this
CPU/offline audit. Test behavior and test metrics remain intentionally unknown.

Concrete blockers: none identified for the final-test gate. The final test
must remain one-shot and must be invoked with the recorded freeze SHA; this
review did not invoke it.

ACCEPTED_FOR_FINAL_TEST
