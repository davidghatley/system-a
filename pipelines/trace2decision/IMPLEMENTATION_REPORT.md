# Trace2Decision Iteration 1 Implementation Report

## VERIFIED

- Selected `11-47/glm-5.2-coding-and-debugging-traces` at immutable revision
  `1371ed38f8890d0520a53bc7ad850308eb4d7a22`; source JSONL SHA-256 is
  `4e6b11f5b60976368f2a76c252808eb766af2ddcf5958d93f25160e45737d3aa`.
- The current source names `glm-5.2` as teacher in rows and card, identifies
  z.ai as provider, documents the moonshiner harness and Codex acceptance-only
  review, and declares CC-BY-4.0.
- The source has 1,821 cumulative decisions and 207 explicit SHA-256 trajectory
  boundaries. Longest cumulative rows contain 2,063 structured calls and 1,668
  actual result-role messages, all linked to an earlier call ID; 395 calls lack
  a matched result.
- The converter rejects all 277 decisions whose prefixes contain an unresolved
  call ID. It emits 1,544 clean decisions across all 207 trajectories, including
  1,337 decisions with a prior tool result.
- Each output has exactly `state`, `questions`, and `gold`; `next_action` is a
  shared-contract `choice` with six fixed criteria and a one-hot
  observed-behavior distribution. Labels are behavioral cloning, not optimality.
- State uses only pre-target messages. Reasoning, full-trajectory `tools_used`,
  source IDs, split/category metadata, and the target turn are excluded.
  Per-row target-canary mutation found zero leakage failures.
- Splits are normalized-task-text grouped and deterministic: train 1,235 rows /
  164 trajectories, dev 159 / 20, test 150 / 23. Trajectory, task group, and
  state-hash crossings are zero.
- Obvious duplicate checks found zero normalized-task duplicate groups, exact
  complete-trajectory duplicate groups, and exact rendered-state duplicates.
- Labels are execute 670, read 289, edit 237, search 182,
  respond-or-finish 166, other-tool 0. Median state length is 11,628 characters
  and p95 is 17,204.
- Three unit tests pass. The independent verifier and the concrete
  `shared.typed_decisions` validator accepted all 1,544 records.
  A full rerun reproduced the report exactly and reproduced train/dev/test/sample
  hashes exactly. Evidence is `artifacts/trace2decision_i1/rerun_verification.json`.
- Twelve deterministic hash-selected samples were inspected; task, prior call,
  result, and action labels were coherent in the inspected records.
- No training or Iteration 2 work was performed.

## INFERRED

- The serialized tool results are consistent with genuine harness observations:
  they use the result role and call IDs and contain concrete command/file output.
  This supports observation-conditioned behavioral cloning after linkage
  filtering.
- GLM-5.2 is a strong teacher by current public positioning and the source's
  verification pipeline, but this project did not benchmark the teacher.
- Grouping normalized initial task text is an appropriate leakage boundary for
  this release because every measured normalized task is unique. It is not an
  authenticated repository-family or task-template clustering guarantee.

## UNKNOWN

- Physical executions were not replayed, signed, or independently authenticated.
- Why 395 serialized calls lack ID-matched results is unknown. No affected
  prefix is included, but the source card's broad retention wording is stronger
  than the measured artifact.
- Dataset-level CC-BY-4.0 metadata does not independently settle all rights in
  embedded generated tasks, repository snippets, or model outputs.
- Teacher identity is publisher/row attribution; `model_attested` is null in
  inspected rows and there is no cryptographic attestation.
- No claim is made about action optimality, task success causality, calibration,
  downstream learning, or transfer beyond this coding harness.

## RECOMMENDED

- Use `artifacts/trace2decision_i1/output/{train,dev,test}.jsonl` as the frozen
  Iteration 1 derivative and retain `output/report.json` beside it.
- Require `pipelines/trace2decision/verify.py` to pass before consumption; it
  invokes the shared validator over every record as well as independent checks.
- Preserve trajectory-macro evaluation because row counts per trajectory vary.
- Keep the 277 unresolved-prefix rows excluded unless authoritative result
  linkage is published and re-audited.
- Do not interpret observed labels as optimal actions and do not train without
  a separately authorized training protocol.
