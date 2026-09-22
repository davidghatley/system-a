# Trace2Decision v2 audit

## VERIFIED

- Source SHA-256 is `4e6b11f5b60976368f2a76c252808eb766af2ddcf5958d93f25160e45737d3aa`.
- The preserved v1 output hashes remain train `d1441e6184d5ef843192f19ed10adaf6c2065fb4971ace478de50d4084240818`,
  dev `202338e20b5a02cc98938b13332da3484d34137c32a4ac5ecf8f20f72b297e8f`,
  test `616d97e562b04723e8566028cb1a9117868580c94680db31d1f270ac334aa5b4`,
  and samples `7f4bf7528993c68dd34a3f68dccb9a5e7afed07a97a96d146f1122ad6de97af1`.
- Pinned `laya.common.build_sequence` and tokenizer were used offline.
- 1,821 source rows were examined; 277 unresolved-prefix rows were excluded,
  leaving 1,544 rows across 207 trajectories and 207 task groups.
- The final 512-token construction was checked without internal state loss for
  every row. Stable IDs are unique, contract validation passes, and trajectory
  and task groups do not cross splits.
- Rerun report and train/dev/test/sample hashes are identical.

## INFERRED

- The protected latest tool result is the most relevant observation for this
  source format. If none exists, the latest non-system, non-task message is
  used.

## UNKNOWN

- Whether the observed action is optimal or causally successful.
- Whether publisher attribution authenticates model identity or physical tool
  execution.
- Whether bounded protected text loses a task detail that matters downstream.

Exact token, split, omission, and file measurements are in `output/report.json`;
rerun evidence is in `rerun_verification.json`.

Repair-cycle-1 additions: the side-effect-free pre-policy candidate audit has
min/mean/max 187/4244.48/16656 tokens; 1504 (97.41%) exceed 512, 1402 (90.8%)
exceed 768, and 1285 (83.23%) exceed 1024. Retained newest-fitting history is
rendered in chronological source order after selection.
