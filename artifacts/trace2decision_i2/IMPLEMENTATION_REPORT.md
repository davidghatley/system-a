# Trace2Decision Iteration 2 implementation report

## VERIFIED

- Implemented `pipelines/trace2decision/i2_convert.py`, a separate v2
  converter. It reads the unchanged v1 local source path and does not modify
  v1 output or shared code.
- Retained 1,544 rows (train/dev/test counts are recorded in
  `output/report.json`), preserving v1 labels and the v1 normalized-task split
  seed. Every row has stable metadata identifiers outside model state.
- Actual pinned Laya construction was used. The 512 sequence audit reports
  complete state preservation and length at or below 512 for every row. The
  same states were measured descriptively at 768 and 1024.
- `i2_verify.py` passed shared validation, leakage checks, stable-ID checks,
  split integrity, and deterministic rerun comparison. Evidence is
  `rerun_verification.json`.
- No training, paid API, broad search, or Iteration 3 work was performed.

## INFERRED

- A tool result is the latest relevant observation in this cumulative trace
  format; it is not an independently authenticated observation.

## UNKNOWN / LIMITATIONS

- Protected fields use deterministic head/tail bounds when needed; the exact
  per-row bounds and any omitted optional messages are retained in metadata.
- 768/1024 are descriptive builder measurements, not model runs.

Repair cycle 1 also adds a side-effect-free full bounded pre-policy audit with
threshold counts at 512/768/1024, retains separate post-policy distributions,
and restores selected newest-fitting history to chronological source order.
