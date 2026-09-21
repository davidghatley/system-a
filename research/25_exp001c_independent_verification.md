# exp_001c Independent Post-Run Verification

Date: 2026-09-20

Overall verdict: **VERIFIED**. No correctness concern was identified.

| Falsification target | Result | Evidence |
|---|---|---|
| Protected splits excluded from selection | VERIFIED | Task-derived proxy assignment precedes tool/conversation parsing; calibration/test rows return immediately. The protected-target sentinel test passed, and emitted selection statistics contain only train/development. |
| No silent truncation | VERIFIED | Complete state uses `truncation=False`; over-budget states are rejected before upstream construction; accepted IDs are asserted equal to the complete expected state. Boundary tests passed and retained maxima equal or remain below each budget. |
| Exact budgets | VERIFIED | Protocol, implementation, and results contain exactly 512, 640, 768, and 1024. |
| 512 reference-only | VERIFIED | 512 has `promotion_eligible=false`; eligible budgets are exactly 640, 768, and 1024. |
| Frozen gates | VERIFIED | Independent arithmetic checks reproduced rates, differences, support, split balance, linkage, no-truncation, and aggregate gate outcomes. Every budget correctly fails label bias. |
| Selection rule | VERIFIED | No eligible budget passes all gates; the passing list is empty and selected budget is null. |
| Linkage clarification | VERIFIED | `51,558 - 22 - 24 = 51,512`: 22 malformed linkage candidates and 24 invalid targets were excluded. Retained-linkage failures are zero at every budget. |
| Hardware scope | VERIFIED from available records | No budget was selected, so no GPU/model smoke was run. The recorded sweep is tokenizer/CPU preprocessing only. |

## Independent Checks

- Protocol SHA-256 matched `protocol.sha256`, provenance, and `checksums.json`.
- Sweep implementation and output artifact hashes and byte counts matched `checksums.json`.
- The bounded assertion suite passed independently.
- Source-row accounting reproduced: `5,624 + 711 + 568 + 743 = 7,646`.
- Source dataset target values were not inspected and the full sweep was not rerun.

Residual limitation: candidate extraction was not independently regenerated from source. Hardware non-use is supported by the hashed implementation and records, but cannot prove that no unrelated command was run outside the recorded workflow.
