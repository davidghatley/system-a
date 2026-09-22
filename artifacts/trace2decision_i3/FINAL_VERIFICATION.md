# Final verification (cycle 1)

**Decision: BLOCKED for training.**

**VERIFIED (independent audit + `i3_tests.py`, 7/7):** Source has 1,821 rows; all 208 no-call targets are excluded; all 76 mixed targets are excluded and every call-order permutation has the same exclusion decision. Output exclusion ledger has 224 `unresolved_prefix_tool_call` rows. Retained counts are exactly train/dev/test **1051/140/122 = 1313** (categories edit/execute/read/search); fixed ontology absent classes are `other_tool`, `respond_or_finish`. V2 assignments are unchanged (`[]`). Actual Laya lengths are **187–512** (mean 473.6), with zero over 512. Prefix-only state, target-string/argument, and metadata-boundary checks pass; no exact or cross-split rendered-state duplicates. Commands/evidence: `commands.log`, `verification.json`; offline rerun reports pass and all five output hashes agree.

**VERIFIED with limitation:** train/dev sample review covers 8 populated strata and agrees structurally with source labels; this is not semantic adjudication. Exact normalized task/trajectory grouping passes (206 groups/trajectories; 163/20/23), but the bounded near-task screen flags one cross-split pair; semantic command/family leakage is **UNKNOWN**. Thus no broad independence claim is justified.

**INFERRED/UNKNOWN:** Tool-interface labels are observed categories, not intent. No-call exclusion is conservative and source-supported only by absent completion markers; completion semantics are not established. `checksums.sha256` does **not** match current output or `verification.json` hashes (independent `sha256sum`), so manifest integrity is not verified.

Only this report was written; no code/data/evidence was altered.
