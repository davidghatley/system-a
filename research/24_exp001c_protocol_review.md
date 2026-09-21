# exp_001c Pre-Execution Protocol Review

Date: 2026-09-20

Scope: independent review of the proposed exp_001c protocol and implementation against corrected exp_001b v0.2.0. No sweep result was available or inspected.

| Question | Verdict | Evidence |
|---|---|---|
| Varies only context budget? | PASS | All arms share candidates, labels, rendering, tokenizer, questions, deduplication, and gates; only `max_len` changes. |
| Avoids calibration/test targets? | PASS | Proxy group and split are computed from task first; protected rows return before tools or conversations are inspected. |
| 512 reference-only? | PASS | It is measured but absent from `ELIGIBLE_BUDGETS`; selection considers only 640, 768, and 1024. |
| Promotion fully determined? | PASS | Selection is the minimum passing eligible budget, with NONE if no eligible budget passes. |
| Linkage clarification consistent? | PASS | v0.2's written stop condition concerned malformed retained pairs; excluding and counting malformed source candidates satisfies that intent. |
| Result-dependent flexibility? | PASS | No budget additions, threshold changes, larger-budget preference, hardware fallback, training, or evaluation are authorized. |

## Blockers And Resolution

1. **Resolved before freeze:** one-sided call/result wrappers were silently skipped instead of counted. The parser now treats the presence of either wrapper in an adjacent gpt/tool pair as a candidate and excludes/counts incomplete linkage.
2. **Resolved before freeze:** the sweep imported predecessor behavior without fail-closed expected hashes. The protocol and implementation now pin and verify the imported v0.2 script, Laya `common.py`, and both tokenizer files.

Final verdict after resolution: **PASS TO FREEZE AND EXECUTE**. The review identified no remaining result-dependent flexibility.
