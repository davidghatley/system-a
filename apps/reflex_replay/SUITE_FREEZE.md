# Iteration 3 replay suite freeze

Frozen before any replay/showcase execution on 2026-09-22, starting from commit
`2d4408d`. The task is **observed next-action prediction/evaluation** on real
Trace2Decision v2 compact-state records.

## Selection rule

1. Include the accepted Iteration 2 integration row because it is the required
   interoperability evidence and a known specialist failure.
2. Starting after that row in `dev.jsonl`, include the first record whose
   observed label equals the training-split majority (`execute`).
3. Include the immediately following record when its observed label differs
   from that majority and it has omitted history. This keeps a simple-rule
   failure and a lossy-context case visible.

This rule selected these records, in fixed order:

| Case | Record ID | Source line | Observed label | Case tags |
| --- | --- | ---: | --- | --- |
| accepted-integration | `t2d-i2-4bba28e1b18498c2d153a95a223fc411` | 1 | `search` | known specialist failure, low-margin prediction |
| majority-success | `t2d-i2-1ea82a027c0f8bb26a6dadd308748d7a` | 3 | `execute` | ordinary, simple-rule success |
| majority-failure | `t2d-i2-23900e0db0295335a7f2487b0acdcf96` | 4 | `edit` | simple-rule failure, omitted-history context |

The first row's result was known because it was an explicit required input.
The other cases were selected by the rule above, not model output. No case may
be removed from the default display because it is unfavorable.

## Reference and ambiguity definitions

- References are observed source-trace actions, not human judgments,
  recommendations, task-success labels, or optimal actions.
- A probabilistic prediction is displayed as `ambiguous/low-margin` when the
  top-two probability margin is at most `0.05`. This is a display threshold,
  not a calibrated uncertainty guarantee.
- The source dataset is machine-generated and publisher-attributed. Physical
  execution, teacher identity, action optimality, task success, and source/model
  overlap are unverified or unknown.
- The suite is a fixed diagnostic example set, not an accuracy estimate.
