# System-A Iterations

## Iteration 1

- **Goal:** Ship a useful Reflex demo, prove bounded local Laya learning, and produce one clean trace-derived typed-decision dataset.
- **Work performed:** Implemented, reviewed, repaired, and independently verified Reflex; implemented and verified a bounded local trainer; audited and converted pinned GLM-5.2 traces.
- **Accepted deliverables:** Reflex API/CLI and offline demo; shared typed-decision validator; Trace2Decision converter and 1,544-record trajectory-grouped derivative.
- **Failed criteria:** Local Trainer native-label choice accuracy regressed from 47.5% to 27.5%, so the frozen +5-point learning criterion failed despite improved NLL.
- **Important measurements:** Reflex median/p95 39.689/40.527 ms on RTX 3060; Trainer 5.508 GiB peak reserved and 39.784 s total bounded run; Trace2Decision 1,544 decisions across 207 trajectories with zero split crossings.
- **Commit SHA:** `936ddb8` (deliverables and evidence).
- **Next options:** Reflex sharing; a separately frozen Trainer retry; Trace2Decision training only after Trainer learning is accepted.
