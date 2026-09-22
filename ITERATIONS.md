# System-A Iterations

## Iteration 1

- **Goal:** Ship a useful Reflex demo, prove bounded local Laya learning, and produce one clean trace-derived typed-decision dataset.
- **Work performed:** Implemented, reviewed, repaired, and independently verified Reflex; implemented and verified a bounded local trainer; audited and converted pinned GLM-5.2 traces.
- **Accepted deliverables:** Reflex API/CLI and offline demo; shared typed-decision validator; Trace2Decision converter and 1,544-record trajectory-grouped derivative.
- **Failed criteria:** Local Trainer native-label choice accuracy regressed from 47.5% to 27.5%, so the frozen +5-point learning criterion failed despite improved NLL.
- **Important measurements:** Reflex median/p95 39.689/40.527 ms on RTX 3060; Trainer 5.508 GiB peak reserved and 39.784 s total bounded run; Trace2Decision 1,544 decisions across 207 trajectories with zero split crossings.
- **Commit SHA:** `936ddb8` (deliverables and evidence).
- **Next options:** Reflex sharing; a separately frozen Trainer retry; Trace2Decision training only after Trainer learning is accepted.

## Iteration 2

- **Goal:** Connect traces, shared validation, Laya item construction, local training, and held-out inference through one verified pipeline.
- **Work performed:** Repaired Reflex around the pinned specialist/shared contract; validated the evaluator with a public base/specialist positive control; implemented and ran one frozen hardware-adapted RLCD retry; produced and verified Trace2Decision v2; completed one parent-owned cross-track repair and integration verification.
- **Accepted deliverables:** Reflex specialist path; shared typed-decision contract with native string-state and metadata support; public positive-control evaluator; accepted B2 checkpoint and evidence; deterministic 1,544-row Trace2Decision v2 derivative; reusable parent integration verifier.
- **Failed criteria:** None. The first parent integration execution exposed rounded-probability and native-string/provenance boundary defects; these were repaired within the one allowed cross-track cycle and the unchanged row then passed. Both models missed that row's action label, which is an accuracy limitation rather than an interoperability criterion.
- **Important measurements:** B1 specialist/base choice 68.0%/35.5%; B2 choice 36.0% to 70.5% (+34.5 points), all-question 38.8% to 73.0%, native-gold NLL 1.5310 to 0.6894, 80 updates, 8.670 GiB peak reserved, 772.8 s total; Trace2Decision v2 1,544 rows with zero split crossings and max 512 tokens; specialist benchmark median/p95 40.150/40.977 ms.
- **Commit SHA:** `7d2caf6` (deliverables and evidence).
- **Next options:** User review. Trace2Decision training and Iteration 3 remain unauthorized.
