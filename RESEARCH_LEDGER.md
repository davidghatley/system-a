# System-A Research Ledger

This is the authoritative experiment index. Detailed artifacts live under `experiments/` and `artifacts/`.

| Experiment | Date | Hypothesis | Protocol | Dataset | Model | Code Commit | Environment | Hardware | Seeds | Primary Metric | Success Criterion | Result | Interpretation | Decision | Artifacts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `substrate_smoke_001` | 2026-09-20 | One full Laya optimizer step fits and updates intended parameters on RTX 3060 | One forward/backward/step, FP16 scale 1.0, max 512 positions | Synthetic typed decision; no experiment data | `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982` | Pending milestone commit | `.venv`; versions in artifact | RTX 3060 12 GB | `42` | Successful guarded optimizer step under 11.5 GiB | Finite gradients; encoder/scorer update; action head unchanged | Pass: 6.56 GiB reserved, 5.55 s measured region | Local full-parameter substrate is feasible for a one-sequence step; longer stability unknown | Proceed with data work, not training | `research/17_laya_optimizer_smoke.md`, `artifacts/gpu_diagnostics/fp16_scale1_optimizer_step.json` |
| `exp_001` | 2026-09-20 | Laya learns task-conditioned next-control decisions and useful abstention from verified Kimi K3 traces | `0.2.0` stopped | `greghavens/kimi-k3-coding-and-debugging-traces@33a874c3affbdb97e142752a9144e6624ef5bd07` | Not trained | Pending milestone commit | Data preflight environment only | RTX 3060 12 GB | None | Complete decision-bundle exact match | Data viability gates must pass before training | Stopped: 0/3,372 prefixes with prior calls contain authentic tool-result observations | Dataset can test action-sequence imitation, not observation-conditioned control | Reject dataset for this experiment; no test scoring | `research/10_kimi_data_audit.md`, `research/12_kimi_audit_reproduction.md`, `artifacts/exp_001_data_profile.json` |
| `exp_001b` | 2026-09-20 | A coherent Hermes six-tool subset can be converted into leakage-resistant Laya examples | Preprocessing prototype draft; training prohibited | `lambda/hermes-agent-reasoning-traces@b92885e4f0161d4b2536512710e004d4892cac6e`, Kimi shard | No model run authorized | Pending milestone commit | Existing local data/tokenizer | CPU preprocessing | `20260920` split hash | All converter integrity gates | Every assertion and bias/support gate in prototype protocol | Not run | Reconciliation found a candidate 28,220-row subset, but provenance and selection-bias risks remain | Authorize preprocessing prototype only | `research/18_hermes_reconciliation.md`, `experiments/exp_001b_hermes_preprocessing/` |

## Ledger Rules

- Add a row before every serious run.
- Never overwrite a result; append a protocol revision or new experiment.
- Record the actual code commit, environment lock, data revision/checksums, and artifact paths after execution.
- Classify outcomes as support, failure for the setup, inconclusive, invalidated, or feasibility failure.
