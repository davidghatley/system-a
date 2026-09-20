# Phase 1-4 Dependency Graph

## Critical Path

```text
A1 architecture + A2 training + A5 calibration audit
                         |
                         v
             A4 one-step GPU measurement
                         |
                         +-------------------+
                                             |
B1 dataset schema + provenance               |
             |                               |
             v                               |
B2 integrity/leakage profile                 |
             |                               |
             v                               v
B3 state renderer + typed target map --> D4 protocol reconciliation
                                             |
C1 prior-art refresh ------------------------+
D1 independent design review ----------------+
D2 statistical critique ---------------------+
D3 leakage critique -------------------------+
                                             |
                                             v
                                  freeze exp_001 protocol
                                             |
                                             v
                                  baselines on train/cal
                                             |
                                             v
                                  bounded 100-update smoke
                                             |
                                             v
                                      authorization gate
                                             |
                              +--------------+--------------+
                              |                             |
                             stop                    one-seed main pilot
                                                            |
                                                            v
                                           independent result reviews
                                                            |
                                                            v
                                           Phase 4 compression boundary
```

## Independent Work Available Now

| ID | Work | Output | Depends On |
|---|---|---|---|
| A1/A2/A5 | Re-audit pinned Laya architecture, loss, checkpoint loading, and calibration behavior | `research/05_laya_substrate_audit.md` | Existing source pins only |
| A4 | Build and run the smallest legitimate single-GPU forward/backward measurement | script plus `artifacts/laya_one_step.json` | A1/A2 may refine implementation; environment setup can start now |
| B1/B2 | Pin and profile Kimi K3 data, including arguments, observations, splits, duplicates, and action distribution | profiler plus `artifacts/exp_001_data_profile.json` | Dataset access only |
| C1 | Refresh closest prior art using primary 2025-2026 sources | `research/06_prior_art_update.md` | None |
| D1/D2 | Independently critique target, metrics, thresholds, and statistical power | `research/07_exp001_design_review.md` | Current protocol |
| D3 | Adversarial leakage and shortcut review | `research/08_exp001_leakage_review.md` | Current protocol and dataset schema |

## Sequential Work

- Do not freeze the tool-family map before B1/B2.
- Do not implement final rendering before the data profile establishes observable fields.
- Do not fit baselines before split manifests and rendering are frozen.
- Do not run the 100-update model smoke before the one-step memory gate.
- Do not inspect sealed-test metrics before checkpoint and calibration threshold are frozen.
- Do not design Phase 5 implementation before Phase 4 synthesis.

## Stop Gates

- Stop Experiment 001 if usable observations are mostly absent, split integrity fails, or minimum row counts are not met.
- Stop full fine-tuning if one-step peak reserved memory exceeds 11.5 GiB or projected runtime exceeds 24 hours after preregistered length reductions.
- Invalidate, rather than reinterpret, any run with target leakage or test-guided tuning.
