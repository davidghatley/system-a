# System-A Project State

Updated: 2026-09-20

## Current Phase

Phase 1 substrate smoke is complete. Phase 2 is selecting and validating an observation-bearing trace source. Phase 3 remains protocol design only; no serious training run is authorized.

## Current Hypothesis

A 421M Laya model can learn task-conditioned next-control decisions from verified frontier-agent traces better than trivial and linear baselines, while calibrated confidence supports useful abstention.

## Active Work

- `exp_001` Kimi K3: stopped before training because the pinned corpus contains no authentic tool-result observations.
- `exp_001b` Hermes preprocessing prototype: authorized for deterministic conversion and integrity checks only.
- Proposed prototype target: six binary tool decisions over one fixed toolset; control mode is derived from the set.
- Main training, calibration, and official evaluation remain prohibited.

## Frozen Decisions

- All work, downloads, caches, environments, and durable evidence remain inside the repository; external temporary or parent-directory access is prohibited.
- Observable control behavior only; hidden reasoning is excluded.
- Local RTX 3060 execution is preferred and must be measured before cloud use.
- Splits are by trajectory/task, never individual prefix rows.
- Calibration uses a held-out calibration partition, not training or sealed test data.
- Main training is prohibited until dataset, representation, manifests, hardware fit, and baselines are frozen.
- Promotion and failure thresholds in `experiments/exp_001/protocol.json` cannot be relaxed after results.

## Confirmed Findings

- Laya has 421,293,827 trainable parameters plus a three-element temperature buffer; it is a ModernBERT-large encoder plus typed decision head, not a causal decoder.
- The public training recipe updates encoder and head with RLCD-style perturbation loss plus soft cross-entropy.
- Official 2xT4 settings cannot run unchanged on the 12 GB RTX 3060.
- A guarded single optimizer step at 512 positions completed locally with FP16 scale 1.0, 6.56 GiB peak reserved VRAM, finite gradients, encoder/scorer updates, and unchanged action head.
- Default FP16 GradScaler initialization overflowed on the same step; scale 1.0 is required for the measured recipe and longer-run stability remains unknown.
- The pinned Kimi K3 release has 582 trajectories and 3,956 rows, but independent audits found zero authentic tool-result messages.
- A reconciled Hermes six-tool subset can yield 28,220 unique representable targets from 5,300 trajectories for preprocessing research, but provenance, semantic task identity, rights, and selection bias remain unresolved.

## Rejected Or Superseded

- Mind2Web as Experiment 001: human demonstrations and a grounding-heavy task do not directly test frontier-agent control distillation.
- A multi-dataset mixture for the first run: schema and provenance confounds would make failure difficult to interpret.
- Generative QLoRA as a substitute for Laya: incompatible with the stated substrate question.
- Kimi K3 as the Experiment 001 control dataset: action history is present but environment observations are absent.

## Unresolved Questions

- Can the Hermes preprocessing prototype satisfy all canary, exclusion-ledger, split, support, and selection-bias gates?
- Is template-proxy-disjoint within-corpus imitation scientifically useful enough to justify training without repository/task identities?
- Can embedded-content rights and publisher-attributed execution provenance support the intended private research use?
- Is FP16 scale-1 training stable beyond one optimizer step?
- Are the current promotion criteria statistically and operationally defensible?

## Next Actions

1. Implement the bounded Hermes converter, exclusion ledger, manifests, and canary tests without model execution.
2. Quantify 512-token selection bias by split, proxy group, source category, and label.
3. Decide whether the narrowed within-corpus claim has enough value and power to justify a new training protocol.
4. Resolve statistical review blockers and baseline definitions before any protocol freeze.
5. Run no training until a new immutable protocol explicitly passes every gate.
