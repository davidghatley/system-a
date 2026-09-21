# System-A Project State

Updated: 2026-09-20

## Current Phase

Phase 1 Laya single-sequence optimizer smoke passed on RTX 3060. The bounded exp_001b preprocessing correction is complete and independently verified. No training or official evaluation is authorized.

## Current Hypothesis

A 421M Laya model can learn task-conditioned next-control decisions from verified frontier-agent traces better than trivial and linear baselines, while calibrated confidence supports useful abstention.

## Active Work

- `exp_001` Kimi K3: stopped before training because the pinned corpus contains no authentic tool-result observations.
- `f2e1666` exp_001b: **INVALIDATED — implementation error** (silent truncation, incorrect canaries, unamended representation). Its FAIL neither supports nor rejects Hermes.
- Corrected `exp_001b` v0.2.0: **FAIL**. It retained 28,930 rows from 5,322 trajectories and 643 proxy groups after 33,610 over-budget and 639 duplicate exclusions. Gates failed on 29 malformed linkage candidates and a 20.7136-point `read_file` positive/negative over-budget-rate difference (limit 20 points). Fail-closed behavior wrote no model-ready splits. Independent review found no implementation defect.
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
- Corrected upstream-Laya preprocessing found 63,179 structurally/target-valid candidates, 33,610 over-budget exclusions, 29,569 representable candidates, 639 duplicates, and 28,930 retained diagnostic rows. This is a valid preprocessing FAIL, not evidence that Hermes is intrinsically unsuitable.

## Rejected Or Superseded

- Mind2Web as Experiment 001: human demonstrations and a grounding-heavy task do not directly test frontier-agent control distillation.
- A multi-dataset mixture for the first run: schema and provenance confounds would make failure difficult to interpret.
- Generative QLoRA as a substitute for Laya: incompatible with the stated substrate question.
- Kimi K3 as the Experiment 001 control dataset: action history is present but environment observations are absent.

## Unresolved Questions

- Whether a new protocol should study the 29 malformed linkage records or accept the measured `read_file` selection distortion; no revision is authorized by this result.
- Is template-proxy-disjoint within-corpus imitation scientifically useful enough to justify training without repository/task identities?
- Can embedded-content rights and publisher-attributed execution provenance support the intended private research use?
- Is FP16 scale-1 training stable beyond one optimizer step?
- Are the current promotion criteria statistically and operationally defensible?

## Next Actions

1. Stop this repair. Preserve v0.2.0 FAIL evidence and the independent verification.
2. Do not design training: corrected preprocessing did not pass every gate.
3. Any later remediation requires an explicit new preprocessing protocol; criteria must not be relaxed post hoc.
