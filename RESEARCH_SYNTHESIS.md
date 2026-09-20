# Research Synthesis: Policy Distillation Into Laya

Date: 2026-09-20

## Decision

Proceed to a **local feasibility and data-preflight phase**, not a main training run.

Experiment 001 will use `greghavens/kimi-k3-coding-and-debugging-traces`, not Mind2Web and not a multi-dataset mixture. It is the smallest currently verified corpus that simultaneously provides a named frontier teacher, real cumulative agent prefixes, observable next assistant tool calls, deterministic verification, trajectory-disjoint source splits, a permissive training license, and a size appropriate for an RTX 3060 pilot.

The first target is deliberately narrow: predict `CALL_TOOL` versus `RESPOND_OR_FINISH` and the set of next tool families. It does not predict prose, hidden reasoning, or tool arguments. Parallel calls are represented by one binary `noul` question per tool family, while control mode is a `choice` question.

## What We Know

### Laya

- The pinned base model is `convaiinnovations/laya` revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` with 421,293,830 parameters.
- It is a bidirectional ModernBERT-large encoder plus a learned decision head, not an autoregressive language model.
- It supports typed `choice`, ordinal `score`, and binary `noul` decisions. Options are encoded in the input and scored from option-mask representations.
- The official notebook updates the encoder and decision head. Its objective combines a REINFORCE-style perturbation loss using proper scoring rewards with soft cross-entropy against teacher distributions.
- The official recipe uses two 16 GB T4 GPUs, 1,024 tokens, micro-batch 8 per GPU, gradient accumulation 4, FP16, and gradient checkpointing. It cannot be copied unchanged to one 12 GB card.
- The shipped calibration path is not sufficient for this experiment: the notebook fits temperatures on training examples, and inherited option-count temperatures can override newly fitted per-type temperatures.
- Repository benchmark numbers are author-reported and were not independently reproduced.

### Hardware

- This machine exposes an RTX 3060 with 12,288 MiB and compute capability 8.6.
- Inference should fit comfortably.
- Full fine-tuning at 512 tokens, micro-batch 1, FP16, and checkpointing is plausible but unverified. The fixed FP32 parameter, gradient, and Adam-state lower bound is about 6.74 GB before activations and runtime overhead.
- Local training is therefore **conditionally feasible**. A measured one-step backward pass is a hard gate, not a formality.

### Jev

- TypeSafe's primary announcement is dated 15 September 2026, not merely “September 2026.” It describes Jev as unstructured state in and typed probabilistic decisions out, with parallel outputs and RLCD.
- TypeSafe's architecture, speed, cost, intelligence, and calibration statements remain vendor claims. The public article does not establish that Jev and Laya share an architecture or a reproducible training implementation.
- The article's published workflow evaluation uses Astra and Fable reference probabilities. That is decision imitation/evaluation evidence, not evidence that a small model can learn an agent's longitudinal control policy.

### Data and prior art

- Public datasets commonly support routing, isolated tool choice, judging, or SFT. Complete control-policy records with state, available actions, executed action, resulting observation, and outcome are much rarer.
- The closest verified public precedent is behavioral cloning from agent trajectories. None of the inspected work establishes the proposed combination of a Laya-style typed encoder, calibrated selective control, and frontier-agent next-action traces.
- `greghavens/kimi-k3-coding-and-debugging-traces` currently reports 582 accepted trajectories and 3,956 cumulative next-step rows. Each row ends at one target assistant turn; it identifies Kimi K3 as teacher, `pi` as runtime, includes train/validation splits, and is CC BY 4.0.
- The dataset is success-filtered. It can test imitation of verified behavior but cannot teach or evaluate failure recovery robustly, and teacher agreement is not equivalent to action optimality.
- Dataset previews show tool names but suspiciously empty serialized arguments in sampled rows. The card says arguments/results are retained; this contradiction must be measured over the full pinned data before training.

## Contradictions Resolved

### Mind2Web versus coding traces

The independent experiment report proposed Mind2Web because its candidate elements and cross-website splits make an excellent action-grounding benchmark. That proposal is rejected for Experiment 001:

- its demonstrations are crowdsourced human trajectories, not frontier-agent traces;
- the proposed QLoRA causal-decoder plan is not Laya training;
- HTML candidate grounding is a large additional representation problem;
- it would test web grounding more than the stated frontier-control distillation hypothesis.

Mind2Web remains a possible later domain experiment.

### Full mixture versus one small corpus

The dataset survey recommended mixing Nebius, Kwai-Klear, Hermes, and Exgentic. That is premature for the first falsifiable run. Their action schemas, harnesses, outcome fields, teacher provenance, and licensing obligations differ. A mixture would make a negative result uninterpretable and add substantial normalization work.

### Generative output versus typed decisions

The earlier protocol proposed a serialized action grammar trained with causal cross-entropy. That silently replaced Laya with a generative model. The frozen protocol now maps the observable next control into Laya's actual `choice` and `noul` primitives.

## Dataset Decision

### Selected

`greghavens/kimi-k3-coding-and-debugging-traces`

Reasons:

- named, current teacher and harness;
- verified successful trajectories;
- one row per next assistant step;
- explicit trajectory identity and task-disjoint split;
- small enough to profile and train locally;
- parallel tool calls are retained;
- clear CC BY 4.0 dataset license.

### Not selected first

- `nebius/SWE-agent-trajectories`: superior success/failure metadata and scale, but older, coding-specific, much larger, and teacher/provenance obligations are more complex.
- `Exgentic/agent-llm-traces`: excellent frontier-model and cross-harness coverage, but only 1,781 heavy OpenTelemetry sessions and no documented task reward in the visible schema.
- `lambda/hermes-agent-reasoning-traces`: broad real tool use and many calls, but no terminal outcome field.
- `trace-commons/agent-traces`: only 30 sessions currently.
- `RESMP-DEV/Fable-GPT-5.5-Distillation-Traces`: very large and heterogeneous; the card claims 9.1M records while the current Hub structure exposes about 2.0M rows, so it is not a clean pilot source.
- `greghavens/fable-5-coding-and-debugging-traces`: the exact identifier supplied in the mission did not resolve during live Hub inspection.
- Mind2Web: excellent later grounding benchmark, but not frontier-agent policy distillation.

## Experiment 001

### Hypothesis

A 421M Laya model can learn task-conditioned next-control decisions from verified Kimi K3 trajectories better than frequency and linear baselines, and its held-out probabilities can support selective execution.

### Inputs

- Task text.
- Observable conversation and tool results strictly before the target assistant turn.
- No target turn, hidden reasoning, verifier label, category, model identity, split, or trajectory identifier.
- Context capped at 512 tokens, retaining the task and most recent observations.

### Targets

- `choice`: `CALL_TOOL` or `RESPOND_OR_FINISH`.
- `noul` per frozen tool family: whether that family appears in the next assistant turn.
- Parallel calls remain multi-label rather than being forced into one class.
- Tool arguments and prose are outside the pilot scope.
- Low-confidence predictions abstain after calibration; synthetic escalation labels are not used.

### Splits

- Preserve the source's trajectory-disjoint validation split as the untouched test set.
- Reserve 20% of source-train trajectories for calibration/checkpoint selection using a deterministic group hash.
- Never split individual prefix rows independently.
- Audit trajectory hashes, exact rendered states, and normalized task duplicates before training.

### Smallest useful run

- Hardware smoke: one optimizer step on two upstream typed-decision cases.
- Data/model smoke: at most 100 trajectories and 500 decision rows, one step then at most 100 updates; no official test scoring.
- Main pilot: all eligible rows if at least 1,500 train and 300 test rows survive preflight.

### Baselines

- Per-question majority.
- TF-IDF plus one-vs-rest logistic regression.
- Untuned pinned Laya.
- Fine-tuned Laya.
- Task-only and history-only fine-tuned ablations.

### Frozen promotion criteria

All conditions are required:

1. Complete decision-bundle exact match improves by at least 5 absolute points over the strongest non-Laya baseline, with trajectory-bootstrap 95% CI lower bound above zero.
2. Full-state Laya beats the better of task-only and history-only ablations by at least 3 points.
3. At a calibration-frozen threshold, selective accuracy is at least 80% at at least 50% coverage, with its 95% lower bound at least 75%.
4. Test ECE after held-out temperature fitting is at most 0.10.
5. Leakage and target-marker audits find zero violations.

One seed can provide pilot support only. It cannot establish H1 generally.

### Frozen failure criteria

- The 95% CI upper bound on baseline gain is below +5 points.
- The full-state model fails to beat both input ablations.
- No frozen threshold reaches 80% selective accuracy at 50% coverage.
- Post-calibration ECE exceeds 0.20.
- Too few eligible rows survive, or the one-step run exceeds 11.5 GiB/projected 24 hours.

Leakage or test-guided tuning invalidates the run rather than falsifying the hypothesis. Results between promotion and failure thresholds are inconclusive. Criteria will not be relaxed after results are seen.

## Unknowns Blocking Training

- Exact dataset revision and checksums.
- Full tool-name distribution and the frozen canonical family map.
- Whether tool arguments and tool observations are materially redacted despite the card's claim.
- Exact train/validation trajectory counts and duplicate rates.
- Measured full-model backward-pass VRAM and throughput.
- Whether the upstream code can be adapted cleanly to multiple binary questions per state without repeated-state inefficiency.
- Whether full fine-tuning offers enough gain over head-only or PEFT approaches to justify its cost. LoRA is technically plausible but not the primary run because upstream support and objective interactions are unverified.

## Interpretation Boundaries

A positive result supports only task-disjoint, one-step control imitation inside one coding harness. It does not show broad policy compression, action optimality, closed-loop success, or cross-harness transfer. A negative but valid result would justify testing H2 with a coherent domain only if the failure analysis points to distribution breadth rather than broken data or infeasible training.
