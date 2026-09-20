# Experiment 001: Can a ~400M Model Learn Useful Frontier-Agent Action Selection?

Status: preregistration draft. Dataset-dependent quantities marked `UNRESOLVED` must be frozen by the preflight script before training. No test labels or test metrics may be inspected while choosing preprocessing, hyperparameters, thresholds, or stopping checkpoints.

## Claim under test

A roughly 400M-parameter, Laya-style causal policy can learn a useful next-action policy from frontier-agent demonstrations, rather than merely reproducing action frequencies, website strings, or annotation artifacts. Here "useful" means that on unseen websites it selects an admissible target and operation better than non-neural and frozen-model baselines, while exposing calibrated uncertainty that permits escalation.

This experiment is deliberately an offline, one-step policy test. It does **not** establish closed-loop task completion, long-horizon recovery, planning, or that the demonstrated teacher action is optimal.

## Why Mind2Web

Experiment 001 uses the public Mind2Web release, with `osunlp/Mind2Web` as the canonical task/trajectory representation and `osunlp/Multimodal-Mind2Web` only as a convenient flattened/indexed derivative if its records are checksum-consistent with the canonical release.

Verified public schema (20 September 2026):

- Task row: `website`, `domain`, `subdomain`, `annotation_id`, `confirmed_task`, `action_reprs`, `actions`.
- Each ordered action has `action_uid`, pre-action `raw_html`, pre-action `cleaned_html`, `operation`, `pos_candidates`, and `neg_candidates`.
- `operation` contains `op`, `original_op`, and `value`; documented normalized operations are `CLICK`, `TYPE`, and `SELECT`.
- Candidate records contain `tag`, `backend_node_id`, and serialized `attributes`; positives additionally identify original/top-level targets.
- Positive candidates can be empty after preprocessing even when the original target remains in `raw_html`.
- The multimodal derivative has one action per row and additionally exposes `target_action_index`, `target_action_reprs`, and `screenshot`. Its public splits are 7,778 train action rows, 1,338 cross-task, 1,030 cross-website, and 4,060 cross-domain rows according to the Hub viewer at inspection time.
- The canonical card reports 1,009 train tasks and official cross-task (252 tasks), cross-website (177), and cross-domain (reported as 912 in prose) test regimes. These task counts and flattened row counts are not interchangeable.

The text-only path is selected for RTX 3060 12 GB feasibility and to isolate action policy from visual grounding. Screenshots are not model inputs in Experiment 001.

Sources inspected:

- Mind2Web dataset card and schema: https://huggingface.co/datasets/osunlp/Mind2Web
- Multimodal-Mind2Web schema/splits: https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web
- Original paper: https://arxiv.org/abs/2306.06070
- Independent metric implementation description: https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/mind2web/

## Unit and information boundary

The unit is an action step, but every split is made at `annotation_id` (whole trajectory/task). No steps from one trajectory may cross train/validation/test boundaries. Each example includes:

1. `confirmed_task`.
2. Prior gold action representations strictly before the current action, truncated from the left if needed.
3. A deterministic, document-order list of candidates derived only from the current pre-action `cleaned_html` and the released positive/negative candidate records.
4. Candidate text formed from an allowlist of visible/semantic attributes (`tag`, visible text if recoverable, `aria-label`, `title`, `name`, `placeholder`, `role`, `type`, and option text). `backend_node_id`, candidate source (positive/negative), original/top-level flags, `action_uid`, `annotation_id`, website/domain labels, and target indices are never rendered to the model.

Candidates are assigned ephemeral indices after deterministic ordering. The output grammar is one of:

```text
ACT(candidate_index, CLICK)
ACT(candidate_index, TYPE, value)
ACT(candidate_index, SELECT, value)
ESCALATE(reason_code)
```

Maximum context length, candidate cap, HTML-to-candidate extraction rules, and truncation direction are `UNRESOLVED_PREFLIGHT`. The selected target must remain present when constructing an answerable training example; examples where truncation removes every admissible target are labeled `ESCALATE(target_not_observable)`, never `ACT`.

## Target hierarchy

The policy is evaluated hierarchically rather than treating a serialized teacher string as one class:

1. **Decision:** `ACT` versus `ESCALATE`.
2. **Grounding:** if acting, select a candidate element.
3. **Operation:** select `CLICK`, `TYPE`, or `SELECT` conditional on the element.
4. **Argument:** emit the value for `TYPE`/`SELECT`; no argument for `CLICK`.

Training uses masked causal cross-entropy over the canonical output only. The demonstrated operation/value supplies supervision. For grounding, any released `pos_candidates` member retained in the candidate set is a valid target; one deterministic representative is serialized for SFT, but evaluation gives credit to the entire admissible set. A sensitivity run serializes the original target when available versus a deterministic top-level positive. Divergence is reported, not selected post hoc.

### Teacher action is not synonymous with good action

Mind2Web records one crowdsourced trajectory. It neither proves uniqueness nor optimality, and alternative actions may make progress. Therefore:

- Call the label a **demonstrated action**, not an optimal action.
- Element correctness is set-valued over released positives.
- Exact operation/value agreement measures demonstration imitation.
- A blinded manual audit labels sampled predictions as `good`, `bad`, or `indeterminate` given task, history, and state. A good action must be plausibly progress-making and not introduce irreversible harm; it need not match the teacher.
- Report both benchmark imitation and audited good-action rate. Do not use audit outcomes to tune Experiment 001.

Audit size is 200 uniformly sampled valid predictions from cross-website test plus all disagreements between the trained policy and the strongest baseline up to 200 additional cases. Two raters, blinded to system identity and teacher action; adjudicate disagreements. Exact sample size may only change before training and is `UNRESOLVED_HUMAN_BUDGET`. Report Cohen's kappa and binomial confidence intervals. `indeterminate` is excluded from the main good-action denominator and included as bad in a conservative sensitivity analysis.

## Abstain and escalate

`ESCALATE` means hand control to a stronger policy/human; it is not task success. Reasons are `target_not_observable`, `ambiguous`, `unsupported_operation`, and `low_confidence`.

Training abstention examples may include natural empty-positive/truncated-target states and task-state mismatches. Synthetic mismatches are capped at 20% of training instances, generated within the same website and operation-frequency stratum, with task text length matched, to reduce obvious artifacts. They are training aids only and cannot validate abstention.

The genuine selective-policy evaluation uses a frozen, blinded human annotation of 300 examples sampled from official cross-website states before model scoring: 200 ordinary released states and 100 deliberately enriched from empty-positive, severe-truncation, and candidate-ambiguity strata. Annotators answer whether at least one displayed candidate/action is safely progress-making. This yields `ACTABLE`, `SHOULD_ESCALATE`, or `INDETERMINATE`. Sampling weights restore prevalence for aggregate reporting. Size and strata are `UNRESOLVED_HUMAN_BUDGET`; absent this annotation, abstention results are exploratory and cannot satisfy the support criterion.

The confidence score is the normalized probability of the best valid `ACT` serialization versus `ESCALATE`, computed with constrained decoding. The escalation threshold is selected once on internal validation to minimize escalation subject to a lower 95% confidence bound of 80% selective step success. It is frozen for all official tests.

## Splits and leakage controls

- Training pool: official Mind2Web train tasks only.
- Internal split: group by `annotation_id`, seeded hash (`sha256(annotation_id + seed)`), 80% train / 20% validation. If near-duplicate task templates span groups, connected components under normalized-task MinHash/Jaccard threshold `UNRESOLVED_PREFLIGHT` are assigned together.
- Primary generalization test: official **cross-website**, where websites are unseen in training. This is the required genuine generalization test.
- Secondary tests: official cross-task and cross-domain, reported without affecting the primary decision.
- No random step split. No train/validation row duplication. No fitting vocabulary, candidate cap, class weights, calibration, thresholds, or checkpoints on official tests.
- Hash normalized task text, raw/cleaned HTML, candidate renderings, and full examples. Report exact and near-duplicate overlap by split. Remove train examples that exactly duplicate any official test input without removing test examples; record counts.
- Do not expose `website`, `domain`, `subdomain`, IDs, target flags, positive/negative list membership, or `target_action_index` in prompts.
- Strip/normalize volatile timestamps, session IDs, UUID-like strings, and backend IDs using rules frozen from train/validation only. Report an ablation with task text and candidate order independently shuffled on validation to detect shortcuts.

## Imbalance and artifacts

Report counts by operation, website, domain, trajectory length quartile, step position quartile, candidate-count quartile, target rank, positive count, and escalation reason. Main training samples tasks uniformly, then steps uniformly, preventing long trajectories from dominating. No operation reweighting in the primary run; a class-balanced sampler is diagnostic only.

Artifact probes, all frozen before test scoring:

- Candidate-only model (no task/history).
- Task-only operation model (no state/candidates).
- Candidate order randomized at evaluation while preserving labels.
- Candidate text masked to tag only.
- Website/domain-name regex scan and ID/target-marker canary scan.
- Synthetic-escalation discriminator evaluated on synthetic versus natural human-labeled escalation cases. High separability is evidence that synthetic negatives are unsuitable as evaluation evidence.

## Model and training budget

The exact "Laya-style" checkpoint/configuration is not present in the local mission context and is `UNRESOLVED_MODEL_ID`. Before training, freeze a public or local causal decoder in the 300M-600M range and document architecture, license, tokenizer, parameter count, pretraining contamination statement, and checksum. If Laya denotes a specific architecture rather than a project shorthand, Experiment 001 must not silently substitute another model.

Feasible primary plan for a confirmed RTX 3060 12 GB:

- QLoRA/NF4 base weights, bf16 if supported otherwise fp16, gradient checkpointing, paged 8-bit optimizer.
- LoRA on attention and MLP projections; rank 16, alpha 32, dropout 0.05.
- Effective batch size 32 via microbatch 1 and accumulation 32.
- Maximum 3 epochs and 2,000 optimizer updates; evaluate every 100 updates.
- Select checkpoint by validation macro step success, tie-broken by lower validation NLL. One seed (`20260920`) for the feasibility gate; three seeds are required before a robust scientific claim.
- Hard limits: peak allocated VRAM <= 11.5 GB and wall-clock <= 24 hours. Preflight a 100-update run; if it exceeds either projection, reduce context/candidate cap before training and freeze the change.

Learning rate (`UNRESOLVED_PREFLIGHT`, grid of at most two values), exact checkpoint, context length, and hardware identity must be resolved in `protocol.json` before any official test inference.

## Baselines

All baselines receive exactly the same rendered inputs and candidates.

1. **Frequency:** validation-majority operation plus first candidate; values copied only when a unique obvious option/text argument exists under the same deterministic parser.
2. **Lexical retrieval:** BM25/token-overlap between task plus history and each candidate; operation chosen by train conditional frequency from candidate tag/attributes; deterministic argument extraction.
3. **Frozen base:** selected ~400M checkpoint, constrained zero-shot decoding, no fine-tuning.
4. **Candidate-only trained ablation:** same adapter budget, task/history removed.
5. **Oracle candidate upper-bound diagnostic:** gold positive element supplied; predicts operation/value only. This is not a competing policy.

The strongest non-oracle baseline is chosen by validation macro step success before official test evaluation.

## Metrics

Compute per-step and task-cluster-bootstrap 95% confidence intervals (10,000 resamples of `annotation_id`). Report micro and task-macro versions where meaningful.

Primary:

- **Step success rate:** admissible element AND correct operation/value, following Mind2Web convention. Exact value normalization rules are frozen in preflight.
- **Element accuracy:** selected candidate is any retained released positive.
- **Operation F1:** token-level F1 for operation/value representation, plus operation-type macro-F1 separately.
- **Task success proxy:** all evaluated steps in a recorded trajectory are successful. Explicitly not closed-loop task success.

Selective safety/calibration:

- Coverage, selective step success (risk-coverage curve), AURC, and risk at 50%, 75%, and 90% coverage.
- Escalation precision/recall/F1 and balanced accuracy against human `SHOULD_ESCALATE` labels.
- AUROC and AUPRC for predicting step failure.
- NLL, Brier score, expected calibration error (15 equal-mass bins), and maximum calibration error.
- Utility at escalation costs 0.05, 0.20, and 0.50, with correct act = +1, incorrect act = -1, escalation = negative cost. These are sensitivity analyses, not the primary endpoint.

Hierarchy and diagnostics:

- Decision accuracy; grounding accuracy conditional on `ACTABLE`; operation accuracy/F1 conditional on correct grounding; normalized exact-match and character-F1 for values.
- Good-action rate from blinded audit; teacher-match rate; rates of good non-teacher actions and bad teacher-matching actions.
- Metrics by operation, website/domain, trajectory position, candidate count, target rank, task length, and natural versus synthetic escalation origin.
- Invalid-output rate, forced-decoding repair rate, latency, tokens/second, peak VRAM, and train wall-clock.

Primary comparison uses paired task-cluster bootstrap difference in cross-website step success between the trained policy and strongest non-oracle baseline. Multiple secondary comparisons use Holm correction. No significance claim rests on action rows treated as independent.

## Preregistered decision logic

### Support

Experiment 001 supports the narrow claim only if all conditions hold on untouched cross-website test:

1. Trained policy step success exceeds the strongest non-oracle baseline by at least 5.0 absolute percentage points and the paired task-bootstrap 95% CI lower bound is above 0.
2. Trained policy element accuracy and operation-type macro-F1 each exceed the strongest non-oracle baseline; neither paired 95% CI lower bound may be below -1.0 point.
3. At the validation-frozen threshold, coverage is at least 50% and the 95% lower confidence bound of selective step success is at least 80% on the human-labeled cross-website subset.
4. Human-audited good-action rate is at least 60%, its 95% CI lower bound exceeds the strongest baseline's point estimate minus 5 points, and at least 10 audited predictions are good non-teacher actions (otherwise that last descriptive claim is underpowered).
5. The full policy exceeds the candidate-only trained ablation by at least 5 points in cross-website step success, guarding against candidate/order artifacts.
6. No leakage canary, duplicate audit, or prompt-field audit finds target-derived information in model inputs.

With one training seed this is **pilot support**, not robust confirmation. Robust support requires all primary-direction results over three preregistered seeds with the mean satisfying condition 1 and no seed underperforming the strongest baseline.

### Falsification / failure to support

The useful-policy claim is falsified for this setup if condition 1 fails with the 95% CI upper bound below +5 points, or if the full model fails to beat the candidate-only ablation (CI upper bound <= 0), indicating no demonstrated task-conditioned gain. It also fails if selective success is below 80% at every coverage >= 50%, or audit good-action rate has a 95% upper bound below 60%.

Any leakage, test-guided tuning, broken grouping, or target-marker artifact invalidates rather than falsifies the experiment. OOM/time-budget failure is feasibility failure. Intermediate outcomes are explicitly inconclusive, not positive.

## Execution order

1. Resolve every blocking field in `protocol.json`; inventory checksums, exact schema, operation/value frequencies, and hardware.
2. Freeze extraction, grouping, duplicate rules, label normalization, model, hyperparameters, prompts, and audit forms. Write split IDs and hashes.
3. Run leakage/artifact tests and 100-update hardware preflight.
4. Train baselines and model using train/internal validation only; freeze checkpoint and threshold.
5. Generate all official-test predictions once and store immutable predictions before scoring.
6. Conduct blinded human audit, then score and bootstrap.
7. Publish exclusions, failures, all subgroup metrics, and protocol deviations.

## Experiment 002 (concept only; do not execute)

If Experiment 001 reaches pilot support, Experiment 002 tests whether offline one-step gains cause closed-loop gains. Deploy the frozen policy in a reproducible browser environment (candidate: BrowserGym/MiniWoB++ or a version-pinned WebArena subset), compare policy-only, policy-with-escalation to a stronger teacher, frozen base, and teacher-only under equal action/token budgets, and score verifier-based task completion, unsafe/irreversible actions, escalation cost, recovery after injected action errors, latency, and cost. Use task-family-held-out templates plus procedurally perturbed layouts as generalization tests. Randomize task order, run at least three seeds, record complete trajectories, and preregister non-inferiority safety plus superiority in cost-adjusted completion. Environment/version, task suite, stronger teacher, and transfer mapping from Mind2Web actions are intentionally `UNRESOLVED`; resolving them belongs to Experiment 002 design, not Experiment 001.
