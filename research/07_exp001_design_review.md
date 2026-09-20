# Experiment 001 Independent Design and Statistical Review

Date: 2026-09-20

## Verdict

Experiment 001 can become a valid pilot of **task-disjoint, offline next-control imitation in one coding harness**, but the current protocol is not ready to freeze. The main blockers are not model implementation: the target is partly redundant, the primary complete-bundle metric is dominated by easy negatives and undefined across changing tool taxonomies, the proposed bundle confidence is not a valid joint probability, calibration has no single estimand, threshold selection has no selection-aware guarantee, and minimum sample sizes are stated in correlated rows rather than independent trajectories.

Even after repair, success would support only this claim:

> On held-out tasks from the same Kimi K3/pi data-generating process, the trained Laya checkpoint predicts the recorded next control bundle more accurately than prespecified frequency and linear predictors, and a separately calibrated selection rule controls the empirical risk of accepted bundle predictions at a prespecified coverage.

It would not establish a useful executable policy, calibrated probabilities for alternative good actions, frontier-policy compression generally, closed-loop success, recovery, action optimality, or transfer beyond this harness. Success-filtering and one observed action per state make those stronger claims unidentifiable.

## Blocking Amendments

These must be resolved and frozen before training or official test inference. They do not require editing the current `protocol.json` during this review.

### 1. Replace the redundant target with one coherent structured label

`control_mode` is deterministically implied by the tool-family vector if every tool call belongs to exactly one frozen family:

\[
Y_0=\mathbb{1}\{\sum_{j=1}^K Y_j>0\}.
\]

Training and scoring both representations double-count the same fact and permit impossible predictions such as `RESPOND_OR_FINISH` plus `call_shell=yes`, or `CALL_TOOL` with no family. The protocol does not say how such outputs are repaired or scored.

Amendment:

- Define the primary label as the canonical set `RESPOND_OR_FINISH` or a nonempty set of frozen tool families.
- Derive control mode from that set; retain separately predicted control mode only as a diagnostic auxiliary head.
- Freeze behavior for unknown, malformed, duplicate, and zero-family tool calls. An unknown family must not silently become `RESPOND_OR_FINISH`.
- Freeze the family map using source-train data only. Map test-only names to a prespecified `OTHER_TOOL` family and report their rate; do not revise the taxonomy after seeing test names or counts.
- Enforce coherence before all model and baseline scoring using one prespecified rule, preferably decoding the highest-probability valid bundle rather than repairing independent argmaxes after the fact.

### 2. Make the primary estimand resistant to all-negative dominance

For `K` tool families, an example with one called family contributes `K-1` easy negative labels. Complete-bundle exact match avoids per-label inflation but remains strongly determined by the prevalence of `RESPOND_OR_FINISH`, single-family calls, and rare families. It gives the same zero credit to one extra family and a completely wrong control decision, and it collapses same-family multiplicity despite claiming parallel-call prediction. Its value is also not comparable if preflight changes `K`.

Amendment:

- Keep bundle exact match as the primary **imitation** endpoint only after `K`, `OTHER_TOOL`, and coherence rules are frozen.
- Define the primary population estimand explicitly as trajectory-macro accuracy:

\[
\widehat A_{\mathrm{traj}}=\frac{1}{G}\sum_{g=1}^{G}\left(\frac{1}{n_g}\sum_{i=1}^{n_g}\mathbb{1}\{\hat B_{gi}=B_{gi}\}\right),
\]

  so long trajectories do not dominate. Report row-micro accuracy secondarily.
- Co-primary descriptive decomposition: mode balanced accuracy; exact family-set match conditional on a tool call; sample-averaged Jaccard score conditional on a tool call; and per-family precision/recall with support. These expose an apparent gain obtained only from `RESPOND_OR_FINISH` or common families.
- Report same-family call multiplicity separately and remove any claim that the primary target preserves that aspect of parallelism.
- Freeze whether examples with unrepresentable labels are excluded. Report exclusions by split and trajectory; exclusion cannot depend on model output.

Exact match is appropriate for teacher agreement, not action quality. Rename `teacher agreement` to `recorded-action agreement`; with one label it is not an independent metric.

### 3. Do not call products of marginal probabilities bundle probabilities

The Laya questions are encoded and scored separately. Marginal outputs `p(Y_j=1|X)` do not identify

\[
P(B=\hat B\mid X),
\]

because family calls and control mode are dependent. Multiplying selected marginals assumes conditional independence and, with the redundant mode question, counts the same evidence twice. Taking the minimum or mean also does not produce a joint probability.

Amendment:

- Freeze a scalar raw confidence score without probabilistic overclaim, for example the coherent bundle's summed log marginal score excluding the redundant mode head.
- On data not used to train or choose the checkpoint, fit a one-dimensional calibrator from that score to `Z=1{predicted bundle is exactly correct}`. The resulting `q(X)` has the operational interpretation `P(Z=1 | score)` under exchangeability; it is not a full joint distribution over bundles.
- Report question-level probabilities separately. For mode, use binary NLL/Brier/reliability. For each family, use binary NLL/Brier and prevalence-weighted and macro summaries. Do not pool all family-question pairs into one ECE because that obscures rare-family failures and treats dependent questions as independent.
- If the scientific claim truly requires a joint distribution, replace the independent heads with an explicitly normalized structured model over valid observed bundles. That is a different model design and should be preregistered, not inferred post hoc.

Hard one-hot labels are legitimate for behavioral cloning but do not reveal the teacher's latent probabilities or whether another action would be good. Proper scoring rules are proper for the distribution of **recorded labels conditional on the observed state**, not automatically for action quality. Wording about teacher probabilities must be changed to empirical one-hot action labels. Label smoothing or fabricated soft targets would not solve this identification problem.

### 4. Define calibration and selective risk before choosing a threshold

`ECE <= 0.10` is presently ambiguous: it could refer to mode confidence, all marginal answers, the highest marginal, or bundle correctness. ECE is bin-dependent, biased, and can look small under imbalance. It is not a sufficient promotion criterion [Nixon et al., 2019; Kumar et al., 2019].

Amendment:

- Primary operational calibration target: bundle-correctness score `q`; report test Brier score, log loss with frozen clipping, a 10-bin equal-mass reliability table, ECE as descriptive only, and the calibration intercept/slope.
- Secondary calibration targets: mode and each supported family as specified above. State exactly which target the temperature acts on. A single temperature on separate heads does not calibrate bundle correctness.
- Replace “coverage at a threshold reaching 80% accuracy” with selective risk. For threshold `t`, define

\[
C(t)=G^{-1}\sum_g n_g^{-1}\sum_i \mathbb{1}\{q_{gi}\ge t\},
\]

\[
R(t)=\frac{\sum_g n_g^{-1}\sum_i \mathbb{1}\{q_{gi}\ge t\}(1-Z_{gi})}
{\sum_g n_g^{-1}\sum_i \mathbb{1}\{q_{gi}\ge t\}},
\]

  with the same trajectory-macro weighting as the primary estimand. Accuracy is `1-R(t)`.
- Freeze tie handling (`q=t` accepted), behavior when no case is accepted, and interpolation conventions for the risk-coverage curve and AURC.
- Do not use the same calibration cases to fit temperature/calibrator, compare checkpoints, and search thresholds without adjustment. Preferred split: train-fit; development for checkpoint/hyperparameters; calibration-fit for temperatures/score calibrator; calibration-select for the threshold. All are trajectory-disjoint. If data are too small, use cross-fitting within source-train for score calibration, then reserve a final trajectory-disjoint threshold set.
- Select the largest-coverage threshold whose **simultaneous one-sided 95% upper confidence bound** on risk is at most 0.20. Pointwise intervals after searching thresholds are anti-conservative. Predefine a finite threshold grid and obtain a simultaneous clustered bootstrap band using the maximum studentized deviation, or use a distribution-free risk-control procedure with familywise correction [Geifman and El-Yaniv, 2017; Angelopoulos et al., 2021].
- On the sealed test, evaluate the frozen threshold once. Promotion requires `coverage >= 0.50` and a one-sided 95% upper bound for risk `<= 0.25` only if that weaker 75% guarantee is genuinely the claim. If the claim is 80% selective accuracy, require the test upper risk bound `<= 0.20`; the current point estimate 80% plus lower bound 75% does not support an 80% guarantee.

Selection should be based on the primary bundle-correctness event, not a mixture of per-question correctness events. Risk-coverage curves and AURC remain useful secondary summaries [El-Yaniv and Wiener, 2010; Ding et al., 2020].

### 5. Use a paired trajectory bootstrap and define every interval

“Trajectory-cluster bootstrap” is directionally correct but incomplete. The independent sampling unit is a source trajectory/task, while rows within it are repeated cumulative prefixes. Resampling rows would be pseudoreplication. Resampling trajectories while computing row-micro accuracy changes the estimand toward long trajectories.

Amendment:

- Resample the `G` sealed-test trajectories with replacement and retain every eligible row from each sampled trajectory.
- For model comparisons, use the same sampled trajectory indices for both systems and compute the paired difference on every replicate.
- Use the trajectory-macro statistic above as primary. Publish row-micro results with the same cluster resampling as secondary.
- Freeze interval construction. Use percentile 95% paired cluster-bootstrap intervals with 10,000 replicates and seed `20260920`; report achieved quantiles. Avoid BCa unless its cluster-level jackknife is explicitly implemented and stable.
- For a frozen selective threshold, recompute both numerator and denominator inside each trajectory resample. Report intervals for risk and coverage separately; a risk interval conditional on the observed accepted count does not capture coverage variation.
- Do not bootstrap ECE as if bins were fixed independently of predictions. Freeze bin edges from calibration for test reliability summaries, or recompute equal-mass bins per replicate and state that the estimand includes binning. Prefer Brier/NLL for inference.
- If the source split is not approximately a random sample of tasks from a stated superpopulation, label bootstrap intervals as sampling-stability intervals, not universal generalization guarantees.

### 6. Replace row-count gates with a power and precision gate based on trajectories

`1,500` train rows and `300` test rows say little about precision because 3,956 prefixes arise from only 582 trajectories and adjacent prefixes are highly correlated. Power for a paired difference depends on test trajectory count, trajectory sizes, baseline/model discordance, and between-trajectory variance. No current calculation establishes that a 5-point gain or 50%-coverage selective claim is detectable.

Amendment:

- After data preflight but before model results, freeze `G_train`, `G_dev`, `G_cal_fit`, `G_cal_select`, and `G_test`, plus row-count and length distributions.
- Run a blinded design analysis using labels and baseline predictions only. Estimate the variance of trajectory-level baseline correctness and simulate paired binary outcomes over a grid of model gains and within-trajectory correlations. Freeze the minimum detectable effect (MDE) at 80% power and two-sided alpha 0.05 for the paired cluster-bootstrap decision rule. Do not use Laya predictions in this calculation.
- Also compute precision by bootstrap simulation: expected 95% interval width for a 5-point paired gain and for selective risk at 50% coverage. Require at least 30 accepted test trajectories and report the number of trajectories with at least one accepted row; this is a bare stability floor, not a power guarantee.
- If the test split cannot provide 80% power for a scientifically meaningful effect, explicitly classify the run as estimation-only. Do not convert a non-significant result into evidence of no benefit.
- Training adequacy cannot be certified by a generic row minimum. Freeze minimum family supports and require enough positive train trajectories per family; otherwise merge to `OTHER_TOOL` or mark that family descriptive before training.

No defensible numeric power result can be derived until the preflight supplies trajectory counts, lengths, family prevalence, and baseline discordance. This is itself a blocker.

### 7. Make promotion and failure logically symmetric

Current promotion requires an observed gain of at least 5 points but only a confidence lower bound above zero. This supports positive gain, not a true gain of at least 5 points. Failure uses an upper bound below +5 points. Consequently, an estimate of +4 with a narrow interval above zero is neither promoted nor failed, while “failure” can include a real positive but sub-5-point effect. The ablation rule is also inconsistent: promotion compares with the better ablation by 3 points, while failure says “fails to beat both,” which can mean several different contrasts.

Amendment: choose one claim and one two-one-sided decision scheme.

- Recommended practical-superiority claim: `Delta = A_Laya - A_baseline > 0.05`.
- Support if the two-sided 95% paired interval lower bound exceeds `+0.05`.
- Falsify practical superiority if its upper bound is at or below `+0.05`.
- Otherwise inconclusive.

If that standard is infeasible for a pilot, weaken the claim explicitly:

- Pilot support if point estimate `Delta >= 0.05` and lower bound `>0`.
- Evidence against any benefit if upper bound `<=0`.
- Evidence against a five-point benefit if upper bound `<0.05`.
- Everything else inconclusive.

For ablations, define `Delta_context = A_full - max(A_task-only, A_history-only)` on point estimates and bootstrap the complete max operation within each paired replicate. Promotion and failure must use that same contrast and a frozen margin. Testing two separate ablation contrasts without multiplicity handling is not equivalent.

Remove ECE alone as a failure criterion. Poor calibration can falsify the calibrated-selective claim, but it does not falsify learning next-control decisions. Hardware/runtime and insufficient data are feasibility stops, not scientific failures.

### 8. Make baseline competition fair and prespecified

The per-question majority predictor can emit an invalid bundle and is weaker than a natural structured frequency baseline. One-vs-rest logistic regression receives no stated regularization search, class handling, thresholding, coherence decoder, or calibration budget. “Strongest non-Laya baseline” can also become a moving target.

Amendment:

- Add a structured empirical-frequency baseline predicting the most frequent valid complete bundle in source-train, plus a mode-conditional most frequent family set.
- Give logistic regression the same rendered state, truncation, labels, valid-bundle decoder, development split, and threshold/calibration protocol as Laya. Freeze TF-IDF vocabulary/`ngram_range`, regularization grid, class weighting, convergence rule, and tie-breaking before test scoring.
- Fit majority/frequency statistics on train only, never calibration or test.
- Select one primary non-Laya comparator on development data using a prespecified ordering and metric. Freeze it before any test predictions. Report comparisons with all baselines; apply Holm correction to confirmatory secondary comparisons.
- Match selection budgets. If Laya uses one copied hyperparameter setting, say so. If Laya receives multiple checkpoints or learning rates, permit a comparable bounded development search for logistic regularization.
- Untuned Laya is a model ablation, not a non-Laya baseline. Task-only/history-only models test information use, not architecture superiority, and should use the same optimization budget or be clearly labeled unequal-budget diagnostics.

## Optional Improvements

- Weight trajectories equally during training or sample trajectories uniformly then prefixes uniformly; otherwise long successful trajectories dominate both the learned policy and row-level loss.
- Report step-position quartiles. Later cumulative prefixes may be easier because the teacher's earlier actions reveal its plan, so task-disjointness alone does not establish task-conditioned reasoning.
- Add a task-shuffled diagnostic and a history-shuffled-within-tool-stratum diagnostic on development data. These test whether gains come from task semantics or local harness syntax.
- Report tool-call cardinality and exact set match by cardinality. A model that never predicts multi-family bundles may score well if they are rare.
- Report pre-calibration and post-calibration scores. Calibration must not conceal degradation in discrimination; include AUROC/AUPRC for bundle correctness as descriptive measures.
- Use three seeds before making a model-training claim beyond pilot feasibility. Bootstrap intervals quantify test-sample uncertainty, not optimization-seed uncertainty.
- Consider a later annotation study with multiple acceptable next actions or executable replay. It is required to move from recorded-action imitation to action-quality claims.
- Compare CE-only with RLCD+CE only if claiming benefit from the Laya training objective. The current experiment can test the checkpoint architecture/training pipeline, not attribute gains to RLCD.

## Claim-to-Falsifier Table

| Claim | Required observation | Observation that falsifies or fails to support it |
|---|---|---|
| Laya learns recorded next-control bundles better than the frozen primary baseline | Positive paired test gain under the frozen trajectory-macro exact-match estimand | Paired 95% CI upper bound `<= 0`; for a five-point practical claim, upper bound `<= 0.05` rules out gains greater than five points under the chosen convention |
| The gain uses task and observable history | Full-state exceeds `max(task-only, history-only)` under the frozen paired contrast | CI upper bound for the full-minus-max contrast `<= 0`; a point loss to either ablation is descriptive warning but must follow the exact frozen rule |
| The model predicts tool-family sets, not only control mode | Improvement persists on tool-call-only exact set match/Jaccard and supported families | No gain, or material degradation, conditional on tool calls while aggregate gain is driven by `RESPOND_OR_FINISH` |
| Bundle confidence is calibrated | Frozen test Brier/NLL and reliability analysis for `Z=bundle correct` meet prespecified targets | Calibration slope/intercept or risk estimates violate frozen tolerances; ECE alone is not decisive |
| Confidence supports useful abstention | Frozen threshold attains coverage `>=0.50` with one-sided test upper risk bound at the claimed limit | Coverage lower than 0.50, or risk upper bound above the limit; searching another test threshold invalidates the claim |
| The result generalizes to unseen tasks in this harness | Source validation is genuinely task/trajectory-disjoint and all gates pass | Duplicate/task leakage invalidates the result; it does not count as scientific falsification |
| Demonstrations yield a useful executable policy | Not tested by this protocol | No Experiment 001 observation can establish this; closed-loop verifier outcomes are required |
| Laya captures Kimi K3's probability distribution or optimal policy | Not identifiable from one success-filtered hard action per state | No Experiment 001 observation can establish this; repeated samples/action values or outcome comparisons are required |

## Reconciliation Checklist

The protocol is ready to refreeze only when all items below are fixed without looking at Laya test results:

1. Dataset revision, trajectory counts, source-split semantics, duplicate handling, and state exclusions.
2. Train-only tool-family taxonomy, `OTHER_TOOL` behavior, coherent target/decoder, and malformed-label rules.
3. Primary trajectory-macro estimand and conditional decomposition metrics.
4. Development, calibration-fit, calibration-select, and sealed-test group manifests.
5. Exact bundle-correctness confidence score and calibrator; no unsupported marginal-product interpretation.
6. Threshold grid, simultaneous risk-control method, claimed risk level, coverage target, tie handling, and test confidence bound.
7. Paired cluster-bootstrap algorithm, interval type, seed, and treatment of selective denominators.
8. Trajectory-level power/precision analysis and minimum family supports.
9. One symmetric promotion/falsification scheme, including the exact max-ablation contrast.
10. Structured and logistic baseline tuning, calibration, coherence, and selection budgets.
11. Claim wording restricted to offline recorded-action imitation and selective prediction in this harness.

Once these are frozen, the design can proceed without result-driven choices. Until then, official test scoring should remain prohibited.

## Statistical Sources

- El-Yaniv, R., and Wiener, Y. (2010). “On the Foundations of Noise-free Selective Classification.” *JMLR* 11:1605-1641. https://jmlr.org/papers/v11/el-yaniv10a.html
- Geifman, Y., and El-Yaniv, R. (2017). “Selective Classification for Deep Neural Networks.” *NeurIPS 2017*. https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
- Ding, Y., Liu, J., Xiong, J., and Shi, Y. (2020). “Revisiting the Evaluation of Uncertainty Estimation and Its Application to Explore Model Complexity-Uncertainty Trade-Off.” *CVPR 2020*. https://openaccess.thecvf.com/content_CVPR_2020/html/Ding_Revisiting_the_Evaluation_of_Uncertainty_Estimation_and_Its_Application_to_CVPR_2020_paper.html
- Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., and Lei, L. (2021). “Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.” arXiv:2110.01052. https://arxiv.org/abs/2110.01052
- Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., and Tran, D. (2019). “Measuring Calibration in Deep Learning.” *CVPR Workshops 2019*. https://arxiv.org/abs/1904.01685
- Kumar, A., Liang, P. S., and Ma, T. (2019). “Verified Uncertainty Calibration.” *NeurIPS 2019*. https://proceedings.neurips.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html
- Vaicenavicius, J., Widmann, D., Andersson, C., Lindsten, F., Roll, J., and Schön, T. B. (2019). “Evaluating Model Calibration in Classification.” *AISTATS 2019*. https://proceedings.mlr.press/v89/vaicenavicius19a.html
- Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press. Cluster resampling must follow the independent sampling unit, here the trajectory.
