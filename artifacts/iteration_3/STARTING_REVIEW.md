# Iteration 3 Starting Skeptical Review

Date: 2026-09-22
Starting commit: `2d4408d`
Verdict: **BLOCK observed-action training/evaluation until target construction is repaired and re-reviewed.**

## Boundary and method

This review used only the directive, source schema/rows, historical Trace2Decision v2 code/data/reports, and Iteration 2 trainer/integration evidence. It did not inspect any concurrent Iteration 3 labels, predictions, or results; the concurrently present `exploration/` path was not inspected. No model, GPU, network, or write-producing verifier was used. CPU analyses used at most four math threads. This file is the only review-owned write.

Read-only scans recomputed target-event structure, v2 membership, split/group counts, exact target-argument/call-ID occurrence in state, and cheap shortcut diagnostics from `artifacts/trace2decision_i1/source/traces.jsonl` plus historical v2 JSONL. The shortcut scores below are diagnostics, not eligible comparison results, because previous action and source step were read from source structure rather than reconstructed solely from the model-visible state.

## Severity-ordered findings

### P0: v2 labels contradict the required whole-target-turn semantics (VERIFIED)

`pipelines/trace2decision/i2_convert.py:157-158` maps only `names[0]`. In all 1,821 source rows, 277 targets have multiple calls: 201 map to one category and 76 map to multiple categories. In the current 1,544-row v2 population, 216 are multi-call targets: 151 homogeneous and **65 mixed-category** (train/dev/test **51/3/11**). The mixed combinations in the full source are read+search 34, execute+read 27, and execute+search 15. Call order can therefore change 65 retained labels even though the observed target event is unchanged as a set.

Under the directive, those 65 rows are not single-label examples. Excluding them would leave **1,479 rows (1,184/156/139) and 206 trajectories (163/20/23)** before any other semantic exclusions. Training or evaluating current v2 can reward learning serialization order rather than observed action category and can change both class support and the strongest conclusion.

Required repair tests:

- Map every call in the target turn, retain homogeneous bundles, and exclude mixed bundles with explicit split/trajectory/category-combination counts.
- Prove permutation invariance for homogeneous and mixed bundles; every permutation must retain the same label or exclusion decision.
- Independently recompute target decisions from source and require zero mismatches.

Acceptance gate: reviewed deterministic output and hashes, with no mixed-category row in train/dev/test and preserved group assignments for all retained rows. Stop if target parsing still depends on first-call order after two repair cycles.

### P0: `respond_or_finish` is partly inferred from absence, not established event semantics (VERIFIED risk; completion interpretation INFERRED)

The converter assigns every target with no `tool_calls` to `respond_or_finish` without checking content/event completeness. The source has 208 no-call targets; v2 retains 166 (133/16/17). All have non-empty content, 16 have non-empty `reasoning_content`, 207/208 are final assistant steps, and one is non-final (`behavior-dependency-planning-0111`, step 7/9) despite saying it is waiting for confirmation. Target call objects and argument JSON are structurally valid, but the schema exposes no stream-completion or finish-reason field. Thus 165 retained final responses are consistent with completion, but this is publisher-structure evidence rather than a direct completion marker; the non-final case is ambiguous under the requested rule.

The 277 unresolved-prefix exclusions are also selection, not proof of malformed targets: they disproportionately remove execute 149, read 52, finish 42, and edit 34 targets. This can make the retained workflow easier and alters class/position distributions.

Required repair tests:

- Define positive source evidence that distinguishes a completed no-tool response from internal reasoning, partial stream, or missing tool record. Do not infer solely from an empty call list.
- Audit all no-call targets and separately report final, non-final, reasoning-bearing, empty/malformed, and ambiguous cases; exclude ambiguity explicitly.
- Report exclusions by split, trajectory, source step, and would-be category, including unresolved prefixes and mixed targets.

Acceptance gate: every retained `respond_or_finish` has documented source-supported completion semantics; the non-final case is explicitly adjudicated. Stop if the source cannot support that distinction; narrow the estimand to tool-bearing completed events instead.

### P1: workflow position and transition structure can produce a misleading learning win (VERIFIED diagnostic; compliant score UNKNOWN)

The historical labels are highly positional. A train-only step-mode diagnostic gives dev accuracy/macro-F1 **0.579/0.349** versus majority **0.440/0.102**; a train-fitted previous-action transition diagnostic gives **0.598/0.352**. On historical test these are **0.507/0.340** and **0.513/0.326**, versus majority **0.400/0.095**. Transition modes are start→search, search→read, and read/edit/execute→execute. Neither diagnostic predicts edit on dev/test; transition also never predicts finish. High accuracy can therefore coexist with weak action coverage.

These scores are deliberately not eligible baselines: source step and source-derived previous action can contain information unavailable after v2 truncation. They do falsify any assumption that row-level improvement necessarily represents task/state understanding. A fine-tune can learn common workflow grammar or repeated harness wording and still fail the intended added-value question.

Required repair tests:

- Fit majority, rendered-state-only previous-action/transition, and TF-IDF logistic baselines on train groups only. Baselines must consume the exact frozen state text available to Laya; no `source_step`, trajectory ID, omitted history, raw source event, target arguments, or metadata.
- Report fixed-six-label macro-F1, accuracy, per-class support/recall, confusion matrices, NLL/Brier where distributions are defined, and group-level paired differences.
- Add diagnostic ablations using only task, only latest observation, and state with obvious workflow-position/previous-action cues masked. Keep these exploratory and do not tune the primary experiment from test outcomes.

Acceptance gate: “learning” requires fine-tuned Laya to beat the identical base; “added value” additionally requires at least the predeclared five-point macro-F1 gain over the dev-selected inexpensive baseline, with paired group uncertainty reported. Stop the added-value claim if the gain vanishes under cue masking, is confined to common classes, or is matched by the sparse/transition baseline.

### P1: nominal row count greatly overstates the uncertainty unit (VERIFIED) and repository-family independence is unproved (UNKNOWN)

Current v2 has 1,544 adjacent prefixes but only 207 trajectories/task groups; after mandatory mixed-event filtering the likely maximum is 206. The final historical test has only **23 groups** (139 corrected rows), and dev has 20. Exact normalized task and complete-trajectory duplicates/crossings are zero, which is a real pass. It is not family independence: task stems and languages cross splits (for example `go`, `ts`, `rs`, `bash`, and `behavior-dependency-planning`), and normalized task text does not cluster shared recipe/repository/template ancestry. Source rows were selected from accepted trajectories, so future success/review affected population inclusion even though those future fields are not rendered.

Row bootstrap or row-level significance would be invalid. The transition diagnostic's per-trajectory accuracy is highly dispersed (dev mean 0.508, SD 0.293; test mean 0.396, SD 0.266), illustrating how a few trajectories can dominate a row aggregate.

Required repair tests:

- Audit near-duplicate prompts and repository/recipe/template families across splits, not only exact normalized tasks; document a defensible broader group key.
- Compute paired bootstrap or randomization intervals by the broader group, with trajectory as the minimum unit. Report group count, support, and seed variation separately.
- Report macro-F1 both row-weighted and as a trajectory-level macro summary; include leave-one-family-out sensitivity where families cross splits.

Acceptance gate: no split crossing at the accepted broadest defensible group and uncertainty based on that unit. If only 23 final groups remain, label the result exploratory and avoid precise generalization claims. Stop if material near-duplicate family leakage is found after results have been viewed; repair splits and rerun under a newly frozen protocol rather than reusing those results confirmatorily.

### P1: the historical test is already research-visible (VERIFIED), so it cannot support a fresh confirmatory claim

Historical reports expose test label counts, hashes, and full verifier scans. The committed deterministic v2 samples include three test rows with labels, and prior review directly inspected target/source pairs including a parallel call. The directive also records prior dev/test inspection. The current test may be used once as exploratory held-out evidence after freezing, but not called sealed, unseen, or independent of protocol development. A reshuffle of these same 207 trajectories would not manufacture a new test.

Required repair test: before reading any new corrected-test predictions, freeze data revision, renderer, labels, seeds, checkpoint selection, baseline selection budget, metrics, smoothing, group unit, and comparison rule; retain a timestamp/hash record. Keep all selection on train/dev.

Acceptance gate: final wording says “previously inspected exploratory held-out split.” Stop any confirmatory or fresh-test claim; a confirmatory result requires newly sourced, group-independent trajectories whose labels were not inspected during development.

### P1: information parity can silently favor either baselines or Laya (VERIFIED interface risk)

Laya receives `state` plus the fixed choice question/options. Metadata is excluded at the integration boundary, which passed. A baseline using source step/raw previous action would be advantaged; conversely a sparse baseline given only task text while Laya gets latest observation/history would be disadvantaged. The previous-action baseline is especially delicate because 1,233/1,544 states omit history and protected fields are head/tail bounded. Although 97.41% of full bounded candidates exceed 512 tokens, the v2 report describes loss as “only optional older history”; protected task/system/observation text can also be reduced. Every retained latest tool observation was the final prefix event in this source scan, which reduces one suspected future-selection risk, but does not restore omitted context.

Exact target leakage checks passed: target arguments and target call IDs appeared verbatim in zero v2 states; structural rendering uses `messages[:-1]`. Future-derived source metadata is not rendered. These are converter-level passes, not proof against semantic duplication of expected commands from task text.

Required repair tests:

- Materialize one canonical model-input record per example and prove each comparator derives features only from it; record any unavoidable representation difference.
- Add canaries for every target field, malformed target variants, and target-argument substrings; inspect semantic command templates separately from exact-string leakage.
- Correct truncation reporting to distinguish omitted messages from bounded protected fields, and stratify results by truncation/history retention.

Acceptance gate: parity audit passes for every method and no method consumes metadata or source-only fields. Stop comparison if parity cannot be demonstrated or if target/future fields enter any model input.

### P2: missing class support and population filtering can inflate macro conclusions (VERIFIED)

`other_tool` has zero positives in all v2 splits, yet the primary metric requires a fixed six-label list. Historical majority macro-F1 is only 0.102 on dev when the absent class is scored as zero. Mixed-event and unresolved-prefix filtering is label- and position-dependent. A large aggregate gain may therefore reflect execute/read/search while edit and finish remain poor, or may vary solely with whether the absent class is included.

Required repair tests: freeze the six-label macro-F1 implementation with zero for missing classes; also report observed-class macro-F1 as explicitly secondary, supports/recalls, and corrected-population exclusion sensitivity. Do not delete `other_tool` after seeing results.

Acceptance gate: every table names its fixed label list and missing-class rule. Stop a broad added-value conclusion if gains are limited to common actions or any class has materially worse recall without an explicitly narrow claim.

### P2: Iteration 2 trainer evidence validates machinery, not this task or two-seed stability (VERIFIED)

The accepted B2 result is real evidence that the local evaluator/trainer can learn its public typed-decisions slice: one seed improved choice accuracy 36.0%→70.5%. It used a different dataset/workflow, all 421M parameters, and one completed seed; its dev set was the same set evaluated before and after training. It establishes capability and cost (772.8 s total, 8.670 GiB peak reserved), not Trace2Decision learning, calibration, class-balanced value, or seed stability. Integration tested one v2 row and both models missed it; that is interoperability only.

Required repair tests: two predeclared seeds if budget permits, otherwise declare a single-seed pilot before training; evaluate every completed seed and separate seed variation from group uncertainty. Select checkpoints on corrected dev only and perform exactly one final corrected-test batch.

Acceptance gate: all runs, including failures, are reported. Stop at pilot/unresolved if seed sign changes, the budget cannot support the declaration, or checkpoint selection touches test.

### P2: latency provenance is inconsistent across summaries (VERIFIED)

The `39.689/40.527 ms` pair is the Iteration 1 **base** checkpoint verification (`convaiinnovations/laya@1c5edc...`), not the specialist. Specialist measurements are separate runs: implementation `41.093/41.893`, repair artifact `41.089/42.455`, final verification rerun `40.150/40.977` ms. `PROJECT_STATE.md:22` calls the Iteration 1 pair the “pinned public checkpoint” result without identifying base/run, while Iteration 2 records correctly use the final specialist rerun. The committed `specialist_benchmark.json` contains the repair run, not the final-verification numbers. The 6.339 s integration figure is load plus one inference, not warm latency. All warm figures measure tokenization plus inference for the 449-token example and exclude load; none establishes full-record latency on corrected Trace2Decision.

Required repair test: every latency row must carry checkpoint ID/revision, input/sequence tokens, device/software, warmups/samples, statistic definition, included stages, and exact artifact/run. Measure all compared models on the same corrected records and synchronized harness; report cold load separately.

Acceptance gate: no unattributed or cross-run latency comparison. Stop any speed/production claim if the quoted number cannot be traced to a matching immutable artifact or if question-level and record-level scopes differ.

## Decision gates

1. **Data gate:** whole-turn semantics, completion evidence, exclusion ledger, order invariance, leakage tests, deterministic hashes, and stratified train/dev review pass before substantive training.
2. **Protocol gate:** corrected revision, renderer, fixed labels, two seeds or declared pilot, metrics, group unit, baseline budget, checkpoint rule, and five-point added-value threshold are frozen before corrected-test predictions.
3. **Parity gate:** majority, transition, TF-IDF, base Laya, and fine-tuned Laya consume the same available compact-state evidence; source-only metadata is forbidden.
4. **Inference gate:** paired differences use trajectories or broader families, not rows; report only 20 dev and 23 test groups unless correction changes them, and disclose the effective-family count as unknown until audited.
5. **Claim gate:** “learning” means improvement over the identical base. “Added value” additionally means at least +0.05 absolute macro-F1 over the frozen inexpensive comparator with uncertainty compatible with practical benefit. Neither implies task completion, optimality, general intelligence compression, or reliable agent control.

## Stop conditions

- Stop training if target repair is not accepted within two cycles, mixed-category targets remain, no-call completion cannot be established, leakage/parity fails, or corrected train/dev classes are not viable.
- Stop final evaluation if test predictions/labels influence any design choice, broad grouping reveals leakage, or required seeds/baselines are incomplete.
- Stop positive claims if gains disappear under group resampling, are matched by cheap shortcuts, depend on common classes only, or do not reproduce across declared seeds.
- Preserve a valid negative result: if the sparse/transition baseline wins, the supported conclusion is that workflow grammar explains the benchmark better than Laya-specific fine-tuning, not that the pipeline failed.

## Reproduction record

Inputs inspected are at commit `2d4408d`, especially `System-A_Next_Iteration_Prompt.md`, `pipelines/trace2decision/{convert.py,i2_convert.py,i2_verify.py}`, `artifacts/trace2decision_i{1,2}/`, `artifacts/train_i2/`, `artifacts/iteration_2/`, source README/manifest/JSONL, `ITERATIONS.md`, and `PROJECT_STATE.md`. Read-only Python scans used `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=4`; no model was loaded.

Evidence classification summary: event counts, historical shortcut diagnostics, split/group counts, exact leakage scans, sample split membership, and latency/artifact identities are **VERIFIED** from committed bytes. Completion semantics, latest-observation relevance, and accepted-trajectory authenticity are **INFERRED**. Broader repository-family independence, effective independent-family count, corrected-model performance, calibration, seed stability, and downstream utility are **UNKNOWN**.
