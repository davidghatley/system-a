# Iteration 3 saved-dev probability diagnostics

## Scope and reproducibility

CPU/offline analysis only. Inputs were `artifacts/experiment_i3/seed42_recovery.json`, `seed314159_run.json`, and `artifacts/baseline_i3/preflight/corrected_v3/dev_predictions.jsonl` (TF-IDF logistic method). Group IDs for TF-IDF records were joined by record ID to the saved seed-42 dev prediction metadata; no text, train, or test records were opened. No inference, model load, CUDA, training, or held-out test use occurred.

Commands: `python scripts/test_i3_dev_diagnostics.py`; `python -m py_compile scripts/i3_dev_diagnostics.py scripts/test_i3_dev_diagnostics.py`; `python scripts/i3_dev_diagnostics.py > artifacts/release_i3/run.log`. Outputs: `metrics.json` (full detail), `reliability.svg`. Re-run from repository root with Python 3 and NumPy installed. `matplotlib` is not required. The pytest package is unavailable in this environment; the targeted regression test is executable directly and passed.

## Schema, fixed label and loss checks (verified by read-only inspection)

- `training/laya_trace_i3/protocol.py`: `FIXED_LABELS = [read, search, edit, execute, other_tool, respond_or_finish]`; label-to-choice indices are derived by locating these labels in `CHOICE_OPTION_ORDER`. Protocol checks the config labels and this mapping.
- `training/laya_trace_i3/experiment.py`: prediction rows convert model probabilities through `option_keys` into the configured fixed label order; evaluation consumes mappings in that order. Training objective uses `loss_rlcd + cross_entropy(choice_logits[6], one_hot_gold)`; CE target uses the label-to-choice mapping. RLCD reference floors log probabilities at -9.21 and CE uses probability floor 1e-12 (`six_class_rlcd_reference`); training uses torch log_softmax/cross_entropy. Saved dev NLL uses 1e-12 floor in evaluation.
- Saved I3 probabilities are six-key label mappings. Baseline JSONL contains `gold`, `id`, and method predictions/probability mappings. Script explicitly orders by fixed labels, verifies ranges/sums through diagnostics and reports normalization deviation. Per-row normalization is applied before analysis.

## Verified raw dev summaries

All three sources have 140 rows, with fixed-six gold supports read=27, search=16, edit=27, execute=70, other_tool=0, respond_or_finish=0. The latter classes are absent from this dev slice: their class NLL is correctly unavailable, not zero evidence. Exact per-class error counts, NLL sums/contributions, confidence measures, top 20 wrong predictions, per-row sums and zero counts are in `metrics.json`.

| Source | Accuracy | Fixed-six macro-F1 | NLL (1e-12 floor) | Brier | Mean max confidence correct / wrong |
|---|---:|---:|---:|---:|---:|
| seed 42 | 0.6786 | 0.4453 | 3.4056 | 0.5327 | 0.9425 / 0.8271 |
| seed 314159 | 0.7143 | 0.4419 | 3.7481 | 0.5525 | 0.9889 / 0.9762 |
| TF-IDF logistic | 0.6071 | 0.2927 | 0.9519 | 0.5150 | 0.6093 / 0.5319 |

Metrics computed from saved predictions reproduce their embedded experiment metric values to floating-point rounding for both neural seeds. The six-class macro F1 includes absent classes as zero.

## Temperature assessment

A single positive scalar temperature is fitted by bounded deterministic golden-section minimization of NLL over `T in [0.01,100]`, using each fold's training rows only. Five deterministic size-balanced folds are made from indivisible `task_group`s; all available trajectories are checked to map to one group. There are 20 task groups, so this remains a small, coarse grouped-CV estimate. Exact fold memberships are deterministic from descending group size then string key; fold results and each held-out raw/scaled NLL, Brier, accuracy and macro-F1 are in the JSON. Raw/scaled pooled foldwise scores are also recorded.

**Correction:** an earlier version concatenated held-fold scaled probabilities in fold order while scoring labels in source order. That misalignment corrupted pooled CV scores. The implementation now assigns each calibrated probability back to its original row index, verifies that held folds partition every input row exactly once, checks that reconstructed raw pooled metrics equal direct metrics, and asserts positive temperature scaling preserves every argmax. A targeted synthetic regression test deliberately creates shuffled/interleaved fold ordering and validates aligned pooled scores.

With aligned rows, pooled foldwise raw→scaled held-out NLL is seed 42: 3.4056→1.0503; seed 314159: 3.7481→0.9484; TF-IDF: 0.9519→0.9684. CV accuracy/macro-F1 are unchanged by positive scalar scaling: seed 42 0.6786/0.4453, seed 314159 0.7143/0.4419, TF-IDF 0.6071/0.2927. TF-IDF worsens in pooled CV NLL; its fold-specific outcomes remain mixed. The corrected outputs overwrite the prior JSON and SVG. Full-dev fitted temperatures and resulting scores in the JSON are explicitly marked **in-sample exploratory** and are not unbiased estimates or selection evidence. Temperatures are not adopted as a release choice.

No exact zero probabilities occurred in these saved inputs; thus the `1e-300` clipping used only to make log/temperature operations finite did not alter any saved zero entries (count=0 for all). NLL metric reporting uses the protocol's `1e-12` target-probability floor. Row normalization deviations are reported per source. If future exports round a probability to zero, calibration is defined on normalized saved probabilities with zero replaced by `1e-300` before log transform; this is numerically stable but cannot recover information discarded by rounding, and metrics should not be interpreted as original-logit calibration.

## Assessment and limits

**Recommendation:** treat temperature scaling as exploratory only; do not use these diagnostics to select a release candidate. On this grouped CV, scalar scaling materially lowers neural NLL while preserving class predictions; TF-IDF NLL slightly worsens overall. Neural probabilities are highly overconfident and wrong-case mean confidence is high. Only 20 groups and two absent classes limit conclusions; temperature fitting optimizes NLL and is not a release-selection procedure. Per-class NLL contribution is decomposable, not causal. No test conclusions are made. No candidate selection or model changes were performed.

**Verified:** file contents, fixed label mapping and loss definitions inspected; saved dev metrics and group-aware calculation performed by the script.
**Inferred:** the positive temperatures and apparent neural held-out fold NLL reductions suggest overconfidence on this dev distribution; this may not transfer.
**Unknown:** behavior on held-out test or future deployment data; whether this coarse grouped-CV estimate generalizes; calibration under changed class prevalence. No test files or results were accessed.
