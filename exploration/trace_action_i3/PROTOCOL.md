# Iteration 3 exploration protocol

Frozen before measurement on 2026-09-22. Starting commit:
`2d4408de85587a046c3f14fec1835dd9e69940c9`.

## Inputs and boundary

- Inputs are only Trace2Decision v2 `train.jsonl` and `dev.jsonl` under
  `artifacts/trace2decision_i2/output/`.
- The test split is not read.
- CPU only, one process, standard-library Python, and at most four threads
  (the implementation itself is single-threaded).
- Outputs are confined to `artifacts/exploration_i3/`.
- These are exploratory measurements and cannot alter the primary experiment.
- V2 labels are potentially affected by the known target bug: the converter
  labels a multi-tool target from its first call rather than excluding targets
  whose calls map to different action categories.

## Nominated ideas

1. Measure whether workflow position and the previous observed action explain
   dev labels. This tests whether a cheap workflow-state component is more
   useful than broad state understanding.
2. Measure whether the latest observation adds information beyond the task
   text using a fixed sparse classifier. This tests whether preserving the
   latest observation is justified and whether an observation router is a
   useful narrow application.
3. Test choice-order stability. Not selected because there are no applicable
   model predictions in the allowed inputs and running a model is prohibited.

## Probe 1: workflow signal

Fit on train and evaluate on dev: train majority; exact `source_step` majority
with global fallback; coarse position-bin majority (1, 2, 3, 4-5, 6-10, 11+)
with global fallback; and a previous-action transition baseline. A previous
action is available only when the preceding retained row in the same trajectory
has exactly `source_step - 1`; otherwise use a START fallback. Also measure a
position-bin plus previous-action table with transition then majority fallback.

Report accuracy, fixed-six-label macro-F1, per-label recall/support, and the
fraction of dev rows with a contiguous retained predecessor. Use 5,000
deterministic trajectory-cluster bootstrap replicates for paired accuracy and
macro-F1 differences from majority (seed 20260922).

Decision rule: recommend carrying explicit workflow state into the next
iteration if either previous action or position plus previous action improves
dev macro-F1 by at least 0.05 over majority and the direction is not solely an
artifact of unavailable predecessor rows. Otherwise do not prioritize it.

## Probe 2: latest-observation increment

Parse the protected `TASK` and `LATEST RELEVANT OBSERVATION` sections from the
existing rendered state. Fit two independently trained multinomial naive Bayes
classifiers with fixed settings: lowercase regex word tokens, word unigrams and
bigrams, train document frequency at least 2, additive alpha 1.0, and empirical
class priors with alpha 1.0. Compare TASK against TASK plus latest observation.

Report the same classification metrics plus NLL and multiclass Brier score.
Use 5,000 deterministic trajectory-cluster bootstrap replicates for the paired
accuracy and macro-F1 differences (seed 20260922).

Decision rule: preserve and investigate latest-observation routing if adding it
improves dev macro-F1 by at least 0.05. If the effect is smaller or negative,
do not claim that this dataset establishes incremental observation value; favor
workflow-state investigation if Probe 1 meets its rule.
