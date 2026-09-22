# Iteration 3 exploration report

## Boundary and evidence

This exploratory run started at commit
`2d4408de85587a046c3f14fec1835dd9e69940c9`. It read only the existing v2
train and dev files: 1,235 train rows across 164 trajectories and 159 dev rows
across 20 trajectories. `results.json` records their SHA-256 hashes and states
that test was not read. The frozen protocol is
`exploration/trace_action_i3/PROTOCOL.md`.

All findings are potentially affected by the v2 target bug. The v2 converter
uses the first target tool call as the label, while mixed-category multi-call
targets should be excluded. These measurements are therefore exploratory and
must not modify the frozen primary experiment.

## Probe 1: workflow position and previous action

| Train-fitted method | Dev accuracy | Dev macro-F1 | Macro-F1 difference vs majority (trajectory bootstrap 95%) |
|---|---:|---:|---:|
| Majority (`execute`) | 0.4403 | 0.1019 | reference |
| Exact source step | 0.5849 | 0.3493 | +0.2474 [+0.1997, +0.2972] |
| Position bin | 0.6038 | 0.3593 | +0.2574 [+0.2139, +0.2992] |
| Previous action | 0.5975 | 0.3522 | +0.2503 [+0.2058, +0.2949] |
| Position bin + previous action | **0.6226** | **0.4261** | **+0.3242 [+0.2508, +0.4040]** |

Previous action means the preceding retained target only when its source step
is exactly contiguous; otherwise the feature is `START`. This feature was
available for 1,071/1,235 train rows (86.72%) and 139/159 dev rows (87.42%).
The best method recalled search/read/edit/execute at 84.21%/51.85%/37.04%/
84.29%, but recalled no `respond_or_finish` rows. `other_tool` had zero support
in both splits and contributes zero under the fixed six-label macro-F1.

**Observation:** workflow position and recent action explain substantially more
dev variation than the global class distribution, and their combination is the
best cheap probe result.

**Limitation:** only 20 dev trajectories support the clustered interval;
position can encode this coding harness's stereotyped workflow rather than a
portable policy; missing/noncontiguous prior rows use `START`; and buggy v2
targets can affect both current and previous labels.

**Decision changed:** explicit workflow position and previous observed action
should become a required inexpensive baseline and candidate state feature in
the next iteration. Broad text understanding should not receive credit for
performance already available from workflow phase.

## Probe 2: latest observation increment

| Fixed multinomial NB input | Dev accuracy | Dev macro-F1 | NLL | Brier |
|---|---:|---:|---:|---:|
| TASK | 0.4214 | 0.1091 | 10.4908 | 1.1494 |
| TASK + latest observation | **0.4654** | **0.2143** | **9.6229** | **1.0373** |

Adding the observation changed accuracy by +0.0440 (trajectory-bootstrap 95%
percentile interval +0.0149 to +0.0816) and macro-F1 by +0.1052 (+0.0510 to
+0.1649). The fixed feature sets contained 19,124 and 21,919 train-derived
unigram/bigram features, respectively.

**Observation:** the latest observation adds predictive information beyond the
rendered task under this fixed sparse learner and passes the predeclared
+0.05 macro-F1 threshold.

**Limitation:** this is one untuned naive Bayes specification, its probabilities
are not calibrated, absolute macro-F1 remains low, task truncation can vary by
step, and the same 20 dev trajectories plus target bug constrain inference.

**Decision changed:** retain the protected latest-observation field and test it
in a corrected-data ablation. Do not treat the current sparse classifier as a
useful broad predictor or discard workflow state in favor of text.

## Recommendation

On corrected targets, make a workflow baseline using explicit position and
previous observed action the mandatory comparator, then ablate the latest
observation on top of that same information budget. The most promising narrow
application is workflow-phase telemetry or routing for tool-action families,
not autonomous next-action control: the cheap workflow baseline is already
strong for search/read/execute but fails to identify completion and has not
been linked to task success.

This recommendation is evidence-backed by the best dev macro-F1 of 0.4261 and
the separate +0.1052 incremental observation effect. It does not alter the
primary experiment.

## Reproduction and resources

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python exploration/trace_action_i3/run_probes.py \
  --train artifacts/trace2decision_i2/output/train.jsonl \
  --dev artifacts/trace2decision_i2/output/dev.jsonl \
  --output artifacts/exploration_i3/results.json
```

The successful run used one configured thread, 24.66 seconds wall time, 24.52
CPU seconds, and reported 5,328,572 KiB maximum RSS (5.08 GiB), within the 6 GiB
limit but with little memory headroom. `/usr/bin/time` was unavailable; the
script recorded these measurements with Python `time` and `resource`. The
failed wrapper attempt and successful command are preserved in `commands.log`.

## Uncertainties

- The unknown number and class distribution of mixed-category v2 targets could
  change both probes after repair.
- Dev has no `other_tool` support and only 20 independent trajectory groups.
- No test data, neural model, network, GPU, or choice-order prediction probe was
  used.
- The probes establish prediction associations, not causal usefulness or
  improved downstream agent outcomes.
