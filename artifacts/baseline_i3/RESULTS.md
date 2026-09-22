# Iteration 3 baseline finalization

Status: **VERIFIED train/dev baseline run; no test data or test results were read.**

## Frozen protocol

- Starting commit: `2d4408de85587a046c3f14fec1835dd9e69940c9`
- Input revision: corrected Trace2Decision i3 `output/train.jsonl` and `output/dev.jsonl`.
- Fixed labels and order: `read, search, edit, execute, other_tool, respond_or_finish`.
- Primary metric: fixed-list macro-F1; absent classes contribute zero.
- Candidate budget: four TF-IDF logistic candidates, frozen in
  `preflight/manifest_corrected_v3.json` before fitting. Selection is DEV macro-F1,
  exact ties by ascending candidate ID.
- Count smoothing: additive alpha 1.0 for count baselines. Logistic probabilities
  are softmax outputs and are not claimed to be calibrated.
- Bootstrap: 2,000 paired resamples of whole trajectory groups, seed 314159.
- All predictors consume only the canonical `state` text. The transition baseline
  uses a mapped prior tool name only when recoverable from the state; unavailable
  rows are explicitly counted invalid and use the declared training-majority
  fallback, never hidden metadata.

## Corrected train/dev population

| Split | Rows | Trajectories | SHA-256 |
|---|---:|---:|---|
| train | 1,051 | 163 | `86442271b3ddb74ce517ee335f6d89fbc80f94441ff1198250da5bf9ab4dda3c` |
| dev | 140 | 20 | `ade8262625a46eef78dcb2b11b42eaced444e1604bab1af5423412a67ae92783` |

The corrected converter report records 76 mixed-category target turns, 208
no-call records without positive completion evidence, and 224 unresolved-prefix
exclusions. These are VERIFIED from `artifacts/trace2decision_i3/output/report.json`;
the resulting baseline population is narrower and tool-interface labels describe
observed interfaces, not inferred optimal intent.

## DEV comparison

| Method | Macro-F1 | Accuracy | NLL | Brier |
|---|---:|---:|---:|---:|
| Training majority | 0.1111 | 0.5000 | 1.2346 | 0.6638 |
| Previous-action transition | 0.1977 | 0.5643 | 1.1674 | 0.6142 |
| TF-IDF logistic (`word12-c2`) | **0.2927** | **0.6071** | **0.9519** | **0.5150** |

The selected primary inexpensive comparator is **TF-IDF logistic `word12-c2`**,
selected on DEV only. Its paired macro-F1 difference versus transition is
0.0950, bootstrap 95% percentile interval [0.0366, 0.1641]; versus majority it
is 0.1816, interval [0.1007, 0.2709]. There are 20 independent trajectory
groups. These are DEV selection/comparison measurements, not confirmatory claims.

The complete fixed-label per-class support/recall and confusion matrices are in
`preflight/corrected_v3/result.json`; the fixed `other_tool` and
`respond_or_finish` classes have zero DEV support and remain in the macro average.

For the selected comparator, DEV support/recall is
`read 27/0.4444`, `search 16/0.1250`, `edit 27/0.1481`,
`execute 70/0.9571`, `other_tool 0/0`, `respond_or_finish 0/0`.
Its confusion rows (gold order above; columns are the same order) are:
`[[12,0,0,15,0,0],[1,2,0,13,0,0],[0,0,4,23,0,0],
[2,0,1,67,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`.

## Artifacts and cost

- Model: `preflight/corrected_v3/selected_model.json.gz`
- Frozen selected config: `preflight/corrected_v3/selected_config.json`
- DEV predictions: `preflight/corrected_v3/dev_predictions.jsonl`
- Full result JSON: `result.json` (same result as the preflight corrected-v3 JSON)
- Measured fit/evaluate wrapper: 23.478256 seconds wall time, 684,768 KiB
  maximum resident set size, CPU-only with four-thread environment variables.
  No GPU was used. RSS is a Linux `resource.getrusage` measurement.

## Reproduction

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m baselines.trace_action_i3.cli \
  --manifest artifacts/baseline_i3/preflight/manifest_corrected_v3.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s baselines/trace_action_i3/tests -v

sha256sum artifacts/trace2decision_i3/output/train.jsonl \
  artifacts/trace2decision_i3/output/dev.jsonl \
  artifacts/baseline_i3/preflight/manifest_corrected_v3.json \
  artifacts/baseline_i3/preflight/corrected_v3/result.json \
  artifacts/baseline_i3/preflight/corrected_v3/selected_model.json.gz \
  artifacts/baseline_i3/preflight/corrected_v3/selected_config.json \
  artifacts/baseline_i3/preflight/corrected_v3/dev_predictions.jsonl
```

## Evidence boundaries

**VERIFIED:** deterministic six-label metrics, candidate selection, trajectory
bootstrap intervals, data hashes, model/config serialization, CPU/RSS wrapper
measurement, and passing unit tests. **INFERRED:** compact-state transition
extraction is a fair information-parity representation where the marker parser
finds a tool name; the fallback is appropriate for unavailable prior action.
**UNKNOWN:** neural/base-Laya comparison, seed variation for neural fine-tuning,
downstream task utility, calibration, and generalization beyond this corrected
train/dev population.
