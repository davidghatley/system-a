# Iteration 3 baseline preflight verification

## Scope

Verified against starting commit `2d4408d`. Only Trace2Decision v2 `train.jsonl` and `dev.jsonl` were opened for interface understanding and the optional smoke. No test examples or test results were read. This is not corrected-v3 evidence and is not a held-out evaluation.

All three methods consume the same compact `state` text. Majority ignores its content. TF-IDF uses only that string. Transition parameters are fitted from consecutive labels within training trajectories; at prediction time its previous-action key is extracted only from `RECENT HISTORY` or `LATEST RELEVANT OBSERVATION` in `state`. Metadata is limited to ordering, leakage checks, IDs, and trajectory-level resampling.

## Verified observations

The v2 smoke used 1,235 train rows in 164 trajectories and 159 dev rows in 20 trajectories. The selected four-candidate model was `word12-c2` by dev macro-F1 with candidate-ID ascending as the exact tie-break.

| Method | Macro-F1 | Accuracy | NLL | Brier |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.1019 | 0.4403 | 1.4533 | 0.7255 |
| Previous-action transition | 0.1876 | 0.4969 | 1.3881 | 0.6856 |
| TF-IDF logistic | 0.3183 | 0.5597 | 1.1439 | 0.5843 |

The transition baseline recovered previous-action evidence from compact state in 33/159 rows and used its declared smoothed training-majority fallback in 126/159. TF-IDF minus transition macro-F1 was 0.1306 with a trajectory-bootstrap percentile interval of [0.0816, 0.1898], using 1,000 resamples over 20 dev trajectories. TF-IDF minus majority was 0.2164 [0.1504, 0.2924]. These intervals quantify this v2 dev sample only and do not correct dev-selection optimism.

The fixed `other_tool` class had zero dev support. The fixed-list macro-F1 still includes it as zero, as required. Full supports, recalls, confusion matrices, candidate outcomes, and probability metrics are in `v2_interface_smoke/result.json`.

Two complete smoke executions produced identical substantive artifact hashes:

```text
b2ce60255fcc4daa4fbea873e30a5bcf9275c975b133b99ecfc691ff82be0e15  selected_model.json.gz
52ef8cf15a4d9f9af3482f0fbfbbc52e7aae20823d5d4137515368b53cefe608  dev_predictions.jsonl
9f532d88cbd13b22852c813f1e433691c2b1bea3b29a379e3062071a073874dd  selected_config.json
```

Per-candidate fit-plus-dev measurements were about 4.5 to 8.9 seconds in the recorded run. `/usr/bin/time` is unavailable in this environment, so peak RSS was not independently measured; the implementation bounds vocabulary at 8,000 and stores six dense weight vectors, making the 8 GiB limit non-binding by construction, but that statement is inference rather than a measured peak.

## Commands

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m unittest discover -s baselines/trace_action_i3/tests -v
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m baselines.trace_action_i3.cli --manifest artifacts/baseline_i3/preflight/manifest_v2.json
sha256sum artifacts/baseline_i3/preflight/v2_interface_smoke/selected_model.json.gz artifacts/baseline_i3/preflight/v2_interface_smoke/dev_predictions.jsonl artifacts/baseline_i3/preflight/v2_interface_smoke/selected_config.json
git diff --check
```

## Uncertainties and next input

The substantive question remains unresolved until corrected v3 train/dev arrive. V2 target construction can label mixed-category turns by their first call, so none of these scores should select a final baseline or support a claim about corrected observed actions. For v3, retain the fixed labels and candidate budget, point a new manifest at train/dev only, inspect class support and transition extraction coverage, and defer any held-out evaluation to the integrator's frozen comparison.
