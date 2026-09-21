# Repair 1 Record

Date: 2026-09-21

## Scope

The frozen subset and preserved 32-update run were not rerun or changed. The exact-choice metric is now explicitly native `gold[qid].label` hard-label accuracy; probability argmax is retained only as diagnostic metadata.

## Verified metric reconstruction

`training/laya_local/metrics.py` independently reconstructs targets from `subsets/dev.jsonl` and joins them to the preserved prediction files. The test and `reconstruction_native_metric.json` agree:

- baseline: 19/40 (0.475)
- post-training: 11/40 (0.275)
- absolute change: -0.200; **not a pass**

The tied question `agent_trace_observability_000003/action` has native label `human_review` (index 1), while distribution argmax is index 0. This is included in the test.

## Verified bounded resume smoke

`training/laya_local/resume.py` is an actual continuation entry point. New checkpoints save model, optimizer, scaler, step, data cursor/order, and Python/NumPy/Torch/CUDA RNG state. The smoke checkpoint under `resume_smoke/` is derived from the preserved checkpoint with the deterministic legacy cursor/RNG fields supplied; the preserved failed-run checkpoint remains untouched.

The single authorized GPU update produced `artifacts/train_i1/resume_smoke/result.json`: source step 32 advanced to 33, gradients were finite, intended parameters changed, and the excluded action head was unchanged. This smoke output is not used to alter primary results.

## Checks

Verified commands:

```text
.venv/bin/python -m unittest discover -s training/laya_local/tests -v  # 4 passed
.venv/bin/python -m py_compile shared/typed_decisions/schema.py training/laya_local/data.py training/laya_local/metrics.py training/laya_local/prepare_subset.py training/laya_local/train.py training/laya_local/resume.py training/laya_local/tiny_overfit.py
```

No 32-step experiment was rerun, no frozen subset was changed, and no acceptance pass is claimed.
