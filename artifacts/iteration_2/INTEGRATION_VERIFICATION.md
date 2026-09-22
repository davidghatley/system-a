# Iteration 2 Parent Cross-Track Integration Verification

Date: 2026-09-21
Role: parent verifier

## Decision

**ACCEPTED.** One unchanged Trace2Decision v2 row passed through the shared
parser and JSONL loader, Trainer construction and collator, the accepted B2
checkpoint, and the pinned Reflex specialist. Stable provenance was preserved
outside model tensors and did not alter model-facing construction.

This closes the parent-only cross-track gate in
`research/27_iteration_2_acceptance.md`. It is an interoperability result, not
an accuracy result: both models made a wrong native-label prediction on the one
selected row.

## Selected Row

- Source: `artifacts/trace2decision_i2/output/dev.jsonl`, line 1.
- Raw line SHA-256:
  `87cb076da3a9233363d94d9007c2fe2a17b3eb20c04bef3b9cacd460bc75aa68`.
- Canonical content SHA-256 before and after all paths:
  `dd6322ae9630285aabeeae220b2a4669b8490e028e6ab50bfd09077e85268d7d`.
- Stable ID: `t2d-i2-4bba28e1b18498c2d153a95a223fc411`.
- Trajectory ID:
  `5da9e1adc9465f5d3f1d55a6832c2ad0bbfd60ede685d1b7c7d44d1c2575b5fb`.
- Task group:
  `12195029f8542bdcd05c0759686031001c8cc4038d274d509ce3874888dd08b0`.
- Source step: 1.
- Native gold label: `search`.

## Verified Path

**Shared contract:** `parse_record(raw_record)` and `load_jsonl(source)[0]`
returned equal records, retained the complete metadata object, and did not
mutate the input. Native plain-string state now remains a string, while Hub
JSON-string state continues to decode.

**Trainer boundary:** `to_laya_items` resolved `record_id` from `metadata.id`
and retained the original metadata in the item's out-of-band fields. Constructing
the same item after removing metadata produced identical `ids`, `markers`,
`qtype`, target, labels, and option keys. `collate_items` exposed metadata only
under `meta`; model tensors were exactly `input_ids`, `attention_mask`,
`marker_pos`, `marker_mask`, and `qtype`.

**Accepted B2 checkpoint:** strict checkpoint loading succeeded. The row used
492 input tokens and six option markers. Prediction was `read`; native gold was
`search`; gold NLL was `2.0902161598205566`. This failed prediction does not
invalidate construction or execution interoperability.

**Reflex specialist:** the pinned
`convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`
loaded offline and consumed the unchanged row directly. Prediction was
`other_tool`; native gold was `search`. Load plus inference took
`6.338612936902791` seconds in this run.

The complete probabilities and machine-readable assertions are in
`artifacts/iteration_2/integration_result.json`, SHA-256
`07f4f1efad39116a13687965800b718045ab598105d844e8fd7f81457e6b42fb`.

## Cross-Track Repair Cycle

The first parent integration attempt stopped before producing a pass artifact.
It exposed three connected boundary defects, repaired together as the one
allowed cross-track repair cycle:

1. `shared.typed_decisions.parse_record` attempted JSON decoding for every
   string state even though native string state is valid; the v2 row therefore
   could not pass through `load_jsonl`.
2. Trainer construction only read legacy top-level `id`; v2 stable identity in
   `metadata.id` became `None`. It now remains available out of band without
   entering tokenization or tensors.
3. Laya rounds each choice probability to four decimals independently. Reflex
   required a `1e-6` sum tolerance and rejected the six-option specialist
   response. Reflex now permits only the mathematically bounded four-decimal
   rounding error (`K * 5e-5`), rejects larger errors, and normalizes accepted
   values to an exact distribution.

The first post-repair GPU attempt reached Reflex and failed with:

```text
ReflexModelError: Laya returned an invalid response:
response.answers.next_action.probabilities must sum to 1
```

That failed attempt is retained here rather than silently omitted. No training,
checkpoint selection, data change, or model retry occurred. The same frozen row
and checkpoints were used after the boundary repair.

## Commands And Results

CPU tests after repair:

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 15/15 passed.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
```

Result: 21/21 passed.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=pipelines/trace2decision .venv/bin/python -B -m unittest pipelines/trace2decision/i2_tests.py -v
```

Result: 4/4 passed.

Integration command:

```text
env TMPDIR=$PWD/data/cache/tmp HF_HOME=$PWD/data/cache/huggingface HF_HUB_CACHE=$PWD/data/cache/huggingface/hub TRANSFORMERS_CACHE=$PWD/data/cache/huggingface/transformers TORCH_HOME=$PWD/data/cache/torch XDG_CACHE_HOME=$PWD/data/cache/xdg HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 PYTHONPATH=apps/reflex .venv/bin/python scripts/verify_iteration_2_integration.py
```

Final result: `status=passed`; output written to
`artifacts/iteration_2/integration_result.json`. `git diff --check` passed.

Final implementation hashes:

```text
3e6b4f6116f56b24b59fc4bff804b2702b46e210a8f2c2778b3d77b4d2ad1dbc  shared/typed_decisions/schema.py
ea7d55e39d60deef940eb8baf4b3d451a7bd8dec9f7813420b0821ea8a4b3311  training/laya_local/data.py
aac336d54cae2d9ecd27e45ed4adc89ec5f279afca10df97c95fab9d02995985  apps/reflex/reflex/core.py
f2151b2ef0a486a70adf3d5503840ddf4bc6b83d702be57d5a55c17495978c99  scripts/verify_iteration_2_integration.py
```

## Evidence Classification

**Verified:** source-row hashes and immutability; shared parse/load equality;
metadata retention; model-field equality with metadata removed; collator tensor
boundary; strict B2 checkpoint execution; pinned Reflex specialist execution;
recorded predictions/probabilities; all stated test results.

**Inferred:** the source trace's latest action is an appropriate supervision
target for its state. The converter's separate verification supports the
mapping, but this integration run did not authenticate the original execution.

**Unknown:** action optimality, task success, accuracy over the complete v2
derivative, calibration on the trace domain, and generalization beyond this one
row. Neither wrong prediction is promoted to an accuracy claim.
