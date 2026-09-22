# B1 Public Positive-Control Review

Date: 2026-09-21. This is the corrected Cycle 1 result; the earlier
`positive_control.json` and its invalid derivative predeclaration are preserved
as historical evidence and are not used here.

## Determination

**PASS.** On the predeclared public `agent_trace_observability/test` subset,
the specialist is 136/200 choice answers (0.680) versus the base's 71/200
(0.355): **+32.5 absolute points and +65 correct answers**. This exceeds the
unchanged gate of >=5 points / >=8 additional correct choice answers. B1
permits the parent to consider B2; B2 was not started.

## Frozen construction and provenance (VERIFIED)

- Dataset/config/split: `LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2`, `agent_trace_observability/test`.
- File: `artifacts/train_i1/source_test.parquet`; 100 rows, 500 questions.
- Source SHA-256: `833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d`.
- Predeclaration full-file SHA-256: `5bb9eeb19c033265a91b0f20c131aeca6307f0566e158cef2cde1d31bf26a7c2`.
- Recorded predeclaration content hash (hash excluding its self-reference line): `a8d824c7767ce220147bb98c3d1788b6e3540db63cf2f7f5233ab98debd166a6`.
- The descriptive total correction and reproducible hash procedure are recorded
  in `artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION_ERRATA.md`;
  the immutable as-run predeclaration is unchanged.
- Base: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Specialist: `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.
- Local Laya source: `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
- Identical settings: `max_len=512`, `head_max_len=192`, native tokenizer and
  sequence builder, public JSON fields decoded without reshaping, file order
  and all five questions retained.

The upstream base card specifies 512 context; the specialist card specifies
1024 context and reports the typed-decisions recipe as RLCD fine-tuning from
the base on 1,200 training cases / 6,000 decisions. 512 was therefore chosen
and disclosed as the common fairness setting, not as reproduction of the
specialist's 1024-token card benchmark.

## Measurements (VERIFIED)

| model | choice | score | noul | all-question hard | mean gold NLL |
|---|---:|---:|---:|---:|---:|
| base | 71/200 = 0.355 | 65/200 = 0.325 | 58/100 = 0.580 | 194/500 = 0.388 | 1.3373310093 |
| specialist | 136/200 = 0.680 | 148/200 = 0.740 | 84/100 = 0.840 | 368/500 = 0.736 | 0.8008518995 |

All-question hard accuracy uses returned probability argmax: native choice
labels for `choice`, integer level keys for `score`, and synthesized
`false=1-noul`, `true=noul` probabilities for `noul`. Gold NLL uses the
returned probability assigned to each native gold label. Per-workflow/type
records are in the JSON output (this subset has one workflow).

## Commands and checks (VERIFIED)

1. `.venv/bin/python -m unittest training.laya_local.tests.test_typed_decisions -v`
   — 6 tests run; 5 passed, 1 pre-existing failure in
   `test_rejects_non_distribution` because the shared validator says “sum to
   1” while the old test expects “sum to one”. New public JSON and score/noul
   semantic tests passed.
2. `.venv/bin/python -m py_compile training/laya_local/positive_control.py training/laya_local/metrics.py training/laya_local/data.py training/laya_local/tests/test_typed_decisions.py`
   — passed.
3. `sha256sum artifacts/train_i1/source_test.parquet` — required source hash
   matched.
4. `mkdir -p data/cache/tmp artifacts/train_i2 && env TMPDIR=$PWD/data/cache/tmp HF_HOME=$PWD/data/cache/huggingface HF_HUB_CACHE=$PWD/data/cache/huggingface/hub TRANSFORMERS_CACHE=$PWD/data/cache/huggingface/transformers TORCH_HOME=$PWD/data/cache/torch XDG_CACHE_HOME=$PWD/data/cache/xdg .venv/bin/python training/laya_local/positive_control.py`
   — completed on NVIDIA GeForce RTX 3060 12 GiB; base 16.13 s, specialist
   13.22 s; no training.

Output: `artifacts/train_i2/positive_control_public.json`, SHA-256
`6bfed0f71e395d8b1f29b3e9f8a462d942c7900a45b305517ede6a81be9d29dc`.

## Evidence limits

The gate is verified only for this predeclared 100-case public workflow test
subset and this common 512-token inference setting. The result does not claim
the specialist's published aggregate benchmark is reproduced. No B2 training
or evaluation was performed.
