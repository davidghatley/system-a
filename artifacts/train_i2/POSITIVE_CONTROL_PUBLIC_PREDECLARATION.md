# B1 Public Positive-Control Predeclaration

Frozen before any new model is loaded and before any new prediction: 2026-09-21.

## Immutable source and pins

- Dataset: `LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2`
- Config/split: `agent_trace_observability/test`
- Existing frozen file: `artifacts/train_i1/source_test.parquet`
- Source SHA-256: `833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d`
- Base: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`
- Specialist: `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`
- Laya source: `d113dca2512fb3eaca313534bc54c7162d87c1d4`

The predeclaration content hash is recorded as the SHA-256 of this file with
the `Predeclaration content SHA-256` line removed (this avoids an impossible
self-referential file hash): `a8d824c7767ce220147bb98c3d1788b6e3540db63cf2f7f5233ab98debd166a6`.

## Exact rows and order

All 100 rows are retained in Parquet file order, with the public `id`,
`workflow`, `state`, `questions`, and `gold` fields preserved. The JSON-string
columns are decoded only at evaluation input; no Trace2Decision reshaping,
question filtering, or reordering occurs.

```text
agent_trace_observability_000000, agent_trace_observability_000001, agent_trace_observability_000002, agent_trace_observability_000003, agent_trace_observability_000004, agent_trace_observability_000005, agent_trace_observability_000006, agent_trace_observability_000007, agent_trace_observability_000008, agent_trace_observability_000009, agent_trace_observability_000010, agent_trace_observability_000011, agent_trace_observability_000012, agent_trace_observability_000013, agent_trace_observability_000014, agent_trace_observability_000015, agent_trace_observability_000016, agent_trace_observability_000017, agent_trace_observability_000018, agent_trace_observability_000019, agent_trace_observability_000020, agent_trace_observability_000021, agent_trace_observability_000022, agent_trace_observability_000023, agent_trace_observability_000024, agent_trace_observability_000025, agent_trace_observability_000026, agent_trace_observability_000027, agent_trace_observability_000028, agent_trace_observability_000029, agent_trace_observability_000030, agent_trace_observability_000031, agent_trace_observability_000032, agent_trace_observability_000033, agent_trace_observability_000034, agent_trace_observability_000035, agent_trace_observability_000036, agent_trace_observability_000037, agent_trace_observability_000038, agent_trace_observability_000039, agent_trace_observability_000040, agent_trace_observability_000041, agent_trace_observability_000042, agent_trace_observability_000043, agent_trace_observability_000044, agent_trace_observability_000045, agent_trace_observability_000046, agent_trace_observability_000047, agent_trace_observability_000048, agent_trace_observability_000049, agent_trace_observability_000050, agent_trace_observability_000051, agent_trace_observability_000052, agent_trace_observability_000053, agent_trace_observability_000054, agent_trace_observability_000055, agent_trace_observability_000056, agent_trace_observability_000057, agent_trace_observability_000058, agent_trace_observability_000059, agent_trace_observability_000060, agent_trace_observability_000061, agent_trace_observability_000062, agent_trace_observability_000063, agent_trace_observability_000064, agent_trace_observability_000065, agent_trace_observability_000066, agent_trace_observability_000067, agent_trace_observability_000068, agent_trace_observability_000069, agent_trace_observability_000070, agent_trace_observability_000071, agent_trace_observability_000072, agent_trace_observability_000073, agent_trace_observability_000074, agent_trace_observability_000075, agent_trace_observability_000076, agent_trace_observability_000077, agent_trace_observability_000078, agent_trace_observability_000079, agent_trace_observability_000080, agent_trace_observability_000081, agent_trace_observability_000082, agent_trace_observability_000083, agent_trace_observability_000084, agent_trace_observability_000085, agent_trace_observability_000086, agent_trace_observability_000087, agent_trace_observability_000088, agent_trace_observability_000089, agent_trace_observability_000090, agent_trace_observability_000091, agent_trace_observability_000092, agent_trace_observability_000093, agent_trace_observability_000094, agent_trace_observability_000095, agent_trace_observability_000096, agent_trace_observability_000097, agent_trace_observability_000098, agent_trace_observability_000099
```

## Common settings and frozen metrics

Both checkpoints use identical code, source row order, and settings:
`max_len=512`, `head_max_len=192`, native Laya tokenizer/sequence builder,
and one `predict(state, questions)` call per row. The specialist card
publishes a 1024-token context, while the base card publishes 512; 512 is
chosen here as the common fair comparison and is not a reproduction of the
specialist card benchmark.

- Primary: native-label choice accuracy over the 200 choice questions.
- Type-correct all-question hard accuracy: returned probability argmax for
  choice, score, and noul (400 questions total).
- Per-type and per-workflow accuracy, plus mean gold NLL
  `-log(returned_probability[gold_label])`, are reported.
- Unchanged gate: specialist must improve primary choice accuracy by at least
  5 absolute points, equivalent to at least 8 additional correct choice
  answers. Otherwise B1 is FAIL/BLOCK and B2 is not started.

No rows, settings, pins, metrics, or gate will change after predictions.
