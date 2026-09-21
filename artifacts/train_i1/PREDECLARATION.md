# Local Trainer Iteration 1 Predeclaration

Frozen before baseline or post-training evaluation on 2026-09-21.

- Source: `LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2`, config `agent_trace_observability`.
- Train subset: first 64 rows in pinned train Parquet source order.
- Dev subset: first 20 rows in pinned test Parquet source order.
- Primary metric: exact choice accuracy over every `choice` question in the fixed dev rows. Improvement threshold: at least 5 absolute percentage points.
- Diagnostic selected before results: mean gold-distribution NLL over all fixed dev questions. A tiny-set overfit may only diagnose target/gradient correctness and does not independently satisfy acceptance.
- Run: seed 42, 32 optimizer updates, microbatch 1, gradient accumulation 2, FP16 autocast with scaler initial scale 1.0, max length 512, head budget 192, encoder LR 2.5e-5, scorer LR 1e-4.
- No development record or metric may be changed after baseline inspection.
