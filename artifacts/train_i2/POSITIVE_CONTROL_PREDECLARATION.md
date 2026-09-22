# Iteration 2 B1 Positive-Control Predeclaration

Frozen before loading either model or reading model predictions: 2026-09-21.

## Fixed evaluation

- Workflow: `agent_trace_observability` only. The pinned Iteration 1 derivative has
  no other typed workflows; no additional data will be searched for in B1.
- Subset: all 150 rows of
  `artifacts/trace2decision_i1/rerun2/test.jsonl`, in file order. This is the
  fixed public test derivative, with 150 choice questions and no score/noul
  questions.
- Subset SHA-256:
  `616d97e562b04723e8566028cb1a9117868580c94680db31d1f270ac334aa5b4`.
- Native-label target: the declared `gold[question_id].label`, mapped through
  the native question's option order. No probability argmax substitution.
- Evaluator/data construction: the same pinned Laya source
  `d113dca2512fb3eaca313534bc54c7162d87c1d4`, tokenizer, sequence builder,
  context (`max_len=512`, `head_max_len=192`), question order, and metric code
  will be used for both models.
- Models: base
  `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982` and
  specialist
  `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.

## Frozen metrics and decision rule

1. Primary: native-label choice accuracy (correct / 150).
2. Secondary: all-question native accuracy (same native hard labels; here it is
   also over 150 questions because this subset is choice-only).
3. Diagnostic: mean gold NLL, `-log(p_native_label)` using the model's returned
   native option distribution.

B1 is **PASS** only if the specialist clearly outperforms the base on the
primary metric under this identical evaluator. For this bounded subset,
“clearly” means a positive absolute improvement of at least 5 percentage
points (at least 8 additional correct answers), with the direction also
reported for all-question accuracy and NLL. Otherwise B1 is **BLOCK** and no
B2 work may start.

No threshold, row, seed, model, evaluator, or metric will be changed after
model predictions are loaded.
