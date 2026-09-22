# B1 Positive-Control Review

Date: 2026-09-21

## Decision

**BLOCK — B1 PASS criterion not met.** The specialist is directionally better
than the base, but its native-label accuracy gain is only 4/150 (+2.67 absolute
points), below the predeclared +5-point / 8-question threshold. B2 was not
started.

This is a verified bounded measurement, not a claim about the public benchmark
or other workflows.

## Frozen evaluation and provenance

- Input: `artifacts/trace2decision_i1/rerun2/test.jsonl`, 150 rows / 150
  choice questions, choice-only.
- Input SHA-256:
  `616d97e562b04723e8566028cb1a9117868580c94680db31d1f270ac334aa5b4`.
- Trace2Decision source revision: `1371ed38f8890d0520a53bc7ad850308eb4d7a22`.
- Laya source: `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
- Base: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Specialist: `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.
- Identical construction: pinned source, native JSON records, option order,
  `max_len=512`, `head_max_len=192`, and one-question-at-a-time `Agent.predict`.
- Native labels came from `gold[question_id].label`; the evaluator used the
  returned probability assigned to that native label for NLL.

The public specialist card was read before implementation. Its published
benchmark figures are context only; these local results use the frozen subset
and do not import those claims.

## Results (VERIFIED)

| Model | Choice accuracy | All-question native accuracy | Mean gold NLL | Correct / total | Wall time |
|---|---:|---:|---:|---:|---:|
| Base | 0.173333 | 0.173333 | 2.608410 | 26 / 150 | 16.090 s |
| Specialist | 0.200000 | 0.200000 | 1.981532 | 30 / 150 | 14.378 s |
| Specialist - base | +0.026667 | +0.026667 | -0.626878 | +4 | 30.468 s total |

All-question accuracy equals choice accuracy because the frozen workflow
subset contains no `score` or `noul` questions. The specialist improves NLL,

## Commands and checks

Evaluation (RTX 3060, repository-local caches):

```bash
mkdir -p data/cache/tmp artifacts/train_i2
env TMPDIR="$PWD/data/cache/tmp" HF_HOME="$PWD/data/cache/huggingface" HF_HUB_CACHE="$PWD/data/cache/huggingface/hub" TRANSFORMERS_CACHE="$PWD/data/cache/transformers" TORCH_HOME="$PWD/data/cache/torch" XDG_CACHE_HOME="$PWD/data/cache/xdg" .venv/bin/python training/laya_local/positive_control.py
```

Observed twice; the second run produced the measurements recorded in
`positive_control.json`. No paid API, training, or cloud compute was used.

Bounded verification:

```bash
.venv/bin/python -m unittest discover -s training/laya_local/tests -v
.venv/bin/python -m py_compile training/laya_local/positive_control.py training/laya_local/metrics.py training/laya_local/data.py
```

## Limitations (UNKNOWN / INFERRED)

- **UNKNOWN:** Whether the specialist would pass on a larger, independently
  frozen subset or the other typed-decision workflows; no additional data was
  searched or evaluated for B1.
- **INFERRED:** The substantial NLL reduction with a small accuracy gain is
  consistent with improved probability alignment without enough evidence for a
  clear hard-label win. It does not establish causality.
