# Reflex Replay Iteration 3 commands and results

All commands ran from repository root on 2026-09-22.

## Primary command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay \
  python -m reflex_replay
```

Result: exit 0; three frozen records displayed. Summary:

```text
accepted-integration:
  majority -> execute, observed search, WRONG, margin 0.2421
  convaiinnovations/laya-typed-decisions@f9ab0b... -> other_tool,
  observed search, WRONG, AMBIGUOUS/LOW-MARGIN, margin 0.0216,
  stored elapsed 6.338613s (model load plus inference)
majority-success:
  majority -> execute, observed execute, CORRECT, margin 0.2421
majority-failure:
  majority -> execute, observed edit, WRONG, margin 0.2421
REPLAY ELAPSED: 0.003031s
```

The terminal output also printed each complete compact state, all six
probabilities, model revisions, prediction provenance, and both warnings.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay \
  python -m unittest discover -s apps/reflex_replay/tests -v
```

Result: exit 0; 6/6 passed. Covered correct, wrong, low-margin/ambiguous,
invalid-distribution, named-case, and CLI error paths.

## Static checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q apps/reflex_replay
git diff --check
```

Result: both exit 0 with no output.

## Input measurement command

The training-label counts and deterministic dev candidates were inspected with
a read-only Python standard-library command over:

```text
artifacts/trace2decision_i2/output/train.jsonl
artifacts/trace2decision_i2/output/dev.jsonl
```

Observed counts: `{'search': 143, 'execute': 540, 'read': 241, 'edit': 178,
'respond_or_finish': 133}`; no `other_tool` positives; total 1,235.

## Resource boundary

Only Python standard-library CPU commands ran. There was no model loading,
GPU use, network access, training, inference, or dependency installation.
