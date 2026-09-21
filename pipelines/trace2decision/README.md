# Trace2Decision Iteration 1

Deterministic conversion of the pinned GLM-5.2 coding-agent traces into the
native contract: `state`, one `choice` question, and a one-hot `gold`
distribution. Generated rows validate with `shared.typed_decisions`.

```bash
python3 pipelines/trace2decision/convert.py \
  --source artifacts/trace2decision_i1/source/traces.jsonl \
  --output-dir artifacts/trace2decision_i1/output
python3 -m unittest discover -s pipelines/trace2decision -p 'test_*.py'
```

The labels reproduce observed teacher behavior for behavioral cloning. They do
not assert that the selected action was optimal. No reasoning content is used.
