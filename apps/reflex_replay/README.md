# Reflex Replay

Reflex Replay answers one narrow developer-useful question:

> Given the compact state available at one recorded agent step, what next action
> did a stored model prediction select, and did it match the action subsequently
> observed in the source trace?

It is a CPU-only terminal inspection tool for real Trace2Decision v2 records and
stored prediction results. It shows the exact compact input, immutable model
identity, full probabilities, observed reference, argmax correctness, top-two
margin, elapsed-time scope, and provenance warnings. It also compares every case
with the training-split majority rule.

It does not load a model, execute tools, control an agent, recommend an action,
or establish reliable predictive quality. Agreement with an observed action is
not evidence that the action was correct or optimal.

## Run

From the repository root, with Python 3.10+ and no dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay \
  python -m reflex_replay
```

The command displays all three frozen cases, including unfavorable cases. To
inspect a shorter success or failure while retaining the default suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay \
  python -m reflex_replay --case majority-success

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay \
  python -m reflex_replay --case accepted-integration
```

## Stored Result Ingestion

Use `--predictions PATH` for a JSONL file. One result has this contract:

```json
{
  "record_id": "stable suite record ID",
  "model": {"id": "model identity", "revision": "immutable revision"},
  "probabilities": {
    "edit": 0.1,
    "execute": 0.2,
    "other_tool": 0.1,
    "read": 0.3,
    "respond_or_finish": 0.1,
    "search": 0.2
  },
  "elapsed_seconds": 0.04,
  "elapsed_scope": "inference for one complete record",
  "provenance": {"source": "path/to/evidence.json"}
}
```

This is the future live-adapter boundary: an adapter may write this schema after
inference, then the replay remains model-independent. Duplicate results for a
record are allowed so models can be compared. Each stored result must also have
an `integrity` envelope. Its artifact identity binds source path/line, frozen
source-line SHA-256, canonical compact-state SHA-256, record ID, and observed
reference. The envelope digest covers that identity plus every other prediction
field (including probabilities, model, latency, and reference), using sorted-key
UTF-8 JSON with compact separators. Unknown record IDs, malformed
distributions, non-finite values, missing identity, or digest mismatches fail
closed with exit 2. This provides integrity against accidental or unauthorised
edits, not authenticity: the hash is not a signature and cannot prove who
created the file.
The checked-in stored result is copied from accepted Iteration 2 evidence; no
new inference was performed.

## Fixed Evidence

- `SUITE_FREEZE.md` records selection before showcase execution.
- `examples/frozen_suite.json` pins source line hashes and observed labels.
- `examples/stored_predictions.jsonl` preserves the accepted specialist result.
- The simple comparator is `execute`, the v2 training majority: 540/1,235 rows.
  Its displayed probabilities are the six training frequencies; `other_tool`
  has no training positive in this derivative.
- Low-margin means a top-two gap of at most 0.05. This fixed display flag is not
  a calibrated confidence or validated deferral policy.

The suite has three dev records and is deliberately too small and selected to
estimate accuracy. The specialist result exists for only one case and is wrong
and low-margin there. Missing specialist results for the other records are not
imputed. The source is machine-generated and publisher-attributed; physical
execution, teacher identity, rights, action optimality, task success, and
source/model overlap were not independently authenticated.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay \
  python -m unittest discover -s apps/reflex_replay/tests -v
```

Tests cover correct, wrong, ambiguous/low-margin, malformed-distribution, and
CLI error behavior without weights, GPU, network, or third-party packages.
