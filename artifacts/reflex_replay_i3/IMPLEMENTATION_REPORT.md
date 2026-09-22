# Reflex Replay Iteration 3 implementation report

Date: 2026-09-22
Starting commit: `2d4408d`
Owned writes: `apps/reflex_replay/**`, `artifacts/reflex_replay_i3/**`

## Result

Implemented a standard-library, CPU-only terminal replay for one task:
**observed next-action prediction/evaluation from real compact-state records**.
It replays stored predictions without loading Reflex/Laya and compares them with
a frozen training-majority rule. It displays the exact compact state, model and
revision, full probabilities, observed action, argmax correctness, a fixed
low-margin flag, elapsed time and scope, and explicit advisory/provenance
warnings.

The honest developer workflow is inspecting whether a model or cheap baseline
agreed with what a traced coding agent did next, especially on failures and
low-margin outputs. This can debug model/data behavior and compare stored runs.
It cannot control an agent or establish that any action is advisable.

## Frozen suite

`apps/reflex_replay/SUITE_FREEZE.md` was written before executing the replay.
The manifest pins three real v2 dev rows by line SHA-256:

| Case | Observed reference | Available result | Outcome |
| --- | --- | --- | --- |
| accepted-integration | `search` | majority and accepted specialist | both wrong; specialist top-two margin 0.0216 |
| majority-success | `execute` | majority | correct |
| majority-failure | `edit` | majority | wrong |

Reference labels are observed source-trace actions. The suite was deliberately
selected and has only three correlated records from one task trajectory, so no
accuracy is computed or implied. The specialist result exists for only the
accepted integration row; absent results are not fabricated or imputed.

## Verified measurements

- Frozen v2 training counts: edit 178, execute 540, other_tool 0, read 241,
  respond_or_finish 133, search 143; total 1,235. Thus `execute` is the simple
  majority comparator with probability 0.4372.
- Accepted specialist prediction: `other_tool`; observed `search`; wrong.
  Top-two margin 0.0216, below the predeclared 0.05 display threshold.
- Accepted specialist elapsed value: 6.338613 seconds. This is copied from the
  Iteration 2 integration and covers model load plus inference, not replay time.
- Full three-case CPU replay elapsed 0.003031 seconds in the verification run.
  This covers local file loading, hash/schema validation, rule construction,
  and rendering. It is one run, not a benchmark.
- Tests: 6/6 passed. Compileall and `git diff --check` passed.

The first named-case execution exposed a filtering defect: full-suite stored
predictions looked foreign after the suite was prematurely narrowed. Filtering
was moved after full-suite ingestion and a regression test was added. The
subsequent full verification passed.

## Evidence classification

**Verified:** source hashes and observed labels; accepted specialist values are
unchanged from `artifacts/iteration_2/integration_result.json`; majority counts;
displayed argmax/margins; test and command results above.

**Inferred:** replaying compact inputs and stored outputs is useful for developer
inspection and future run comparison.

**Unknown:** source physical execution, teacher identity, rights, task success,
action optimality, probability calibration, source/model overlap, broad model
quality, and generalization. The model result is one row and is not an accuracy
estimate. The fixed margin is a display heuristic, not validated uncertainty.

## Changed paths

- `apps/reflex_replay/README.md`
- `apps/reflex_replay/SUITE_FREEZE.md`
- `apps/reflex_replay/reflex_replay/`
- `apps/reflex_replay/examples/`
- `apps/reflex_replay/tests/`
- `artifacts/reflex_replay_i3/IMPLEMENTATION_REPORT.md`
- `artifacts/reflex_replay_i3/COMMANDS_RESULTS.md`

No model was loaded; no GPU, network, package installation, or shared/existing
application modification was used. No commit was created.
