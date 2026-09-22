# Iteration 3 Reflex Replay Review

## Decision: CHANGES_REQUIRED

**VERIFIED:** `SUITE_FREEZE.md` documents pre-replay selection, pins three dev rows by SHA-256, and keeps majority success/failure plus known specialist failure visible. CLI shows exact compact state, observed (not optimality) reference, probabilities, model revision, correctness, low-margin flag, elapsed scope, and warnings. The specialist's 6.338613s is attributed to model-load-plus-inference; replay time is separate. `unittest` 6/6, compileall, diff-check, full CLI, named failure CLI, and integrity assertions passed CPU-only/offline.

**FALSIFICATION:** the stored JSON is schema-validated, but its source SHA only describes `integration_result.json`; prediction values are not bound to that evidence. I changed probabilities in memory, retained the SHA, and replay accepted the altered prediction. “Preserves the accepted specialist result” is therefore unenforceable. The three cases are diagnostic, not quality evidence; broader usefulness is UNKNOWN.

**Required repair:** add and verify a tamper-evident stored-prediction manifest (or validate every claimed source field against the accepted artifact), plus a regression test rejecting altered probabilities. Re-run and record the suite.

Commands: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python -m unittest discover -s apps/reflex_replay/tests -v`; same environment `python -m reflex_replay`, `--case majority-failure`; `python -m compileall -q apps/reflex_replay`; `git diff --check`; read-only hash/in-memory tamper checks.
