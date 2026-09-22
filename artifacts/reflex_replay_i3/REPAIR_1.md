# Repair 1

**VERIFIED:** Stored predictions now require a SHA-256 envelope. Canonical bytes
are UTF-8 JSON, recursively sorted keys, compact separators, and
`allow_nan=false` (`json-sort-keys-utf8-v1`). The digest covers the complete
prediction payload plus an artifact identity containing source path/line,
frozen source-line SHA-256, canonical compact-state SHA-256, record ID, and
observed reference. Replay recomputes the identity from the frozen source and
fails closed on identity, digest, or bound-reference mismatch.

**VERIFIED:** Added tamper tests for probabilities, model, reference, latency,
and compact source hash. Existing success, wrong, ambiguous, and error paths
remain present. `REVIEW.md` was not modified.

Commands/results (repository root, CPU/offline):

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python -m unittest discover -s apps/reflex_replay/tests -v  -> 8/8 OK
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python -m reflex_replay -> exit 0; 3 cases
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python -m reflex_replay --case majority-failure -> exit 0
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q apps/reflex_replay -> OK
git diff --check -> OK
```

**LIMITATION:** This is tamper integrity, not authenticity: the local hash is
not a signature and cannot identify who generated or edited the file.
