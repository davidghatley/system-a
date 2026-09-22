# Public Positive-Control Predeclaration Erratum

Date: 2026-09-21

This erratum corrects the descriptive total in the immutable as-run file
`artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md`. That file is
preserved byte-for-byte; its original full-file SHA-256 is:

`5bb9eeb19c033265a91b0f20c131aeca6307f0566e158cef2cde1d31bf26a7c2`

The text `(400 questions total)` in the all-question metric description was
incorrect. The public subset contains 500 questions in total: 200 `choice`,
200 `score`, and 100 `noul`. The 400 in the original description was a
descriptive/documentation error; it does not alter rows, predictions, settings,
pins, metrics, or the gate.

## Reproducible content-hash procedure

For the immutable as-run file, run this from the repository root:

```text
sha256sum artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md
```

This produces the original full-file value recorded above. The predeclaration's
recorded content-hash method is to hash the file bytes after removing only the
complete self-reference hash line (line 17), while retaining the surrounding
paragraph and all other bytes. Reproduce that method with:

```text
.venv/bin/python -c 'from pathlib import Path; import hashlib; p=Path("artifacts/train_i2/POSITIVE_CONTROL_PUBLIC_PREDECLARATION.md"); lines=p.read_bytes().splitlines(keepends=True); print(hashlib.sha256(b"".join(x for i,x in enumerate(lines,1) if i != 17)).hexdigest())'
```

For the immutable as-run file, that procedure produces
`bdf907ff22833aaa183f9e0f548809c089938a3eca9390d1a121c200033f321c`.
This is a reproducible post-run documentation check, not pre-run cryptographic
proof that the file existed before model loading or prediction.

If the descriptive total had been corrected before applying that same
procedure, the expected corrected content hash would be
`f94bd744fc667ad37b8891c33321ff5e1374cc9887440a990b53807d8c710e29`.
The frozen as-run file is intentionally not rewritten to apply that correction.
