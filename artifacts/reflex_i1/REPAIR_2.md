# Reflex repair cycle 2 evidence

Date: 2026-09-21. Final allowed repair. Changes are limited to `apps/reflex/**` and this artifact.

## VERIFIED

- The integrity check permits ignored generated `__pycache__/*.pyc`/`.pyo` entries.
- It rejects tracked modifications and untracked/ignored `.py` payloads, plus native module payloads (`.so`, `.pyd`, `.dll`, `.dylib`). Tests cover each required behavior.
- Full test command passed: `PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v` (15 tests, OK).
- Bounded offline CUDA example passed from the existing checkout: `PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --benchmark --warmup 1 --samples 2` (exit 0; CUDA device `cuda:0`; 2 samples; JSON output included benchmark and answers).
- `git status --short` and `git diff -- apps/reflex artifacts/reflex_i1` were inspected before completion; no files outside the permitted write paths were edited by this repair. Existing unrelated workspace changes remain untouched.

## INFERRED

- Ignoring only bytecode cache entries preserves normal post-import Python behavior without allowing source or native import substitution.

## UNKNOWN

- This check does not prevent a payload from being changed after the Git status check and before import.
- Cross-platform native-loader behavior beyond the recognized suffixes was not exercised.

## CONTEXT

- Pinned Laya source revision: `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
- Pinned checkpoint revision: `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
