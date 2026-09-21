# Reflex Iteration 1 Final Verification

Date: 2026-09-21. Verifier: independent final verifier. No implementation or documentation files were modified.

## Status: ACCEPTED

All acceptance items in `REVIEW.md` were rechecked after the two permitted repairs. The repaired malformed-response handling, normalized binary probabilities, and source-integrity policy are verified below.

## Commands and verified results

Working directory: `/home/davidhatley/Projects/research/system_a`

```text
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
```

**VERIFIED:** 15 tests passed in 0.251 s.

```text
.venv/bin/python -m compileall -q apps/reflex/reflex apps/reflex/tests
PYTHONPATH=apps/reflex .venv/bin/python -m reflex --help
.venv/bin/python -m json.tool apps/reflex/examples/agent_state.json
.venv/bin/python -m json.tool artifacts/reflex_i1/example_benchmark.json
```

**VERIFIED:** compile, help, and both JSON checks exited 0.

```text
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader,nounits
.venv/bin/python -c "import importlib.metadata as m, platform; print(platform.python_version()); print({x:m.version(x) for x in ['torch','transformers','safetensors','huggingface-hub','numpy']})"
```

**VERIFIED environment:** Python 3.14.7; NVIDIA GeForce RTX 3060, driver 610.57.04, 12288 MiB, compute capability 8.6; torch 2.11.0+cu128, transformers 5.17.0, safetensors 0.8.0, huggingface-hub 1.32.0, numpy 2.5.3.

```text
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --benchmark --warmup 3 --samples 10
```

**VERIFIED:** exit 0, offline, device `cuda:0`, 3 warmups and 10 samples. Measured scope was tokenization plus inference, excluding model load: min 38.966 ms, median 39.689 ms, nearest-rank p95 40.527 ms, max 40.527 ms. Selected answers were `true` and `publish`; both disagreed with the supplied gold argmaxes, consistent with the documented advisory/non-policy limitation.

**VERIFIED pinned assets:** source revision `d113dca2512fb3eaca313534bc54c7162d87c1d4`; checkpoint revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`; model SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.

## Targeted behavioral checks

**VERIFIED:** direct fake-agent checks returned `ReflexModelError` with actionable field paths for `{}`, `NaN`, and out-of-range `noul`; no traceback was produced by the error boundary. A fake `noul=0.12345` produced `{"false": 0.8765, "true": 0.1235}`, sum exactly `1.0`.

**VERIFIED:** source status was only ignored generated `laya/__pycache__/`; the 15-test suite independently covers tracked Python changes, untracked Python payloads, ignored native payloads, and permitted bytecode caches. The offline load therefore exercised the permitted clean source-integrity path.

## Residual risks / unknowns

- **INFERRED:** fixed-environment repeated inference should be stable because evaluation mode has no sampling; cross-hardware, driver, and library bitwise stability is not established.
- **UNKNOWN:** calibration, accuracy, decision utility, CPU latency, and behavior on very long production states were not measured.
- **UNKNOWN:** the integrity check cannot prevent a file race after status validation and before import, and native suffix coverage is not cross-platform exhaustive.
- The dependency lock constrains the tested software environment but does not guarantee bitwise portability.

## Write-scope check

The only requested artifact written by this verification was `artifacts/reflex_i1/VERIFICATION.md`. No commit was created; implementation paths were not edited.
