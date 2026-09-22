# Reflex I2 Final Verification

Date: 2026-09-21
Status: **ACCEPTED**

## Scope and provenance

Verified from `research/27_iteration_2_acceptance.md`, `artifacts/reflex_i2/REVIEW.md`, `artifacts/reflex_i2/REPAIR_1.md`, the final shared/Reflex source and README, and the existing Reflex I2 artifacts.

- Laya source checkout: `data/laya` at `d113dca2512fb3eaca313534bc54c7162d87c1d4`; only ignored generated `__pycache__` files were present.
- Default specialist: `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.
- Explicit base: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Specialist `model.safetensors` SHA-256: `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.
- Installed versions match `apps/reflex/requirements-lock.txt`: torch `2.11.0+cu128`, transformers `5.17.0`, safetensors `0.8.0`, huggingface-hub `1.32.0`, numpy `2.5.3`.

These are VERIFIED local measurements. Cross-hardware portability and upstream compatibility outside the pinned source/checkpoints remain UNKNOWN.

## Commands and verified results

All commands ran from `/home/davidhatley/Projects/research/system_a`.

```text
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
Ran 20 tests ... OK; exit 0

.venv/bin/python -m compileall -q apps/reflex/reflex shared/typed_decisions
exit 0
```

The full Reflex suite passed, including the newly required malformed gold-less question cases, legacy-gold rejection, native Trace2Decision metadata handling, and model-boundary assertions.

Direct validation rerun:

```text
goldless malformed type REJECTED
goldless malformed instructions REJECTED
goldless malformed criteria REJECTED
legacy gold REJECTED ValidationError
model state keys [] metadata in questions False
trace result ... argmax_agrees=True
```

Thus malformed questions are rejected before model invocation even when `gold` is absent; the bare Iteration-1 gold mapping is rejected; native `{type, label, probabilities}` is accepted; and Trace2Decision metadata is retained by parsing but excluded from Laya inputs. This is VERIFIED.

Offline commands rerun without rewriting the existing evidence files (stdout was observed directly):

```text
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --checkpoint base
exit 0; selected publish / true; input_tokens 449; base revision 1c5edc17...

... same command with --checkpoint specialist
exit 0; selected publish / true; input_tokens 449; specialist revision f9ab0b22...

... specialist with --benchmark --warmup 3 --samples 10
exit 0; device cuda:0; warmup 3; samples 10; min 39.625 ms; median 40.150 ms; p95 nearest-rank 40.977 ms; max 40.977 ms; model load excluded
```

The base and specialist consumed the unchanged Iteration-1 semantic example and produced the same selected answers. This establishes execution compatibility, not accuracy. The specialist benchmark is a bounded local RTX 3060 measurement, not a production-latency claim.

## Documentation and acceptance checks

- README default specialist output matches the rerun: `true`/`publish`, probabilities `0.4196/0.5804` and `0.2362/0.1734/0.2678/0.3226`, specialist revision documented. The separate base comparison is explicitly labeled and matches its rerun. VERIFIED.
- `shared.typed_decisions` is the Reflex validation contract; no Reflex-private alternate gold validator remains. VERIFIED by source inspection and tests.
- Exact model, source, dependency, and local weight pins are recorded above and in the adjacent artifacts. VERIFIED.
- Warning behavior, actionable errors, offline loading, and metadata exclusion passed. VERIFIED.

## Residuals

Accuracy, calibration, representative-domain performance, CPU latency, long-state behavior, and cross-hardware/library bitwise stability were not measured. They are UNKNOWN, not acceptance failures. The unchanged example is a semantic smoke test and is not a specialist benchmark of decision quality. No evidence of paid APIs, training, broad dataset search, or Iteration 3 work was found in the reviewed Reflex record; absence beyond that reviewed scope is not independently proven.

## Change-scope confirmation

Before this report, the working tree already contained the Iteration 2 implementation/artifact changes listed by `git status --short`. During this verification I did not modify implementation or existing files, and reruns wrote no artifact files. The only file created by me is this file: `artifacts/reflex_i2/VERIFICATION.md`. No commit was made.
