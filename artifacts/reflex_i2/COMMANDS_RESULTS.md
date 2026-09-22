# Reflex Iteration 2 Commands and Results

Working directory: `/home/davidhatley/Projects/research/system_a`.

## Provenance and compatibility

Hub source inspected at revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`:
`hf://models/convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2/rl_agent_config.json`.
It reports `model_name=laya-typed-decisions`, ModernBERT-large, and `max_len=1024`.

```text
git -C data/laya rev-parse HEAD
d113dca2512fb3eaca313534bc54c7162d87c1d4

HF_HOME="$PWD/data/cache/huggingface" .venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('convaiinnovations/laya-typed-decisions', revision='f9ab0b228f0fc0f14d873dbc99038f135c2da1b2', cache_dir='data/cache/huggingface/hub'))"
.../models--convaiinnovations--laya-typed-decisions/snapshots/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2

sha256sum data/cache/huggingface/hub/models--convaiinnovations--laya-typed-decisions/snapshots/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2/model.safetensors
4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader,nounits
NVIDIA GeForce RTX 3060, 610.57.04, 12288, 11081, 8.6
```

## Tests and compatibility test

```text
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
Ran 18 tests ... OK
.venv/bin/python -m compileall -q apps/reflex/reflex shared/typed_decisions
exit 0
```

The suite includes native gold, optional gold, stable Trace2Decision fields, and an assertion that metadata is not passed into model state. Tests require no weights.

## Same unchanged input, base and specialist

Input: `apps/reflex/examples/agent_state.json` (unchanged from Iteration 1).

```text
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --checkpoint base
exit 0; full stdout: artifacts/reflex_i2/base_output.json

PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --checkpoint specialist
exit 0; full stdout: artifacts/reflex_i2/specialist_output.json
```

Base selected `publish` and `true`; specialist selected `publish` and `true`. Both full JSON outputs include checkpoint revision, device, usage, probabilities, and reference evaluation.

## Specialist benchmark

```text
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --checkpoint specialist --benchmark --warmup 3 --samples 10
exit 0; full stdout: artifacts/reflex_i2/specialist_benchmark.json
```

VERIFIED result: device `cuda:0`; warmup 3; samples 10; min 40.150 ms, median 41.093 ms, p95 nearest rank 41.893 ms, max 41.893 ms; scope tokenization plus inference, excluding model load.

Output SHA-256: base `5d2f6024dcb6f349f3d9507962402096d2d5e28d2b89faa989f4d20598a9bc35`; specialist `bdde3fd047b6354273d439e4b28645abc232f7fde9b75855d0837fc601ec0337`; benchmark `1b2e03e40f21b588cdf1559a0c950a62c9d87673fafa282c876b6af37d88170e`.
