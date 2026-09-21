# Reflex Iteration 1 Commands

Working directory for every command: `/home/davidhatley/Projects/research/system_a`

```bash
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
PYTHONPATH=apps/reflex .venv/bin/python -m reflex --help
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader,nounits
.venv/bin/python -c "import importlib.metadata as m, platform; print(platform.python_version()); print({x:m.version(x) for x in ['torch','transformers','safetensors','huggingface-hub','numpy']})"
git -C data/laya rev-parse HEAD
sha256sum "data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/model.safetensors"
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --benchmark --warmup 3 --samples 10
.venv/bin/python -m json.tool apps/reflex/examples/agent_state.json
.venv/bin/python -m json.tool artifacts/reflex_i1/example_benchmark.json
.venv/bin/python -m compileall -q apps/reflex/reflex apps/reflex/tests
```

The inference command was executed twice: once before adding explicit reference-agreement output and once after it. The final run is `example_benchmark.json`; the earlier latency observation is retained in `preliminary_benchmark.json`. The command both runs the realistic example and measures latency. No network or paid API is used.
