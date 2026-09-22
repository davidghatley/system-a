# Reflex I2 Repair Cycle 1

Date: 2026-09-21

## VERIFIED

- `shared.typed_decisions` now validates every question before optional `gold`; gold-less records with unsupported types, blank instructions, or invalid criteria fail before model invocation.
- Gold entries, when present, require native `{type, label, probabilities}` and bare Iteration-1 distributions are rejected. Reflex reference evaluation consumes only that native shape.
- `apps/reflex/examples/agent_state.json` has native gold objects. A comparison against `HEAD` verified `state_questions_unchanged=True`; labels and probability values remain `inspect`/`false` with the original one-hot distributions.
- README expected output now describes the default specialist and separately reports the base comparison.
- The test suite passed: 20 tests, exit 0. Compileall passed, exit 0.
- The unchanged-semantic example runs passed offline on `cuda:0` for both checkpoints. Both selected `publish` and `true`, with 449 input tokens. Specialist output reports `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`; base output reports `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Specialist benchmark passed offline with 3 warmups and 10 samples, excluding model load: min 39.933 ms, median 41.089 ms, p95 nearest-rank 42.455 ms, max 42.455 ms.

## Commands and results

All commands were run from `/home/davidhatley/Projects/research/system_a`.

```text
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
Ran 20 tests ... OK (exit 0)

.venv/bin/python -m compileall -q apps/reflex/reflex shared/typed_decisions
exit 0

PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --checkpoint base
exit 0; stdout: artifacts/reflex_i2/base_output.json; stderr: artifacts/reflex_i2/base_stderr.txt

PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --checkpoint specialist
exit 0; stdout: artifacts/reflex_i2/specialist_output.json; stderr: artifacts/reflex_i2/specialist_stderr.txt

PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --checkpoint specialist --benchmark --warmup 3 --samples 10
exit 0; stdout: artifacts/reflex_i2/specialist_benchmark.json; stderr: artifacts/reflex_i2/specialist_benchmark_stderr.txt

python -c 'import json,subprocess; old=json.loads(subprocess.check_output(["git","show","HEAD:apps/reflex/examples/agent_state.json"])); new=json.load(open("apps/reflex/examples/agent_state.json")); print("state_questions_unchanged=", old["state"] == new["state"] and old["questions"] == new["questions"]); print("gold_labels=", {k:v["label"] for k,v in new["gold"].items()}); print("gold_probabilities=", {k:v["probabilities"] for k,v in new["gold"].items()})'
state_questions_unchanged= True
gold_labels= {'next_control_mode': 'inspect', 'allow_publish_now': 'false'}
gold_probabilities= {'next_control_mode': {'inspect': 1.0, 'act': 0.0, 'publish': 0.0, 'escalate': 0.0}, 'allow_publish_now': {'false': 1.0, 'true': 0.0}}
```

Output hashes from the rerun are:

```text
base_output.json       5d2f6024dcb6f349f3d9507962402096d2d5e28d2b89faa989f4d20598a9bc35
specialist_output.json bdde3fd047b6354273d439e4b28645abc232f7fde9b75855d0837fc601ec0337
benchmark.json         2ca4f28a1c8e1d28ca2daf5b1c762bb74b6bcc0f664ddbceac7995c1c8dc2411
```

## INFERRED

- Native gold migration preserves runtime semantics because state/questions, labels, distributions, selected answers, and model probabilities are unchanged in the recorded comparison. This is a semantic-preservation inference supported by the verification above, not an accuracy claim.

## UNKNOWN

- Accuracy, calibration, CPU latency, representative-domain performance, cross-hardware bitwise stability, and long-state behavior remain unmeasured.
- The benchmark is a bounded local measurement and does not establish production latency.
