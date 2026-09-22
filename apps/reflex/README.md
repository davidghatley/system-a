# Reflex

Reflex is a small local observability tool: give it an agent `state` and typed `questions`, and it reports the pinned public Laya checkpoint's selected answer and full probability distribution. The example asks whether a release agent should inspect, act, publish, or escalate before its artifact signature is verified.

It is useful as a visible advisory signal for agent orchestration and intervention experiments. It does **not** execute tools, enforce a policy, prove safety, or establish that checkpoint probabilities are semantically calibrated for arbitrary agent states.

## Quickstart

Run from the repository root. Python 3.10+ is recommended.

```bash
python -m venv .venv
PIP_CACHE_DIR="$PWD/data/cache/pip" .venv/bin/pip install -r apps/reflex/requirements.txt -c apps/reflex/requirements-lock.txt
git clone https://github.com/NandhaKishorM/laya.git data/laya
git -C data/laya checkout d113dca2512fb3eaca313534bc54c7162d87c1d4
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" \
  .venv/bin/python -m reflex apps/reflex/examples/agent_state.json
```

The source checkout is pinned to Laya `d113dca2512fb3eaca313534bc54c7162d87c1d4`. The default specialist is `convaiinnovations/laya-typed-decisions` at `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`; it uses the same `laya.agent.Agent` contract as the explicit base comparison `convaiinnovations/laya` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Use `--checkpoint base` for that comparison. The lock is the tested dependency environment; it does not claim bitwise portability across hardware or drivers. No paid service or API key is used. Thereafter, prohibit network access explicitly:

```bash
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" \
  .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline
```

The output includes, for every question, `selected`, `probabilities`, and checkpoint `confidence`, plus checkpoint revision, actual device, token usage, and a scope warning. If `gold` is present, `reference_evaluation` makes argmax agreement visible without supplying labels to the model. Output values depend on hardware/library numerics, but evaluation mode has no sampling and repeated runs on a fixed environment are expected to be stable.

Expected output shape for the default specialist (the measured RTX 3060 run selected these answers):

```json
{
  "answers": {
    "allow_publish_now": {
      "selected": "true",
       "probabilities": {"false": 0.4196, "true": 0.5804}
    },
    "next_control_mode": {
      "selected": "publish",
       "probabilities": {"act": 0.2362, "escalate": 0.1734, "inspect": 0.2678, "publish": 0.3226}
    }
  },
  "reference_evaluation": {
    "allow_publish_now": {"gold_argmax": "false", "model_argmax": "true", "argmax_agrees": false},
    "next_control_mode": {"gold_argmax": "inspect", "model_argmax": "publish", "argmax_agrees": false}
  }
}
```

The disagreement is intentional evidence of the limitation: these checkpoint scores should be observed, not obeyed as a safety policy.

For the explicit base comparison (`--checkpoint base`), the corresponding measured probabilities are `false=0.2477, true=0.7523` and `act=0.3133, escalate=0.088, inspect=0.2563, publish=0.3424`; it selected `true` and `publish`. The base revision is `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.

## Python API

```python
import json
from reflex import Reflex

record = json.load(open("apps/reflex/examples/agent_state.json"))
model = Reflex.from_checkpoint(offline=True)
result = model.decide(record)
print(result["answers"]["allow_publish_now"]["probabilities"])
```

`gold` is optional evaluation metadata and is validated but never supplied to the model. Native gold is `{type, label, probabilities}` keyed by question ID. Stable Trace2Decision fields such as `id`, `trajectory_id`, `task_group`, `source_step`, and `metadata` are preserved, but only `state` and `questions` enter Laya.

## Latency

Model load is deliberately excluded; tokenization and inference are included:

```bash
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" \
  .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline \
  --device cuda:0 --benchmark --warmup 3 --samples 10
```

## Errors And Limits

- Malformed JSON/schema errors exit with code 2 and identify the bad field.
- `--offline` fails with a download instruction when the exact revision is not cached.
- The source checkout must be exactly the pinned commit. Changed tracked source and untracked/ignored import payloads (`.py` and native modules) are rejected; generated `__pycache__/*.pyc` files are harmless.
- The checkpoint is a 421M-parameter bidirectional decision model, not a language model or universal agent policy.
- Input wording, truncation, option descriptions, checkpoint training coverage, stale inherited option-count calibration, and domain shift can materially affect scores.
- `confidence` is normalized entropy for choice/score and the larger option probability for binary `noul`; neither is an independently validated probability of correctness here.
- The upstream action/escalation head is intentionally not exposed; its semantics are undocumented for this use.
- CPU works but is substantially slower. Device placement and inference failures are surfaced as actionable errors.

## Tests

Tests do not need model weights:

```bash
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
```
