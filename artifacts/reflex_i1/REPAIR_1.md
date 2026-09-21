# Reflex repair cycle 1 evidence

Date: 2026-09-21. Changes are limited to `apps/reflex/**` and this artifact.

## VERIFIED

- Raw responses are validated for complete IDs/types, required fields, finite numbers, probability ranges/sums, and score output. Malformed post-inference data becomes `ReflexModelError`; the CLI returns 2 without a traceback.
- Binary display rounds the true side once and derives false as its displayed complement.
- The loader checks the pinned revision and rejects dirty, untracked, or ignored source-tree contents.
- `requirements-lock.txt` records the tested versions from the prior environment evidence: torch 2.11.0, transformers 5.17.0, safetensors 0.8.0, huggingface-hub 1.32.0, numpy 2.5.3.

## Commands

Working directory: `/home/davidhatley/Projects/research/system_a`

```text
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
.venv/bin/python -m compileall -q apps/reflex/reflex apps/reflex/tests
PYTHONPATH=apps/reflex .venv/bin/python -m reflex --help
PYTHONPATH=apps/reflex HF_HOME="$PWD/data/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m reflex apps/reflex/examples/agent_state.json --offline --device cuda:0 --benchmark --warmup 1 --samples 2
```

The unit, compile, and help checks passed (12 tests). The bounded offline example was attempted and correctly failed before model loading because the existing checkout contains ignored `data/laya/laya/__pycache__/`; this is the enforced integrity policy, not a model failure. No network or paid service was used.

## INFERRED / UNKNOWN

- The clean-tree policy covers Git-visible tracked, untracked, and ignored files; racing filesystem changes after the check are not prevented.
- The dependency lock improves reproducibility for the tested software environment but does not establish bitwise portability across hardware or drivers.
