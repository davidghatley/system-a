# exp_001b corrective execution

Working directory: `/home/davidhatley/Projects/research/system_a`.
Base code commit: `f2e1666eaa1e367d9ce1722d49466be98ce4704d` plus the corrective working tree; exact executed builder SHA-256 is in run provenance. The final corrective commit contains these bytes and results.
Pinned Laya source: `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
Protocol SHA-256, calculated **before dataset execution**:
`59a35040cb1518c4781d432a8864bc13b8deb770c163fcb028fdc49b2db07e59`.

```bash
sha256sum experiments/exp_001b_hermes_preprocessing/protocol_v0.2.0.json
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$PWD/data/cache/huggingface" XDG_CACHE_HOME="$PWD/data/cache" CUDA_VISIBLE_DEVICES="" .venv/bin/python -B experiments/exp_001b_hermes_preprocessing/scripts/test_preprocessing.py 2>&1 | tee artifacts/exp_001b/preexecution_tests.log
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$PWD/data/cache/huggingface" XDG_CACHE_HOME="$PWD/data/cache" CUDA_VISIBLE_DEVICES="" .venv/bin/python -B experiments/exp_001b_hermes_preprocessing/scripts/build_dataset.py > artifacts/exp_001b/execution_console.log 2>&1
```

Fixture assertions passed before the full converter was started. The converter refuses execution-directory reuse. All processing is CPU/tokenizer-only; no model weights, inference, training, or official evaluation.

Result: **FAIL** (exit 1 by design). Failed gates: `complete_adjacent_linkage`, `label_over_budget_difference_lte_20pp`. Independent bounded verification commands and observed reconciliations are recorded in `research/23_exp001b_independent_verification.md`.
