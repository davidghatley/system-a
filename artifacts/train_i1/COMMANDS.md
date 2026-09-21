# Reproduction Commands

Run from repository root. No paid API or cloud compute was used.

## Unit Tests

```bash
.venv/bin/python -m unittest discover -s training/laya_local/tests -v
.venv/bin/python -m py_compile shared/typed_decisions/schema.py training/laya_local/data.py training/laya_local/prepare_subset.py training/laya_local/train.py training/laya_local/tiny_overfit.py
```

Observed twice: three tests passed; compilation produced no output.

## Pinned Public Data

```bash
curl --fail --location --output artifacts/train_i1/source_train.parquet "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/ea9306458d6e9563628369a3d1e72e362fb381d2/agent_trace_observability/train-00000-of-00001.parquet"
curl --fail --location --output artifacts/train_i1/source_test.parquet "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/ea9306458d6e9563628369a3d1e72e362fb381d2/agent_trace_observability/test-00000-of-00001.parquet"
.venv/bin/python training/laya_local/prepare_subset.py --train-parquet artifacts/train_i1/source_train.parquet --dev-parquet artifacts/train_i1/source_test.parquet
```

The resulting IDs and hashes are in `subsets/manifest.json`. The predeclaration is `PREDECLARATION.md`.

## Bounded Run

```bash
env TMPDIR="$PWD/data/cache/tmp" \
  PIP_CACHE_DIR="$PWD/data/cache/pip" \
  HF_HOME="$PWD/data/cache/huggingface" \
  HF_HUB_CACHE="$PWD/data/cache/huggingface/hub" \
  HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets" \
  TRANSFORMERS_CACHE="$PWD/data/cache/huggingface/transformers" \
  TORCH_HOME="$PWD/data/cache/torch" \
  XDG_CACHE_HOME="$PWD/data/cache/xdg" \
  .venv/bin/python training/laya_local/train.py \
  --config training/laya_local/config.json \
  --train artifacts/train_i1/subsets/train.jsonl \
  --dev artifacts/train_i1/subsets/dev.jsonl \
  --output-dir artifacts/train_i1/run
```

## Permitted Diagnostic

Run only after the primary failed:

```bash
env TMPDIR="$PWD/data/cache/tmp" \
  PIP_CACHE_DIR="$PWD/data/cache/pip" \
  HF_HOME="$PWD/data/cache/huggingface" \
  HF_HUB_CACHE="$PWD/data/cache/huggingface/hub" \
  TRANSFORMERS_CACHE="$PWD/data/cache/huggingface/transformers" \
  TORCH_HOME="$PWD/data/cache/torch" \
  XDG_CACHE_HOME="$PWD/data/cache/xdg" \
  .venv/bin/python training/laya_local/tiny_overfit.py
```

## Revisions

- System-A working base: `fa19b50ebaf2502b1e3908f98602a7d7e20fdf63` (the iteration files are intentionally uncommitted).
- Laya source: `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
- Laya model: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Dataset: `LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2`.

## Repair 1 verification

```bash
.venv/bin/python -m unittest discover -s training/laya_local/tests -v
.venv/bin/python -m py_compile shared/typed_decisions/schema.py training/laya_local/data.py training/laya_local/metrics.py training/laya_local/prepare_subset.py training/laya_local/train.py training/laya_local/resume.py training/laya_local/tiny_overfit.py
env TMPDIR="$PWD/data/cache/tmp" HF_HOME="$PWD/data/cache/huggingface" HF_HUB_CACHE="$PWD/data/cache/huggingface/hub" HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets" TRANSFORMERS_CACHE="$PWD/data/cache/huggingface/transformers" TORCH_HOME="$PWD/data/cache/torch" XDG_CACHE_HOME="$PWD/data/cache/xdg" .venv/bin/python training/laya_local/resume.py --config training/laya_local/config.json --train artifacts/train_i1/subsets/train.jsonl --checkpoint artifacts/train_i1/resume_smoke/checkpoint --output artifacts/train_i1/resume_smoke/result.json
```

The resume command was authorized and bounded to one GPU optimizer update. It did not rerun the frozen 32-step experiment.
