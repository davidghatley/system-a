# Trace2Decision I2 repair cycle 1 evidence

## VERIFIED

- `REVIEW.md` was read first and was not modified.
- The converter now audits the full bounded source candidate before optional
  512 history selection, using the pinned local Laya tokenizer path.
- Pre-policy min/mean/max token lengths are 187/4244.48/16656. Counts over
  512/768/1024 are 1504/1402/1285 (97.41%/90.8%/83.23%). Post-policy final
  distributions are separate in `output/report.json`.
- Newest-fitting selection is unchanged; selected history is rendered in
  source-index order. Tests explicitly catch reversed history.
- 1,544 rows and split counts remain train 1,235, dev 159, test 150; IDs,
  labels, protected-field policy, and v1 files were preserved.
- Deterministic verifier passed: 1,544 rows, 1,544 stable IDs, valid shared
  records, equal report, and equal train/dev/test/sample hashes. New v2 hashes:
  train `79255bd82a26c663845ff8c82a97e3d912b0e19a7231cc8cec2e480e837b5e27`,
  dev `eed09d564f1ae17cabb2ef47bec8ae94aaa08c41ccd6da692149f14b0b47a540`,
  test `01272aa8bb7f1b95b2c82092ae55ebcb037b3e960e70141d7ab1538c34abd9ba`,
  samples `ae359f131b984697f850f1972b3b8be7e45be0d714f06a306b8df1d1ceb64dea`.

## Commands

```text
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=pipelines/trace2decision .venv/bin/python -B pipelines/trace2decision/i2_convert.py --source artifacts/trace2decision_i1/source/traces.jsonl --output-dir artifacts/trace2decision_i2/output
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=pipelines/trace2decision .venv/bin/python -B -m unittest pipelines/trace2decision/i2_tests.py -v
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=pipelines/trace2decision .venv/bin/python -B pipelines/trace2decision/i2_verify.py --source artifacts/trace2decision_i1/source/traces.jsonl --expected-dir artifacts/trace2decision_i2/output --rerun-dir artifacts/trace2decision_i2/rerun --evidence artifacts/trace2decision_i2/rerun_verification.json
```

## UNKNOWN

The tokenizer emitted a non-fatal warning for an unselected pre-policy
candidate above its model maximum; this does not affect final Laya construction.
No training or downstream learnability was tested.
