# Trainer B2 Pre-Run Repair Cycle 1

Date: 2026-09-21

## Outcome

All five changes required by `B2_PRE_RUN_REVIEW.md` were implemented in the
isolated B2 path. The immutable superseding declaration is
`B2_PREDECLARATION_REPAIR_1.md`, full-file SHA-256
`b7a9515266b0bf504b149b664ab7b6dcc644d8f48e6b0d4c7288a564e4500de7`.
Its separate `.sha256` file is required by the runner. The original declaration,
original checksum, review, subset files/manifest, and all B1/I1 evidence were
not modified.

No model was loaded, CUDA was not initialized, and no inference, baseline
evaluation, optimizer step, or training occurred. The B2 main command was not
run and no `artifacts/train_i2/b2_run` output was created.

## Implemented Repairs

1. Named-parameter guards now require `421293827` total, `394781696` encoder,
   and `26512131` non-encoder elements. A separate guard requires `421293830`
   state-dict elements and a registered three-element `temperature` buffer that
   is absent from named parameters.
2. Literal train/dev/manifest hashes are enforced before the Torch import.
   Loaded records and manifest IDs are checked against independently constructed
   ID arrays, and every ordered record is compared with direct reconstruction
   from the already hash-verified Parquets.
3. GradScaler now uses `init_scale=64.0` and `growth_interval=1000`. A scale
   decrease after `step/update` is a hard overflow/skipped-step failure before
   scheduler or update-counter advancement. A CPU tensor test verifies the
   64-way accumulation/scaling algebra.
4. Each live optimizer, scheduler, and scaler state dict is cloned back to CPU
   and compared with saved state after loading. The focused test covers both a
   valid round trip and a deliberately faulty live scaler load.
5. The fixed B2 `71/200` baseline guard and `81/200` gate were removed. `71/200`
   is recorded only as the B1 cross-check; B2 requires 200 finite baseline
   choice results and final correct at least measured B2 baseline correct plus
   10 over the same 200 choices.

All other frozen recipe choices are unchanged.

## Frozen File Hashes

```text
a42fd2fd176ea6bc0e68e371be81eeb8be624caa021f096f594e8bfda7689cb4  training/laya_local/b2_config.json
0cabc5ffdee25c1e5af7c8a980b36dc20897f973eaf490635ef99d1c88660967  training/laya_local/b2_train.py
7b0609536a8aeb6e62d735b0501c1f58cdf94b6a20b20ad391b446108cfaa677  training/laya_local/b2_prepare.py
3ed831526de44adaecdb2a2fa6711ecb9d9128b6c19f276115a0908693077b22  training/laya_local/tests/test_b2.py
d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656  artifacts/train_i2/b2_subset/train.jsonl
73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e  artifacts/train_i2/b2_subset/dev.jsonl
569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056  artifacts/train_i2/b2_subset/manifest.json
b7a9515266b0bf504b149b664ab7b6dcc644d8f48e6b0d4c7288a564e4500de7  artifacts/train_i2/B2_PREDECLARATION_REPAIR_1.md
```

## Verification

CPU-only tests:

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 14 tests passed, including 8 focused B2 tests.

The focused subset test independently verifies literal output/manifest hashes,
pinned source hashes, exact ID arrays, row counts, ordered JSON content, and
direct Parquet equivalence. A separate static audit verifies source ordering of
all hash/declaration/subset checks before `import torch`, repaired constants,
and absence of the former fixed baseline gate. In-memory compilation covered 12
Python files. The superseding declaration checksum verified successfully and
`git diff --check` passed.

These are verified CPU/static results. Full-model structure, GPU fit, B2
baseline, full-run numerical stability, checkpoint behavior at model scale,
runtime, and final accuracy remain unknown until separately authorized
execution.

One preliminary standalone subset-audit command failed from an argument-count
typo in its inline audit expression. It stopped in audit code without changing
files or entering any model path. The corrected independent audit then passed;
the repository subset regression test had already passed before this typo.
