# Trainer B2 Pre-Run Verification After Repair Cycle 1

Date: 2026-09-21
Role: independent repair-cycle verifier

## Decision

**ACCEPTED_FOR_ONE_MAIN_RUN.**

All five repairs required by `B2_PRE_RUN_REVIEW.md` are present, internally
consistent with the superseding predeclaration, and covered by passing CPU-only
tests and independent static/data audits. No model was loaded, CUDA was not
initialized, and no inference, baseline evaluation, optimizer step, or training
was performed during this verification.

This decision authorizes exactly one invocation of the following command, with
no retries:

```text
env TMPDIR=$PWD/data/cache/tmp HF_HOME=$PWD/data/cache/huggingface HF_HUB_CACHE=$PWD/data/cache/huggingface/hub TRANSFORMERS_CACHE=$PWD/data/cache/huggingface/transformers TORCH_HOME=$PWD/data/cache/torch XDG_CACHE_HOME=$PWD/data/cache/xdg .venv/bin/python training/laya_local/b2_train.py
```

## Required Repairs

### 1. Model State Accounting: Verified

- `validate_model_structure` requires `421293827` named-parameter elements,
  split into `394781696` encoder and `26512131` non-encoder elements.
- It separately requires `421293830` state-dict elements and a registered
  three-element `temperature` buffer absent from `named_parameters()`.
- The optimizer partition still covers every named parameter exactly once, and
  the non-encoder group includes `act_head`.
- The focused regression test for these distinctions passes. The full model was
  not instantiated, as prohibited; the exact full-model guard remains a
  run-time check supported by the previously reviewed diagnostic and pinned
  `register_buffer` source.

### 2. Independent Subset Enforcement: Verified

- Before Torch is imported, the runner compares the manifest, train JSONL, and
  dev JSONL against literal hashes independent of the manifest.
- It verifies both pinned Parquet hashes, independently constructs both exact
  ID arrays, and compares those arrays with loaded records and manifest IDs.
- It reconstructs the first 256 train rows and all 100 dev rows directly from
  the pinned Parquets and requires exact ordered serialized-record equality.
  This covers IDs, row/question order, criteria/probability order, and gold
  values.
- The independent rerun reproduced exact source/JSONL equality and counts of
  256/100 rows and 1,280/500 questions.

### 3. FP16 Scaling And Step Guard: Verified

- Config and runner use `init_scale=64.0`, `growth_interval=1000`, accumulation
  64, and exactly 80 updates. Thus `scale(loss/64)` presents the scale-1
  un-divided loss magnitude to each backward while unscale produces the intended
  accumulated average.
- The CPU algebra regression test passes.
- A scale decrease after `scaler.step()`/`scaler.update()` is an immediate
  no-retry failure. Scheduler stepping and update-counter advancement occur only
  after this guard passes. Existing finite objective, gradient, and gradient-norm
  guards remain.

### 4. Live Checkpoint Reload Verification: Verified

- After each `load_state_dict`, the runner reads the live optimizer, scheduler,
  and scaler state, clones it to CPU, and compares it exactly with the saved
  state.
- Serialized-payload equality, model digest equality, and exact
  update/microforward/epoch checks remain.
- The focused round-trip test passes and its deliberately faulty scaler loader
  is rejected.

### 5. Measured B2 Baseline Gate: Verified

- `71/200` remains only the recorded B1 cross-check; no exact B2 baseline guard
  or fixed `81/200` gate remains.
- Baseline evaluation must yield exactly 200 choice results and finite NLL
  metrics before training.
- The final primary gate is exactly `final choice correct >= measured B2
  baseline choice correct + 10`, over the same frozen 200 choices, matching the
  five-point Iteration 2 acceptance criterion.

## Hashes And Pins

The two checksum files verified with `sha256sum -c`. Independently recomputed
SHA-256 values are:

```text
a9f913ff3d09e00361e26169e620e156390013443fb09da328b2dc7e4f4fa8da  artifacts/train_i2/B2_PREDECLARATION.md
2ce52d92c2e80d27740eeb83df24d175bb19a2d9d51983004337b3cbab762836  artifacts/train_i2/B2_PREDECLARATION.sha256
b7a9515266b0bf504b149b664ab7b6dcc644d8f48e6b0d4c7288a564e4500de7  artifacts/train_i2/B2_PREDECLARATION_REPAIR_1.md
0d58fcccae76d04a30a7c53b94413fc418dccd7e8a0ee595c6ca8a952835ec9d  artifacts/train_i2/B2_PREDECLARATION_REPAIR_1.sha256
26caa358d0c65eb0069cbb1188817493a6ab129c3e18e58d8983cfec37a5c972  artifacts/train_i2/B2_PRE_RUN_REVIEW.md
a42fd2fd176ea6bc0e68e371be81eeb8be624caa021f096f594e8bfda7689cb4  training/laya_local/b2_config.json
0cabc5ffdee25c1e5af7c8a980b36dc20897f973eaf490635ef99d1c88660967  training/laya_local/b2_train.py
7b0609536a8aeb6e62d735b0501c1f58cdf94b6a20b20ad391b446108cfaa677  training/laya_local/b2_prepare.py
3ed831526de44adaecdb2a2fa6711ecb9d9128b6c19f276115a0908693077b22  training/laya_local/tests/test_b2.py
d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656  artifacts/train_i2/b2_subset/train.jsonl
73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e  artifacts/train_i2/b2_subset/dev.jsonl
569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056  artifacts/train_i2/b2_subset/manifest.json
f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2  data/laya/laya/common.py
2c37036054dfb92073b518e994cb66c8a9009fb8d15b060d9bdfc6d4f94b1d39  data/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
e7b2f78fe539517c6a66a90c68ed671c6133194a88d27dbff5d84713af46b901  artifacts/train_i1/source_train.parquet
833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d  artifacts/train_i1/source_test.parquet
891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c  cached model.safetensors
ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd  cached rl_agent_config.json
```

The local Laya checkout is clean at
`d113dca2512fb3eaca313534bc54c7162d87c1d4`, matching the pin.

## Import Order And Execution State

Static AST/source inspection confirms the first Torch import in `main` occurs
only after config validation and its literal hash check; frozen subset hashes,
source hashes, IDs, and direct Parquet reconstruction; notebook and Laya-source
hash checks; superseding declaration/checksum verification; and Laya revision
verification. The pre-Torch imports used by subset verification do not import
Torch. Environment/cache settings are then established before Torch import.

`artifacts/train_i2/b2_run` does not exist. No evidence of a prior B2 main
invocation was found.

Filesystem mtimes support this sequence: B1 final authorization at 17:03,
original B2 implementation/subset/freeze by 17:25, independent review at 17:37,
repaired config/runner/tests by 17:44, superseding declaration at 17:45 and
checksum at 17:46, then repair report at 17:48. This is supporting evidence,
not a trusted cryptographic timestamp. The original and superseding declarations
and separate checksums are byte-for-byte consistent with their recorded hashes.

## CPU And Static Verification

All commands ran from the repository root with no model loading, inference, or
training.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 14 tests ran and all 14 passed, including all 8 focused B2 tests.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<in-memory compile of training/laya_local, its tests, and shared/typed_decisions>'
```

Result: 14 Python files compiled in memory.

Independent repository-local Python audits, run with CUDA hidden, reported:

```text
runner static/import-order audit: PASS
independent subset/source audit: PASS
config/schedule audit: PASS
```

`git diff --check` also passed. Existing unrelated worktree changes were left
untouched.

## Remaining Unknowns

Full-model structure at instantiation, actual FP16 baseline, complete-run
numerical stability, peak VRAM, checkpoint behavior at model scale, wall time,
and final accuracy remain unknown until the single authorized main invocation.
These are guarded run-time outcomes, not pre-run acceptance blockers.

This verification wrote only
`artifacts/train_i2/B2_PRE_RUN_VERIFICATION.md`. It did not modify
implementation, config, subsets, predeclarations, prior reports, or source data,
and it created no commit.
