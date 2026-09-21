# Laya Local Trainer Iteration 1 Implementation Report

Date: 2026-09-21

## Outcome

**VERIFIED:** The preserved bounded RTX 3060 training proof completed without OOM, with finite gradients, expected parameter-group behavior, checkpoint state loading, and identical inference after model reload.

**VERIFIED:** Repair 1 defines the primary metric as native `gold[qid].label` hard-label accuracy. Independent reconstruction on the unchanged fixed 40-choice dev subset is `19/40 = 0.475` to `11/40 = 0.275`, an absolute change of `-0.200`. Iteration 1 therefore does **not** satisfy the primary acceptance threshold of at least +0.05.

**VERIFIED:** The predeclared diagnostic moved in the learning direction: mean gold-distribution NLL across all 100 fixed dev questions improved from `1.4951586` to `1.2102497`, a reduction of `0.2849089` (19.05%). The post-failure tiny overfit diagnostic reduced one example's NLL from `2.2463284` to `1.1899292` and changed its prediction from the wrong option to the gold argmax.

**UNKNOWN:** No independent reviewer has found the fixed subset too small or noisy, so the NLL and tiny-overfit evidence cannot replace the failed primary criterion under `research/26_iteration_1_acceptance.md`.

## Implementation

**VERIFIED:** `shared/typed_decisions/schema.py` validates and loads the public native `state`, `questions`, and `gold` probability contract. It accepts the Hub's JSON string columns, requires matching question/gold keys, enforces supported `choice`/`score`/`noul` structures, preserves option order, and checks finite normalized probabilities.

**VERIFIED:** `training/laya_local/data.py` converts every native question to a Laya item through pinned Laya `build_sequence`, preserving probability values and option order. The measured run converted 64 train records to 320 items and 20 dev records to 100 items.

**VERIFIED:** `training/laya_local/train.py` implements gradient checkpointing, microbatch 1, accumulation 2, FP16 autocast with `GradScaler(init_scale=1.0)`, loss/gradient/VRAM logging, full-model checkpoint save, and checkpoint inference reload. New checkpoints also save data cursor/order and Python/NumPy/Torch/CUDA RNG state.
**VERIFIED:** `training/laya_local/resume.py` is a bounded continuation entry point restoring model, optimizer, scaler, step, cursor, and RNG state. One authorized resumed GPU update is recorded separately and does not alter primary results.

**VERIFIED:** The exact configuration was frozen in `config.json` and copied into `artifacts/train_i1/run/result.json`. The subsets were frozen before baseline in `artifacts/train_i1/PREDECLARATION.md` and `artifacts/train_i1/subsets/manifest.json`.

## Measurements

| Measurement | Result |
|---|---:|
| Optimizer updates | 32 |
| Forward/backward microbatches | 64 |
| Baseline plus training through synchronization (`train_seconds`, mislabeled in this run) | 19.258 s |
| Total measured run | 39.784 s |
| Peak CUDA allocated | 5.453 GiB |
| Peak CUDA reserved | 5.508 GiB |
| Gradients finite | yes |
| FP16 scale | 1.0 throughout |
| Baseline native-label exact choice accuracy | 0.475 |
| Post-training exact choice accuracy | 0.275 |
| Absolute primary change | -0.200 |
| Baseline gold NLL | 1.495159 |
| Post-training gold NLL | 1.210250 |
| Reload predictions exactly equal | yes |
| One-step resumed smoke | finite gradients; intended parameters changed; action head unchanged; step 32 → 33 |

The complete 32-point loss curve and pre-clip gradient norms are in `artifacts/train_i1/run/result.json`. Per-question baseline and post predictions are in `baseline.json` and `post_training.json`.

## Parameter Evidence

| Group | SHA-256 before | SHA-256 after | Result |
|---|---|---|---|
| Encoder | `f7aa4c8a40271b2ade427b18938155e6dc78998125051eb56ab112c3960a4308` | `32115417feee1541d328b70a347ab7ff0cc7ec41cbb66a9ee7dc3d24e8f7f03e` | changed as intended |
| Scorer/head | `5e1269c1d9dbb381248adeb9923bdb9604548373ad3da1e3967ebc717e7e004c` | `9699269317ba9f0fbbf0450d0acf2da9fb3034762d2775119386c838b40fd9a3` | changed as intended |
| Action head | `baad5de68b70b686f58b9718ea7a8de82085d43450a17cfc591cffd7b1490d2d` | `baad5de68b70b686f58b9718ea7a8de82085d43450a17cfc591cffd7b1490d2d` | unchanged as intended |

## Provenance And Reproduction

**VERIFIED:** Laya source revision was `d113dca2512fb3eaca313534bc54c7162d87c1d4`; model revision was `1c5edc17a7acd8701df6fc341c0d179f1c62c982`; dataset revision was `ea9306458d6e9563628369a3d1e72e362fb381d2`. Source Parquet, subset, and checkpoint hashes are in `artifacts/train_i1/CHECKSUMS.sha256`; exact commands are in `artifacts/train_i1/COMMANDS.md`.

**VERIFIED:** Hardware was NVIDIA GeForce RTX 3060 12 GiB, driver `610.57.04`; software included Python 3.14.7, PyTorch 2.11.0+cu128, Transformers 5.17.0, safetensors 0.8.0, NumPy 2.5.3, and PyArrow 25.0.1.

## Limitations

**VERIFIED:** PyTorch warned that memory-efficient and flash-attention backward kernels are nondeterministic despite `warn_only=True` deterministic algorithms. This limits bitwise replay claims but does not affect the recorded checkpoint reload equality within this run.

**INFERRED:** The simultaneous NLL improvement and argmax-accuracy decline are consistent with optimization toward noisy soft teacher distributions over too few sampled updates, rather than a broken target or gradient path. This interpretation is not established causally by one seed.

**UNKNOWN:** Whether a larger prespecified training budget, choice-stratified sampler, different optimizer, or multiple seeds would improve fixed-dev choice accuracy was not tested because that would exceed this iteration's bounded-run and diagnosis limits.

**RECOMMENDED:** Treat this iteration as a functioning trainer proof but an acceptance blocker. The native-label metric reconstruction and one-step continuation repair do not constitute a pass. Any later retry should be separately predeclared and must not retroactively replace this frozen run.
