# Trainer B2 Implementation Report

Date: 2026-09-21

## Outcome

Trainer B2 is implemented and frozen as a separate path. No model was loaded,
CUDA was not initialized, and no inference, baseline evaluation, optimizer
step, or training was run. The B2 main command remains unexecuted.

The predeclaration was completed before any model execution and has ordinary
full-file SHA-256
`a9f913ff3d09e00361e26169e620e156390013443fb09da328b2dc7e4f4fa8da`,
stored separately in `artifacts/train_i2/B2_PREDECLARATION.sha256`.

## Implemented Files

- `training/laya_local/b2_config.json`: complete immutable pins, schedule,
  objective constants, metric gate, and hardware limit.
- `training/laya_local/b2_prepare.py`: deterministic Parquet-to-JSONL freezer
  that preserves source row and nested insertion order and permits output only
  under `artifacts/train_i2`.
- `training/laya_local/b2_train.py`: isolated single-GPU B2 runner with no CPU
  fallback or retry path.
- `training/laya_local/tests/test_b2.py`: CPU-only tests for schedule/config
  guards, subset hashes/order/counts, reward math, perturbation projection,
  advantage centering, zero action gradient, and checkpoint-state copying.
- `artifacts/train_i2/b2_subset/train.jsonl`: first 256 pinned train rows.
- `artifacts/train_i2/b2_subset/dev.jsonl`: all 100 pinned test rows.
- `artifacts/train_i2/b2_subset/manifest.json`: exact IDs, source/output hashes,
  and row/question/type counts.
- `artifacts/train_i2/B2_PREDECLARATION.md`: immutable pre-run protocol.
- `artifacts/train_i2/B2_PREDECLARATION.sha256`: separate full-file hash.
- `artifacts/train_i2/B2_IMPLEMENTATION_REPORT.md`: this durable record.

No existing Iteration 1 runner file was changed. No Iteration 1 artifact, B1
artifact, or independent-review file was edited.

## Verified Notebook Details

Direct inspection of the pinned notebook, rather than prose alone, established:

- Preprocessing calls `build_sequence` before the later training-script
  `1024/256` assignments, so actual notebook items use the base config's
  `512/192` and default `truncate_left=False` (`ipynb` JSON lines 98-121 and
  225-245).
- The notebook uses four epochs and sets sigma once per epoch by linear
  interpolation from 0.4 to 0.1, producing 0.4, 0.3, 0.2, 0.1 (lines 248-255,
  273-281).
- Every name containing `encoder.` enters the encoder AdamW group and every
  other named parameter enters the head group (lines 257-263). This includes
  `act_head`.
- Perturbations are masked Gaussian noise projected to zero option mean;
  perturbed logits start from detached logits; rewards use
  `proper_reward(..., w_sph=0.75, w_rps=1.0)`; advantages are group-centered
  then divided by one default PyTorch standard deviation; the Gaussian policy
  loss is added to full-weight soft CE (lines 299-320).
- `+ 0.0 * act.sum()` preserves zero action task gradient. Because the action
  parameters are in AdamW and receive zero rather than absent gradients,
  decoupled weight decay remains notebook-faithful.

The reward implementation was also checked directly in pinned
`data/laya/laya/common.py`: target-weighted log score with floor -9.21,
spherical score, and score-only RPS normalized by `K-1` (lines 140-166).

## Frozen Data

| split | rows | choice | score | noul | total questions | JSONL SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| train | 256 | 512 | 512 | 256 | 1,280 | `d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656` |
| dev | 100 | 200 | 200 | 100 | 500 | `73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e` |

Manifest SHA-256:
`569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056`.
The source Parquet hashes remain
`e7b2f78fe539517c6a66a90c68ed671c6133194a88d27dbff5d84713af46b901`
and `833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d`.

## Runner Guards

The runner verifies fixed config bytes, all source/model/notebook/common hashes,
the Laya Git revision, subset and manifest hashes, source Parquet hashes,
predeclaration hash, exact row/item counts, marker survival, and exact model
parameter counts/group partition before evaluation or training. CUDA absence is
a hard failure; there is no CPU fallback.

The base must reproduce exactly 71/200 choice answers before training starts.
The training loop permits exactly 5,120 one-sequence microforwards, accumulation
64, and 80 updates. It fails on non-finite objective/gradient values, missing
gradients, count drift, or peak reserved memory above 11.5 GiB. There is no OOM
retry or alternate configuration.

Only two evaluations exist: baseline and final after reload. Raw logits are
used, no calibration is fitted or applied, and no intermediate development
metric can influence training. Final pass requires at least 81/200 native
choice answers. Choice, score, noul, all-question and workflow native hard
metrics, native-gold-label NLL, and soft-target cross-entropy are emitted.

The checkpoint includes model, optimizer, scheduler, scaler, and stopping
state. Optimizer state is cloned to CPU before the original model is released.
The model state digest and complete training state must reload exactly before
the sole final evaluation.

## Verification

Final CPU-only test command:

```text
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 11 tests ran, all 11 passed in 1.134 seconds. This includes the six
existing B1/schema tests and five focused B2 tests.

Final in-memory compile command:

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from pathlib import Path; files=[*Path("training/laya_local").glob("*.py"),*Path("training/laya_local/tests").glob("*.py")]; [compile(p.read_bytes(),str(p),"exec") for p in files]; print(f"in-memory compile passed: {len(files)} files")'
```

Result: `in-memory compile passed: 12 files`. No bytecode was written.

Predeclaration verification:

```text
sha256sum -c artifacts/train_i2/B2_PREDECLARATION.sha256
```

Result: `artifacts/train_i2/B2_PREDECLARATION.md: OK`.

Two preliminary failures are retained in this chronology rather than hidden:
the first subset command stopped before row I/O because the standalone script
lacked the repository root on `sys.path`; after that was fixed, generation
succeeded. The first focused test run exposed an incorrect test callback
signature for the independent reward reference; the test adapter was corrected
and all subsequent focused and full runs passed. Neither failure loaded a model,
initialized CUDA, ran inference, or trained.

## Unverified Until Authorized Execution

Because the main command was intentionally not run, RTX 3060 fit under 11.5 GiB,
reproduction of the 71/200 raw-logit baseline, finite full-model gradients,
checkpoint reload behavior at full scale, wall time, and final accuracy remain
unknown. These are runtime gates, not implementation claims. There is no current
implementation blocker; execution is simply outside this task's authorization.
