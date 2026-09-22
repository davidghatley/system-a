# Trainer B2 Independent Post-Run Review

Date: 2026-09-21
Role: independent post-run reviewer for the sole Trainer B2 run

## Decision

**ACCEPTED.**

The sole recorded B2 run satisfies the frozen and repaired protocol. Native-label
choice accuracy improved from `72/200 = 0.360` to `141/200 = 0.705`: `+69`
correct answers and `+0.345` absolute (`+34.5` percentage points). This exceeds
the repaired gate of measured baseline plus 10 and the Iteration 2 requirement
of at least five absolute points. The run completed exactly 5,120
microforwards and 80 optimizer updates, stayed below the 11.5 GiB reserved-VRAM
limit, and produced a structurally complete, exactly reload-checked checkpoint.

No repair is required. The absence of a captured command/stdout/stderr log and
the update-sampled rather than per-microforward numeric record are evidence
limitations described below; neither contradicts the successful guarded run.

## Scope And Contract

Reviewed the frozen criteria, B1 authorization, original and superseding B2
declarations and checksums, pre-run review and verification, current runner,
config, preparer, focused tests, frozen subsets, pinned source files, and every
file under `artifacts/train_i2/b2_run`, including safetensors metadata and the
training-state structure.

The controlling protocol is the original declaration plus
`B2_PREDECLARATION_REPAIR_1.md`. In particular, the repaired primary gate is
`final choice correct >= measured B2 baseline correct + 10`; the original fixed
`71/200` and `81/200` B2 guards are superseded. B1 was independently accepted
and explicitly authorized one B2 run before the B2 declarations and execution.

## Independently Verified Results

The two prediction files each contain exactly 500 entries. Their ordered
`(record_id, workflow, question_id, type, gold_label)` tuples exactly equal the
ordered questions and native `gold[qid].label` values decoded directly from the
hash-pinned development Parquet. Baseline and final identity tuples also match
one another exactly. Every recorded `correct` value equals
`prediction == gold_label`.

Metrics recomputed solely from the prediction entries are:

| metric | baseline | final | change |
|---|---:|---:|---:|
| choice | 72/200 = 0.360 | 141/200 = 0.705 | +69 / +0.345 |
| score | 65/200 = 0.325 | 140/200 = 0.700 | +75 / +0.375 |
| noul | 57/100 = 0.570 | 84/100 = 0.840 | +27 / +0.270 |
| all questions | 194/500 = 0.388 | 365/500 = 0.730 | +171 / +0.342 |
| native-gold NLL | 1.5309924923181535 | 0.6893509099334478 | -0.8416415823847057 |
| soft-target NLL | 1.5481649901866912 | 0.9644585451185703 | -0.5837064450681209 |

The sole workflow, `agent_trace_observability`, exactly matches the all-question
result. Recomputed summaries and NLL means exactly equal `baseline.json`,
`final.json`, and both compact copies in `result.json`. All 1,000 per-prediction
native-gold and soft-target NLL values are finite and nonnegative.

## Schedule And Numeric Evidence

- `result.json` records 256 train rows, 1,280 train items, 100 dev rows, 500 dev
  items, 5,120 microforwards, and 80 updates. These equal four complete
  1,280-item epochs with accumulation 64 and 20 updates per epoch.
- The 80 update records are consecutive. Epoch blocks are exactly 20 updates
  each and use sigma `0.4, 0.3, 0.2, 0.1` as frozen.
- Every recorded loss, RL loss, CE loss, reward mean, pre-clip gradient norm,
  scale, and learning rate is finite. Recorded total loss equals RL plus CE.
- Recorded total loss ranges from `-0.6420596838` to `2.5848536491`, reward mean
  from `-2.0948991776` to `0.5232933750`, and pre-clip gradient norm from
  `6.4137611389` to `76.2886581421`. Large pre-clip norms are expected to be
  clipped by the frozen global norm limit of 1.0; they are finite.
- FP16 scale is exactly `64.0` for all 80 updates. The checkpoint scaler has
  scale 64, growth tracker 80, and growth interval 1,000, establishing no
  recorded overflow-driven scale decrease or skipped step.
- Both recorded learning-rate sequences exactly equal the independently
  calculated `CosineAnnealingLR(T_max=80, eta_min=1e-6)` values after each
  update. They begin after update 1 at `2.4990748434888676e-5` and
  `9.996183729391579e-5`, and both end at `1e-6` after update 80.
- Peak reserved memory is `9,309,257,728` bytes = `8.669921875` GiB, below the
  frozen 11.5 GiB limit by `2.830078125` GiB.
- Recorded training time is `712.4538953680312` seconds and total timed run is
  `772.804736902006` seconds. Both are finite and positive; total exceeds
  training by `60.3508415339748` seconds.

The frozen runner checks every microforward's loss components, reward tensor,
and advantage tensor for finiteness, and every update's complete gradients and
gradient norm before advancing. A successful final result cannot be emitted
through this code after one of those guards fails.

## Hashes And Pins

All original and repaired frozen hashes independently recompute exactly:

```text
a9f913ff3d09e00361e26169e620e156390013443fb09da328b2dc7e4f4fa8da  B2_PREDECLARATION.md
2ce52d92c2e80d27740eeb83df24d175bb19a2d9d51983004337b3cbab762836  B2_PREDECLARATION.sha256
b7a9515266b0bf504b149b664ab7b6dcc644d8f48e6b0d4c7288a564e4500de7  B2_PREDECLARATION_REPAIR_1.md
0d58fcccae76d04a30a7c53b94413fc418dccd7e8a0ee595c6ca8a952835ec9d  B2_PREDECLARATION_REPAIR_1.sha256
26caa358d0c65eb0069cbb1188817493a6ab129c3e18e58d8983cfec37a5c972  B2_PRE_RUN_REVIEW.md
a42fd2fd176ea6bc0e68e371be81eeb8be624caa021f096f594e8bfda7689cb4  b2_config.json
0cabc5ffdee25c1e5af7c8a980b36dc20897f973eaf490635ef99d1c88660967  b2_train.py
7b0609536a8aeb6e62d735b0501c1f58cdf94b6a20b20ad391b446108cfaa677  b2_prepare.py
3ed831526de44adaecdb2a2fa6711ecb9d9128b6c19f276115a0908693077b22  test_b2.py
d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656  train.jsonl
73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e  dev.jsonl
569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056  manifest.json
f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2  data/laya/laya/common.py
2c37036054dfb92073b518e994cb66c8a9009fb8d15b060d9bdfc6d4f94b1d39  pinned public notebook
e7b2f78fe539517c6a66a90c68ed671c6133194a88d27dbff5d84713af46b901  source_train.parquet
833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d  source_test.parquet
891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c  pinned base model.safetensors
ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd  pinned rl_agent_config.json
```

The local Laya checkout reports exactly
`d113dca2512fb3eaca313534bc54c7162d87c1d4` and is clean. The model snapshot,
dataset revision, config, and source revisions in `result.json` match the frozen
values. Direct Parquet reconstruction exactly matches both JSONL subsets, IDs,
order, and content. Checkpoint config is semantically identical to both the
current frozen config and the config embedded in `result.json`.

Run-artifact SHA-256 values independently measured during this review are:

```text
e66d915fb21e7cc76bde1ca2568061ad429022b8e279494cf1683edd2c1f202b  baseline.json
3f33805b7a9ca461db88f92cb95678ab63083b971403e993f5a8c62cf91d9c92  final.json
964dc43f0c856a0026a85b6057d4c8355ed91a452ff28efc90c3dced83b94ffa  result.json
9f75a91ce9c3d6539415c3240a57be5dac6ed2a7db79e31568be2bcc04bd39f2  checkpoint/config.json
6fe7cfb383491d1a740c1de398afda71a5f6bf4ae3bee47dad398a983ba4fde5  checkpoint/model.safetensors
8302bfb9a91863c4a9c4e70fab5156cd477589bee61941b76b5523b2c01f6ae9  checkpoint/training_state.pt
```

## Checkpoint Audit

- Safetensors header inspection finds 206 tensors, all FP32, with exactly
  421,293,830 elements. `temperature` is a three-element FP32 tensor. Tensor
  names and shapes exactly match the pinned base checkpoint; the pinned base
  stores 205 parameter tensors as FP16 while the trained checkpoint stores the
  model state as FP32, consistent with model construction/loading and training.
- Streaming all saved tensors on CPU through the runner's independently
  transcribed state-digest algorithm reproduces
  `bba78de2392b63c81c7a54f7572f015746e6e4f93890baab0bbe83b37c66f390`
  exactly, matching `result.json`.
- The training-state archive has the exact top-level keys `optimizer`,
  `scheduler`, `scaler`, `updates`, `microforwards`, and `epochs`, with stopping
  state `80`, `5120`, and `4`.
- Optimizer state has two groups containing 170 encoder and 35 non-encoder
  parameters. All 205 parameters have `step`, `exp_avg`, and `exp_avg_sq` state;
  every step scalar is exactly 80, and each moment family spans all 421,293,827
  named-parameter elements. Initial rates are `2.5e-5` and `1e-4`, final rates
  are both `1e-6`, and both groups retain weight decay `0.01`.
- Scheduler state records `T_max=80`, `eta_min=1e-6`, `last_epoch=80`,
  `_step_count=81`, and final rates `[1e-6, 1e-6]`. Scaler state records scale
  64, growth tracker 80, and growth interval 1,000.
- The frozen runner performs strict model loading, compares the pre-save and
  post-load model digest, compares the serialized training-state payload, and
  reads back and exactly compares each live optimizer, scheduler, and scaler
  state before final evaluation. `result.json` records both reload checks true.

## Chronology And One-Run Evidence

Filesystem timestamps support the declared sequence: B1 authorization at
17:03:34, original declaration/checksum at 17:25, pre-run review at 17:37,
repaired source by 17:44, superseding declaration/checksum at 17:45-17:46,
repair report at 17:48, and independent one-run authorization at 17:55:21.
Run artifacts then form one continuous sequence: baseline at 17:56:27,
checkpoint weights at 18:08:27, training state/config at 18:08:35, final after
reload at 18:09:03, and result at 18:09:06. Only the expected six files exist
under the run directory. The checkpoint's `exist_ok=False` creation, exact
stopping state, consistent hashes, and continuous timestamps support one
completed invocation without retry or fallback.

This chronology is **INFERRED**, not cryptographically proven. There is no
captured shell invocation log, process record, or trusted timestamp that could
exclude an unrecorded failed attempt before checkpoint creation. No repository
evidence indicates such an attempt, retry, alternate profile, CPU fallback,
intermediate evaluation, or second completed run.

## Warnings And Evidence Limits

- **UNKNOWN:** stdout/stderr was not retained. The runner uses
  `torch.use_deterministic_algorithms(True, warn_only=True)`, so whether PyTorch
  emitted a determinism warning cannot now be audited. This does not alter the
  exact artifact metrics or checkpoint checks, but future runs should preserve
  the command log.
- **VERIFIED WITH LIMITATION:** `loss_curve` stores the objective terms and
  reward mean from the 64th microforward at each update, not all 5,120 values or
  an accumulation-window mean. Thus all 80 retained samples, all retained
  gradient norms, and all scales are independently finite; finiteness of omitted
  microforward tensors is established by the frozen fail-fast runner and
  successful completion, not by a complete numeric trace.
- **VERIFIED WITH LIMITATION:** NLL means were independently recomputed from the
  500 retained per-question NLL values. Probabilities/logits were not retained,
  so those per-question NLL values cannot be re-derived independently from raw
  model outputs without forbidden repeat inference.
- **INFERRED:** the CUDA peak and `perf_counter` timings are internally
  consistent and emitted by the frozen runner, but no external GPU telemetry or
  wall-clock command log was retained.
- **UNKNOWN OUTSIDE BOUND:** no claim is made beyond the fixed 100-row
  `agent_trace_observability` development source, the pinned hardware/software
  environment, or this one seed and recipe.

These limitations do not replace or weaken the primary criterion and do not
justify a repair or rerun.

## CPU-Only Review Commands

All review commands ran from the repository root with
`CUDA_VISIBLE_DEVICES=''` where Python/Torch was involved. No model inference,
training, or CUDA initialization was performed.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 14 tests passed, including all 8 focused B2 tests.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<in-memory compile of training/laya_local, its tests, and shared/typed_decisions>'
```

Result: 14 files compiled in memory.

Repository-local inline Python audits decoded the pinned Parquet, aligned every
prediction and native label, recomputed every metric, checked all schedule and
numeric records, parsed both checkpoint containers, streamed the complete model
state digest on CPU, and inspected training state with CPU mmap/fake tensors.
`sha256sum` verified all listed files. `git diff --check` passed.

Two preliminary audit assertions were corrected and retained here for
transparency. One expected private GradScaler key names rather than the public
saved keys; the corrected structural audit passed. Another initially required
base and trained checkpoint dtypes to match; inspection showed identical names
and shapes but the expected FP16-base to FP32-trained serialization difference,
after which the corrected schema/config audit passed. Neither command wrote a
file, loaded CUDA, changed evidence, or affected the decision.

## Boundary Confirmation

This review wrote only `artifacts/train_i2/B2_RUN_REVIEW.md`. It did not modify
implementation, config, subsets, declarations, run artifacts, or checkpoints;
it did not train, rerun inference, initialize CUDA, or create a commit.
