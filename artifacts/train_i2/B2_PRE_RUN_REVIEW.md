# Trainer B2 Pre-Run Review

Date: 2026-09-21
Role: independent pre-run code and protocol reviewer

## Decision

**CHANGES_REQUIRED. Do not start the B2 main run.**

The current runner is guaranteed to stop at its parameter-count guard before
baseline evaluation, and two additional issues weaken subset immutability and
FP16 accumulation safety. No model was loaded, CUDA was not initialized, and no
inference, optimizer step, or training was run during this review.

## Required Repairs

### 1. Blocker: impossible named-parameter count

`training/laya_local/b2_train.py:263-269` computes counts from
`model.named_parameters()` but requires totals that include the three-element
registered `temperature` buffer. The executable condition currently requires:

```text
encoder named parameters     394,781,699
non-encoder named parameters  26,512,131
total named parameters       421,293,830
```

The pinned model actually has:

```text
encoder named parameters     394,781,696
non-encoder named parameters  26,512,131
total named parameters       421,293,827
temperature buffer elements            3
state elements               421,293,830
```

This is independently preserved in
`artifacts/gpu_diagnostics/fp16_scale1_optimizer_step.json:99-107`, and follows
directly from `temperature` being created with `register_buffer` in
`data/laya/laya/common.py:102`. The optimizer partition is otherwise correct:
every named parameter goes to exactly one group and `act_head` is in the
non-encoder group.

Precise repair: change the named-parameter assertions to `421293827` total and
`394781696` encoder, retain `26512131` non-encoder, and separately assert that
the complete `state_dict()` has `421293830` elements and that
`temperature.numel() == 3` with `temperature` absent from
`named_parameters()`. Add a focused regression test. Re-freeze runner/config
hashes and issue a superseding pre-run declaration before execution; do not
rewrite the existing predeclaration in place.

### 2. High: frozen subset hashes are not enforced by the runner

The predeclaration freezes train JSONL, dev JSONL, and manifest hashes at
`B2_PREDECLARATION.md:49-51`, but `b2_train.py:170-181` trusts each JSONL hash
read from the mutable manifest and never compares the manifest or JSONLs to
those frozen values. A coordinated edit to a JSONL and its manifest would pass
the current checks as long as row/item counts and source Parquet hashes remain
valid. The runner also does not compare loaded record IDs/order to the frozen
manifest IDs or reconstruct the selected rows from the pinned Parquets.

Precise repair: make these expected values independent runner inputs and fail
before importing Torch unless all three match exactly:

```text
train.jsonl d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656
dev.jsonl   73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e
manifest    569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056
```

Also compare loaded IDs, row order, questions, criteria order, probability
order, and gold values against the first 256/all 100 rows decoded directly
from the already hash-verified Parquets, or enforce equivalent frozen hashes
and exact ID arrays in a test that cannot take expected hashes from the same
manifest under test. The current `test_b2.py:24-36` has the same circular trust
for output hashes.

### 3. High: effective FP16 backward scale is 64 times below the tested safe path

The public notebook uses the default enabled `GradScaler` and divides loss by
accumulation 4 (`ipynb` JSON lines 267 and 320-330). The B2 runner deliberately
uses scale 1.0 because the default 65536 overflowed in an earlier diagnostic;
scale 1.0 with an un-divided loss was subsequently measured finite and able to
update the model. B2 now divides every loss by 64 before `backward`
(`b2_train.py:351`), making the per-microforward scaled backward signal
`loss / 64`. That numerical path was not tested and is 64 times smaller than
the verified scale-1 smoke path. FP32 master gradients do not remove FP16
activation-backward underflow risk.

Precise repair: use `init_scale=64.0` with a growth interval greater than the
entire 80-update run (or disable growth while retaining overflow detection).
Then `scale(loss / 64)` presents approximately the already-tested unscaled
loss magnitude to each FP16 backward, while `unscale_` produces the intended
average accumulated gradient. Keep the existing finite-gradient hard stop.
Add a CPU tensor test for the accumulation/scaler algebra and accurately
predeclare this as a hardware adaptation, not the notebook's default scaler.

The existing evidence supports this repair: default scale 65536 overflowed;
scale 1 with an un-divided loss had finite gradients, completed one AdamW step,
and peaked at 6.56 GiB reserved. It does not establish safety for scale
`1 / 64` over 5,120 backward passes.

### 4. Medium: checkpoint verification checks the file payload, not loaded objects

After calling `load_state_dict`, `b2_train.py:405-410` compares the saved CPU
tree with `loaded_training_state`, which is merely the deserialized file tree.
It does not compare `optimizer.state_dict()`, `scheduler.state_dict()`, or
`scaler.state_dict()` after those objects have loaded the state. Therefore
`training_state_reload_exact: true` overstates what is checked.

Precise repair: after all three `load_state_dict` calls, clone each live
object's state back to CPU and compare it with the corresponding saved state.
Retain the update/microforward/epoch checks. Add a focused round-trip test that
would fail if a live object's loaded state differs.

### 5. Medium: the exact 71/200 baseline guard changes execution precision

B1's `71/200` base result was produced through `Agent`, which selects BF16 on
this Ampere GPU from the pinned base config and batches all five row questions.
B2 evaluates one question at a time under explicit FP16
(`b2_train.py:288-328`). Positive temperature scaling does not change argmax,
but precision and batching can. Thus `71/200` is a useful expected result, not
an already verified exact result for this B2 evaluation path. An incidental
one-answer difference would terminate the sole authorized command before
training even when pins and native metric semantics are correct.

Precise repair: keep `71/200` as a recorded B1 cross-check, but define the B2
primary gate from the baseline measured by the frozen B2 evaluator before any
update: final choice correct must be at least `baseline_choice_correct + 10`
over the same 200 examples. Continue to fail before training on pin, source,
ID/order, item-count, or non-finite baseline errors. If an exact baseline guard
is retained instead, it must first be frozen from this exact FP16,
one-sequence evaluation path under separate authorization.

## Verified Protocol Details

- B1 authorization is present and accepted: specialist `136/200` versus base
  `71/200`, a `+0.325` choice-accuracy delta.
- Current hashes match the predeclaration for the notebook, Laya common source,
  model weights/config, config, preparer, runner, focused tests, train/dev
  JSONLs, manifest, and source Parquets. The separate predeclaration checksum
  verifies as `a9f913ff...`.
- Filesystem chronology supports B1 verification first, then config/preparer,
  subset generation, runner/tests, predeclaration, checksum, and implementation
  report. This is supporting evidence, not a trusted cryptographic timestamp.
  No `artifacts/train_i2/b2_run` output exists.
- The pinned public RLCD transcription is correct: masked Gaussian
  perturbations have zero valid-option mean; perturbed logits begin at detached
  logits; reward is target-weighted floored log score plus `0.75` spherical
  score and score-only subtraction of `1.0 * RPS/(K-1)`; advantages are centered
  per item over `G=4` and divided by one default global standard deviation; the
  Gaussian score-function term is added to full-weight soft CE; the action
  contribution is exactly zero.
- Shapes are coherent for the B2 microbatch: logits/target/mask are `[1,K]`,
  perturbations and rewards are `[4,1,K]` and `[4,1]`, and option masking is
  applied to perturbation projection, softmax, Gaussian log probability, CE,
  and RPS.
- Target semantics match the notebook: soft native distributions train the
  model; notebook-style argmax is retained only as an item field; evaluation
  uses native `gold.label`. The reviewed train/dev subsets contain 15/11 cases
  respectively where native label differs from soft-distribution argmax, and
  B2 correctly evaluates the native label.
- The optimizer grouping and timing otherwise match the declared adaptation:
  encoder LR `2.5e-5`, all other named parameters including `act_head` at
  `1e-4`, AdamW weight decay `0.01`, clip after unscale, optimizer then scaler
  update then cosine scheduler step, `T_max=80`, and `eta_min=1e-6`.
- The in-place epoch shuffle matches the single-rank notebook schedule:
  successive seeds 42, 43, 44, and 45 mutate the prior order. Counts are exactly
  1,280 microforwards and 20 full updates per epoch, 5,120/80 total, with no
  partial update.
- Sequence construction is notebook-faithful at effective `512/192` with
  default right truncation. CPU tokenization of all frozen rows found 1,280
  train and 500 dev items, no marker loss, lengths 124-199, and no state
  truncation. Options are 256/100 binary and 1,024/400 four-way for train/dev.
- Model save/reload digest logic is bit-exact for the complete model state. The
  live optimizer/scheduler/scaler verification needs repair as described above.
- CUDA absence, OOM, non-finite objective/gradients/norm, missing gradients,
  peak reserved memory above 11.5 GiB, count drift, strict model reload failure,
  and final primary failure are hard stops with no fallback. Prior measured
  one-step peak reserved memory was 6.56 GiB, but full B2 fit and stability
  remain unknown until the authorized run.
- Baseline/final reporting covers native choice, score, noul, all-question and
  workflow accuracy, native-label NLL, and soft-target CE. No calibration or
  intermediate dev evaluation is present.

## CPU-Only Checks Run

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 11 tests passed. These tests do not instantiate the full model and did
not expose the impossible parameter-count assertion.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<in-memory compile of trainer/tests/shared modules>'
```

Result: 14 files compiled in memory.

Additional repository-local CPU audits verified all declared SHA-256 values,
the clean pinned Laya checkout at
`d113dca2512fb3eaca313534bc54c7162d87c1d4`, exact JSONL-to-Parquet row
equality, IDs/order/counts/types, native-label semantics, option markers, and
token lengths/truncation. No network access or external path was used.

## Re-Review Gate

Do not authorize the main command until repairs 1-4 are implemented and a
superseding immutable protocol/checksum records all changed hashes and scaler
semantics. Repair 5 must either be adopted or supported by an exact-path frozen
baseline. A follow-up reviewer should rerun CPU tests/static audits and verify
that no model, CUDA, inference, or training execution occurred during repair.
