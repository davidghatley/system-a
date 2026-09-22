# Trainer B2 Final Verification

Date: 2026-09-21
Role: final independent verifier after post-run review

## Decision

**ACCEPTED.**

Trainer B2 learning is accepted for Iteration 2. Cross-track integration may
proceed, subject to the parent verifier's separate cross-track acceptance.

The frozen held-out native-label choice result improved from
`72/200 = 0.360` to `141/200 = 0.705`, an independently recomputed gain of
`69/200 = 0.345` absolute. This exceeds both the repaired predeclared gate of
10 additional correct choices and the Iteration 2 requirement of five absolute
percentage points.

## Checklist Decision

| Trainer B2 requirement | Decision | Verification basis |
|---|---|---|
| Run only after B1 passes | **PASS** | `B1_FINAL_VERIFICATION.md` accepts B1 and explicitly authorizes B2. Its mtime precedes both B2 declarations, one-run authorization, and all run evidence. Chronology is supported rather than cryptographically timestamped. |
| Immutable predeclared recipe | **PASS** | The original and superseding declarations and both checksum files verify. Current config, runner, preparer, tests, subsets, manifest, source Parquets, notebook, Laya source, and cached base model/config reproduce their declared hashes. Seed, optimizer, rates, context, epochs/updates, batching, objective, primary metric, and stopping rule are explicit. |
| Bounded public-recipe adaptation | **PASS** | Notebook and `common.py` spot-checks confirm the declared sequence construction, AdamW/cosine schedule, four-perturbation RLCD equations, proper reward, CE term, and zero action contribution. The fixed 256-row/four-epoch/single-GPU adaptations, accumulation 64, and scaler 64 repair are bounded and predeclared; no search or fallback path exists. |
| Finite one-main-run completion | **PASS WITH EVIDENCE LIMITATION** | Evidence records exactly 5,120 microforwards, 80 consecutive updates, four 20-update epoch blocks, and fixed sigmas `0.4, 0.3, 0.2, 0.1`. All 80 retained loss components, rewards, gradient norms, scales, and learning rates are finite and schedule-consistent. The frozen runner rejects any non-finite microforward objective, reward, advantage, gradient, or gradient norm before successful completion. Only update-boundary samples are retained, and absence of an unrecorded failed invocation cannot be proven without a command log. |
| Reloadable checkpoint | **PASS** | The checkpoint has 206 FP32 tensors and 421,293,830 elements; names and shapes match the pinned base state, including the three-element `temperature` buffer. Independent CPU streaming reproduces model-state digest `bba78de2392b63c81c7a54f7572f015746e6e4f93890baab0bbe83b37c66f390`. Training state contains all 205 parameter states, two optimizer groups, scheduler/scaler state, and stopping state `80/5120/4`. The frozen runner performs strict model reload and exact live optimizer/scheduler/scaler comparisons before final evaluation; the emitted result records both checks true. |
| Metrics, loss, VRAM, wall time | **PASS WITH EVIDENCE LIMITATION** | Prediction-derived metrics and NLL means recompute exactly. Retained loss records are finite and satisfy `loss = loss_rl + loss_ce`. Peak reserved VRAM is `9,309,257,728` bytes (`8.669921875 GiB`), below 11.5 GiB. Training and total times are finite at `712.4538953680312 s` and `772.804736902006 s`. VRAM and timing are runner-reported without independent telemetry; loss is sampled at each update rather than retained for all microforwards. |
| At least five-point native-choice improvement | **PASS** | Direct recount gives baseline `72/200`, final `141/200`, and `+0.345` absolute. The repaired gate required at least `82/200`; final exceeds it by 59 correct answers. |
| Diagnostic NLL and all-question reporting | **PASS** | Baseline/final native-gold NLL are `1.5309924923181535` and `0.6893509099334478`; soft-target NLL are `1.5481649901866912` and `0.9644585451185703`. All-question accuracy is `194/500 = 0.388` and `365/500 = 0.730`. Choice, score, noul, workflow, and all-question totals are present. The primary criterion remains native-label choice accuracy. |
| Evidence limitations disclosed | **PASS** | The post-run review accurately identifies missing stdout/stderr and external telemetry, update-sampled numeric evidence, inability to rederive NLL from unretained logits, inferred one-run chronology, and no generalization claim beyond the frozen 100-row development source. These remain limitations, not contradictions of the guarded result. |

## Independent Recomputations

Both prediction files contain exactly 500 entries. Their ordered
`(record_id, workflow, question_id, type, gold_label)` tuples match all ordered
questions and native labels decoded directly from the pinned development
Parquet. Every `correct` field equals `prediction == gold_label`, and both
retained NLL values for all 1,000 predictions are finite and nonnegative.

| metric | baseline | final | change |
|---|---:|---:|---:|
| choice | 72/200 = 0.360 | 141/200 = 0.705 | +69 / +0.345 |
| score | 65/200 = 0.325 | 140/200 = 0.700 | +75 / +0.375 |
| noul | 57/100 = 0.570 | 84/100 = 0.840 | +27 / +0.270 |
| all questions | 194/500 = 0.388 | 365/500 = 0.730 | +171 / +0.342 |
| native-gold NLL | 1.5309924923181535 | 0.6893509099334478 | -0.8416415823847057 |
| soft-target NLL | 1.5481649901866912 | 0.9644585451185703 | -0.5837064450681209 |

The recomputed summaries exactly match `baseline.json`, `final.json`, and the
compact copies in `result.json`. Checkpoint config is semantically identical to
the frozen config and the result's embedded config. The 80 learning-rate pairs
match independently calculated `CosineAnnealingLR(T_max=80, eta_min=1e-6)`
values, ending at `1e-6` for both groups. FP16 scale remains 64 for all retained
updates; checkpoint scaler state records growth tracker 80 and growth interval
1,000.

The run directory contains only the expected six files. Their SHA-256 values
independently reproduce the post-run review:

```text
e66d915fb21e7cc76bde1ca2568061ad429022b8e279494cf1683edd2c1f202b  baseline.json
3f33805b7a9ca461db88f92cb95678ab63083b971403e993f5a8c62cf91d9c92  final.json
964dc43f0c856a0026a85b6057d4c8355ed91a452ff28efc90c3dced83b94ffa  result.json
9f75a91ce9c3d6539415c3240a57be5dac6ed2a7db79e31568be2bcc04bd39f2  checkpoint/config.json
6fe7cfb383491d1a740c1de398afda71a5f6bf4ae3bee47dad398a983ba4fde5  checkpoint/model.safetensors
8302bfb9a91863c4a9c4e70fab5156cd477589bee61941b76b5523b2c01f6ae9  checkpoint/training_state.pt
```

Filesystem chronology is continuous and consistent with one authorized run:
authorization at 17:55:21, baseline at 17:56:27, checkpoint files at
18:08:27-18:08:35, final at 18:09:03, and result at 18:09:06. This is inferred
support, not proof that no unrecorded failed attempt occurred.

## CPU And Static Checks

All Python commands used `CUDA_VISIBLE_DEVICES=''` and
`PYTHONDONTWRITEBYTECODE=1`. The checkpoint audit asserted that Torch CUDA was
not initialized before or after inspection. No model inference or training was
performed.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s training/laya_local/tests -v
```

Result: 14 tests ran; all 14 passed, including all eight focused B2 tests.

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<in-memory compile of trainer, trainer tests, and shared typed decisions>'
```

Result: 14 Python files compiled in memory.

Independent repository-local CPU audits performed these checks:

- Recomputed all frozen and run-artifact SHA-256 values, including the cached
  base weights/config and both declaration checksum files.
- Reconstructed frozen JSONL subsets directly from the pinned Parquets through
  the shared parser and verified exact IDs, order, content, rows, and question
  counts.
- Aligned all prediction identities/native labels and recomputed every hard
  metric and retained NLL mean.
- Checked counts, update/epoch/sigma sequence, finite numeric records, objective
  decomposition, cosine learning rates, primary gate, memory, timing, and exact
  run-directory contents.
- Streamed the complete safetensors checkpoint on CPU to reproduce the model
  digest and inspected the memory-mapped training state without CUDA.
- Confirmed the Laya checkout is clean at
  `d113dca2512fb3eaca313534bc54c7162d87c1d4` and `git diff --check` passes.

## Evidence Classification

**Verified:** artifact hashes and contents; source/subset/prediction alignment;
all retained metrics and NLL means; primary improvement; update schedule and
retained finite numeric values; checkpoint tensor schema/digest and complete
training-state structure; CPU tests and static compilation.

**Inferred from guarded execution:** finiteness of the 5,040 microforward
objective records not retained in `loss_curve`; successful strict live reload
before final evaluation; one uninterrupted authorized invocation; runner-reported
VRAM and elapsed times.

**Unknown:** whether deterministic-algorithm warnings appeared on stdout/stderr;
whether an unrecorded failed invocation occurred; behavior beyond this pinned
dataset slice, seed, hardware/software environment, and recipe. Raw logits and
external GPU telemetry were not retained.

These limitations are explicitly bounded and do not overturn any Trainer B2
acceptance criterion.

## Boundary Confirmation

This verification wrote only `artifacts/train_i2/B2_VERIFICATION.md`. It did not
edit implementation, configuration, subsets, declarations, run artifacts,
checkpoints, or prior reviews. It did not initialize CUDA, infer, train, or
create a commit.
