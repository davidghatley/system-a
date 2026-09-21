# Independent Review: Local Trainer Iteration 1

Date: 2026-09-21

## Recommendation

**BLOCKED.** The trainer proof is substantially reproducible and the optimization path is live, but the frozen learning criterion was not met: exact choice accuracy fell from 20/40 (0.500) to 11/40 (0.275), an absolute -0.225. The predeclared NLL improvement is useful diagnostic evidence, not sufficient acceptance evidence under the frozen checklist. No criteria were relaxed.

## Findings (ordered by severity)

### 1. BLOCKER — frozen primary learning criterion failed (VERIFIED)

`run/result.json` and the per-question evaluation files show 20/40 to 11/40 choice accuracy. This is below the required +0.05 absolute improvement. The fixed dev set has 40 choice questions across 20 records, so I do not find evidence that it is demonstrably too small or noisy enough to invoke the checklist's exception. That judgment is necessarily an assessment, not a statistical proof.

### 2. HIGH — reported exact-choice target is not the native `gold.label` (VERIFIED; acceptance impact INFERRED)

`data.py` sets `item["label"]` to the first maximum probability index and `train.py` evaluates against that value. The native schema separately requires a `gold[qid]["label"]`, which is ignored by conversion/evaluation. In the frozen dev data, one of 40 choice questions has a tied distribution whose declared label is `human_review` while Python's first argmax is `continue`; recomputing against declared labels gives baseline 19/40 and post 11/40. The pass/fail outcome does not change here, but the metric identity is ambiguous and the artifact's `gold_argmax` naming does not establish that argmax, rather than the native label, was the predeclared exact-choice metric.

### 3. MEDIUM — checkpoint reload is verified; actual resume/continuation is not implemented or verified (VERIFIED)

The script saves model, optimizer, scaler, and step state, reloads them, and reproduces inference exactly. It does not expose a resume mode or perform a post-reload optimizer update. Thus `optimizer_state_loaded: true` proves state deserialization, not resumability. This partially meets the checklist's save/resume configuration requirement, while the separate reload-and-inference requirement is met.

### 4. LOW — timing and VRAM claims are bounded but limited (VERIFIED)

The recorded RTX 3060, peak allocated/reserved memory, and wall times are supported by `result.json`; checksum verification also reproduced all six listed hashes. `train_seconds` measures the training loop through CUDA synchronization (not a warmup-normalized benchmark), and the run has one seed and one bounded 32-update sample. These are adequate run measurements, but should not be generalized performance claims.

## Component checks

- **Target construction — VERIFIED:** native records validate; question order and probability values are preserved for choice/score/noul; the model-side loss is masked to valid markers and uses soft gold distributions. The label/argmax issue above remains.
- **Objective and optimizer — VERIFIED:** masked log-softmax cross-entropy against the distribution, Adafactor parameter groups, gradient clipping, and FP16 scaling are present. The second mask is redundant but harmless.
- **Accumulation — VERIFIED by code/evidence:** two microbatches are accumulated, loss is divided by two, gradients are checked after unscale, then one optimizer step is taken; the artifact records 64 microbatches for 32 updates.
- **Parameter inclusion/exclusion — VERIFIED:** encoder and non-encoder/non-action-head parameters changed; action-head hash stayed identical; the partition overlap guard is present.
- **Baseline/post identity — VERIFIED:** the same frozen dev items and evaluation function are used before and after training; post-checkpoint predictions exactly match post-training predictions.
- **Subset predeclaration — VERIFIED:** first 64 pinned train rows and first 20 pinned test rows were declared before baseline, with matching source/output hashes and distinct IDs.
- **Finite/OOM path — VERIFIED:** bounded run completed on the recorded GPU, all recorded gradients were finite, and peak reserved memory was 5.508 GiB of 12 GiB.
- **Basic checks — VERIFIED:** independently rerun 3 unit tests and Python compilation successfully on 2026-09-21.
- **Alternate NLL criterion — VERIFIED as diagnostic, NOT ACCEPTED as learning criterion:** all-question gold NLL improved 1.495159 to 1.210250, but this is explicitly the predeclared diagnostic. I do not approve it as clear learning evidence sufficient to override the failed exact-choice criterion; the accuracy regression and single short run leave the interpretation INFERRED, not established.

## Repair brief (maximum two repairs)

### Repair 1

- **FINDING:** Exact-choice evaluation uses probability argmax while native records carry a declared label, with a tie in the frozen dev set.
- **WHY:** The acceptance metric must have one unambiguous native identity; otherwise the reported 20/40 baseline is not independently reproducible from the contract.
- **REQUIRED CHANGE:** Explicitly choose and predeclare native-label accuracy or distribution-argmax accuracy, update conversion/evaluation and artifacts consistently, and do not alter the frozen subset or retroactively claim a pass.
- **ACCEPTANCE TEST:** A no-training metric test independently reconstructs every dev target from JSONL and exactly matches the stored baseline/post counts, including the tied record.

### Repair 2

- **FINDING:** State reload is tested, but there is no resumable continuation path.
- **WHY:** Saving optimizer/scaler state without proving a subsequent update does not satisfy a resume claim.
- **REQUIRED CHANGE:** Add a bounded, deterministic resume entry point that restores model, optimizer, scaler, step, and data cursor/RNG state, without changing this failed run's results.
- **ACCEPTANCE TEST:** A bounded CPU-independent smoke test (or an explicitly bounded GPU continuation only if authorized) reloads the checkpoint, performs one update, and verifies finite gradients plus a changed intended parameter and unchanged excluded action head.

## Evidence and commands

Reviewed `research/26_iteration_1_acceptance.md`, all source files under `training/laya_local/`, `shared/typed_decisions/`, and `artifacts/train_i1/`, plus the pinned Laya `common.py` model/collator implementation for target and mask semantics. Independently ran:

```text
.venv/bin/python -m unittest discover -s training/laya_local/tests -v
.venv/bin/python -m py_compile shared/typed_decisions/schema.py training/laya_local/data.py training/laya_local/prepare_subset.py training/laya_local/train.py training/laya_local/tiny_overfit.py
sha256sum source_train.parquet source_test.parquet subsets/train.jsonl subsets/dev.jsonl run/checkpoint/model.safetensors run/checkpoint/training_state.pt
```

No GPU training job was launched. No implementation files were modified.
