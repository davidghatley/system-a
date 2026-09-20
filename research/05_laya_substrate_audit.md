# Pinned Laya Substrate Audit for System-A Phase 1

Audit date: 2026-09-20.

## Scope and Pins

- **VERIFIED:** Upstream source is GitHub `NandhaKishorM/laya` at commit `d113dca2512fb3eaca313534bc54c7162d87c1d4` (`git rev-parse HEAD`). Citations below refer to that immutable tree.
- **VERIFIED:** Phase 1 names `convaiinnovations/laya` revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` and 421,293,830 parameters (`experiments/exp_001/protocol.json:113-130`). The specialist comparison revision is `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` (`research/01_laya_training_and_hardware.md:5-10`).
- **VERIFIED:** Primary artifacts compared were the library implementation (`laya/common.py`, `laya/agent.py`), the upstream Kaggle notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`), and both pinned HF `rl_agent_config.json` files.
- **UNKNOWN:** The notebook installs `laya>=0.1.6` and unbounded newer dependencies rather than the audited commit/package version (notebook cell 3, JSON line 66). It also downloads model and dataset without `revision=` (cell 4, lines 92-102). Thus executing it today is not a pinned reproduction.

Labels mean: **VERIFIED** is directly established by pinned code/config or a deterministic calculation; **INFERRED** follows from those facts but was not measured end-to-end; **UNKNOWN** lacks evidence; **RECOMMENDED** specifies Phase 1 behavior.

## Architecture and Exact Parameters

- **VERIFIED:** Each question is encoded as its own sequence: `[CLS] type + instructions [SEP] [MASK] option ... [SEP] state [SEP]`. Options are capped at 48 tokens initially, then jointly shortened when the head budget is exceeded; state truncation is right-side by default and left-side only with `truncate_left=True` (`laya/common.py:49-86`).
- **VERIFIED:** The backbone is the pinned ModernBERT-large config: hidden size 1,024, 28 layers, 16 attention heads, FFN size 2,624, local attention 128, and maximum positions 8,192 (HF base revision `encoder/config.json`). Laya requests SDPA when constructing it (`laya/common.py:129-137`).
- **VERIFIED:** The decision head adds a three-entry type embedding, two PyTorch `TransformerEncoderLayer`s with 16 heads and FFN width 4,096, an option scorer `LayerNorm -> Linear(1024,1024) -> GELU -> Linear(1024,1)`, and an action head `Linear(1028,256) -> GELU -> Linear(256,2)` (`laya/common.py:89-103`).
- **VERIFIED:** Exact parameter accounting is:

| Group | Parameters | Derivation |
|---|---:|---|
| ModernBERT encoder | 394,781,699 | published total minus the audited head |
| Two transformer head layers | 25,192,448 | `2 * (12*d^2 + 13*d)`, `d=1024` |
| Type embedding | 3,072 | `3 * 1024` |
| Option scorer | 1,052,673 | LayerNorm plus two linear layers |
| Action head | 263,938 | `(1028*256+256) + (256*2+2)` |
| **Total** | **421,293,830** | matches pinned safetensors metadata/report |

- **VERIFIED:** All 421,293,830 parameters have `requires_grad=True` by construction and the notebook puts every named parameter into AdamW. It partitions names containing `encoder.` at LR 2.5e-5 and all remaining parameters at LR 1e-4 (notebook cell 5, lines 257-263).
- **VERIFIED:** The action-distribution features use `softmax(logits.detach())`; therefore answer logits receive no gradient through those four features. The pooled encoder representation is not detached (`laya/common.py:119-126`).
- **VERIFIED:** The notebook does not train an action objective: `+ 0.0 * act.sum()` creates zero action-path gradients (cell 5, lines 316-322). Because action-head parameters remain in AdamW with weight decay 0.01, they can still change through decoupled decay despite receiving no task signal.
- **RECOMMENDED:** For the Phase 1 one-step runner, omit the action head from optimizer groups or set its group weight decay to zero. Record it as intentionally frozen; System-A's declared targets are answer distributions, not Laya's undocumented action/escalation target.

## Loss Audit

- **VERIFIED:** For reported distribution `q`, soft target `y`, option mask `m`, and `K=sum(m)`, library reward is

  `R = sum_i y_i max(log(max(q_i, 1e-12)), -9.21) + w_sph * <y,q>/||q||_2 - I[type=score] * w_rps/(K-1) * sum_i (CDF(q)_i-CDF(y)_i)^2`.

  The implementation is `laya/common.py:140-166`; defaults are `w_sph=0.5`, `w_rps=1.0`. The notebook calls it with `w_sph=0.75`, `w_rps=1.0` (cell 5, lines 310-314).
- **VERIFIED:** Notebook exploration samples `G=4` masked Gaussian perturbations, projects each to zero mean over valid options, and forms `z = stop_gradient(logits) + epsilon`, `q=softmax(z)` (cell 5, lines 304-308). Advantage is centered per item over the group, then divided by one scalar standard deviation over the whole group and batch (lines 310-314).
- **VERIFIED:** The policy term is `-mean(A * logp)`, where `logp = -sum_i(z_i-logit_i)^2/(2*sigma^2)`, and the total also contains full-weight soft cross-entropy `-mean(sum_i y_i log_softmax(logits)_i)` (cell 5, lines 316-320). `z` contains detached logits, so this is a score-function/finite-perturbation gradient path, not differentiation through reward.
- **VERIFIED:** The action output contributes exactly zero to the scalar objective. The notebook uses `find_unused_parameters=True`, apparently to accommodate that path (cell 5, lines 243 and 320).
- **VERIFIED:** A tiny local reward test could not execute because this workspace has no `torch` module. Formula inspection confirms its implementation matches `proper_reward`; the existing test also checks finite gradients and that an ordinal distribution concentrated near the target outranks one concentrated farther away (`experiments/laya_smoke_test/smoke_loss.py:7-35`).
- **INFERRED:** With one-hot/hard System-A targets, CE alone already supplies a direct supervised gradient. The incremental value and stability of the perturbation term are unvalidated for this task.
- **RECOMMENDED:** Preserve the upstream combined loss for the substrate smoke, but log `loss_ce`, `loss_rl`, reward mean/std, and gradient norms separately. Keep a later CE-only ablation as required by `research/01_laya_training_and_hardware.md:127-129`; do not claim RLCD benefit from the combined run alone.

## Checkpoint Loading, Precision, and Batching

- **VERIFIED:** `build_model` constructs an encoder from local config with normal PyTorch parameter dtype, then safetensors are loaded on CPU and copied strictly before `model.to(device)` (`laya/agent.py:174-206`; notebook cell 5, lines 232-241). The pinned encoder config says `dtype: float32`; the checkpoint file is approximately 842.6 MB, consistent with FP16 storage.
- **VERIFIED:** Loading FP16 tensors into the newly constructed FP32 module does not make trainable parameters FP16. The notebook then uses FP16 autocast and a CUDA GradScaler (cell 5, lines 267 and 290-322). Parameters, gradients, and Adam moments remain FP32 in this implementation.
- **VERIFIED:** Inference similarly moves the FP32 model to the device without casting it; `amp_dtype` controls autocast only (`laya/agent.py:194-206,266-277`). The pinned base config requests BF16, while the notebook explicitly trains in FP16.
- **VERIFIED:** `collate_items` flattens groups into independent sequences and pads every sequence to the longest sequence and every option set to the largest option count (`laya/common.py:218-250`). `Agent.system_one` creates one full state-bearing sequence per question, batches those sequences, and performs one model call (`laya/agent.py:254-277`). It does not encode the state once and share it across questions.
- **INFERRED:** Phase 1 memory and throughput scale with the number of questions/tool families per decision bundle. `micro_batch=1` is ambiguous unless defined as one question sequence rather than one trajectory row/bundle.
- **RECOMMENDED:** Define optimizer micro-batch in sequences. For feasibility, run both (a) one 512-token sequence and (b) one complete maximum-question bundle flattened into `Q` sequences, because production bundle memory is the latter. Bucket by sequence length and cap total tokens per forward pass; preserve all questions from a bundle for bundle-level metrics but no shared forward-state assumption.

## Single-GPU Adaptation Hazards

- **VERIFIED:** The notebook is intrinsically DDP/NCCL: it asserts at least two GPUs, initializes NCCL, sets CUDA rank, wraps DDP, and shards examples by rank (cells 2 and 5, lines 33-52 and 214-246). It cannot be reduced to one GPU merely by changing `nproc_per_node` while retaining the assertion/setup cell unchanged.
- **VERIFIED:** Encoder checkpointing is genuinely enabled with `gradient_checkpointing_enable(use_reentrant=False)` (cell 5, line 238).
- **VERIFIED:** Head checkpointing is not implemented. `DecisionModel` defines `self.head_checkpointing=False`, but `forward` directly invokes every head layer and never reads the flag (`laya/common.py:103-113`). The notebook's `model.head_checkpointing=True` (cell 5, line 239) has no effect.
- **VERIFIED:** The notebook preprocesses sequences using the downloaded base config's 512/192 budgets (cell 4, lines 98-121), saves those token IDs, and only afterward changes the training config to 1024/256 (cell 5, lines 225-245). Training therefore remains on precomputed 512/192 sequences; the stated 1024/256 training setting is ineffective for this notebook run.
- **VERIFIED:** Gradient accumulation does not reduce single-forward activation memory. The protocol's accumulation 32 changes effective batch/update frequency, not whether one sequence or one multi-question bundle fits (`experiments/exp_001/protocol.json:119-128`).
- **RECOMMENDED:** Remove distributed initialization/DDP for the one-GPU runner; load directly on `cuda:0`; enable only verified encoder checkpointing; use `optimizer.zero_grad(set_to_none=True)`; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialization; and fail rather than silently falling back to CPU.
- **RECOMMENDED:** Set token budgets before preprocessing. Do not claim head checkpointing unless `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` is actually wrapped around each head layer and tested.
- **UNKNOWN:** Whether full-parameter FP32-master AdamW at 512 tokens fits below the protocol's 11.5 GiB limit on the RTX 3060. No full model backward pass has been measured.

## Calibration Precedence

- **VERIFIED:** Runtime first looks up `temperature_by_options[temp_bucket(type,K)]` and falls back to `temperature[type]` only when no bucket exists (`laya/agent.py:300-305`; bucket definitions at `laya/common.py:209-211`).
- **VERIFIED:** The specialist config retains all six base option-count temperatures while replacing only the three per-type values. Consequently the new per-type values are bypassed for every represented bucket: `choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, `score:3-5`, and `noul:2` (pinned specialist `rl_agent_config.json`).
- **VERIFIED:** Notebook calibration selects `all_items[::15][:400]` from training data and fits only per-type temperatures (cell 5, lines 345-388). It neither fits nor removes inherited option buckets, so its fitted temperatures generally do not control inference.
- **RECOMMENDED:** Phase 1 must fit on trajectory-disjoint calibration data only. Choose exactly one precedence scheme before test access: simplest is delete `temperature_by_options` and fit three per-type temperatures; option buckets require enough calibration examples per bucket and must replace, not inherit, stale values. Persist the resolved calibration map and assert with a unit test which temperature each Phase 1 question uses.

## Contradictions and Corrections to Existing Report

1. **VERIFIED CONTRADICTION:** `research/01_laya_training_and_hardware.md:63` says the official notebook uses 1,024 max tokens and 256 head tokens. Those values are assigned only after preprocessing; actual saved training sequences use the base config's 512/192 budgets (notebook cells 4-5, lines 98-121 and 225-245).
2. **VERIFIED CONTRADICTION:** `research/01_laya_training_and_hardware.md:93-95` and the smoke config treat “both checkpointing modes”/`head_checkpointing=true` as a memory control. Only encoder checkpointing operates; the library never consumes the head flag (`laya/common.py:103-113`).
3. **VERIFIED CLARIFICATION:** `research/01_laya_training_and_hardware.md:17,29-33` correctly reports the total and architecture but not the exact split. The audited split is 394,781,699 encoder plus 26,512,131 head parameters.
4. **VERIFIED CLARIFICATION:** The existing report correctly says answer logits are detached in action features, but omits that the notebook supplies no action-learning signal while AdamW can still decay action-head weights.
5. **VERIFIED CLARIFICATION:** “One forward pass answers all questions” (`research/01_laya_training_and_hardware.md:136`) is technically true only as batching: each question duplicates and independently encodes the state. It is not one shared state representation.
6. **VERIFIED SUPPORT:** Existing findings about FP32 master parameters (`research/01_laya_training_and_hardware.md:79-83`), hybrid RLCD+CE (`:49-59`), and stale calibration precedence (`:103-109`) agree with pinned code.

## Minimal One-Step Implementation Specification

An engineer should be able to implement/review the runner solely from this section and the cited functions.

1. **Pins and environment.** Check out source commit `d113dca2512fb3eaca313534bc54c7162d87c1d4`. Download model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` with an explicit `revision`; record resolved snapshot path, file hashes, Python/PyTorch/Transformers/Safetensors/CUDA versions, GPU name, and free/total VRAM. Use one visible GPU and FP16 autocast plus GradScaler. No DDP.
2. **Model.** Read pinned `rl_agent_config.json`, override `max_len=512` and `head_max_len=192` before tokenization, call `build_model(cfg, encoder_dir=<snapshot>/encoder)`, strict-load `<snapshot>/model.safetensors`, enable encoder checkpointing with `use_reentrant=False`, move to `cuda:0`, and call `train()`. Assert total parameters `421293830`, encoder `394781699`, non-encoder `26512131`, and FP32 parameter dtype.
3. **Trainable groups.** Optimize encoder parameters at `2.5e-5`; optimize `head`, `type_emb`, and `scorer` at `1e-4`; exclude `act_head` and the non-parameter temperature buffer. AdamW weight decay is 0.01. Assert every intended parameter appears exactly once and action-head parameters do not appear.
4. **Input.** Construct deterministic Phase 1-shaped questions: one `choice` control-mode question and at least one binary `noul` tool-family question over the same rendered state, with hard one-hot targets. Use `build_sequence(..., max_len=512, head_max_len=192, truncate_left=True)` because the protocol requires retaining the task and recent history; if the task must always survive, implement and unit-test a task-prefix plus recent-history renderer before this call rather than relying on blind left truncation. Assert every expected marker survives.
5. **Batch semantics.** First use exactly one 512-token-padded question sequence for the minimum feasibility result. Then repeat with one complete representative/worst-case bundle flattened to `Q` sequences. Pad/collate as in `collate_items`; report `Q`, tensor shape, actual tokens, and option counts. Do not call a bundle “micro-batch 1” without reporting `Q`.
6. **Loss.** Seed Python/NumPy/PyTorch/CUDA deterministically. Set `G=2`, `sigma=0.4`, `w_sph=0.75`, `w_rps=1.0`. Implement notebook cell 5 lines 299-320 exactly: zero-mean masked perturbations from detached logits, `proper_reward`, per-item group centering, global advantage standardization, Gaussian policy loss, and full-weight soft CE. Exclude action output from loss. Assert all loss/reward/gradient values are finite and log CE and RL terms separately.
7. **One optimizer step.** For memory feasibility set accumulation to 1; accumulation 32 does not lower peak memory and would make “one step” mean 32 forward/backward passes. Reset CUDA peak stats immediately before forward; run one autocast forward, scaled backward, unscale, clip global norm to 1.0, optimizer step, scaler update, and zero gradients. Synchronize CUDA around timing.
8. **Pass/fail evidence.** Save no model. Emit JSON containing pins, resolved config, seeds, parameter counts/dtypes, batch/token shapes, losses/reward/gradient norm, step time, and `max_memory_allocated`/`max_memory_reserved` in bytes and GiB. Pass only if the optimizer step completes, intended encoder and scorer parameters have finite nonzero gradients/updates, excluded action parameters are unchanged, and peak reserved memory is at most 11.5 GiB. OOM is a recorded failure, not a retry with silent CPU fallback.

## Audit Conclusion

- **VERIFIED:** The substrate architecture, full-parameter training path, combined loss, multi-question batching semantics, checkpoint dtype behavior, and calibration precedence are sufficiently specified for a one-step runner.
- **RECOMMENDED:** Do not implement Phase 1 from the notebook verbatim. Use its loss equations, but correct pinning, single-GPU setup, pre-tokenization budgets, ineffective head checkpointing claims, action-head optimization, bundle accounting, and calibration state as specified above.
- **UNKNOWN:** Measured RTX 3060 fit remains the sole substrate execution gate; this audit performed no weight download, cloud action, long training, or destructive edit.
