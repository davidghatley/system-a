# Laya Training and RTX 3060 12 GB Feasibility

Research date: 2026-09-20. The requested parent mission prompt and its literal 18 questions were not present in the empty workspace. The 18 questions below reconstruct the requested scope; this is an **UNKNOWN** limitation on verbatim prompt-claim verification.

## Source Pins

- GitHub `NandhaKishorM/laya`, commit [`d113dca2512fb3eaca313534bc54c7162d87c1d4`](https://github.com/NandhaKishorM/laya/tree/d113dca2512fb3eaca313534bc54c7162d87c1d4), committed 2026-09-20.
- HF base model revision [`1c5edc17a7acd8701df6fc341c0d179f1c62c982`](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982).
- HF specialist revision [`f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`](https://huggingface.co/convaiinnovations/laya-typed-decisions/tree/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2).
- Dataset revision [`ea9306458d6e9563628369a3d1e72e362fb381d2`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/ea9306458d6e9563628369a3d1e72e362fb381d2).
- NVIDIA product page: [GeForce RTX 3060 family](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/).

Labels have precise meanings: **VERIFIED FACT** is directly supported by inspected code/config/data or a command result; **INFERENCE** is reasoned from those facts; **UNKNOWN** lacks direct evidence; **RECOMMENDATION** is proposed action.

## Executive Finding

**VERIFIED FACT:** This machine exposes `NVIDIA GeForce RTX 3060, 12288 MiB, compute capability 8.6` through `nvidia-smi`. Laya is a 421,293,830-parameter ModernBERT-large encoder plus a two-layer transformer decision head. The published weights occupy 842.6 MB in FP16.

**INFERENCE:** Inference should fit comfortably in 12 GB. Full-parameter AdamW fine-tuning at 512 tokens, micro-batch 1, gradient checkpointing, and gradient accumulation is plausible on the RTX 3060. The public 2xT4 notebook cannot run unchanged: it asserts two GPUs, uses DDP/NCCL, 1,024-token sequences, and micro-batch 8 per 16 GB T4. A 1,024-token full fine-tune on 12 GB is **UNKNOWN** until a measured one-step test.

**RECOMMENDATION:** Start at the supplied smoke profile, measure peak allocation, then increase token length or micro-batch one at a time. Do not begin a multi-hour run before a one-step forward/backward and held-out calibration design pass.

## The 18 Questions

### 1. What exactly is Laya?

**VERIFIED FACT:** Laya is not an autoregressive LLM. It uses a bidirectional encoder and emits typed probability distributions for `choice`, ordinal `score`, and binary `noul` questions. Its sequence format is `[CLS] type/instructions [SEP] [MASK] option ... [SEP] state [SEP]`; each option's `[MASK]` hidden state is scored. See [`laya/common.py#L33-L126`](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/laya/common.py#L33-L126).

### 2. What is the architecture and parameter count?

**VERIFIED FACT:** The base and typed-decisions repos report 421,293,830 parameters. The encoder config has hidden size 1,024, 28 layers, 16 heads, intermediate size 2,624, and maximum position embeddings 8,192. Laya adds two `nn.TransformerEncoderLayer`s, a three-entry question-type embedding, an MLP option scorer, and a 256-hidden-unit action head. Sources: [base encoder config](https://huggingface.co/convaiinnovations/laya/blob/1c5edc17a7acd8701df6fc341c0d179f1c62c982/encoder/config.json), [`DecisionModel`](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/laya/common.py#L89-L126).

**VERIFIED FACT:** The action head consumes the pooled token plus detached answer-distribution features, so answer logits do not receive gradients through those action features.

### 3. What checkpoints exist and which should be trained?

**VERIFIED FACT:** The family card lists English ModernBERT-large (421M, configured 512 context), multilingual mmBERT-base (322M, configured 1,024), and typed-decisions ModernBERT-large (421M, configured 1,024). The specialist was fine-tuned for four typed-decisions workflows. Sources: [family card](https://huggingface.co/convaiinnovations/laya/blob/1c5edc17a7acd8701df6fc341c0d179f1c62c982/README.md), [specialist card](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2/README.md).

**RECOMMENDATION:** Fine-tune the English base for a new English domain, not the benchmark specialist, unless the target exactly matches its synthetic workflows. Use multilingual only when language coverage is required.

### 4. What data did the official specialist use?

**VERIFIED FACT:** `LocalLLaMA/typed-decisions` is synthetic, English, Apache-2.0, and has four workflows. Config `all` contains 1,200 train and 400 test cases; each case has five questions, yielding 6,000 train and 2,000 test decisions. The per-workflow configs are alternate views, not extra independent examples. Sources: [dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/ea9306458d6e9563628369a3d1e72e362fb381d2), [notebook preprocessing](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).

### 5. How are records converted into training examples?

**VERIFIED FACT:** The notebook parses JSON `state`, `questions`, and `gold`, creates one sequence per question, normalizes teacher probabilities, and uses the probability argmax as the hard label. Unlike the more complete Hub `rl_common.py`, the notebook does not shuffle option order or train multi-turn prefix slices. Source: [official notebook](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).

### 6. Is this reinforcement learning, supervised learning, or both?

**VERIFIED FACT:** It is hybrid. Four zero-mean Gaussian perturbations of detached logits are scored; a group-mean standardized advantage drives a REINFORCE-style Gaussian policy loss. A full-weight soft cross-entropy term against teacher distributions is added. The public notebook therefore is not “pure policy gradients,” despite a markdown sentence calling it that.

**INFERENCE:** Improvements cannot be attributed solely to RLCD because the CE coefficient is 1.0 and no ablation is supplied.

### 7. What is the reward/loss exactly?

**VERIFIED FACT:** `proper_reward` is log score (floored at -9.21) plus `w_sph *` spherical score; ordinal `score` examples additionally subtract `w_rps *` ranked probability score. Defaults are `w_sph=0.5`, `w_rps=1.0`; the fine-tuning notebook overrides spherical weight to 0.75. Source: [`laya/common.py#L140-L166`](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/laya/common.py#L140-L166).

**INFERENCE:** Positive weighting of strictly proper scores remains strictly proper, but clipping the log score and combining it with finite-sample exploration/CE means the broad marketing statement “the only way to maximise reward” should not be read as an end-to-end calibration guarantee.

### 8. What are the official optimization settings?

**VERIFIED FACT:** The notebook sets 4 epochs, micro-batch 8/GPU, accumulation 4, two GPUs (effective 64 sequences), group size 4, encoder LR `2.5e-5`, head LR `1e-4`, AdamW weight decay 0.01, cosine decay to `1e-6`, FP16 autocast/scaler, gradient clipping 1.0, exploration sigma 0.4 to 0.1, 1,024 max tokens, 256 head tokens, and encoder/head gradient checkpointing.

### 9. Does the published checkpoint prove the notebook produced it?

**VERIFIED FACT:** No. The current specialist config records `updates=7313`, `epochs_completed=1`, `hours=1.96`, `world_size=1`. The notebook specifies 4 epochs and world size 2. The base config contains the same training record. Source: [specialist config](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2/rl_agent_config.json).

**UNKNOWN:** Which exact code, data revision, seed, GPU, and hyperparameters produced the published specialist weights. The notebook saves `fine_tuned`, temperatures, and sizing fields but does not populate that `training` object.

### 10. How long does official training take?

**VERIFIED FACT:** The specialist model card says about 4–5 hours on 2xT4. A notebook markdown cell says 4–6 minutes. The checkpoint config says 1.96 hours on world size 1 for a different recorded run.

**UNKNOWN:** A reproducible wall-clock time. The 4–6 minute claim is incompatible with the model card and is likely erroneous.

**INFERENCE:** An RTX 3060 is materially slower than 2xT4 DDP for this workload; expect hours, not minutes, after establishing a configuration that fits.

### 11. How much fixed memory does full fine-tuning require?

**VERIFIED FACT:** The FP16 checkpoint is 842.6 MB. The notebook constructs a default FP32 model and loads the checkpoint into it; autocast does not turn master parameters into FP16. A rough lower bound for FP32 parameters + FP32 gradients + two FP32 Adam moments is `421.3M * 16 bytes = 6.74 GB`, excluding activations, temporary tensors, CUDA context, allocator fragmentation, and possible optimizer implementation overhead.

**INFERENCE:** 12 GB leaves roughly 5 GB for everything dynamic. Gradient checkpointing and micro-batch 1 are essential. The estimate is not a measured peak and cannot certify fit.

### 12. Will inference fit and run on an RTX 3060 12 GB?

**VERIFIED FACT:** The local GPU has 12,288 MiB and Ampere compute capability 8.6; the runtime uses BF16 for capability 8+, SDPA attention, and an ~843 MB FP16 checkpoint. Source: local `nvidia-smi`; [`laya/agent.py#L194-L207`](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/laya/agent.py#L194-L207).

**INFERENCE:** One checkpoint should fit comfortably for inference. Preloading all three checkpoints is not recommended on 12 GB because resident weights plus model structures and inference workspaces could consume several GB and offer little benefit for a single-domain service.

### 13. Will full fine-tuning fit on the RTX 3060 12 GB?

**INFERENCE:** Probably at 512 tokens, micro-batch 1, both checkpointing modes enabled, and accumulation. It may fit at 1,024 tokens with micro-batch 1, but this is **UNKNOWN** without the actual dependency stack and a measured backward pass. The official micro-batch 8 at 1,024 is not credible on 12 GB.

**RECOMMENDATION:** If full tuning OOMs, first lower `max_len` to 384/256, then freeze the encoder or train only the head. The official code has no LoRA/PEFT path; adding one is a separate engineering and validation task.

### 14. Does BF16 work on this GPU?

**VERIFIED FACT:** RTX 3060 is Ampere (local compute capability 8.6), and Laya selects BF16 on capability 8+. The official notebook nevertheless explicitly uses FP16 autocast and `GradScaler`.

**RECOMMENDATION:** Reproduce the notebook with FP16 first. Treat BF16 as an optional measured variant; it usually avoids scaling concerns but does not halve FP32 Adam/master-state memory in this implementation.

### 15. Is calibration implemented correctly?

**VERIFIED FACT:** Inference selects `temperature_by_options` first and only falls back to per-type `temperature`. The specialist fitted per-type temperatures near 1.0, but inherited base `temperature_by_options` values remain present and therefore override them for all represented buckets. The specialist card explicitly acknowledges this. Sources: [`laya/agent.py#L300-L305`](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/laya/agent.py#L300-L305), [specialist config](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2/rl_agent_config.json).

**VERIFIED FACT:** The notebook calibrates on `all_items[::15][:400]`, a subset of training examples, not a held-out calibration split. It fits only three per-type temperatures, not option-count buckets.

**RECOMMENDATION:** Reserve an untouched calibration split, remove stale bucket temperatures or fit those same buckets, and report pre/post NLL, Brier, ECE, and reliability plots on a separate test set.

### 16. Are the reported quality claims independently verified?

**VERIFIED FACT:** Repository-authored results report specialist accuracy 0.766, soft accuracy 0.471, Brier 0.062, ECE 0.213, and score MAE 0.242 on 400 test cases. Base accuracy is 0.362, below the 0.461 majority baseline. The cards state Jev numbers are third-party, use differing prompts/sample sizes, and were not measured by this project. Sources: [specialist card](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2/README.md), [`BENCHMARKS.md`](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/BENCHMARKS.md).

**UNKNOWN:** Independent reproduction of 0.766 from the pinned model/data. No long evaluation or 843 MB model download was performed here.

**INFERENCE:** “Beats Jev” is indicative, not a controlled head-to-head result. “Surpasses teacher ceiling” is possible for noisy labels but does not establish generalization beyond this synthetic distribution.

### 17. What important model/data limits affect deployment?

**VERIFIED FACT:** The specialist is English-only and trained on four synthetic workflows. Its option texts share a 256-token head budget; high-cardinality choices are truncated to as few as four tokens/option. The project reports Banking77 accuracy 0.492 for the specialist versus Jev's separately published 0.870, and recommends under ~20 options. `score` is the weakest primitive in broader benchmarks. Sources: [family card](https://huggingface.co/convaiinnovations/laya/blob/1c5edc17a7acd8701df6fc341c0d179f1c62c982/README.md), [`build_sequence`](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/laya/common.py#L49-L86).

**RECOMMENDATION:** Validate on real, chronologically held-out target-domain data; test option-order permutations, class imbalance, drift, abstention/escalation, and failure costs before production.

### 18. What is the safest reproducible path on this machine?

**RECOMMENDATION:** Run `experiments/laya_smoke_test/run.sh`. It performs dependency-free source/config/GPU preflight and runs exact tiny reward tests only if PyTorch is already available. Then create an isolated environment, pin package/model/data revisions, download the model, and run exactly one forward/backward optimizer step with the supplied conservative config while recording `torch.cuda.max_memory_allocated()` and `max_memory_reserved()`. Only after that should a short subset-overfit test and then a real run be authorized.

**RECOMMENDATION:** For credible science, use three splits (train/calibration/test), fixed seeds, immutable raw examples, dataset hashes, logged package/CUDA versions, saved resolved config, and metrics by primitive/workflow/class. Compare CE-only against RLCD+CE; otherwise the specific value of RLCD remains unmeasured.

## Claim Audit

| Claim | Status | Finding |
|---|---|---|
| 421M parameters | **VERIFIED FACT** | HF safetensors metadata: 421,293,830. |
| One forward pass answers all questions | **VERIFIED FACT** | Questions are separate sequences collated into one batch/forward call, not one shared encoded state. Runtime scales with question count. |
| “Nothing to hallucinate” | **INFERENCE** | No text generation, but confidently wrong classifications remain possible and are documented. |
| Mathematically calibrated probabilities | **INFERENCE** | Proper scoring reward supports truthful distributions in theory; shipped ECE and stale temperature precedence do not guarantee calibration. |
| 0.766 typed-decisions accuracy | **VERIFIED FACT** as a repository claim; **UNKNOWN** independently | Result files/card report it; not reproduced here. |
| Beats Jev 0.727 | **INFERENCE** | Not a controlled same-run comparison. |
| 4–5 hours on 2xT4 | **UNKNOWN** | Card claim conflicts with notebook and checkpoint metadata. |
| Public notebook reproduces published checkpoint | **UNKNOWN** | Config provenance conflicts materially. |
| RTX 3060 12 GB can train it | **INFERENCE** | Constrained full tuning is plausible; unchanged official notebook cannot run and fit is unmeasured. |

## Smoke-Test Interpretation

The supplied smoke test intentionally does not download weights or launch training. `preflight.py` verifies the pinned remote JSON/configs, local GPU, and rough fixed-state memory. `smoke_loss.py` checks finite rewards/gradients and the ordinal RPS behavior with tiny tensors. `config.json` is a proposed first measured one-step profile, not an official supported CLI config.

## Commands Actually Executed

```text
git ls-remote https://github.com/NandhaKishorM/laya.git HEAD
git clone --depth 1 https://github.com/NandhaKishorM/laya.git /tmp/opencode/laya-investigation
git rev-parse HEAD
git log -1 --format='%H%n%cI%n%s'
jq -r '.cells[] | select(.cell_type=="code") | .source[]' notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python -c 'import torch; ...'                 # failed: torch not installed
python -c 'import transformers, safetensors; ...'  # failed: modules not installed
firecrawl --status
firecrawl search 'site:nvidia.com GeForce RTX 3060 12 GB specifications memory bandwidth CUDA cores' --limit 3 --json
```

Read-only Hugging Face MCP/API calls inspected both model repos, all file listings, cards, configs, Hub revisions, and dataset structure. No weights were downloaded, no environment was modified, and no training/evaluation was launched.
