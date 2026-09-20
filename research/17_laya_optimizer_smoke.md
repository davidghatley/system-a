# Laya FP16 Scale-1 Optimizer Smoke Test

Measurement date: 2026-09-20.

## Outcome

- **VERIFIED:** Exactly one FP16-autocast forward, one `GradScaler(init_scale=1.0)` backward, and one AdamW optimizer step completed. The persisted counters are `1 / 1 / 1 attempted / 1 completed`, with `max_steps=1` asserted.
- **VERIFIED:** No retry, BF16 comparison, checkpoint save, or training continuation occurred.
- **VERIFIED:** Combined loss `2.2289695739746094`, cross-entropy loss `2.478915214538574`, RL loss `-0.24994561076164246`, and all rewards were finite before backward.
- **VERIFIED:** All optimized gradients were finite before and after unscale. Their global norm was `293.410888671875` before clipping.
- **VERIFIED:** Encoder and scorer parameters changed, while the excluded action head remained byte-identical.
- **VERIFIED:** Peak CUDA allocation was `6.481866359710693 GiB`; peak reservation was `6.556640625 GiB`.

## Parameter Evidence

| Group | SHA-256 before | SHA-256 after | Result |
|---|---|---|---|
| Encoder | `f7aa4c8a40271b2ade427b18938155e6dc78998125051eb56ab112c3960a4308` | `50ecd038bfb22786189cce0a38d9bb8a2accd524794a74e9640dbedc27e42e60` | Changed |
| Scorer | `5e1269c1d9dbb381248adeb9923bdb9604548373ad3da1e3967ebc717e7e004c` | `68b94018e9495e8f54c8abf77924a4d6e03f0ae6c39a7c13cb71f797f6e50b9b` | Changed |
| Action head | `baad5de68b70b686f58b9718ea7a8de82085d43450a17cfc591cffd7b1490d2d` | `baad5de68b70b686f58b9718ea7a8de82085d43450a17cfc591cffd7b1490d2d` | Unchanged |

Sampled encoder parameter `encoder.embeddings.tok_embeddings.weight` changed by `2.500065602362156e-05`. Sampled scorer parameter `head.layers.0.self_attn.in_proj_weight` changed by `-9.994208812713623e-05`.

## Timing And Memory

| Measurement | Value |
|---|---:|
| Total measured step region | `5.554485362954438 s` |
| Optimizer step and synchronization | `0.17018723901128396 s` |
| Baseline allocated / reserved | `1,686,235,136 / 1,700,790,272 bytes` |
| Peak allocated / reserved | `6,959,851,008 / 7,040,139,264 bytes` |

The total region includes diagnostic parameter hashing and is not a throughput benchmark.

## Reproduction Record

Structured evidence is in `artifacts/gpu_diagnostics/fp16_scale1_optimizer_step.json`; complete terminal output and the exact command are in `artifacts/gpu_diagnostics/fp16_scale1_optimizer_step_command.log`.

```bash
env TMPDIR="$PWD/data/cache/tmp" \
  PIP_CACHE_DIR="$PWD/data/cache/pip" \
  HF_HOME="$PWD/data/cache/huggingface" \
  HF_HUB_CACHE="$PWD/data/cache/huggingface/hub" \
  HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets" \
  TRANSFORMERS_CACHE="$PWD/data/cache/huggingface/transformers" \
  TORCH_HOME="$PWD/data/cache/torch" \
  XDG_CACHE_HOME="$PWD/data/cache/xdg" \
  .venv/bin/python scripts/run_laya_one_step.py \
  --fp16-scale1-optimizer-step \
  --output artifacts/gpu_diagnostics/fp16_scale1_optimizer_step.json
```

The run used Laya source revision `d113dca2512fb3eaca313534bc54c7162d87c1d4`, model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, seed 42, and one 48-token sequence padded to `[1, 512]`. All paths and caches remained repository-local.

## Caveat

- **VERIFIED:** PyTorch warned that memory-efficient attention backward used a nondeterministic algorithm despite deterministic algorithms being requested with `warn_only=True`.
- **UNKNOWN:** This single authorized smoke step does not establish stability over continued training or every replay. No continuation was attempted.
