# FP16 Initial-Loss-Scale Diagnostic

Measurement date: 2026-09-20.

## Outcome

- **VERIFIED:** Exactly one FP16 autocast forward and one scaled backward completed with `GradScaler(init_scale=1.0)`. No retry, BF16 run, training run, or optimizer step occurred.
- **VERIFIED:** The combined loss (`2.2289695739746094`), cross-entropy loss (`2.478915214538574`), RL loss (`-0.24994561076164246`), and all rewards were finite before backward.
- **VERIFIED:** Every optimized gradient was finite both before and after unscale. There was no first offending gradient and the complete offending-name lists were empty. The pre- and post-unscale global gradient norms were both `293.410888671875`, as expected at scale 1.0.
- **VERIFIED:** Optimizer steps attempted/completed remained `0/0`. Exact SHA-256 hashes over every named parameter were identical before and after backward (`9286915c980c8b11e8941bbc8d8e6face4da721b9a74fdd5607e5e6652d4fcf8`).
- **INFERRED:** The prior run's non-finite gradients are consistent with initial loss-scale overflow. Holding the pinned model, input, seed, FP16 autocast, and software environment fixed while reducing the GradScaler initial scale from its default `65536.0` to `1.0` produced entirely finite gradients. This isolates loss scale as the changed numerical condition, subject to the nondeterministic attention-backward caveat below.

## Reproduction

The complete command and terminal output are preserved in `artifacts/gpu_diagnostics/fp16_scale1_command.log`. Structured measurements are in `artifacts/gpu_diagnostics/fp16_scale1.json`.

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
  --diagnostic-fp16-scale1 \
  --output artifacts/gpu_diagnostics/fp16_scale1.json
```

`scripts/run_laya_one_step.py` diagnostic mode initializes the scaler at 1.0, records gradients before and after `unscale_`, hashes all parameters before and after backward, asserts zero optimizer-step counters, and returns before the script's only `scaler.step` call.

## Measurements

| Measurement | Value |
|---|---:|
| Forwards / backwards | `1 / 1` |
| Optimizer steps attempted / completed | `0 / 0` |
| Forward | `0.5620462030055933 s` |
| Backward | `0.22748225001851097 s` |
| Unscale | `0.026488795992918313 s` |
| Total measured region | `5.60869649000233 s` |
| Baseline CUDA allocated / reserved | `1,686,235,136 / 1,700,790,272 bytes` |
| Peak CUDA allocated / reserved | `3,750,020,608 / 3,873,439,744 bytes` |
| Peak CUDA allocated / reserved | `3.492478847503662 / 3.607421875 GiB` |

The total region includes parameter hashing and other diagnostics in addition to forward, backward, and unscale, so it is not a training-step throughput measurement.

## Scope And Caveat

- **VERIFIED:** Model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, Laya source revision `d113dca2512fb3eaca313534bc54c7162d87c1d4`, seed 42, one 48-token sequence padded to `[1, 512]`, FP32 master parameters, and FP16 autocast were used.
- **VERIFIED:** All paths, caches, command logs, and outputs remained under `/home/davidhatley/Projects/research/system_a`.
- **VERIFIED:** PyTorch warned that memory-efficient attention backward used a nondeterministic algorithm despite deterministic algorithms being requested with `warn_only=True`.
- **UNKNOWN:** Because of that kernel nondeterminism, this one diagnostic cannot prove that every possible replay at scale 1.0 is finite. It does show that the one authorized scale-1 diagnostic was finite throughout.
