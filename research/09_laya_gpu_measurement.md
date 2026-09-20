# Laya RTX 3060 One-Step Measurement

Measurement date: 2026-09-20.

## Outcome

- **VERIFIED BLOCKER:** The smallest configured Laya run completed one CUDA forward and one scaled backward, but the runner detected non-finite gradients after unscaling and refused the optimizer step. Optimizer steps attempted/completed: `0/0`.
- **VERIFIED:** There was no CPU fallback, architecture substitution, DDP, cloud compute, or run beyond the single forward/backward. No checkpoint was saved.
- **VERIFIED:** Peak CUDA memory was 3,750,020,608 bytes (3.493 GiB) allocated and 3,873,439,744 bytes (3.607 GiB) reserved. Memory was not the blocker and stayed below 11.5 GiB.
- **UNKNOWN:** A successful optimizer-step time and post-step encoder/head update cannot be reported because the finite-gradient guard correctly stopped before `optimizer.step()`.
- **UNKNOWN:** Loss component values were finite enough to reach backward, but the first runner version did not persist them before the gradient guard raised. The runner now persists loss and offending-gradient diagnostics before raising; it was not rerun because this task permits one measured attempt.

## Reproduction

The command and terminal output are preserved in `artifacts/laya_one_step_commands.log`; structured evidence is in `artifacts/laya_one_step.json`.

```bash
env TMPDIR="$PWD/data/cache/tmp" \
  PIP_CACHE_DIR="$PWD/data/cache/pip" \
  HF_HOME="$PWD/data/cache/huggingface" \
  HF_HUB_CACHE="$PWD/data/cache/huggingface/hub" \
  HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets" \
  TRANSFORMERS_CACHE="$PWD/data/cache/huggingface/transformers" \
  TORCH_HOME="$PWD/data/cache/torch" \
  XDG_CACHE_HOME="$PWD/data/cache/xdg" \
  .venv/bin/python scripts/run_laya_one_step.py
```

The runner enforces `max_steps=1`, one sequence, accumulation one, a 512-position tensor, explicit pinned paths/revisions, CUDA-only execution, and action-head exclusion. `--help` and `--dry-run` were verified before downloading or execution.

## Pins And Environment

| Item | Verified value |
|---|---|
| Laya source | `NandhaKishorM/laya@d113dca2512fb3eaca313534bc54c7162d87c1d4` |
| Model | `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982` |
| Dataset pin | `LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2` (not downloaded or used) |
| GPU | NVIDIA GeForce RTX 3060, compute capability 8.6, 12,478,185,472 bytes |
| Driver / CUDA runtime | 610.57.04 / PyTorch CUDA 12.8 |
| Python / PyTorch | 3.14.7 / 2.11.0+cu128 |
| Transformers | 5.17.0 |
| Safetensors | 0.8.0 |
| Hugging Face Hub | 1.32.0 |
| NumPy | 2.5.3 |

All downloads, caches, source, environment files, and command working directories remained under the repository. The source checkout is `data/laya`, the environment is `.venv`, and caches are under `data/cache`.

## Measured Configuration

- Seed 42; Python, NumPy, PyTorch, and CUDA seeded; deterministic algorithms requested with `warn_only=True`.
- One deterministic binary `choice` sequence: tensor shape `[1, 512]`, 48 actual tokens, 2 options.
- FP32 master parameters, FP16 autocast, GradScaler, encoder gradient checkpointing, AdamW, and perturbation group size 2.
- Encoder LR `2.5e-5`; decision head LR `1e-4`; weight decay `0.01`; global clip target `1.0`.
- Action head excluded from AdamW. Since the optimizer did not run, encoder, decision head, and action head all remained unchanged.

## Parameter Accounting Correction

The pinned runtime exposes 421,293,827 named parameters: 394,781,696 encoder and 26,512,131 non-encoder. Its state dictionary has 421,293,830 elements because the non-parameter `temperature` buffer contributes three elements. The earlier audit’s 421,293,830 “parameter” assertion conflated state elements with parameters. Of the non-encoder parameters, 26,248,193 were eligible for optimization and 263,938 action-head parameters were excluded.

## Failure Analysis

- **VERIFIED:** Forward and backward each ran exactly once. Peak memory was measured after model/optimizer setup and through backward.
- **VERIFIED:** PyTorch warned that memory-efficient attention backward selected a non-deterministic algorithm despite `warn_only=True`; exact bitwise reproducibility is therefore not established.
- **VERIFIED:** At least one optimized parameter had a NaN or infinity gradient after GradScaler unscale. The optimizer was not attempted, preventing parameter corruption.
- **INFERRED:** The immediate issue is FP16 numerical stability, not VRAM capacity. BF16 autocast is the smallest likely follow-up because Ampere supports BF16 and the architecture would remain unchanged, but that is unmeasured and was not attempted under the one-attempt boundary.
- **UNKNOWN:** The exact first offending parameter and finite loss values were not retained by the attempted runner version. The implementation now records these on future failures.

## Conclusion

This run does not establish a successful Laya optimizer step on the RTX 3060. It establishes that one 512-position forward/backward fits comfortably, while the pinned FP16 path fails the finite-gradient gate before AdamW. The precise reproducible blocker is non-finite unscaled gradients under FP16 with the versions and command above.
