# Executor Repair 1

Date: 2026-09-22. Scope: `training/laya_trace_i3/`, this artifact, and
`research/28_iteration_3_experiment.md` only.

## Changes

- Replaced the retained 64-element autograd `window` with immediate scaled
  backward per microforward. The divisor is derived from the positional
  accumulation window, so every full window uses 64 and every epoch tail uses
  its actual count even after shuffling. The declared schedule remains
  `ceil(1051 / 64) * 4 = 68` optimizer/scheduler updates.
- Preserved AMP/scaler, finite-loss and finite-gradient guards, clipping,
  scheduler stepping, RLCD+CE, no-retry behavior, and checkpoint stopping
  state. Loss logging retains detached scalar values only, so closed windows do
  not retain computation graphs and fit the measured 12 GiB-card contract.
- Replaced the runner's row-only final-test route with a composed evaluator:
  CPU freeze provenance and deterministic dev-candidate/reload validation,
  result-exists refusal, nonblocking GPU lock, atomic one-shot marker, one
  test decode, strict selected-checkpoint load, fixed metrics/predictions,
  task-group bootstrap, synchronized latency, and atomic result publication.
  Any failure after marker creation leaves the consumed marker in place; no
  retry path exists.

## CPU Evidence

Command:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
```

Result: **24/24 passed**. The added tests prove positional boundaries
`[4, 4, 2]` on a shuffled 10-record order, selection rejection before lock or
test access, and lock -> marker -> test-decode ordering while stopping before
real test rows. No CUDA, model import, test record, or actual final marker was
consumed.

Additional model-free checks:

```text
... .venv/bin/python -B training/laya_trace_i3/runner.py validate-config
... .venv/bin/python -B training/laya_trace_i3/runner.py dry-run
... .venv/bin/python -B training/laya_trace_i3/runner.py repair-preflight
```

All returned success with `torch_imported=false`, `model_loaded=false`, and
`test_opened=false`. The actual-model `base-eval` command was not authorized
and therefore was not run as model work.

## Hashes

These are SHA-256 values from the repaired worktree at report creation:

```text
training/laya_trace_i3/experiment.py f37891af485b30f94e3c507fa97a8d08866a113cef45c02a93ccffdec078f92b
training/laya_trace_i3/runner.py a92ae3a9937d2cbbd4be5cfb8b6fe376bd51d7911b6574afa34d471acfe5f0bb
training/laya_trace_i3/test_i3.py 869ecfd62e30a9b7a71806ccd93ff1f8f01d1ff54f272c5a7651f1a537f631bd
training/laya_trace_i3/config.json 748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae
artifacts/experiment_i3/preflight/data_manifest.v3.json 158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2
artifacts/experiment_i3/preflight/actual_model_pilot.json 2323e509f35e9d2e0e2fba64bd12dcb28a488f6086715c0ac4e115226a65d511
```

The measured pilot artifact remains unchanged. Its seed-42 measurements
(`training_seconds=19.15747142105829`, `cold_load_seconds=6.082518487004563`,
`dev_evaluation_seconds=7.365298995980993`, `updates=2`) project, using the
declared 68 updates and `1.25` multiplier, to `831.002307248709` GPU seconds
per seed and `1662.004614497419` for both declared seeds. This is a projection,
not a completed training result.
