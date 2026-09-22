# Iteration 3 pre-run repair 1

Date: 2026-09-22

## Decision

**READY FOR INDEPENDENT RE-REVIEW; NOT READY FOR GPU WORK.** The execution gate
remains blocked. This repair used CPU/offline checks only, did not import Torch,
initialize CUDA, load model weights, run inference/training/smoke, or open final
test records, labels, or results. `PRE_RUN_REVIEW.md` was not modified.

## Verified

- Chronology is reconciled to the accepted manifest and structural verification:
  1,051/140/122 records and 163/20/23 task groups. The older 1,184/156/139
  summary is superseded. The source provenance file is identified as inherited
  i2 lineage, while the manifest and `verification.json` attest the v3 renderer.
- The estimand is narrowed to retained whole target turns containing at least
  one recognized tool call and exactly one category. Absence-based no-call
  completion remains excluded; the frozen six labels are unchanged.
- Renderer choice-logit order is frozen as `edit, execute, other_tool, read,
  respond_or_finish, search`. Reporting labels map to indices `3,5,0,1,2,4`.
- The zero ancillary-action placeholder is removed. The operational objective is
  six-choice Gaussian score-function RLCD plus unit-weight cross-entropy. For
  each class, the CPU reference produced a finite RLCD loss with absolute value
  above `1e-6` and an RLCD-only gradient L1 norm above `1e-6`; exact values are
  in `preflight/repair_preflight.json`.
- Every train/dev record has a unique stable row ID and valid 64-hex trajectory
  and task-group IDs; each trajectory maps to one group. The preflight proved
  1,051 rows/163 groups and 140 rows/20 groups. Bootstrap unit is `task_group`.
- A real nested `fcntl.LOCK_NB` probe rejected lock contention, and acquisition
  after a simulated Torch import was rejected. Dev selection now
  rejects candidates without strict, bit-exact model-state and exact optimizer,
  scheduler, scaler, and stopping-state reload evidence. Latency validation
  rejects missing synchronization, phase, warmup, sample-count, positivity, or
  finiteness evidence.
- Fifteen CPU unit tests passed. Config, accepted train/dev files, static compile,
  both predeclaration hashes, and whitespace checks passed. `base-eval` remained
  blocked before model/CUDA and reported `test_opened=false`.

## Inferred

- The validated contracts should prevent a future executor from selecting a
  checkpoint on asserted-only reload status or reporting latency from incomplete
  timing boundaries. This is inferred from reject-by-default validators and
  unit tests, not from a model runtime.
- Task-group resampling preserves all rows from each selected group; in the
  accepted train/dev data each observed task group currently corresponds to one
  trajectory.

## Unknown

- Actual strict checkpoint reload, CUDA synchronization, warm/cold latency,
  peak VRAM, smoke duration, and GPU-lock behavior across separate GPU workers
  remain unmeasured.
- Semantic/family leakage and independent acceptance of the narrowed estimand
  remain unknown.
- Test labels/results, model metrics, seed variation, and comparative added
  value remain unknown and were not inspected.

## Commands and results

All Python commands used `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and four-thread CPU limits.

```text
.venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
# 15 tests passed in 0.337s

.venv/bin/python -B training/laya_trace_i3/runner.py validate-config
# config_valid; torch_imported=false; model_loaded=false

.venv/bin/python -B training/laya_trace_i3/runner.py validate-accepted-manifest --manifest-sha256 158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2
# accepted; train=1051, dev=140; test not opened

.venv/bin/python -B training/laya_trace_i3/runner.py repair-preflight --report artifacts/experiment_i3/preflight/repair_preflight.json
# repair_preflight_passed; all six objective proofs; train/dev group proofs;
# torch_imported=false; model_loaded=false; test_opened=false

.venv/bin/python -B training/laya_trace_i3/runner.py base-eval
# blocked_before_model_or_cuda; expected_updates_per_seed=68;
# cuda_initialized=false; test_opened=false

.venv/bin/python -B -m py_compile training/laya_trace_i3/__init__.py training/laya_trace_i3/metrics.py training/laya_trace_i3/protocol.py training/laya_trace_i3/experiment.py training/laya_trace_i3/runner.py training/laya_trace_i3/test_i3.py
# passed without output

sha256sum -c artifacts/experiment_i3/PREDECLARATION.sha256
sha256sum -c artifacts/experiment_i3/PREDECLARATION.v1.sha256
# both OK

git diff --check
# passed without output
```

## Re-freeze

Version 1 is preserved byte-for-byte at `PREDECLARATION.v1.md`, SHA-256
`63e46b2f4d8180aa3ed98f5165c138e0756a7d0729b6fd5cc88259c9f94371ac`.
Version 2 was necessarily re-frozen before GPU work because the frozen config
had inherited `0.0 * act.sum()` and did not specify the six-logit permutation.
Version 2 SHA-256 is
`5ca24d90774a7a5b2ec42901f39738b7ec87681a6fcb2b415253bea0f3dfc99b`.

## Changed files and SHA-256

```text
361e205d73d0ab20b475b51a2bbb050473bcc4f767e30cd42412fbeece134adc  training/laya_trace_i3/README.md
748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae  training/laya_trace_i3/config.json
6332086af100b26d1a0605cc7754108b41697ab71d083cadba6aae00ac108185  training/laya_trace_i3/experiment.py
9f3653412a3641da6935c72015f32b2cec5f9a2780853732d4480d518c1c4631  training/laya_trace_i3/protocol.py
10b65a0ea98afdb493fa49701d73d463642eb4432c555b48474fc3dec13d2450  training/laya_trace_i3/runner.py
05b787568ce532ee20b5284cf1adb779daee336e3c083d8c87502ef2af8303b9  training/laya_trace_i3/test_i3.py
5ca24d90774a7a5b2ec42901f39738b7ec87681a6fcb2b415253bea0f3dfc99b  artifacts/experiment_i3/PREDECLARATION.md
5b097dc0f64bfe676aedde5a917c3894941628b24a836d995bb4158ab186a329  artifacts/experiment_i3/PREDECLARATION.sha256
63e46b2f4d8180aa3ed98f5165c138e0756a7d0729b6fd5cc88259c9f94371ac  artifacts/experiment_i3/PREDECLARATION.v1.md
9d55749de2c6e65953b561775b38c665ba78a5f08d4e922f00766d6ab1166eef  artifacts/experiment_i3/PREDECLARATION.v1.sha256
ec0f23485c371d6e00d763c8b288e958b3a5c180ae42281e43688fde4a0441a1  artifacts/experiment_i3/preflight/COMMANDS.md
7cc0e659853df74e47c922cb756f1e469ef3c1a9183b394cb4b5f6dc0db08192  artifacts/experiment_i3/preflight/PREFLIGHT_REPORT.md
1c6225f0ab2c941479ef02774e311b4bc067024b3ee1292f47360f258c41a17a  artifacts/experiment_i3/preflight/repair_preflight.json
3e839c013002d348d97cd23866b20160fc8c75f75db1bd863a9e24eaab71ef60  artifacts/experiment_i3/preflight/guard_probe/gpu.lock
49fa5e4a88388d03abc27175d38ee12c06f41b879080177398bc0bffaf8a3417  research/28_iteration_3_experiment.md
```

`REPAIR_1.md` is omitted from its own hash list. `py_compile` refreshed ignored
bytecode under the authorized training directory; bytecode is not evidence and
is excluded from the durable file list. No files outside the three authorized
paths were modified.
