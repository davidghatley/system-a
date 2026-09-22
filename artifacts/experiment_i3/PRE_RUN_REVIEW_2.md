# Iteration 3 independent pre-run review 2

Date: 2026-09-22

## Scope

CPU/offline review only. I did not import Torch, initialize CUDA, load model
weights, run smoke/training/inference, or open final-test records, labels, or
results. I modified only this report.

## Verified

- Frozen specification v2 hashes to
  `5ca24d90774a7a5b2ec42901f39738b7ec87681a6fcb2b415253bea0f3dfc99b`.
  Its source revision, source hash, renderer revision/hash, Laya revision, base
  revision, model/config hashes, six labels, seeds, optimizer, schedule, FP16,
  clipping, smoke projection rule, metrics, and exploratory test-once policy are
  explicit. The retained v1 freeze is not treated as the active specification.
- Corrected chronology is coherent: the accepted manifest hashes to
  `158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2`
  and records 1,051 train, 140 dev, and 122 test rows in 163/20/23 task groups.
  The current verification evidence hashes to
  `70ff4ca75d7e682fd2911b081b125b5d0753a914572327843a6a9a6d8db97378`,
  reports 1,313 retained rows and a deterministic pass, and is hash-bound by
  the manifest. The stale 1,184/156/139 review chronology is superseded by the
  subsequent integrity repair and refreshed provenance.
- The estimand is now explicitly restricted to retained whole target turns with
  a recognized tool call and exactly one category. No-call targets are excluded;
  the frozen `respond_or_finish` class is allowed to have zero support.
- Reporting labels map to renderer logits as `read:3, search:5, edit:0,
  execute:1, other_tool:2, respond_or_finish:4`. The dependency-free reference
  perturbs all six logits and produced finite RLCD losses and nonzero RLCD
  gradient L1 norms for every class. The zero `act_head` placeholder is absent
  from the repaired declared objective.
- Train/dev records have unique stable row IDs and valid trajectory/task-group
  IDs. The CPU proof found 1,051 rows/163 groups and 140 rows/20 groups, with one
  task group per trajectory. The declared bootstrap unit is `task_group`.
- `fcntl.LOCK_EX | LOCK_NB` contention is exercised with a genuinely nested
  lock and rejected. Lock acquisition is also rejected when `torch` is already
  present in `sys.modules`.
- Fixed-label metrics, missing-class behavior, count-derived 68 updates/seed,
  two-seed default, smoke failure/memory/time checks, latency evidence shape,
  and dev-only deterministic selection reject paths pass their current tests.

## Blocking findings

1. **The smoke path is not operational and therefore cannot enforce lock-before-Torch.**
   `runner.py` routes `smoke` (like every GPU command) to `planned_run`, which
   always returns `blocked_before_model_or_cuda`; no execution path acquires
   `GpuLock` and then imports Torch. The pure-Python objective is a reference,
   not an implemented training step. The lock primitive is sound in isolation,
   but there is no bounded smoke worker to prove that the primitive, six-class
   objective, and resource guards are composed in the required order.

2. **Checkpoint reload evidence remains self-asserted rather than bound to
   checkpoint bytes or a reload operation.** `select_dev_checkpoint` checks
   caller-provided booleans and equal digest strings, but does not require the
   checkpoint path/files to exist or recompute their hashes. A CPU probe passed
   a nonexistent checkpoint with fabricated matching values and it was selected.
   This does not establish strict, bit-exact model reload or exact optimizer,
   scheduler, scaler, and stopping-state restoration.

3. **The final-test-once guard is bypassable.** `consume_final_test_attempt`
   atomically creates a marker, but `load_split(..., "test", ..., allow_test=True)`
   opens test bytes without requiring that marker or a matching dev-freeze hash.
   No final-test command composes marker consumption and test opening, and the
   marker behavior has no unit test. The primitive does not currently guarantee
   marker-before-open or no retry.

4. **Frozen lifecycle and budget fields are not fully validated or enforced.**
   `validate_config` accepted mutations setting `iteration_max_elapsed_seconds`
   to `1`, changing `final_test_policy` to `disabled`, and replacing the warm
   latency definition with `unsynchronized`. The 14,400-second iteration ceiling
   has no runtime reference. The 7,200 GPU-second value is used only for the
   pre-run projection, not as an aggregate elapsed-time stop, and latency
   validation accepts caller assertions rather than measuring synchronized
   boundaries. These are insufficient runtime guards for GPU execution.

## Exact commands and results

All Python commands used `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`,
`MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `PYTHONDONTWRITEBYTECODE=1`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 -v
# 15 tests passed in 0.307s; OK

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py validate-config
# config_valid; config 748b960d...dfb3ae; torch_imported=false; model_loaded=false

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py validate-accepted-manifest --manifest-sha256 158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2
# validated_no_model_load; train=1051; dev=140; test_opened=false;
# torch_imported=false; model_loaded=false; expected_updates_per_seed=68

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py repair-preflight
# repair_preflight_passed; six objective proofs; train/dev 163/20 groups;
# torch_imported=false; model_loaded=false; test_opened=false

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B training/laya_trace_i3/runner.py base-eval
# blocked_before_model_or_cuda; train=1051; dev=140; updates=68;
# torch_imported=false; cuda_initialized=false; model_loaded=false; test_opened=false

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -c 'import copy,json,sys; from pathlib import Path; from training.laya_trace_i3.protocol import validate_config; p=Path("training/laya_trace_i3/config.json"); c=json.loads(p.read_text()); c["budget"]["iteration_max_elapsed_seconds"]=1; c["evaluation"]["final_test_policy"]="disabled"; c["latency"]["warm_definition"]="unsynchronized"; validate_config(c); print("mutated_lifecycle_config_accepted=true"); print(f"torch_imported={str(chr(116)+chr(111)+chr(114)+chr(99)+chr(104) in sys.modules).lower()}")'
# mutated_lifecycle_config_accepted=true; torch_imported=false

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -c 'import sys; from training.laya_trace_i3.protocol import FIXED_LABELS,select_dev_checkpoint; d="a"*64; e={"strict_model_load":True,"model_state_before_sha256":d,"model_state_after_sha256":d,"training_state_file_sha256":d,"optimizer_exact":True,"scheduler_exact":True,"scaler_exact":True,"stopping_state_exact":True}; c={"seed":42,"split":"dev","macro_f1":0.5,"nll":0.8,"checkpoint_path":"artifacts/experiment_i3/preflight/runs/nonexistent/checkpoint","weights_file_sha256":d,"model_state_sha256":d,"training_state_file_sha256":d,"labels":FIXED_LABELS,"reload_evidence":e}; print("nonexistent_checkpoint_accepted="+str(select_dev_checkpoint([c],FIXED_LABELS)["selected_seed"]==42).lower()); print(f"torch_imported={str(chr(116)+chr(111)+chr(114)+chr(99)+chr(104) in sys.modules).lower()}")'
# nonexistent_checkpoint_accepted=true; torch_imported=false

sha256sum -c artifacts/experiment_i3/PREDECLARATION.sha256
# artifacts/experiment_i3/PREDECLARATION.md: OK

sha256sum training/laya_trace_i3/config.json training/laya_trace_i3/experiment.py training/laya_trace_i3/protocol.py training/laya_trace_i3/runner.py training/laya_trace_i3/test_i3.py artifacts/experiment_i3/preflight/data_manifest.v3.json artifacts/trace2decision_i3/verification.json artifacts/trace2decision_i3/provenance.json pipelines/trace2decision/i3_convert.py
# 748b960d...dfb3ae config; 6332086a...adc1 experiment;
# 9f365341...63a1 protocol; 10b65a0e...2450 runner; 05b78756...8303 tests;
# 158b6fa3...5ed2 manifest; 70ff4ca7...7378 verification;
# 9083f386...e854 provenance; 5e80bee8...79f renderer

git rev-parse HEAD
# 2d4408de85587a046c3f14fec1835dd9e69940c9

git diff --check
# passed without output
```

## Remaining limitations

- Actual model compatibility, strict checkpoint reload, CUDA synchronization,
  warm/cold latency, peak VRAM, smoke duration, and cross-process GPU-worker
  behavior remain unmeasured by design.
- Semantic command leakage, near-duplicate task-family leakage, and qualitative
  label correctness remain unknown; no broad independence claim is supported.
- Final-test labels/results, model metrics, seed variation, and comparative added
  value remain unknown and were not inspected.

CHANGES_REQUIRED
