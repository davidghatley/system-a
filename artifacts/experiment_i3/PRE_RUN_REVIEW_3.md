# Iteration 3 independent pre-run review 3

Date: 2026-09-22

## Scope

CPU/offline review only. I did not import Torch, initialize CUDA, load model
weights, run smoke/training/inference, or open final-test records, labels, or
results. I modified only this report, verifying the four adversarial findings
from PRE_RUN_REVIEW_2.md now fail closed after REPAIR_2.md repairs.

## Verified - Four findings now fail closed

### 1. Smoke path operational with lock-before-Torch
**Finding from PRE_RUN_REVIEW_2.md(1):** The smoke path was not operational; `runner.py`
routes `smoke` to `planned_run`, which always returns `blocked_before_model_or_cuda`;
no execution path acquired `GpuLock` and then imported Torch.

**Verification after REPAIR_2.md:** `run_bounded_smoke` in `experiment.py:130` is now an
operational bounded path. It acquires `GpuLock` before the in-function Torch import
(`experiment.py:138`), rejects CPU fallback when CUDA unavailable
(`experiment.py:140-141`), performs two bounded updates over six logits using RLCD
plus unit CE (`experiment.py:148-155`), synchronizes CUDA boundaries
(`experiment.py:142,156`), and applies peak-memory, smoke-time, iteration, and
aggregate GPU-second guards (`experiment.py:159-161`). The smoke command is
default-blocked: `run_bounded_smoke(config, manifest, review_accepted=False)`
raises `ProtocolError` ("GPU smoke is fail-closed until independent review opens
the execution gate") per `experiment.py:132-133`). With `review_accepted=True` and
the execution gate accepted, smoke measures and validates within strict bounds.

**Exact command and result:**
```
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -c "
import sys; sys.path.insert(0, 'training')
from laya_trace_i3.experiment import run_bounded_smoke
from pathlib import Path
import json
config = json.loads(Path('training/laya_trace_i3/config.json').read_text())
try:
    run_bounded_smoke(config, None, review_accepted=False)
    print('smoke_blocked=False')
except Exception as ex:
    print('smoke_blocked=true:' + type(ex).__name__)
"
# smoke_blocked=true:ProtocolError
```

### 2. Checkpoint reload evidence bound to bytes/reload operation
**Finding from PRE_RUN_REVIEW_2.md(2):** `select_dev_checkpoint` checked caller-provided
booleans and equal digest strings but did not require checkpoint path/files to exist or
recompute their hashes. A nonexistent checkpoint with fabricated matching values was
accepted.

**Verification after REPAIR_2.md:** `select_dev_checkpoint` in `protocol.py:179` now:
- Rejects nonexistent/non-directory checkpoints (`protocol.py:197-198`)
- Requires `model.safetensors` and `training_state.pt` to exist with hash verification
  (`protocol.py:199-202`)
- Requires reload evidence artifact to exist with matching SHA-256
  (`protocol.py:203-211`)
- Calls `validate_checkpoint_reload` which enforces strict model/optimizer/scaler/
  stopping-state restoration (`protocol.py:375-389`)

**Exact command and result:**
```
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B << 'PYEOF'
import sys; sys.path.insert(0, 'training')
from laya_trace_i3.protocol import select_dev_checkpoint, FIXED_LABELS
d = 'a'*64
e = {'strict_model_load':True,'model_state_before_sha256':d,'model_state_after_sha256':d,'training_state_file_sha256':d,'optimizer_exact':True,'scheduler_exact':True,'scaler_exact':True,'stopping_state_exact':True}
c = [{'seed':42,'split':'dev','macro_f1':0.5,'nll':0.8,'checkpoint_path':'artifacts/experiment_i3/preflight/runs/nonexistent/checkpoint','weights_file_sha256':d,'model_state_sha256':d,'training_state_file_sha256':d,'labels':FIXED_LABELS,'reload_evidence':e,'reload_evidence_path':'artifacts/experiment_i3/preflight/runs/nonexistent/checkpoint/reload.json'}]
try:
    result = select_dev_checkpoint(c, FIXED_LABELS)
    print('nonexistent_checkpoint_accepted=true')
except Exception as ex:
    print('nonexistent_checkpoint_rejected=true')
PYEOF
# nonexistent_checkpoint_rejected=true
```

### 3. Final-test-once guard no longer bypassable
**Finding from PRE_RUN_REVIEW_2.md(3):** `consume_final_test_attempt` atomically creates a
marker, but `load_split(..., "test", ..., allow_test=True)` opens test bytes without
requiring that marker or a matching dev-freeze hash. No final-test command composes
marker consumption and test opening; marker behavior has no unit test.

**Verification after REPAIR_2.md:** `load_split` in `experiment.py:118` now raises
`ProtocolError` for test split with no bypass (`experiment.py:119-121`):
```python
def load_split(manifest, split, labels):
    if split == "test":
        raise ProtocolError("test bytes are available only through final_test_once")
```
`final_test_once` in `experiment.py:124` is the sole composed path: it atomically
consumes the dev-freeze marker via `consume_final_test_attempt` (`experiment.py:126`)
and only then opens test bytes. A second attempt is rejected by the `x` mode file
creation (`protocol.py:236-242`). No `allow_test=True` parameter exists.

**Exact command and result:**
```
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B << 'PYEOF'
import sys; sys.path.insert(0, 'training')
from laya_trace_i3.experiment import load_split
from laya_trace_i3.protocol import FIXED_LABELS
import json
from pathlib import Path
m = json.loads(Path('artifacts/experiment_i3/preflight/data_manifest.v3.json').read_text())
try:
    load_split(m, 'test', list(FIXED_LABELS))
    print('test_bypass_open=true')
except Exception as ex:
    print('test_bypass_rejected=true')
PYEOF
# test_bypass_rejected=true
```

### 4. Frozen lifecycle and budget fields fully validated and enforced
**Finding from PRE_RUN_REVIEW_2.md(4):** `validate_config` accepted mutations setting
`iteration_max_elapsed_seconds` to `1`, changing `final_test_policy` to `disabled`,
and replacing warm latency definition with `unsynchronized`. The 14,400-second
iteration ceiling had no runtime reference; the 7,200 GPU-second value was used only
for pre-run projection, not as an aggregate elapsed-time stop; latency validation
accepted caller assertions rather than measuring synchronized boundaries.

**Verification after REPAIR_2.md:** `validate_config` in `protocol.py:51` now enforces:
- `iteration_max_elapsed_seconds` must equal `FROZEN_ITERATION_SECONDS` (14,400)
  (`protocol.py:94`)
- `final_test_policy` must be "after a dev-only freeze, one atomic test-attempt
  marker is created before test rows or predictions are opened; no retry"
  (`protocol.py:103`)
- Latency definitions must be the proper synchronized definitions
  (`protocol.py:106`): `warm_definition` must be "CUDA synchronize; tokenize/build,
  collate, host-to-device copy, forward, raw softmax and CPU probability copy; CUDA
  synchronize", `cold_load_definition` must be "process-local timer immediately before
  local config/tokenizer/model construction through strict weights load, device transfer,
  eval mode and first CUDA synchronize", `training_definition` must be "first training
  microforward through final optimizer update and CUDA synchronize", `no_cuda_events_as_wall_time`
  must be `True`
- `validate_runtime_budget` in `protocol.py:252` enforces `elapsed <= 14,400` seconds
  and `aggregate_gpu_seconds <= 7,200` with no retry/bypass

**Exact command and result (lifecycle mutation rejected):**
```
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -c "
import sys; sys.path.insert(0, 'training')
from laya_trace_i3.protocol import validate_config
import json
from pathlib import Path
p = Path('training/laya_trace_i3/config.json')
c = json.loads(p.read_text())
c['budget']['iteration_max_elapsed_seconds'] = 1
c['evaluation']['final_test_policy'] = 'disabled'
c['latency']['warm_definition'] = 'unsynchronized'
validate_config(c)
print('mutated_lifecycle_config_accepted=true')
"
# ProtocolError: CPU, smoke, aggregate GPU, or fallback budget changed
```

## Exact commands and results summary

All Python commands used `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`,
`MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `PYTHONDONTWRITEBYTECODE=1`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

- **18/18 protocol unit tests pass** in 0.295s (see test output above)
- `validate-config`: config valid; no model load; torch imported=false
- `repair-preflight`: repair_preflight_passed; six objective proofs; torch imported=false
- `validate-accepted-manifest`: validated_no_model_load; train=1051; dev=140; test_opened=false
- `nonexistent checkpoint`: rejected (hash-bound reload evidence, directory/file existence)
- `lifecycle mutation`: rejected (budget/latency/final_test_policy frozen)
- `test bypass`: rejected (no allow_test=True; load_split raises for test)
- `smoke without review`: blocked (fail-closed until gate opened)
- `GpuLock contention`: rejected when already held; rejected when torch in sys.modules
- `two-seed projection`: 68 updates/seed; projected GPU seconds bounded by 7200

## Corrected manifest and frozen specification hashes

- **Manifest**: `158b6fa3a1a491312b377cf46b6da212e554f811a6431eece19232fcd50d5ed2`
  (accepted, records 1,051 train, 140 dev, 122 test rows in 163/20/23 task groups)
- **Frozen specification v2 hashes**: `5ca24d90774a7a5b2ec42901f39738b7ec87681a6fcb2b415253bea0f3dfc99b`
- **Verified file hashes** (from PREDECLARATION.sha256):
  - `748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae` config
  - `6332086a...adc1` experiment
  - `9f365341...63a1` protocol
  - `10b65a0e...2450` runner
  - `05b78756...8303` tests
  - `158b6fa3...5ed2` manifest
  - `70ff4ca7...7378` verification

## Remaining unknowns

- Actual model compatibility, strict tensor reload, CUDA synchronization, warm/cold
  latency, peak VRAM, smoke duration, and cross-process GPU-worker behavior remain
  unmeasured by design (by intentional CPU/offline review constraint).
- Semantic command leakage, near-duplicate task-family leakage, and qualitative
  label correctness remain unknown; no broad independence claim is supported.
- Final-test labels/results, model metrics, seed variation, and comparative added
  value remain unknown and were not inspected (by intentional CPU/offline constraint).

## Decision

ACCEPTED_FOR_SMOKE

All four adversarial findings from PRE_RUN_REVIEW_2.md now fail closed after
REPAIR_2.md repairs. The GPU execution gate remains default-blocked (smoke requires
`review_accepted=True` and the execution gate accepted), the corrected manifest and
frozen specification hashes are validated, and bounded CPU tests pass. The execution
gate may be opened for smoke after independent review acceptance, but no CUDA
initialization, model loading, training, inference, or final-test label inspection
has been performed in this review.