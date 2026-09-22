# Executor Review

Date: 2026-09-22

Scope: CPU/offline review of `EXECUTOR_IMPLEMENTATION.md` and
`training/laya_trace_i3/`. No CUDA/model work was run, and no final-test
records were inspected.

## Checks

- Unit suite: `21/21` passed with the documented offline command.
- Unauthorized `actual-model-pilot` invocation: failed closed with
  `ProtocolError: actual-model execution requires explicit independent review authorization`.
- Lock ordering: `GpuLock` rejects an already-imported Torch process; the
  authorized pilot test confirms owner construction and lock acquisition occur
  before Torch/model work.
- Pilot wiring: `run_actual_model_pilot` passes `pilot_updates=2`; the runtime
  performs exactly two optimizer-loop iterations for the pilot and has no CPU
  fallback.
- Listed implementation hashes: config, experiment, runner, tests, and
  manifest match the six recorded values except `protocol.py`.

## Finding

The recorded `protocol.py` hash in `EXECUTOR_IMPLEMENTATION.md` is
`fb52e9e2cc3915c1b3d58a20f9fdc291db97859eb0071dd222344801e9fdc0`, while the
current file hashes to
`fb52e9e2cc3915c1b3d58a20f9fdc29101db97859eb0071dd222344801e9fdc0`.
The implementation checkpoint is therefore not reproducibly bound to the
current implementation. Update the evidence or restore the corresponding
file before authorizing the actual pilot.

CHANGES_REQUIRED
