# Trainer B2 Predeclaration Repair 1

Date frozen: 2026-09-21

## Supersession And Execution State

This immutable declaration supersedes only the repaired B2 protocol details in
`artifacts/train_i2/B2_PREDECLARATION.md`. It references rather than rewrites
that original declaration, whose full-file SHA-256 remains
`a9f913ff3d09e00361e26169e620e156390013443fb09da328b2dc7e4f4fa8da`.
All original choices not explicitly changed below remain frozen.

This repair declaration was completed before any B2 model load, CUDA
initialization, inference, baseline evaluation, optimizer step, or training.
The B2 main command has not been run, and no `artifacts/train_i2/b2_run`
directory exists. CPU-only tests and static/data audits were run with CUDA
hidden. Any further protocol change requires another superseding declaration;
this file and its separate checksum must not be edited.

The independent review requiring these repairs is
`artifacts/train_i2/B2_PRE_RUN_REVIEW.md`, SHA-256
`26caa358d0c65eb0069cbb1188817493a6ab129c3e18e58d8983cfec37a5c972`.

## Repaired Frozen Hashes

- Config changed from `ef876a9719d3b835d3ceed2a2e061f67fc05b3866e58ad60330e9b8fdded1377`
  to `a42fd2fd176ea6bc0e68e371be81eeb8be624caa021f096f594e8bfda7689cb4`.
- Runner changed from `67b2a7a96e441b108c6aea291a515044fd24c299c01543d31476774b2c228552`
  to `0cabc5ffdee25c1e5af7c8a980b36dc20897f973eaf490635ef99d1c88660967`.
- Focused B2 tests changed from
  `1b514d1149727f308d502abb67c346e5c08d5cb4e289fbd6a65d8afec5017f92`
  to `3ed831526de44adaecdb2a2fa6711ecb9d9128b6c19f276115a0908693077b22`.
- Preparer is unchanged at
  `7b0609536a8aeb6e62d735b0501c1f58cdf94b6a20b20ad391b446108cfaa677`.
- Train JSONL is unchanged at
  `d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656`.
- Dev JSONL is unchanged at
  `73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e`.
- Subset manifest is unchanged at
  `569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056`.
- Original declaration checksum file is unchanged at
  `2ce52d92c2e80d27740eeb83df24d175bb19a2d9d51983004337b3cbab762836`.

All model, source, notebook, dataset, Parquet, recipe, schedule, objective,
optimizer, memory, checkpoint, and no-retry pins not changed below retain the
values in the original declaration.

## Repair 1: Model State Accounting

The runner now freezes named parameters separately from registered buffers:

```text
encoder named parameters      394781696
non-encoder named parameters   26512131
total named parameters        421293827
temperature buffer elements           3
state_dict elements           421293830
```

`temperature` must be present as a registered three-element buffer and absent
from `named_parameters()`. The optimizer still partitions every named
parameter exactly once, with the unchanged encoder/non-encoder grouping.

## Repair 2: Independent Subset Enforcement

Before importing Torch, the runner independently requires the literal train,
dev, and manifest SHA-256 values recorded above. It does not source expected
output hashes from the manifest. It also verifies the two pinned Parquet hashes,
reconstructs the first 256 train rows and all 100 dev rows directly from those
Parquets, and compares ordered serialized records exactly. This enforces record
IDs and row order as well as question, criterion, probability, and gold-value
content/order. The expected ID arrays are independently constructed as
`tr_agent_trace_observability_000000..000255` and
`agent_trace_observability_000000..000099` and must match both loaded records
and the frozen manifest.

## Repair 3: FP16 Scaling And Step Guard

The CUDA GradScaler initial scale changes from `1.0` to `64.0`, with frozen
`growth_interval=1000`, greater than the complete 80-update run. This is a
hardware adaptation, not the public notebook's default scaler. With accumulation
64, `scale(loss/64)` presents the previously tested scale-1 un-divided loss
magnitude to each FP16 backward while unscale restores the intended average
accumulated gradient. The existing finite objective, gradient, and norm guards
remain. Any overflow-induced optimizer-step skip, detected by a scale decrease,
is now an immediate no-retry failure; scheduler and update counters advance only
after that guard passes.

## Repair 4: Live Checkpoint Reload Verification

The serialized CPU payload comparison remains, but is no longer reported as a
live-state check. After loading, the runner independently reads each live
`optimizer.state_dict()`, `scheduler.state_dict()`, and `scaler.state_dict()`,
clones it to CPU, and requires exact nested equality with its saved state. The
model digest and update/microforward/epoch checks remain unchanged.

## Repair 5: Measured B2 Baseline Gate

`71/200` is retained only as the B1 cross-check. It is not an exact guard for
the B2 FP16 one-sequence evaluator. Before updates, the frozen B2 evaluator must
produce exactly 200 choice results and finite NLL metrics. The final primary
gate is frozen as:

```text
final native-label choice correct >= measured B2 baseline correct + 10
```

Both values are over the same 200 frozen choices. No fixed `81/200` gate
remains. The two-evaluation schedule, no calibration, and all other metric
semantics remain unchanged.

## Frozen Execution Command

The separately authorized command remains:

```text
env TMPDIR=$PWD/data/cache/tmp HF_HOME=$PWD/data/cache/huggingface HF_HUB_CACHE=$PWD/data/cache/huggingface/hub TRANSFORMERS_CACHE=$PWD/data/cache/huggingface/transformers TORCH_HOME=$PWD/data/cache/torch XDG_CACHE_HOME=$PWD/data/cache/xdg .venv/bin/python training/laya_local/b2_train.py
```

The repaired runner requires this superseding declaration and its separate
full-file checksum. This command was not run during repair or freeze.
