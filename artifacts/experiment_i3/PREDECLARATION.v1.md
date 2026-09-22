# Iteration 3 frozen experiment specification

Date: 2026-09-22. This specification is immutable before any result-bearing
evaluation. The corrected v3 manifest is `preflight/data_manifest.v3.json`.

## Frozen inputs and model

- Source: `11-47/glm-5.2-coding-and-debugging-traces` at
  `1371ed38f8890d0520a53bc7ad850308eb4d7a22`, source SHA-256
  `4e6b11f5b60976368f2a76c252808eb766af2ddcf5958d93f25160e45737d3aa`.
- Renderer: `pipelines/trace2decision/i3_convert.py`, schema
  `trace2decision-i3-v3`, SHA-256
  `5e80bee8711ea7c8dd260f38705791979736ff6ee6488c572028cfabd374c79f`;
  Laya source revision `d113dca2512fb3eaca313534bc54c7162d87c1d4`, tokenizer/base
  revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, max/head lengths 512/192.
- Base model: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`;
  accepted Iteration 2 weights/config/common/notebook pins remain those in
  `training/laya_trace_i3/config.json`.
- Fixed labels/order: read, search, edit, execute, other_tool,
  respond_or_finish. One whole target turn and exactly one `next_action` choice.

## Frozen training and comparison

Seeds are `[42, 314159]`; four epochs; micro-batch 1, accumulation 64, group
size 4; AdamW; encoder/non-encoder LR 2.5e-5/1e-4; weight decay .01; cosine
schedule, eta-min 1e-6; sigmas .4/.3/.2/.1; RLCD weights spherical .75 and RPS
1.0 plus full CE; FP16 scale 64, gradient clipping 1.0, gradient checkpointing;
no OOM retry. A successful measured two-update seed-42 smoke may select seed 42
only when the conservative two-seed projection exceeds 7,200 GPU-seconds;
failed/non-finite/over-budget smoke forbids the main run.

Baselines use the exact rendered state/question interface: training majority,
rendered-state previous-action/transition with declared initial fallback, and
TF-IDF logistic with a small predeclared dev-only budget. The primary comparator
is selected by highest dev macro-F1, then lowest dev NLL, then lexical method
name. Fine-tuning selection is highest dev macro-F1, then lowest NLL, then seed.
Macro-F1 is primary over all six labels (absent classes score zero); accuracy,
support/recall, confusion, clipped raw-softmax NLL, Brier, latency, wall time,
and peak VRAM are required. Paired uncertainty is a group/trajectory bootstrap,
not a row bootstrap. Added value is at least +0.05 absolute dev macro-F1.

The existing test split is previously inspected and **test-once is exploratory**:
one atomic marker is created after the dev-only freeze and before test bytes are
opened; no retry or confirmatory claim is permitted.

## Gate

Substantive GPU/model inference and training are explicitly blocked until
independent data verification and an independent pre-run review both accept the
manifest and protocol. This iteration runs only CPU, offline validation, dry-run,
unit tests, and finite-value checks; no test labels or predictions are read.
