# Trainer B2 Predeclaration

Date frozen: 2026-09-21

## Chronology And Execution State

This file was written before any B2 model load, CUDA initialization, inference,
baseline evaluation, or training. Before this freeze, the pinned notebook and
research records were inspected, the deterministic JSONL subsets were generated
from the existing Parquets, and CPU-only objective/config/subset tests were run
with `CUDA_VISIBLE_DEVICES=''`. No model code path was executed. The B2 main
command has not been run.

After this Markdown file is complete, its ordinary full-file SHA-256 is recorded
in the separate `artifacts/train_i2/B2_PREDECLARATION.sha256`. The Markdown does
not contain a self-referential hash.

## Immutable Pins And Hashes

- Laya source: `NandhaKishorM/laya@d113dca2512fb3eaca313534bc54c7162d87c1d4`
- `data/laya/laya/common.py`: `f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2`
- Public notebook: `data/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
- Public notebook SHA-256: `2c37036054dfb92073b518e994cb66c8a9009fb8d15b060d9bdfc6d4f94b1d39`
- Base model: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`
- Base `model.safetensors` SHA-256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
- Base `rl_agent_config.json` SHA-256: `ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd`
- Dataset: `LocalLLaMA/typed-decisions@ea9306458d6e9563628369a3d1e72e362fb381d2`
- Dataset config: `agent_trace_observability`
- Existing train Parquet SHA-256: `e7b2f78fe539517c6a66a90c68ed671c6133194a88d27dbff5d84713af46b901`
- Existing test Parquet SHA-256: `833ad1364237b9a3fe05be2c62e632f0cb34ba372284b975140d362ab246f70d`
- Frozen config SHA-256: `ef876a9719d3b835d3ceed2a2e061f67fc05b3866e58ad60330e9b8fdded1377`
- Frozen runner SHA-256: `67b2a7a96e441b108c6aea291a515044fd24c299c01543d31476774b2c228552`
- Frozen preparer SHA-256: `7b0609536a8aeb6e62d735b0501c1f58cdf94b6a20b20ad391b446108cfaa677`
- Focused B2 tests SHA-256: `1b514d1149727f308d502abb67c346e5c08d5cb4e289fbd6a65d8afec5017f92`

## Immutable Subsets

Selection is by physical Parquet source order without shuffling or filtering.
JSON object insertion order is preserved, including question and option order.

- Train: first 256 rows of `artifacts/train_i1/source_train.parquet`, IDs
  `tr_agent_trace_observability_000000` through
  `tr_agent_trace_observability_000255`, 1,280 questions total: 512 choice,
  512 score, and 256 noul.
- Dev: all 100 rows of `artifacts/train_i1/source_test.parquet`, IDs
  `agent_trace_observability_000000` through
  `agent_trace_observability_000099`, 500 questions total: 200 choice,
  200 score, and 100 noul.
- Train JSONL SHA-256: `d1f7c06ba773d53746d390dba66ab0b18bdf5e7b174325f75c901cbd732f5656`
- Dev JSONL SHA-256: `73a82a45ca517d843444a9f01887c6ecc11b1d0a9aa2e8ba6cb095d3fdc66f3e`
- Manifest SHA-256: `569d3f280bdc0968ea6948cd96dde8a70fa1b39ccd1f303f2bed7bfb1776f056`

The exact IDs and split metadata are in
`artifacts/train_i2/b2_subset/manifest.json`. Train and dev come from the
dataset's public train and test splits respectively; there is no row overlap.

## Frozen Construction And Schedule

- Seed: 42 for Python, NumPy, PyTorch, and all CUDA generators.
- Sequence construction: public notebook path, `max_len=512`,
  `head_max_len=192`, and `truncate_left=False` (default right truncation).
- Four epochs; epoch shuffle uses the notebook schedule `42 + epoch` and mutates
  the prior epoch's order in place.
- One question item sequence per microforward, accumulation 64, 1,280
  microforwards per epoch, 20 complete updates per epoch, exactly 5,120
  microforwards and 80 optimizer updates total. There are no partial updates.
- RLCD group size `G=4`; sigma by epoch is exactly `0.4, 0.3, 0.2, 0.1`.
- FP16 autocast and CUDA GradScaler with initial scale 1.0.
- Encoder gradient checkpointing with `use_reentrant=False`. The ineffective
  public `head_checkpointing` flag is not claimed or used.

## Frozen Objective

For each item, the implementation transcribes notebook cell 5 exactly:

1. Draw four masked Gaussian logit perturbations and project each to zero mean
   across valid options: `eps=(eps-sum(eps)/K)*mask`.
2. Set `z=stop_gradient(logits)+eps` and `q=softmax(mask(z))`.
3. Use pinned public `proper_reward`: target-weighted log score floored at
   `-9.21`, plus spherical score weighted `0.75`, and for `score` questions
   subtract RPS weighted `1.0` and normalized by `K-1`.
4. Center reward over `G` for each item, then divide by the single default
   PyTorch standard deviation over the complete group/item tensor plus `1e-6`.
5. Use Gaussian policy term
   `-mean(advantage * (-sum((z-logits)^2*mask)/(2*sigma^2)))` plus soft-target
   cross-entropy at weight 1.0.
6. Preserve notebook action behavior with `+ 0.0 * act.sum()`.

The optimizer groups preserve the notebook behavior: every named parameter
containing `encoder.` uses encoder LR `2.5e-5`; every other named parameter,
including `act_head`, uses non-encoder LR `1e-4`. All parameters remain
trainable. The action head receives zero task gradient, but its zero gradient is
present, so AdamW weight decay applies. AdamW weight decay is `0.01`; global
gradient clipping is `1.0`. CosineAnnealingLR has `T_max=80` and
`eta_min=1e-6` and steps after every optimizer update.

## Evaluation And Decision

There is no calibration fitting or application; metrics use raw-logit softmax.
The only evaluations are one baseline before training and one final evaluation
after saving and exactly reloading the checkpoint. No development evaluation
occurs during training, and there is no metric-based early stopping.

- Primary: native-label choice accuracy over all 200 frozen dev choices.
- Frozen baseline guard: the pre-training base must reproduce exactly `71/200`.
  If not, training does not start.
- Pass: final native-label choice accuracy must be at least `81/200`, an
  improvement of at least 10 correct choices or 5 absolute percentage points.
- Diagnostics: native hard correct/total/accuracy for choice, score, noul, all
  500 questions, and workflow; mean native-gold-label NLL; and mean soft-target
  cross-entropy are reported for baseline and final.

## Failure And Stopping Guards

This is one bounded main run with no retries, fallback profiles, or CPU fallback.
It fails immediately on a pin, full-file hash, subset, row/item count, option
marker, parameter count/group, baseline, or predeclaration mismatch; unavailable
CUDA; any non-finite loss, reward, advantage, gradient, or gradient norm; any
missing parameter gradient; peak CUDA reserved memory above 11.5 GiB; any count
other than 5,120 microforwards and 80 updates; any non-exact model or complete
optimizer/scheduler/scaler checkpoint reload; or final primary result below
`81/200`. It never reduces context, changes batching, retries an OOM, or falls
back to CPU. The checkpoint must strictly reload and match the pre-save model
state digest byte-for-byte before the sole final evaluation.

The frozen command for a separately authorized execution is:

```text
env TMPDIR=$PWD/data/cache/tmp HF_HOME=$PWD/data/cache/huggingface HF_HUB_CACHE=$PWD/data/cache/huggingface/hub TRANSFORMERS_CACHE=$PWD/data/cache/huggingface/transformers TORCH_HOME=$PWD/data/cache/torch XDG_CACHE_HOME=$PWD/data/cache/xdg .venv/bin/python training/laya_local/b2_train.py
```

This command was not run during implementation and freeze.
