---
language: en
library_name: pytorch
pipeline_tag: text-classification
tags:
  - experimental
  - laya
  - fixed-schema
  - behavioral-cloning
---

# System-A Laya observed-next-action prototype

## What it does

This is a **fixed-schema, six-output next-tool-category classifier** for rendered coding-agent trace states. It accepts a nonempty text `state` and exactly one `next_action` choice question with the six frozen descriptions and instruction from `inference.py` in the documented order. It predicts the *observed* next whole-turn category among `read`, `search`, `edit`, `execute`, `other_tool`, and `respond_or_finish`. It does not select an optimal action, execute a tool, finish a task, accept arbitrary candidate schemas, or guarantee safe behavior. Use for research or advisory analysis with human oversight; do not use as an authorization or production control policy.

System-A fine-tuned the English Laya checkpoint `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982` using source implementation `d113dca2512fb3eaca313534bc54c7162d87c1d4` and a corrected whole-turn derivative of `11-47/glm-5.2-coding-and-debugging-traces@1371ed38f8890d0520a53bc7ad850308eb4d7a22`. The selected training seed was 42. This adaptation is a narrow observed-action specialization, not a new Laya architecture, a proof of RLCD benefit, or evidence for the upstream model's general calibration claims.

## Input and output

The runnable API is `release/i3/inference.py` and the example is `release/i3/quickstart.py`. Input must contain only `state` (text) and `questions` (one `next_action` choice); `gold`, labels, IDs, provenance metadata and arbitrary extra keys are rejected. Criteria insertion order is `edit, execute, other_tool, read, respond_or_finish, search`; returned dictionary order is `read, search, edit, execute, other_tool, respond_or_finish`. The bundle uses Laya's own `build_sequence` with maximum sequence length **512** and question-head length **192**, including its frozen truncation rules; long input may omit earlier context. Validate that critical context is represented before trusting a prediction.

The API exposes **raw** softmax and **dev-temperature-scaled** probabilities from the same six logits. Temperature `6.203483215500731` was fitted on all 140 dev records only after exploratory group-cross-validation; it is not independently validated. Scaling does not change the winning label in the observed dev experiment. Neither output is warranted as a calibrated real-world probability. Two classes, `other_tool` and `respond_or_finish`, have **zero dev and historical final support**, so this model has demonstrated neither capability. No-call completions were excluded from the training estimand.

## Evidence

The corrected data used 1,051 train / 140 dev / 122 historical exploratory test records with 163/20/23 task groups. Primary metric is macro-F1 over all **six** classes, including two absent classes as zero. Values below are dev metrics recomputed from saved predictions (see `artifacts/release_i3/evidence_audit.json` and `DEV_DIAGNOSTICS.md`):

| Dev method | Fixed-six macro-F1 | Accuracy | NLL | Multiclass Brier |
|---|---:|---:|---:|---:|
| TF-IDF logistic baseline | 0.2927 | 0.6071 | 0.9519 | 0.5150 |
| Laya seed 42 raw | 0.4453 | 0.6786 | 3.4056 | 0.5327 |
| Laya seed 314159 raw | 0.4419 | 0.7143 | 3.7481 | 0.5525 |

On 20 dev task groups, five-fold grouped exploratory temperature assessment for **selected seed 42** changed out-of-fold NLL from 3.4056 to 1.0503 and Brier from 0.5327 to 0.5017 with unchanged accuracy and macro-F1. The checkpoint had already been selected on dev, and the final temperature was refit on that same dev, so these are not independent generalization estimates. High-confidence mistakes dominate raw NLL. Improvements in classification do not imply better probability quality or calibration under domain shift.

The one-shot historical final-test attempt was already consumed before this package. It reported selected raw model macro-F1 0.4575 and accuracy 0.6803 on 122 records, but is **exploratory only**: the executed loader did not bind decoded test bytes to the manifest hash, the split had been inspected previously, comparator results were omitted, and its old latency boundary was invalid. No calibrated historical final metrics or test gains over the baseline are available. This package did not reopen test records.

Bundle weight SHA-256: `9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947`. CPU inference on seven existing dev records reproduced their saved raw probabilities exactly; an isolated repository-local copy of the bundle ran the documented quickstart. This small parity check does not establish test or deployment performance. Full commands and evidence are in `artifacts/release_i3/`.

The full local distribution is bound by `MANIFEST.sha256.json`; the loader rejects missing, extra, symlinked or hash-mismatched assets before model construction. On this workstation, 3 CPU warmups and 10 end-to-end batch-one samples use a synthetic input, with validation, tokenization/sequence build, CPU collation, inference-mode forward, both probability conversions and CPU output inside the timer. Cold load is separate. The exact CPU timings and nearest-rank p95 definition are in `artifacts/release_i3/cpu_latency.json`; this is not a GPU/production latency claim.

## Data, provenance and rights

The upstream Laya source includes Apache-2.0 LICENSE; the pinned model snapshot README declares `license: apache-2.0`. The pinned `11-47` dataset README declares **CC BY 4.0**, expressly permitting training and derivative use with attribution. These are publisher declarations; embedded third-party task/code/content rights and source teacher/physical-execution claims are not independently attested. The bundle is a **local review artifact** pending an owner rights decision; no public model-weight upload is authorized yet. Attribute both Laya and **GLM 5.2 Agent Traces** (`11-47/glm-5.2-coding-and-debugging-traces`, pinned revision) and preserve source license/modification notices if published after review. The full source URLs, README hashes, and attribution discrepancy are recorded in `artifacts/release_i3/RIGHTS.md`. Dataset content, training optimizer state, and private logs are excluded from the inference bundle.

## Known failure cases and limits

- Raw probabilities can approach 1.0 on incorrect dev predictions; e.g., confident `read`/`execute` confusion and `edit`→`execute` errors contribute heavily to NLL.
- `respond_or_finish` and `other_tool` lack evaluation support and must not be advertised as validated capabilities.
- Source traces reflect observed recorded tool use, not successful or optimal outcomes; source teacher identity and physical execution are publisher-attributed.
- The input format is tied to a coding-agent harness and one fixed question. Other languages, unrelated domains, changed label descriptions, or arbitrary candidate sets are untested.
- Training-history recovery for seed 42 did not preserve full-run timing, loss curve, or optimizer trajectory; checkpoint bytes and dev inference are verified separately.
