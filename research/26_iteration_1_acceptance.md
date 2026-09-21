# Iteration 1 Acceptance Checklist

Date frozen: 2026-09-21

This checklist was declared before Iteration 1 implementation and measurement. Existing exp_001 through exp_001c evidence remains historical and is not rewritten by this iteration.

## Shared Rules

- [ ] No paid API use.
- [ ] External model and dataset revisions are pinned and recorded.
- [ ] Commands, configurations, measurements, and important limitations are durable repository artifacts.
- [ ] Claims distinguish verified observations from inference and unknowns.
- [ ] Tests and examples use the native conceptual contract: `state`, `questions`, and `gold` distributions where labels exist.
- [ ] Iteration 2 work does not begin.

## Reflex

- [ ] Minimal Python API and CLI exist.
- [ ] One realistic agent-state example runs against the existing typed-decisions checkpoint after local model availability.
- [ ] Inference is local/offline after model download and surfaces probabilities clearly.
- [ ] Repeated demo output is sufficiently deterministic for sharing.
- [ ] Missing model, malformed input, and inference failures produce actionable errors.
- [ ] README explains usefulness within 60 seconds, quick start, output, non-goals, and checkpoint limitations.
- [ ] Basic automated tests pass.
- [ ] Latency is measured on available hardware with command, warmup/sample count, and device recorded.
- [ ] Fresh-environment instructions are reproducible and require no paid service.

## Local Trainer

- [ ] Loader validates native `state + questions + gold probabilities` records.
- [ ] Conversion produces Laya training items without changing target semantics.
- [ ] Configuration supports gradient checkpointing, microbatching, accumulation, safe FP16, save/resume, VRAM measurement, and loss logging.
- [ ] Fixed train/dev subsets and the primary metric are declared before training results are inspected.
- [ ] Baseline evaluation is recorded before training and identical evaluation runs after training.
- [ ] One bounded single-GPU run completes without OOM and has finite gradients.
- [ ] Intended encoder/head parameters update and intentionally unused parameters remain unchanged.
- [ ] Saved checkpoint reloads and performs inference.
- [ ] Predeclared dev accuracy improves by at least 5 absolute percentage points, or a predeclared alternate metric provides reviewer-approved clear learning evidence.
- [ ] Loss curve, metric, VRAM peak, wall time, configuration, commands, and source commit are recorded.

Predeclared primary metric: exact choice accuracy on a fixed development subset. Predeclared diagnostic if the primary threshold fails: gold-label negative log likelihood, plus a tiny-set overfit test solely to diagnose target/gradient correctness. The diagnostic cannot by itself satisfy acceptance unless an independent reviewer explicitly finds the fixed dev subset too small or noisy.

## Trace2Decision

- [ ] Audit considers at most three candidates and stops when one clearly meets requirements.
- [ ] Selected public dataset has a pinned revision, named teacher, real calls and results, trajectory boundaries, usable provenance/license, and manageable size.
- [ ] Deterministic converter emits one `next_action` choice with a documented compact ontology and one-hot gold distribution.
- [ ] State contains only information available before the target action; leakage checks find zero target-turn inclusion.
- [ ] History truncation, if any, is deterministic and documented.
- [ ] Train/dev/test split is trajectory/task grouped with zero trajectory overlap and obvious duplicates checked.
- [ ] Tool results are genuinely present where expected.
- [ ] Label distribution, state lengths, split integrity, sample records, provenance, transformation, and limitations are reported.
- [ ] Preferred output is at least 500 decisions across 100 trajectories; a smaller honest clean result is allowed and must be reported as such.
- [ ] Output validates against the same native contract consumed by Local Trainer.
- [ ] Converter rerun reproduces content hashes and statistics.

## Review Protocol

Each track receives an implementer review followed by no more than two repair cycles. An independent verifier must reproduce the critical path before acceptance. A track that still misses criteria after two repair cycles is marked BLOCKED rather than having its criteria relaxed.
