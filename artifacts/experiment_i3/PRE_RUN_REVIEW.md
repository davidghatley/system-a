# i3 pre-run review

**Decision: CHANGES_REQUIRED.** No model, CUDA, smoke, or test labels/results
were inspected; test was hash-checked only.

## VERIFIED

- Frozen spec hash passes: `63e46b...371ac`; source is `1371ed3...4d7a22`,
  renderer `5e80...4c79f`, and model is pinned base `laya@1c5edc...c982`.
- Accepted manifest currently hashes to `158b6f...d5ed2`; train/dev hashes and
  counts 1051/140, and test hash is `3a22...f6c8` (manifest says 122).
  CPU checks (10/10 unit tests, config, accepted-manifest, `git diff --check`)
  passed; Torch/model/test-opened remained false.
- Six labels and one `next_action` question are fixed. Retained targets are
  explicit one-hot distributions; renderer uses strict-prefix state and excludes
  mixed categories. Base/fine-tuned pins and data interface match.
- AdamW, LRs, cosine schedule, four epochs, seeds, accumulation, RLCD+CE
  weights, FP16, clipping, 2-update smoke, 7200 GPU-second/14400-second guards,
  nonblocking lock, dev selection, reload hashes, and atomic final-test marker
  are declared. Metrics define macro-F1, NLL, Brier, confusion, support/recall,
  and latency.

## CHANGES_REQUIRED / UNKNOWN

1. Resolve the count/hash chronology conflict: review reports
   1184/156/139, while manifest/files are 1051/140/122; refresh manifest,
   verification, checksums, provenance, and predeclaration evidence together.
2. Review P0 remains: no-call completion has no positive finish
   marker. Either prove one source marker with an audit ledger or retain the
   current exclusion and explicitly narrow the estimand to completed tool-bearing
   turns; obtain independent acceptance.
3. Prove RLCD's six-class action/reward mapping and nonzero objective
   (config declares `0.0 * act.sum()`), plus group-bootstrap IDs and runtime
   reload/lock/latency guards. Semantic/family leakage remains UNKNOWN. No smoke
   until reverified.

Evidence: `REVIEW.md`, `verification.json`, manifest, predeclaration hash.
Commands: offline unittest, config/manifest validation, `sha256sum`, `wc -l`,
`git diff --check`.
