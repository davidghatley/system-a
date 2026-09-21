# Final Verification: Local Trainer Iteration 1

Date: 2026-09-21

## Final status

**BLOCKED.** The implementation and evidence path are substantially verified, but the frozen primary learning criterion fails. Native-label exact choice accuracy changed from `19/40 = 0.475` to `11/40 = 0.275`, an absolute change of `-0.200`, versus the required `+0.050`.

No training update was launched by this verification, and no frozen data or prior run artifact was changed.

## Exact independent checks

All commands were run from the repository root with `PYTHONDONTWRITEBYTECODE=1` where applicable.

1. **Unit tests — VERIFIED:**
   `.venv/bin/python -m unittest discover -s training/laya_local/tests -v`
   Result: 4 tests passed.
2. **Compilation — VERIFIED:**
   `.venv/bin/python -m py_compile shared/typed_decisions/schema.py training/laya_local/data.py training/laya_local/metrics.py training/laya_local/prepare_subset.py training/laya_local/train.py training/laya_local/resume.py training/laya_local/tiny_overfit.py`
   Result: successful with no output.
3. **Checksums — VERIFIED:** independently recomputed SHA-256 values for both pinned Parquet files, both frozen JSONL subsets, the preserved model safetensors, and the preserved training state. All six values exactly match `CHECKSUMS.sha256`.
4. **Native metric reconstruction — VERIFIED:** independently parsed `subsets/dev.jsonl`, used each native `gold[qid].label` and question criterion order, joined predictions by `(record_id, question_id)`, and obtained baseline `19/40` and post-training `11/40`. The tied `agent_trace_observability_000003/action` target is native index `1` (`human_review`), while probability argmax is index `0`.
5. **Fixed subset — VERIFIED:** manifest declares the first 64 pinned train rows and first 20 pinned test rows before baseline; source/output hashes match and train/dev IDs are distinct.
6. **Checkpoint state inspection — VERIFIED with limitation:** the preserved run checkpoint has model weights, optimizer/scaler state, and step `32`; `run/result.json` records exact prediction equality after checkpoint reload. The preserved run checkpoint itself lacks cursor/RNG fields. The separate `resume_smoke/checkpoint` contains those fields and its recorded smoke result is independently consistent with a one-step continuation.

## Technical deliverables

- Native schema validation and conversion to Laya items: **VERIFIED** by source inspection, tests, and recorded run counts (320 train and 100 dev items).
- Gradient checkpointing, microbatching/accumulation, FP16 scaler, clipping, logging, VRAM measurement, and checkpoint save: **VERIFIED** by final code and run evidence.
- Bounded run completion without OOM and finite gradients: **VERIFIED**; 32 updates, 64 microbatches, finite gradients, peak reserved memory `5.508 GiB` on the recorded 12 GiB RTX 3060.
- Intended parameter updates: **VERIFIED** by recorded hashes: encoder and scorer changed; action head hash remained `baad5de6...490d2d` before and after.
- Checkpoint reload and inference: **VERIFIED** by `optimizer_state_loaded`, `scaler_state_loaded`, and `reload_predictions_exact` in `run/result.json`.
- Resume continuation: **VERIFIED for the bounded smoke** by `resume_smoke/result.json`: step `32 -> 33`, cursor `64 -> 65`, finite gradients, intended parameter changed, action head unchanged. The smoke checkpoint was derived with legacy cursor/RNG fields supplied; this is a residual provenance limitation for the original run checkpoint.
- Pinned revisions, commands, configuration, loss curve, metrics, timing, VRAM, and hashes: **VERIFIED** as durable artifacts. Revisions are dataset `ea9306458d6e9563628369a3d1e72e362fb381d2`, Laya source `d113dca2512fb3eaca313534bc54c7162d87c1d4`, and model `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.

## Learning status

- Frozen primary metric: **FAILED** (`-0.200` absolute change; no acceptance threshold met).
- Gold-distribution NLL: **VERIFIED diagnostic improvement only**, `1.4951586 -> 1.2102497` (`-0.2849089`). The frozen checklist explicitly does not permit this diagnostic, or the tiny overfit result, to replace the failed exact-choice criterion without independent reviewer approval; no such approval is evidenced.
- Tiny overfit: **VERIFIED diagnostic only**; it is not acceptance evidence.
- Therefore no learning acceptance pass is justified, despite the optimization path being live.

## Residual risks and unknowns

- **INFERRED:** NLL improvement with choice-accuracy regression may reflect optimization toward soft/noisy targets on a short run; one seed does not establish the cause.
- **UNKNOWN:** Whether a larger predeclared budget, alternate optimizer/sampler, or additional seeds would improve the fixed-dev primary metric; these were not run.
- The recorded timing is a bounded single-run measurement, not a generalized performance claim. Nondeterministic CUDA backward kernels limit bitwise replay claims.
- No evidence was found that the fixed subset is sufficiently small/noisy to invoke the checklist exception.

## Change-scope verification

The pre-verification `git status --short` showed the repository's existing unrelated modified/untracked paths. Verification commands did not alter tracked implementation or prior artifacts. The only file written by this verifier is `artifacts/train_i1/VERIFICATION.md`; no commit was made.
