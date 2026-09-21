# exp_001b Independent Verification

Date: 2026-09-20

## Verdict

**VERIFIED** for the scoped preprocessing correctness claims. No correctness bug was found. The recorded run itself correctly remains **FAIL** and produced no model-ready split files because `complete_adjacent_linkage` and `label_over_budget_difference_lte_20pp` failed. This verification does not override that disposition.

## Findings

| Target | Result | Evidence |
|---|---|---|
| No silent truncation | Verified | `LayaInputs.build` tokenizes the complete serialized state with `truncation=False`, rejects it before construction if any question lacks room, and asserts the upstream result contains the complete state (`build_dataset.py:52-76`). A fresh bounded test passed at the 474-token room boundary and rejected 475 tokens before calling upstream. The artifact reports maximum retained length 512 and 33,610 explicit over-budget exclusions. |
| Exact Laya fidelity | Verified | The builder imports `laya.common.build_sequence` directly and the fresh test confirmed function identity plus exact IDs/markers for all six questions. Laya checkout HEAD is `d113dca2512fb3eaca313534bc54c7162d87c1d4`; `common.py` is unchanged from HEAD and hashes to the provenance value `f948ee...ff7b2`. Its constructor behavior matches the builder's fit/equality assertions (`common.py:49-86`). |
| Canaries test excluded information | Verified | Every representable pre-dedup candidate was covered: 29,569 tested, equal to 63,179 valid candidates minus 33,610 over-budget exclusions; failures were zero. The code mutates the complete target turn, excluded top-level metadata, and two reasoning-only variants, comparing rendered bytes and all six token/marker tuples (`build_dataset.py:192-225`). The fresh fixture also passed an observable-input positive control. |
| Split isolation | Verified | Independent reconstruction from 28,930 retained-manifest rows found zero trajectory and zero proxy-group crossings. The reconstructed per-split identity/group sets exactly equal `split_manifest.json`; reported rows reconcile as train 20,801, development 2,821, calibration 2,555, test 2,753. |
| Deduplication | Verified | Dedup uses the exact ordered tuple of all six `(IDs, markers)` pairs before retention (`build_dataset.py:353-366`). All 28,930 retained `input_sha256` values are unique; 639 duplicate ledger entries equal the profile count, and every duplicate state hash maps to a retained hash. Retained ordering is sorted by `(trajectory_id, message_index)`. |
| Output/count consistency | Verified | `63,239 - 29 linkage - 31 invalid target = 63,179`; `63,179 - 33,610 over budget = 29,569`; `29,569 - 639 duplicates = 28,930`. Ledger reason counts and all split totals agree with the profile and manifests. |
| Fail-closed | Verified | Status is `FAIL`, `approved:false`; no `model_ready` directory exists. The builder gates directory creation on every stop gate and refuses execution-directory reuse (`build_dataset.py:228-241, 424-426, 493-505`). Diagnostic manifests contain identity/group/label/hash metadata, not token sequences or rendered states. |
| Hash/checksum consistency | Verified | All nine files declared by `checksums.json`, including the 508,262,558-byte source shard, matched recorded byte sizes and SHA-256 values. `status.json` correctly hashes `checksums.json`. Current protocol and builder bytes match run provenance/checksums. Fresh deterministic-gzip fixture comparison passed. |

The two failed gates are evidence of fail-closed operation, not implementation defects: 29 malformed linkage candidates triggered the frozen zero-tolerance gate, and the `read_file` positive/negative over-budget-rate difference was 0.207136, above 0.20 (`profile.json:3699-3707, 3756-3768`).

## Residual Limitations

- Full preprocessing was not rerun, as prohibited. Candidate-wide canary and token-fidelity results therefore rely on checksum-verified run artifacts plus fresh bounded fixture execution.
- The diagnostic retained manifest stores sequence hashes rather than raw sequences. Dedup was independently checked at the hash level and against the implementation; SHA-256 collision risk is not empirically eliminated.
- Full-run cross-execution determinism was not tested. Deterministic serialization/gzip construction, pinned revisions, and internal checksum consistency were verified.
- This does not establish semantic task-family isolation, authenticated execution, model learnability, or successful behavior, consistent with the protocol's claim limits.

## Reproduction Record

Only the user-authorized files, compressed manifests, pinned `common.py`, and Laya checkout metadata were inspected. No model weights, inference, training, official evaluation, or full preprocessing were run.

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME="$PWD/data/cache/huggingface" XDG_CACHE_HOME="$PWD/data/cache" \
  CUDA_VISIBLE_DEVICES="" .venv/bin/python -B \
  experiments/exp_001b_hermes_preprocessing/scripts/test_preprocessing.py

git -C data/laya rev-parse HEAD
git -C data/laya diff --quiet HEAD -- laya/common.py
sha256sum data/laya/laya/common.py
test ! -e artifacts/exp_001b/v0.2.0-run1/model_ready
```

A bounded Python stdlib assertion pass read both gzip JSONL manifests, reconstructed split memberships, counted ledger reasons, checked count equations and retained ordering/uniqueness/lengths, and SHA-256/size-verified every entry in `checksums.json` plus the status-to-checksum link. Its observed summary was: 28,930 retained rows and unique input hashes; 35,900 ledger rows; 639 duplicates; 33,610 over-budget; 29 linkage; 31 invalid-target; 16 malformed-reasoning; 1,575 non-exact-toolset; zero trajectory/proxy crossings; nine checksummed files matched.
