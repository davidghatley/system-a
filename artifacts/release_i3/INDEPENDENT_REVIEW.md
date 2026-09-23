# Independent adversarial review — Iteration 3 runner repairs and release bundle

Date: 2026-09-23. Scope limited to independent review of integrated runner repairs, evidence audit/diagnostics, `release/i3`, provenance/rights, freeze/amendment, and available CPU parity evidence. No implementation edits made. No held-out test records or results were opened, and no model inference/training/GPU run was started in this review.

## Verdict

**CONDITIONAL / NOT PUBLIC-RELEASE READY.** The bounded checks I ran passed and I found no reproduced calibration alignment or package label-order defect. The local package’s self-containment and seven-row raw parity are supported by recorded evidence, but not independently repeated here (the parity utility would execute a model). Release remains blocked for public redistribution by unresolved model and dataset rights. There are additional material reproducibility/audit limitations: dev-derived calibration is explicitly exploratory after checkpoint selection; package integrity is recorded in prose/JSON but no enforced whole-bundle checksum gate is evident; and “isolated” copy evidence is local and ignored. Treat as a local review artifact only.

## Verified in this review

- `.gitignore` excludes both `release/i3/bundle/` and `artifacts/release_i3/isolated/`; confirmed via `git check-ignore -v`. This avoids accidentally tracking large weights, but means the release bundle and isolated copy are not present in ordinary tracked source review. Keep checksums/provenance in tracked artifacts and verify them before any transfer.
- Ran `python scripts/i3_evidence_audit.py` from repository root; it exited successfully and wrote/validated its compact audit output. I did not inspect held-out test data/results. The audit report explicitly limits its assertions on reload exactness to reported metadata flags rather than independent live-object proof.
- Ran `python -B scripts/test_i3_dev_diagnostics.py`; synthetic interleaved-fold alignment regression passed.
- Ran `env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s release/i3 -v`; **5/5 contract tests passed**.
- Read `inference.py`: it validates exact fixed schema and criteria order, uses bundled assets via bundle-relative paths, constructs choice logits in frozen order, and remaps to fixed reporting labels. `predict_pair` computes raw and calibrated softmax from the same logits and divides logits by the positive finite temperature. Temperature JSON records dev fit, exploratory status, and raw-output requirement. This supports correct temperature use by inspection, not an independently repeated model run.
- Read the amendment: it states no retroactive test amendment and specifies grouped-CV gates, with calibration only if grouped OOF NLL and Brier improve, argmax remains unchanged, and parity holds. The diagnostics report describes row-index reconstruction and an interleaved-fold regression test, addressing a prior pooled-row alignment bug.
- Read package parity record: it claims exact raw probability parity on seven existing dev rows (`max difference 0.0`) and weight hash. I did not rerun it to honor no model execution. Thus those are **reported evidence**, not reproduced in this review.
- Rights documentation expressly records that public redistribution rights for model weights/tokenizer are unresolved despite upstream model-card metadata, and dataset licensing/restrictions are unknown. Source Apache-2.0 evidence is not a grant for these artifacts. These are blockers, not inferred permissions.

## Concerns / actionable objections

1. **Public distribution blocked (verified documentation; rights unknown).** Keep public upload, repackaging, and weight/tokenizer redistribution blocked until documented rights review establishes permission for model weights, tokenizer, and training dataset/content. An upstream `license: apache-2.0` metadata field and a source-code LICENSE do not establish those rights.
2. **No independent package integrity gate identified.** `PACKAGE_VERIFICATION.md` lists file hashes, and package parity hashes the weights, but I found no verifier that checks every bundle asset (including code/config/calibration/license) against a committed manifest before loading. A modified temperature/config/tokenizer/code could silently change behavior while the weight hash remains unchanged. Add a deterministic full-bundle manifest and a verifier/test that fails closed on missing, extra, or mismatched files before inference; include manifest revision/hash in release documentation.
3. **Calibration is dev-selected and fit after checkpoint selection.** The calibrated choice is explicitly not an independent confirmatory result. Do not present it as deployment-calibrated or as improved generalization; retain raw scores and scaled scores with clear labels and warnings. A new independent holdout and permission are needed for a confirmatory claim.
4. **Evidence/audit limits should remain prominent.** The evidence audit validates dev membership, probability metrics, and hashes, but reload exactness is metadata-reported, and recovery invocation/authorization and storage-resolution history remain unknown. This is appropriately disclosed; do not upgrade those claims to independently proven execution facts.
5. **Provenance for integrated execution remains incomplete.** Reports identify source inputs/revisions but state that dirty training/recovery code revision at execution is unknown. Preserve the existing dirty-source state and command logs; future runs should record a source-tree diff/hash and exact command/environment contemporaneously.
6. **Self-containment evidence is not a clean external handoff.** Report says a repository-local copy ran independently, which is useful evidence, but that copy is git-ignored and the README itself points to evidence elsewhere in the repository. For an actual handoff, retain a tracked package manifest/report and perform verification from a genuinely clean destination with no source-tree/cache dependency. No failure reproduced; this is an evidence-strength objection.

## Inferred / unknown

- **INFERRED:** Fixed-schema label remapping and temperature application are internally consistent based on source inspection; the recorded seven-row exact parity supports but does not prove all-input equivalence.
- **UNKNOWN:** Public rights to model weights, tokenizer, dataset, or embedded content; historical recovery command/authorization; exact dirty source snapshot used in training/recovery; CUDA peak memory; behavior beyond tested dev records; independent calibration/generalization; and whether bundle assets remain byte-identical outside the reported local files.
- **NOT REVIEWED:** Held-out test records/results, by explicit scope. No conclusion about their contents or validity is made here.

## Exact commands and evidence consulted

Commands run from repository root:

```sh
git status --short
find artifacts/release_i3 release -maxdepth 3 -type f | sort
python scripts/i3_evidence_audit.py
python -B scripts/test_i3_dev_diagnostics.py
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s release/i3 -v
git check-ignore -v release/i3/bundle/model/model.safetensors artifacts/release_i3/isolated/bundle/model/model.safetensors
```

Observed test result: package contract suite 5/5 passed; diagnostics alignment regression passed; evidence audit returned successfully. Read `artifacts/release_i3/{EVIDENCE_AUDIT.md,DEV_DIAGNOSTICS.md,PACKAGE_VERIFICATION.md,package_parity.json}`, `research/30_i3_release_protocol_amendment.md`, `release/i3/{inference.py,calibration.json,README.md,test_inference.py}`, and `.gitignore`. No external sources accessed.

## Follow-up review: integrity verifier and frozen distribution (2026-09-23)

This is an append-only follow-up to objection #2 above. Scope: inspect the newly added full-distribution SHA-256 verifier/manifest and loader gate, run only bounded CPU integrity checks, and review freeze/README/rights corrections. No model inference, GPU use, training, or held-out test access occurred.

### Resolution of objection #2

**The concrete missing-integrity-gate objection is resolved for local integrity/tamper detection.** `release/i3/bundle_integrity.py` enumerates the distribution tree, hashes all ordinary files, and rejects unsupported/empty manifests, symlinks, file-set differences, and hash differences. `LocalLayaPredictor.__init__` calls `verify_tree(self.bundle.parent)` before model construction or tokenizer loading. `MANIFEST.sha256.json` contains 20 files and has SHA-256 `9739ec4207218ae55ad2739af324307b2724b985d6eea9894da5c8747ad9cac4`, matching the expected hash pinned in `artifacts/release_i3/RELEASE_FREEZE.md`.

**VERIFIED checks run:**

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B release/i3/bundle_integrity.py
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/test_i3_bundle_integrity.py -v
sha256sum release/i3/MANIFEST.sha256.json
```

Results: verifier printed `bundle integrity verified`; fixture suite passed 1/1, covering matching valid tree and rejection of tampered, extra-file, and missing-file trees; computed manifest hash matched freeze value exactly. The fixture test writes durable fixtures under `artifacts/release_i3/integrity_fixtures/`; these test outputs were retained, not cleaned up. The actual bundle verifier is CPU/file-I/O only and does not import or execute the model.

The manifest excludes itself, `__pycache__`, and `.pyc/.pyo` files. That is reasonable for generated bytecode, and the tracked freeze pins the manifest hash. **Residual limitation, not a reproduced blocker:** the manifest is an integrity mechanism, not a cryptographic signature or trust root if an attacker can replace both manifest and freeze/report. Normal distribution review must independently compare the manifest hash to the freeze record; this mechanism is not authenticated against compromise of the repository/report itself.

### Documentation, rights, and remaining release status

- README now describes the manifest and loader’s fail-closed missing/extra/symlink/hash-mismatch behavior. Freeze identifies the exact expected manifest hash and documents the final isolated copy; this substantially improves handoff evidence. Existing `isolated_rights_final` remains ignored by Git, which is acceptable as local evidence because the tracked freeze and manifest provide the reproducibility anchor; I did not rerun its quickstart.
- Model card now reports dataset publisher’s pinned README declaration of **CC BY 4.0**, attributes both Laya and GLM 5.2 Agent Traces, distinguishes publisher declaration from independently attested rights, and retains the embedded third-party-content caveat. `RIGHTS.md` gives pinned revisions and README hashes, but explicitly notes the dataset publisher’s authority over embedded content is not independently attested and no System-A repository-level code license is identified. Thus the earlier blanket “dataset license unknown” is corrected as to the publisher-declared dataset license; rights scope/authority and System-A code licensing remain unresolved.
- Public upload remains blocked by owner rights decision: model snapshot supplies metadata declaration rather than a separate weight/tokenizer license; dataset CC BY 4.0 declaration does not itself settle embedded third-party rights; and the new System-A adaptation lacks an identified public license. This is a public-rights/legal-review blocker, not an implementation defect found in this follow-up.
- Calibration remains dev-only and exploratory after checkpoint selection, as documented; no evidence here upgrades it to a held-out calibration claim.

### Follow-up verdict

**Integrity implementation: PASS for the reviewed local distribution and tested failure modes. No remaining concrete implementation blocker was reproduced in this narrow follow-up. Overall verdict remains CONDITIONAL / NOT PUBLIC-RELEASE READY**, because owner rights review and licensing decisions remain outstanding (along with previously documented provenance/history unknowns). No held-out results were read or relied upon.

## Final targeted addendum: frozen input contract and final copy (2026-09-23)

### Verified checks

- Recomputed `sha256sum release/i3/MANIFEST.sha256.json`: `51dd67467f195c4c1ef1b3e436146cbbc9081ad5f758fd263c053f773d3b26ee`, exact match to `artifacts/release_i3/RELEASE_FREEZE.md` expected value.
- Ran `env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B release/i3/bundle_integrity.py`: **PASS**, `bundle integrity verified`.
- Ran `env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s release/i3 -v`: **5/5 PASS**.
- Inspected `validate_input`, `FROZEN_INSTRUCTIONS`, `FROZEN_CRITERIA`, `quickstart.py`, and `examples.json`. Validation now requires exact criterion key order, exact descriptions, and exact instruction text; quickstart constructs the same fixed contract from those constants. The examples provide state-only examples intended to be supplied to that fixed question. No concrete mismatch found.
- Read `isolated_contract_final_verify.log` (`bundle integrity verified`) and `isolated_contract_final_quickstart.log` (both finite six-label raw and dev-scaled outputs present). This is **reported recorded execution**, not rerun in this follow-up, avoiding a model run; it supports that the final copied package passed verification and quickstart. Its input is the synthetic quickstart sample, not held-out data.
- Ran `env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m scripts.test_i3_seed_cli_failure -v`: **1/1 PASS**. The composed simulation completes seed 42, injects a synthetic failure before seed 314159 publication, then asserts seed-42 result persists and no seed-314159 result is published.
- Ran `env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3.ProtocolTest.test_final_test_once_rejects_synthetic_test_byte_mismatch_before_decode training.laya_trace_i3.test_i3.ProtocolTest.test_split_loader_rejects_hash_mismatch_before_decode -v`: **2/2 PASS**. These exercise synthetic mismatch bytes and assert validation fails before decoding; they do not access actual held-out records/results.

Two initial test invocations failed due to my invocation errors only: running the script as a filesystem path omitted repository root from `sys.path` (`ModuleNotFoundError: training`), and I used an incorrect test class name (`TestProtocol`/`ProtocolTests` rather than `ProtocolTest`). Corrected module-based invocations above passed; no product failure inferred from those initial errors.

### Final targeted verdict / concrete objections

No concrete implementation objection found in this targeted review. Exact frozen question validation is consistently used by quickstart, the regenerated manifest matches the freeze pin, CPU integrity and contract tests pass, and the inspected isolated final logs show verification plus synthetic quickstart output. Evidence limitation: final isolated quickstart was inspected from retained logs, not repeated here. No model execution or held-out test-data/result access occurred.

Overall verdict remains **CONDITIONAL / NOT PUBLIC-RELEASE READY** for the previously recorded public-rights/licensing questions (model-weight/tokenizer grant, embedded dataset-content scope/authority, and System-A code license) and previously disclosed provenance/history unknowns; these are not implementation defects found in this pass.
