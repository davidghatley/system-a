# Release asset TOCTOU hardening report

Date: 2026-09-24. Scope: the public `release/i3` inference path only. No retraining, GPU workload, historical held-out access, model upload, or push was performed. This report records implementation evidence; it is not a Hugging Face publication-readiness declaration.

## Implementation checkpoint

- Commit: `1439c370bee8a67a6f6d4bcd1d1a9a20b4e1c204`
- Message: `fix: close release asset import and byte races`
- Changed implementation/test files:
  - `release/i3/inference.py`
  - `release/i3/bundle_integrity.py`
  - `release/i3/test_inference.py`
  - `release/i3/MANIFEST.sha256.json`
  - `scripts/test_i3_bundle_integrity.py`

Current source/manifest hashes:

- `a3c51249f6aec1f165aedce30d1e3479bfde03601f1b607a81b593c1740d7f89  release/i3/inference.py`
- `2a8da0839ef7ca732cf22d8423146dd4a00604b5417049e1844239d241254c0c  release/i3/bundle_integrity.py`
- `ce3f28b31d69e2728a628e5313ae75e1b5953e7a807dd9f211dcef6e3ede6a21  release/i3/test_inference.py`
- `d9dcba3ee992b0489695d80522cd0161dbad36c44dd85312aeecd5cb9f99da93  scripts/test_i3_bundle_integrity.py`
- `cf82ed3f137582eb98e149a7ae028cdae8ca8f54f708bb9aabf2b0d1fbeecb1c  release/i3/MANIFEST.sha256.json`

## 1. Root cause

The old constructor performed a point-in-time `verify_tree()` and then reopened security-critical bundle paths:

- `safetensors.torch.load_file(bundle/model/model.safetensors)`;
- `AutoTokenizer.from_pretrained(bundle/tokenizer)`;
- `AutoConfig.from_pretrained(bundle/encoder)` through the bundled Laya `build_model()`;
- pathname reads of `rl_agent_config.json`, `experiment_config.json`, and `calibration.json`.

A file could therefore pass verification as bytes A, be replaced by bytes B, and be consumed by a later pathname-based library. The private Laya source loader already compiled authenticated source bytes, but it did not protect those downstream path consumers.

A fresh independent review also found a second post-verification boundary: quickstart and parity launchers put `release/i3` first on `sys.path`, so a newly planted `transformers.py`/top-level bytecode could be imported after `verify_tree()`.

## 2. Security invariant

For every inference-critical bundle asset, the bytes consumed by inference are the same immutable bytes whose SHA-256 matched the captured manifest snapshot.

The implementation now:

1. captures the manifest and expected digest map before verification;
2. verifies the tree and rechecks the manifest bytes;
3. opens each selected asset through a descriptor-based, no-follow directory walk;
4. reads it once into `bytes`;
5. hashes those exact bytes against the captured digest; and
6. passes those bytes or their parsed in-memory values to the consumer.

No inference-critical consumer receives the original bundle pathname.

## 3. Implementation by asset class

### Immutable-byte authenticated assets

`_read_authenticated_asset()` in `release/i3/inference.py` walks from the filesystem anchor with `O_NOFOLLOW`/`O_DIRECTORY`, requires a regular file, reads once, hashes the resulting immutable buffer, and returns that same buffer. It fails closed on unsupported descriptor-hardening platforms.

The following assets use this primitive:

- `bundle/model/model.safetensors` -> `safetensors.torch.load(bytes)`;
- `bundle/rl_agent_config.json` -> authenticated JSON object;
- `bundle/experiment_config.json` -> authenticated JSON object;
- `bundle/calibration.json` -> authenticated JSON object;
- `bundle/encoder/config.json` -> `AutoConfig.for_model()` and `AutoModel.from_config()`;
- `bundle/tokenizer/tokenizer.json` -> `Tokenizer.from_str()`;
- `bundle/tokenizer/tokenizer_config.json` -> `PreTrainedTokenizerFast(tokenizer_object=...)`.

The tokenizer constructor rejects path-bearing tokenizer configuration fields. The encoder configuration is passed as an in-memory config object; no encoder directory is reopened. The model construction mirrors the pinned Laya `DecisionModel` construction without its path-based `build_model()` call.

### Authenticated private Python source

The existing private namespace/isolation mechanism remains. Its source reader now delegates to the same authenticated-byte primitive. Source bytes are hashed immediately before `compile()` and `exec()`, and previously loaded private source is rehashed before reuse. Relative Laya imports remain process-private and do not alter public `laya` modules.

### Import boundary

`LocalLayaPredictor` initialization and public `predict_pair()` execute inside `_trusted_dependency_imports()`:

- release-root and descendant `sys.path` entries are removed, including empty and byte-valued entries;
- already-loaded modules with an origin inside the release are rejected, except the explicitly local bootstrap/private modules;
- `sys.path` is restored exactly in `finally`, including exceptional exits.

This prevents post-verification `release/i3/transformers.py`, `tokenizers.py`, or bytecode shadowing from satisfying dependency imports. `bundle_integrity.py` additionally rejects regular top-level `.pyc`/`.pyo` files.

### Path-based assets and trust boundary

There are no remaining path-based Transformers, tokenizer, encoder, or safetensors consumers on the `LocalLayaPredictor` path. `verify_tree()` and manifest snapshot handling remain pathname-based control operations, but they are not used as the byte source for inference. Installed Torch/Transformers/tokenizers/safetensors code and the captured manifest are trusted external/pre-start inputs; see limitations below.

## 4. Adversarial regression tests

The release test count is now 17, including the existing import-isolation cases and these new cases:

- post-verification valid replacements/mutations of model weights, all three JSON controls, encoder config, and both tokenizer files: each fails hash authentication before a consumer runs;
- model consumer callback receives the original immutable weight bytes while the pathname is mutated during deserialization; `load_file()` is asserted not to be called;
- tokenizer `from_str()` receives the original serialized bytes while tokenizer files are mutated; path-based tokenizer APIs are not used;
- post-verification symlink substitution is rejected by the descriptor walk;
- manifest mismatch stops before model consumers;
- fresh-process import-shadow regression plants both a top-level source module and a top-level `.pyc`, places the probe directory first on `sys.path`, and proves trusted external modules are selected instead;
- the fresh-process test also preloads a release-root module and proves the origin check rejects it;
- `scripts.test_i3_bundle_integrity` verifies top-level bytecode rejection.

The old path-based implementation would either reopen the mutated pathname, execute the planted dependency shadow, or fail the new byte-consumer assertions.

## 5. Independent adversarial review

### First review

A fresh Space Bunny Free independent reviewer found the post-verification import-shadowing bypass described above. The implementation was repaired before the final review.

### Final review

A second fresh Space Bunny Free independent reviewer returned **READY** for the stated threat model: bundle mutation after initial manifest verification. It reported no concrete in-scope bypass.

Attempted bypasses and outcomes:

| Attempt | Result |
|---|---|
| Plant `transformers.py` after verification | Blocked because release-root paths are absent during imports |
| Plant top-level `transformers.pyc` after verification | Blocked by the same import boundary |
| Plant top-level bytecode before verification | Rejected by `bundle_integrity.py` |
| Preload a module whose origin is inside release | Rejected before imports |
| Use empty/byte/descendant release path entries | Filtered by resolved containment |
| Force an exception inside import boundary | `sys.path` still restored |
| Replace JSON/weights/encoder/tokenizer assets | Hash failure or authenticated original bytes only |
| Replace weights with a symlink | `O_NOFOLLOW` walk rejects it |
| Trick model/tokenizer code into reopening bundle paths | No such call remains |
| Trigger lazy imports during prediction | Prediction remains inside import boundary |
| Reuse mutated private Laya source | Existing private source is rehashed |

The reviewer explicitly scoped its READY result to post-initial-verification bundle mutation and did not treat pre-start replacement of the manifest/bootstrap code as an in-scope bypass.

## 6. Verification

All commands ran from the repository root with CUDA hidden, offline model/transformer settings, and bytecode writes disabled where applicable.

### Full CPU suite

Command:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest training.laya_trace_i3.test_i3 training.laya_trace_i3.test_orchestration scripts.test_i3_dev_diagnostics release.i3.test_inference
```

Result: **93 tests, OK** twice on the final worktree (17.989 s and 17.999 s).

The release/bundle-integrity focused command also passed: **19 tests, OK** (17 release tests plus 2 bundle-integrity tests).

### Commit-derived isolated suite

Command:

```sh
scripts/run_review_cpu_suite.sh 1439c370bee8a67a6f6d4bcd1d1a9a20b4e1c204 commit_1439c37_tocou
```

The exact commit-derived tree passed **93 tests, OK** twice (18.040 s and 18.228 s). Inventory records `object_type=commit` and explicitly omits:

- `artifacts/trace2decision_i3/output/test.jsonl`;
- `artifacts/experiment_i3/preflight/runs/`;
- `release/i3/bundle/model/model.safetensors`;
- `data/cache/`.

Evidence hashes:

- inventory: `7a75218407ec2740b64a32b86686744da6722b219fc5e6a9c2f20813c95d1867`
- run 1: `65076582687fbcceeb11a53152242f452183b40ef3447a81ebcff0b707efc337`
- run 2: `43cce985946ae2c347f5e3f30885594d19282200001196dcc741d6792667ebe1`
- diagnostics log: `19e20dceebdd7e3e6f6dabf507b65bf2efa990aa9b9dbd29cc9867806559fd6b`
- dev audit log: `dc4f05057025c84e001624bbdab43aa2b97d00aecace0ea14f7e61da091efc73`
- missing-weight package-parity log: `31d4df83a761608b88d6c1e2a53e67be93111c2b51dbcd4c886dd302ea707660`
- isolated metrics: `0cfba0dbb9b621319fef37c0731c5fc0093348a003315d4a112672c4398cd563`
- isolated dev audit: `bbc3dc1b09f35d1f887fb39666fd853358bee9d92e1fb8e152defb2032677e0d`

The isolated package-parity command correctly reported unavailable because model weights were intentionally absent; it was not counted as a pass.

### Static/integrity checks

- `py_compile` passed for all changed Python files.
- `release/i3/bundle_integrity.py` reported `bundle integrity verified`.
- Manifest self-check passed for all 20 manifest entries.
- `validate-config` reported `status=config_valid`, `model_loaded=false`, and `torch_imported=false`.
- `git diff --check` passed.
- No model weights, calibration files, frozen evaluation artifacts, or historical data were modified.

## 7. Prediction and parity status

Real CPU package parity was run after the implementation using seven existing dev records only:

- raw maximum absolute difference: `5.066394805908203e-07`;
- calibrated maximum absolute difference: `1.176670855196349e-07`;
- all argmax labels match;
- selected weight SHA-256: `9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947`;
- parity tolerance: `1e-6`;
- parity artifact SHA-256: `1c1491b0caa91106ff20908cb0d68346e613b209cb9f57ef8cc41980e7200a3c`.

The public six-label mapping, raw/calibrated probability semantics, CPU behavior, tokenizer behavior, and model outputs were preserved.

## 8. Remaining limitations and assumptions

1. The manifest present at process initialization is the trust anchor. An attacker able to replace the manifest, `inference.py`, `bundle_integrity.py`, `quickstart.py`, or the process environment **before startup** is outside the reviewed post-verification threat model. Publication tooling should pin the manifest externally.
2. Existing Python bootstrap/bytecode and installed dependency provenance are not all covered by the bundle manifest. Deploy from a trusted, hash-locked environment/source tree.
3. The authenticated reader requires `O_NOFOLLOW` and `O_DIRECTORY` and fails closed where those guarantees are unavailable.
4. Reading the 1.69 GB checkpoint into an immutable byte buffer increases peak memory relative to the old mmap-oriented path.
5. The pinned Laya source still contains an unused path/network-capable `build_model()` fallback. The current release wrapper does not call it; future changes must not reintroduce that call.
6. No real GPU, final-test execution, power-loss injection, multi-host filesystem test, or publication upload was performed. No claim is made here about Hugging Face readiness.

## 9. Git state

- Implementation commit: `1439c370bee8a67a6f6d4bcd1d1a9a20b4e1c204`
- Implementation message: `fix: close release asset import and byte races`
- Branch: `main`
- At report-writing time, `main` was one local commit ahead of `origin/main`.
- No push was performed.
- Unrelated pre-existing modified/untracked worktree files remain untouched and are not part of the implementation commit.
- The follow-up local documentation commit will contain this report and the retained commit-derived inventory/logs; the implementation commit contains only the five implementation/test files listed above.
