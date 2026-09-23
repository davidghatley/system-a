# Iteration 3 package verification and rights audit

## Verified

- Read the current training configuration and interface in `training/laya_trace_i3/config.json`, `experiment.py`, and `protocol.py`.
- Frozen labels are read/search/edit/execute/other_tool/respond_or_finish; model-choice order differs: edit/execute/other_tool/read/respond_or_finish/search.
- Config declares raw softmax, no calibration, max_len 512 and head_max_len 192. Config names upstream model `convaiinnovations/laya` at revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` and upstream source revision `d113dca2512fb3eaca313534bc54c7162d87c1d4`.
- `data/laya/LICENSE` is Apache-2.0; upstream source HEAD is `d113dca2512fb3eaca313534bc54c7162d87c1d4` and `laya/common.py` SHA-256 is `f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2`.
- Hugging Face snapshot README metadata declares `license: apache-2.0`; no separate model-snapshot LICENSE file was found. The metadata declaration is not an independently verified grant for model weights.
- A data-directory license scan found only `data/laya/LICENSE`; no dataset license was located. Dataset rights therefore remain unknown.
- The selected seed-42 checkpoint file (weights only) has SHA-256 `9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947`; training optimizer/state file was deliberately not copied.
- The bundle contains tokenizer and architecture config, frozen experiment config, and the relevant upstream `common.py` implementation. Exact hashes appear below.
- Contract unit tests cover input validation and output probabilities; command: `python -m unittest -v` from `release/i3` (4 passing in the earlier check; rerun after all changes is still required).

## Inferred / implementation notes

- Runtime code is implemented from the training path: `build_sequence`, `build_model`, strict safetensors state loading, single-example collation equivalent, inference-mode forward, raw softmax, and explicit label-order remapping.
- Runtime parity requires exact rendering, tokenizer, truncation, model construction and choice-index mapping. It is not established without an authorized model run.

## Unknown / blockers

- Public redistribution rights for model weights/tokenizer remain unresolved despite upstream README metadata claiming Apache-2.0. Keep this as a local review bundle; do not publish without rights review.
- Dataset license and any applicable dataset restrictions are unknown.
- Parent instructions require coordination before any model execution. Thus no isolated quickstart execution, CPU inference, calibration, parity, or latency check was performed. No held-out test records/results were opened.
- No project `.venv` contains torch/transformers/safetensors (none installed in the checked Python environment); pins in requirements are provisional and offline wheel availability is unknown.
- No verified exploratory evaluation metrics are included because evaluation/test result files were not opened. Existing task report metrics remain to be independently reviewed and reproduced.

## Reproduction

Run `python -m unittest -v` in `release/i3` (CPU, contract-only). To create an isolated environment, install `requirements.txt` from approved local wheels, then run `python quickstart.py` from this directory; that command loads a 1.69GB checkpoint and therefore is gated pending parent authorization. Source revisions/config hashes are recorded in the experiment config. No external sources were downloaded during this task.

## Bundle file SHA-256 manifest

```
model/model.safetensors  9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947
rl_agent_config.json    ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd
experiment_config.json  748b960d470977b250c115ff195f3d4e6dae844474be3cc604cbc14e20dfb3ae
tokenizer/tokenizer.json 6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30
tokenizer/tokenizer_config.json 50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12
encoder/config.json     bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979
upstream/LICENSE        a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9
upstream/laya/common.py f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2
```

## Parent integration verification (2026-09-23; supersedes the earlier pending-execution statements)

**Verified:** the project `.venv` provides Python 3.14, torch `2.11.0+cu128`, transformers `5.17.0`, safetensors `0.8.0`, and NumPy `2.5.3`. With CUDA hidden, `release/i3/quickstart.py` loaded the bundle strictly and produced finite six-class raw probabilities (`artifacts/release_i3/quickstart_cpu.log`). A separate repository-local copy under `artifacts/release_i3/isolated/` ran the documented quickstart with no reference to the original package location (`isolated_quickstart.log`); its output includes both raw and dev-temperature-scaled probabilities. The copy is local/ignored by Git and retained for review.

The reusable CPU parity command was:

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HOME=data/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -B scripts/i3_verify_package.py
```

Its seven **dev-only** records matched saved seed-42 raw probabilities exactly (max absolute difference `0.0`, all argmaxes match) and independently hashed the copied inference weights to `9eaa15bbae116f2e18fd73942d6b76729fdbce4ca40fb5a54d4205e1d8ce8947`. `package_parity.json` contains the row-level evidence. The source code now exposes raw and calibrated distributions from one forward pass; `calibration.json` records temperature `6.203483215500731`, fitted on dev only after grouped-CV diagnostics. This is not a held-out calibration claim.

**Remaining:** public redistribution rights for the training dataset and embedded content are unresolved. Earlier statements above that no model ran or no `.venv` dependencies existed describe the scaffold audit before parent integration, not the final local package state. No held-out test records were opened during package verification. No GPU worker was used; additional GPU-worker wall time remains zero.

**Pinned dataset README correction:** the initial local-only scan found no dataset license file. Parent subsequently fetched the pinned `11-47` README (SHA-256 `c4c15b9a067d478f9a4f23a8078c58755dff04b01af0878e2b55037542dc333b`), which declares **CC BY 4.0** and requires attribution. Embedded third-party rights remain unknown; see `artifacts/release_i3/RIGHTS.md`. This supersedes the earlier statement that the dataset publisher's license itself was unknown.

**Independent-review objection resolved prospectively:** `release/i3/MANIFEST.sha256.json` enumerates every distributed file (including model, tokenizer, config, calibration, source, card and runtime code). `bundle_integrity.verify_tree()` rejects missing/extra/symlinked/hash-mismatched files before model construction. `scripts/test_i3_bundle_integrity.py` verifies valid, corrupted, extra and missing durable fixtures; the full integrated suite passed 60 tests (`integration_tests.log`). A final isolated copy was run after manifest verification; see `isolated_final_verify.log` and `isolated_final_quickstart.log`. This does not resolve rights or provide an external independent calibration estimate.
