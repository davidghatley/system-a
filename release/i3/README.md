# Iteration 3 local inference package

This is a review-ready **local** fixed-schema Laya next-action classifier. Public upload remains subject to the rights review in `MODEL_CARD.md`. The bundle has been loaded and checked using CPU inference.

```text
release/i3/
  inference.py, bundle_integrity.py, quickstart.py, requirements.txt, MODEL_CARD.md
  MANIFEST.sha256.json         # full distribution checksums
  bundle/
    model/model.safetensors      # selected seed-42 inference weights only
    upstream/laya/common.py      # pinned Laya implementation
    upstream/LICENSE            # source Apache-2.0 license
    tokenizer/                  # pinned tokenizer files
    encoder/config.json         # ModernBERT architecture
    rl_agent_config.json        # Laya model configuration
    experiment_config.json      # fixed-schema experiment contract
    calibration.json            # dev-fitted positive scalar temperature
```

Copy the whole `release/i3/` directory to a clean directory. The loader uses only sibling bundle assets; it does not rely on absolute workstation paths or hidden model caches. No optimizer/training state or original trace data is in the inference bundle.
Before model construction, the loader hashes every distributed file and rejects
missing, extra, symlinked, or modified assets. `python bundle_integrity.py`
checks the same manifest explicitly. To rebuild it after an intentional local
change, run `python scripts/i3_make_bundle_manifest.py` at the repository root
and review the tracked manifest diff; never regenerate it merely to dismiss a
transfer mismatch.

## Contract

Input must contain only a nonempty text `state` and a `questions` mapping with exactly one `next_action` choice. Its instructions and six descriptions must match `FROZEN_INSTRUCTIONS` and `FROZEN_CRITERIA` from `inference.py` exactly, in the frozen choice-logit order: `edit, execute, other_tool, read, respond_or_finish, search`. `predict(record, predictor)` returns raw probabilities keyed in the reporting order `read, search, edit, execute, other_tool, respond_or_finish`. `LocalLayaPredictor.predict_pair(record)` returns both raw and dev-temperature-scaled distributions from the **same forward pass**. Gold labels, record IDs and provenance metadata are rejected. This is a fixed-schema coding-harness classifier; arbitrary questions and candidate lists are unsupported.

The Laya sequence builder enforces 512 total tokens and a 192-token question-head limit. Long states can lose context under the pinned truncation rules. Neither raw nor scaled probabilities are independently validated as calibrated in deployment.

`examples.json` contains three original synthetic states representing inspect,
search, and execute contexts; these are API demonstrations, not accuracy
measurements or reproduced trace examples. Known error types (including
confident `read`/`execute` and `edit`/`execute` confusions) are in the model card.

## Quickstart (CPU)

With Python 3.14 and the versions in `requirements.txt` installed, run from this directory:

```sh
python -m unittest -v test_inference
python quickstart.py
```

The latter loads approximately 1.69 GB of weights on CPU and prints both distributions. It ran from a separate repository-local copy (`artifacts/release_i3/isolated_quickstart.log`). Saved dev parity and limitations are recorded in `artifacts/release_i3/package_parity.json` and the model card.

`requirements.txt` pins versions observed in the verified project `.venv`; choose an appropriate platform PyTorch wheel. No online dependency installation was performed during verification. Public redistribution remains blocked pending model/dataset rights review.
