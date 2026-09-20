# Next Actions

## Current Decision

The Kimi K3 Experiment 001 is stopped. The Laya one-step substrate smoke passed. The only authorized next experiment is `exp_001b`, a Hermes preprocessing prototype with no model training or official evaluation.

Follow `experiments/exp_001b_hermes_preprocessing/protocol.json`. If it conflicts with older instructions below, the newer protocol controls.

Do not run the main training job until steps 1-6 are complete and a frozen copy of the resolved protocol exists.

## 1. Run the Existing Safe Preflight

```bash
bash experiments/laya_smoke_test/run.sh
```

Expected result: pinned metadata and GPU checks pass. This does not test model training.

## 2. Create an Isolated CUDA Environment

Use a project-local virtual environment and select the current CUDA-compatible PyTorch command from the official PyTorch installer. Then install the pinned Laya checkout and dataset tooling. Record every resolved version in `artifacts/environment.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate CUDA PyTorch build from https://pytorch.org/get-started/locally/
git clone https://github.com/NandhaKishorM/laya.git data/laya-source
git -C data/laya-source checkout d113dca2512fb3eaca313534bc54c7162d87c1d4
python -m pip install -e data/laya-source
python -m pip install datasets scikit-learn pandas pyarrow
python -m pip freeze > artifacts/environment.txt
```

Do not use the upstream two-GPU notebook unchanged.

## 3. Implement and Run Dataset Preflight

Create `scripts/preflight_exp001_data.py`. It must:

- download `greghavens/kimi-k3-coding-and-debugging-traces` at an explicit commit;
- write file hashes to `artifacts/exp_001_dataset_checksums.json`;
- assert row count, trajectory count, target index, and trajectory-disjoint source splits;
- enumerate target tool names, parallel-call multiplicity, empty targets, response-only turns, and duplicate calls;
- quantify empty/redacted arguments and missing tool-result observations;
- verify target messages and `reasoning_content` are absent from rendered state;
- report exact state and normalized task overlap by split;
- propose, but not silently choose, the minimal tool-family map.

Run:

```bash
source .venv/bin/activate
python scripts/preflight_exp001_data.py \
  --dataset greghavens/kimi-k3-coding-and-debugging-traces \
  --revision <PINNED_COMMIT_SHA> \
  --output artifacts/exp_001_data_profile.json
```

Stop if fewer than 1,500 eligible train rows or 300 test rows remain, if source splits overlap by trajectory, or if observable history is mostly absent.

## 4. Freeze the Representation

Using only the preflight profile, resolve these fields in `experiments/exp_001/protocol.json`:

- dataset revision;
- state template version;
- tool-family map;
- exact source train/calibration/test trajectory counts;
- exclusions and their counts.

Write immutable manifests:

```text
artifacts/exp_001_train_ids.txt
artifacts/exp_001_calibration_ids.txt
artifacts/exp_001_test_ids.txt
artifacts/exp_001_dataset_checksums.json
artifacts/exp_001_protocol_frozen.json
```

The frozen protocol must contain no `UNRESOLVED` values. Hash it before training.

## 5. Implement the One-Step VRAM Test

Adapt the official notebook into a single-GPU script, initially using `experiments/laya_smoke_test/config.json`. It must load the pinned base model, construct two real typed-decision cases, perform one forward/backward/optimizer step, and record:

- peak allocated and reserved VRAM;
- step wall-clock;
- loss components and finite gradients;
- package, CUDA, driver, model, and data revisions.

Run only one step first:

```bash
source .venv/bin/activate
python scripts/run_laya_one_step.py \
  --config experiments/laya_smoke_test/config.json \
  --output artifacts/laya_one_step.json
```

Gate: stop if peak reserved VRAM exceeds 11.5 GiB, gradients are non-finite, or projected main runtime exceeds 24 hours. If it OOMs, reduce sequence length to 384 and then 256. Any change must be recorded before further training.

## 6. Build and Verify Baselines

Implement one shared renderer/label extractor used by every model. Train using train groups only and select thresholds on calibration groups only:

```bash
source .venv/bin/activate
python scripts/run_exp001_baselines.py \
  --protocol artifacts/exp_001_protocol_frozen.json \
  --output artifacts/exp_001_baselines.json
```

Required outputs are majority, TF-IDF/logistic, untuned Laya, task-only, and history-only metrics. Do not score the official test split while changing preprocessing.

## 7. Run the Bounded Data/Model Smoke

Use at most 100 train trajectories, 500 rows, and 100 optimizer updates. Evaluate only on calibration data. Confirm that loss decreases, labels are non-degenerate, saved checkpoints reload, and calibrated probabilities are finite.

Do not alter promotion criteria based on this run. Changes needed for correctness or fit require a protocol deviation recorded before the main run.

## 8. Authorize or Stop

Proceed to the main pilot only if:

- all protocol fields are resolved and hashed;
- data and leakage checks pass;
- one-step and 100-update runs fit the VRAM/time gates;
- baseline artifacts are complete;
- test labels and test metrics have not been inspected for tuning.

Otherwise write `artifacts/exp_001_stop_report.md` with the failed gate. Do not keep tuning until a positive result appears.

## 9. Main Pilot

Train one preregistered seed. Freeze the selected checkpoint and calibration threshold before generating official test predictions. Save raw predictions before scoring. Compute trajectory-cluster bootstrap intervals and every metric listed in the frozen protocol.

Interpret outcomes exactly as preregistered: pilot support, failure for this setup, inconclusive, invalidated, or hardware-feasibility failure.

## 10. Experiment 002 Decision

Do not begin Experiment 002 now. If Experiment 001 validly fails, inspect whether a coherent, sufficiently large domain exists in verified traces. Design/frontend/Three.js/3D is eligible only after obtaining trace data with observable states, actions, and outcomes; interest in the domain is not evidence that suitable public data exists.
