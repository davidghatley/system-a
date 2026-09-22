# Iteration 3 inexpensive action baselines

This workstream compares training-majority, previous-action transition, and sparse TF-IDF multinomial logistic regression on exactly the compact `state` string supplied to Laya. Metadata is used only for trajectory ordering, leakage checks, and grouped uncertainty; it is never a prediction feature. The transition model learns label-to-label transitions from training trajectories, then obtains the previous action from a mapped tool name in the compact state's `LATEST RELEVANT OBSERVATION`. Missing or unmapped evidence uses the declared smoothed training-majority fallback.

The CLI accepts only `train` and `dev` paths, rejects filenames containing `test` or `heldout`, fixes the six-label ontology, permits at most four explicitly listed logistic candidates, and writes only below `artifacts/baseline_i3/preflight/`. Candidate selection maximizes dev macro-F1 and breaks exact ties by ascending candidate ID. Class-count distributions use symmetric additive smoothing; logistic probabilities are softmax outputs and are not described as calibrated.

Run synthetic tests:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m unittest discover -s baselines/trace_action_i3/tests -v
```

Run the v2 interface smoke (not final evidence):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m baselines.trace_action_i3.cli --manifest artifacts/baseline_i3/preflight/manifest_v2.json
```

For corrected v3, create a new manifest under the owned preflight directory pointing only to its train/dev files. Do not tune the declared candidate set after viewing v3 dev outcomes if the objective is a frozen comparison.
