# Iteration 3 release-use amendment (post historical final test)

Date: 2026-09-23. This document preserves `artifacts/experiment_i3/PREDECLARATION.md`, `preflight/runs/freeze.json`, and the consumed attempt marker. It does not amend the historical evaluation retroactively or authorize another test read.

## Release question

Can the already-selected seed-42 checkpoint be distributed as a narrow, experimental fixed-schema next-tool-category classifier with transparent uncertainty and provenance? Its historical test result is exploratory and cannot serve as new confirmation of any post-test calibration or packaging decision. The original dev-only selection rule and six-class macro-F1 remain historical facts.

## Dev-only packaging comparison

Compare the saved raw seed-42 probability vector with positive-scalar temperature scaling on the same six choice logits reconstructed from finite saved softmax probabilities. Use deterministic task-group cross-validation so trajectories remain together. Fit temperature on training groups only; score each held-out group once. Report pooled NLL and Brier plus unchanged accuracy and fixed-six-class macro-F1, raw versus out-of-fold temperature; report per-class supports and the very small number of dev task groups. Any final deployment temperature is refit on all dev examples only after this comparison and is clearly labeled dev-fitted. If saved probabilities have zeros or insufficient numerical precision for a stable logit reconstruction, prefer raw inference until actual logits can be obtained from the existing checkpoint using train/dev only.

Selection rule for this new release-use decision: use calibrated probabilities only if grouped out-of-fold NLL and Brier both improve, argmax classification is unchanged, and inference parity establishes that fitted temperature applies to the actual inference outputs. Otherwise ship raw scores with a prominent uncalibrated warning. This assessment is exploratory because the same dev split selected the checkpoint, and it does not establish calibration on a fresh population. Reliability diagrams are descriptive; no single ECE-like scalar certifies calibration.

The historical test metrics are neither used to fit nor select the temperature. No comparator or latency value omitted from the historical final batch is manufactured by post-hoc test access. A future confirmatory comparison needs a new, independently frozen holdout and separate permission.
