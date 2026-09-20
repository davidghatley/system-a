# Skeptical Review of the Laya One-Step Measurement

Review date: 2026-09-20.

## Classification

- **INCONCLUSIVE:** The measurement does not establish that the pinned FP16 Laya path intrinsically produces non-finite gradients or cannot complete an optimizer step. It establishes only that the first backward at PyTorch's default dynamic-loss scale produced at least one non-finite gradient and that the runner refused the step.
- **VERIFIED:** One 512-position forward and backward fit on the RTX 3060. Peak reserved memory was 3,873,439,744 bytes, below 11.5 GiB (`artifacts/laya_one_step.json:40-43,88-95`). This does not include Adam state initialization during a first completed step.
- **VERIFIED:** This review executed no forward, backward, or optimizer operation. Static syntax parsing and JSON parsing passed; pinned source `HEAD` is `d113dca2512fb3eaca313534bc54c7162d87c1d4` with a clean source worktree. Commands and results are recorded in `artifacts/reviews/gpu_review_checks.json`.

## Attempt to Disprove the Conclusion

### GradScaler semantics

- **VERIFIED:** The runner creates an enabled CUDA GradScaler (`scripts/run_laya_one_step.py:196`), scales the loss for backward, and unscales gradients (`:228-230`). It then manually scans gradients and raises before `scaler.step()` if any are non-finite (`:231-235`).
- **VERIFIED:** Upstream creates the same enabled scaler and scales the backward (`data/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb:267,322`), but after unscale and clipping it calls `scaler.step()` and `scaler.update()` without the runner's terminal finite-gradient guard (`:325-331`). GradScaler's purpose is to skip an overflowed optimizer step and reduce its scale; an overflow on the initial scale is not by itself proof that unscaled FP16 model gradients are intrinsically non-finite.
- **INFERRED:** The most likely immediate cause is overflow introduced by the default initial loss scale, not necessarily FP16 model arithmetic. The prior report's statement that "the pinned FP16 path fails" (`research/09_laya_gpu_measurement.md:65,70`) is therefore too strong.
- **UNKNOWN:** Whether gradients remain non-finite at scale 1.0. The retained artifact cannot distinguish loss-scale overflow from non-finite values generated inside FP16 attention or checkpoint recomputation.

### Missing evidence and provenance

- **VERIFIED:** The result proves that the finite loss/reward guard did not raise before backward, because that guard precedes backward in the runner (`scripts/run_laya_one_step.py:225-229`) and the artifact records one backward (`artifacts/laya_one_step.json:88-93`).
- **UNKNOWN:** Exact combined, CE, RL, reward, and offending-parameter values are absent from the executed artifact (`artifacts/laya_one_step.json:1-96`). The current runner assigns these before raising (`scripts/run_laya_one_step.py:225,232-235`), while the preserved traceback identifies the raise as line 232 (`artifacts/laya_one_step.json:15`; `artifacts/laya_one_step_commands.log:29-35`), now occupied by the diagnostic scan. This confirms the code changed after execution.
- **UNKNOWN:** No hash or immutable copy of the executed runner was retained. Consequently the current source can explain the intended method, but it cannot fully authenticate the exact executed implementation.
- **VERIFIED:** The command ledger records an earlier failed parameter-count attempt followed by the measured attempt (`artifacts/laya_one_step_commands.log:9-22,23-37`). The final JSON safely reports zero optimizer steps attempted/completed (`artifacts/laya_one_step.json:88-95`).

## Upstream Conformance

### Loss, masks, and perturbations

- **VERIFIED:** The loss algebra matches upstream: detached-logit Gaussian perturbations, zero-mean projection over valid options, masked softmax, no-grad proper reward, group-centered/global-standardized advantage, Gaussian score-function term, and full-weight CE (`scripts/run_laya_one_step.py:213-224`; upstream notebook `:304-320`). `proper_reward` itself masks probabilities and implements log, spherical, and score-only ranked terms (`data/laya/laya/common.py:140-166`).
- **VERIFIED:** Marker-mask construction and padding agree with upstream collation: valid markers are true and attention padding is zero (`data/laya/laya/common.py:218-250`; runner `:160-166`). Runner perturbation, softmax, policy loss, and CE all use the marker mask (`scripts/run_laya_one_step.py:209-224`).
- **VERIFIED DIFFERENCE:** Upstream sets perturbation group size `G=4`; the bounded runner uses `G=2` (`data/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb:251,304-308`; `scripts/run_laya_one_step.py:87-88,213`). Sigma 0.4 and projection semantics match. This is a declared smoke adaptation, not an exact upstream training reproduction.
- **INFERRED:** With `G=2`, standardized advantage is especially sensitive to the two sampled rewards, but the formula remains finite due to `+1e-6`. It could alter gradient magnitude relative to upstream and interact with initial scaling; retained values are insufficient to quantify that effect.

### Autocast and gradient checks

- **VERIFIED:** Both implementations limit FP16 autocast to the model forward and cast logits to FP32 before loss construction (`scripts/run_laya_one_step.py:205-208`; upstream notebook `:290-300`). Thus CE, reward, perturbations, and policy arithmetic are FP32, while saved forward activations and model backward can still involve FP16.
- **VERIFIED:** The warning records a non-deterministic memory-efficient attention backward despite deterministic algorithms being requested (`artifacts/laya_one_step_commands.log:27-28`). This defeats bitwise reproducibility but does not itself prove a source of NaN/Inf.
- **VERIFIED:** The runner's post-unscale all-element finite scan is stricter and more informative than upstream's implicit GradScaler handling (`scripts/run_laya_one_step.py:230-235`; upstream notebook `:325-331`). The safety guard was correct for preventing a questionable update; interpreting it as terminal FP16 infeasibility was not.
- **UNKNOWN:** No parameter name, gradient magnitude, GradScaler scale, or separate encoder/head finite count survived the run, so localization is impossible from artifacts.

### Parameter accounting

- **VERIFIED:** The artifact's counts are coherent: 421,293,827 named parameters, comprising 394,781,696 encoder and 26,512,131 non-encoder parameters; the state dictionary has 421,293,830 elements due to the three-element temperature buffer (`artifacts/laya_one_step.json:44-52`; buffer definition at `data/laya/laya/common.py:99-103`). The corrected account in `research/09_laya_gpu_measurement.md:56-58` is valid.
- **VERIFIED DIFFERENCE:** Upstream optimizes every non-encoder named parameter, including the action head (`data/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb:257-263`), even though its objective contribution is multiplied by zero (`:320`). The runner intentionally excludes 263,938 action parameters and optimizes 26,248,193 decision-head parameters (`scripts/run_laya_one_step.py:175-195`; artifact `:44-52`). This follows the earlier audit recommendation but is not exact upstream optimizer accounting.
- **VERIFIED:** The runner checks unique optimizer membership and action-head exclusion (`scripts/run_laya_one_step.py:186-188`). It does not explicitly assert that every non-action named parameter appears once, but its name partition makes that true for this model after the count guards.

## Likely Causes of Non-Finite Gradients

1. **INFERRED, leading:** Default GradScaler initial-scale overflow. The runner's first scaled backward is rejected before dynamic scaling can skip and reduce the scale (`scripts/run_laya_one_step.py:196,228-235`).
2. **INFERRED:** FP16 overflow in encoder attention/backward or checkpoint recomputation. The warning places memory-efficient attention in the backward path (`artifacts/laya_one_step_commands.log:27-28`), but no offending parameter or operator was retained.
3. **INFERRED:** The `G=2` perturbation sample and global advantage standardization may produce a different policy-gradient magnitude from upstream `G=4`, amplifying scale overflow. No retained loss or gradient magnitude permits confirmation.
4. **UNKNOWN:** Corrupt weights, non-finite forward logits, reward, or scalar loss are unlikely because the pre-backward finite guard was passed, but exact values and the executed runner hash were not preserved.

## Single Safest Next Diagnostic

- **RECOMMENDED:** Make one bounded diagnostic run with the identical pin, input, FP16 autocast, checkpointing, masks, `G=2`, and loss, but construct `torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=2000)`. Execute exactly one forward and one scaled backward, call `scaler.unscale_(optimizer)`, record diagnostics, and stop. Do **not** call `scaler.step()`, `optimizer.step()`, or `scaler.update()`.
- **RECOMMENDED:** Persist before stopping: combined/CE/RL losses; reward min/max/mean/std; scaler scale; finite/non-finite gradient-tensor and element counts separately for encoder and decision head; first ten offending parameter names; maximum finite absolute gradient by group; and peak allocated/reserved memory. Hash and preserve the exact diagnostic script beside its JSON result.
- **RECOMMENDED DECISION RULE:** If scale-1 unscaled gradients are all finite, classify the original numerical conclusion **INVALID** and the default initial scale as the blocker. If any remain non-finite, classify FP16 backward instability **VERIFIED** for this input and environment; the next separately authorized experiment may then test BF16. Do not use BF16 first, because it would change two variables and fail to identify whether dynamic loss scaling alone caused the result.

## Final Result

- **INCONCLUSIVE:** Valid evidence supports backward memory fit and a safely blocked first scaled update. It does not support a general conclusion that FP16 Laya gradients are intrinsically non-finite or that a one-step update is infeasible. Stop here pending the precisely bounded scale-1 diagnostic above.
