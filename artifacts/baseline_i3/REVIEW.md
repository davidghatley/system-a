# Independent baseline review

**Decision: ACCEPTED (corrected v3 train/dev baseline only).**

I read directive §5, the claimed artifacts, implementation/tests, manifest, and only corrected-v3 train/dev records; no test data or results were read. Unit tests pass (6/6).

**VERIFIED.** Recomputed hashes: train `86442271…dda3c`, dev `ade82626…92783`, manifest `05e04f46…14bbb`; the stated 1,051/140 rows, 163/20 disjoint trajectories and task groups, fixed six-label metrics, absent-class-zero macro-F1, and all reported candidate metrics match. Selected `word12-c2` is the frozen four-candidate dev winner with ascending-ID tie-break. Recomputed transition counts (883, five gaps, 30 evidence/110 fallback), NLL/Brier, and 2,000-group bootstrap exactly match. Regenerated prediction JSONL and deterministic gzip model bytes are byte-identical. The 23.48 s/684,768 KiB CPU wrapper claim is documented and the CLI enforces four threads.

Information parity is sound at the predictor boundary: all methods receive `state`; metadata/gold/target arguments are not input fields. The transition parser is restricted to observation/history markers and its explicit training-majority fallback is exercised. The selected TF-IDF gain remains vulnerable to dev-selection optimism (four candidates, only 20 trajectories); absent `other_tool` and `respond_or_finish` support makes macro-F1 exploratory. Thus this falsification attempt cannot establish held-out superiority or calibration. No repair is required for this bounded baseline; do not generalize beyond train/dev.
