# Baseline iteration 3 preflight

Everything here is preflight evidence. `manifest_v2.json` uses only v2 train/dev to exercise the interface and explicitly excludes held-out data. Its scores must not be reported as corrected-v3 or final held-out evidence.

The candidate budget was frozen before the v2 smoke: word unigram and word unigram/bigram TF-IDF, each at `C` 0.5 and 2.0, with all optimizer and vocabulary limits in the manifest. Primary selection is fixed-list macro-F1; exact ties resolve by candidate ID ascending. Bootstrap intervals resample complete `trajectory_id` groups.
