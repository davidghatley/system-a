# Selection Bias

Disposition: **FAIL**

Rates measure exclusions above 512 tokens among structurally and target-valid candidates.

| Split | Candidates | Over budget | Rate |
|---|---:|---:|---:|
| train | 45526 | 24181 | 53.11% |
| development | 5986 | 3120 | 52.12% |
| calibration | 5599 | 3008 | 53.72% |
| test | 6068 | 3301 | 54.40% |

Maximum split difference: 2.28% (gate: <= 10.00%).

| Tool | Positive rate | Negative rate | Absolute difference |
|---|---:|---:|---:|
| patch | 72.22% | 52.48% | 19.74% |
| process | 38.75% | 53.29% | 14.54% |
| read_file | 71.67% | 50.96% | 20.71% |
| search_files | 37.70% | 54.51% | 16.81% |
| terminal | 46.68% | 61.93% | 15.25% |
| write_file | 65.64% | 50.33% | 15.31% |

Complete machine-readable tables by split, proxy group, source category, and label are in `../results/profile.json`.
