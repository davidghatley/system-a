# Selection Bias

Disposition: **FAIL**

Rates measure exclusions above 512 tokens among structurally and target-valid candidates.

| Split | Candidates | Over budget | Rate |
|---|---:|---:|---:|
| train | 45547 | 0 | 0.00% |
| development | 5987 | 0 | 0.00% |
| calibration | 5603 | 0 | 0.00% |
| test | 6071 | 0 | 0.00% |

Maximum split difference: 0.00% (gate: <= 10.00%).

| Tool | Positive rate | Negative rate | Absolute difference |
|---|---:|---:|---:|
| patch | 0.00% | 0.00% | 0.00% |
| process | 0.00% | 0.00% | 0.00% |
| read_file | 0.00% | 0.00% | 0.00% |
| search_files | 0.00% | 0.00% | 0.00% |
| terminal | 0.00% | 0.00% | 0.00% |
| write_file | 0.00% | 0.00% | 0.00% |

Complete machine-readable tables by split, proxy group, source category, and label are in `../results/profile.json`.
