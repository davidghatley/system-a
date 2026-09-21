# exp_001c Context-Budget Sweep

Protocol SHA-256: `7b7642f9f2f3942d559db6a25e79cb8b5c90b52050cc642be57a8659d39d8e32`

Only train and development contributed candidates, targets, metrics, gates, and selection. Calibration and test rows were skipped from task-derived split assignment before conversation parsing.

| Budget | Retained | Retention | Worst train tool bias | Worst dev tool bias | Pass |
|---:|---:|---:|---:|---:|---|
| 512 | 23,622 | 47.00% | 20.6035% | 46.7065% | reference |
| 640 | 26,222 | 52.06% | 19.8863% | 41.7635% | FAIL |
| 768 | 28,495 | 56.47% | 18.7458% | 37.6080% | FAIL |
| 1024 | 31,989 | 63.26% | 18.2408% | 30.6040% | FAIL |

Selected budget: **None**

## Budget 512

| Split | Candidates | Over budget | Rate |
|---|---:|---:|---:|
| train | 45,526 | 24,181 | 53.1147% |
| development | 5,986 | 3,120 | 52.1216% |

| Split / tool | Positive | Negative | Positive rate | Negative rate | Difference | Positive retained trajectories |
|---|---:|---:|---:|---:|---:|---:|
| train / patch | 1,702 | 43,824 | 72.0917% | 52.3777% | 19.7140% | 358 |
| train / process | 310 | 45,216 | 40.9677% | 53.1980% | 12.2302% | 118 |
| train / read_file | 4,918 | 40,608 | 71.4925% | 50.8890% | 20.6035% | 890 |
| train / search_files | 3,646 | 41,880 | 37.2737% | 54.4938% | 17.2201% | 1,058 |
| train / terminal | 25,777 | 19,749 | 46.5221% | 61.7196% | 15.1975% | 3,327 |
| train / write_file | 8,438 | 37,088 | 65.5606% | 50.2831% | 15.2774% | 1,880 |
| development / patch | 185 | 5,801 | 69.7297% | 51.5601% | 18.1697% | 44 |
| development / process | 18 | 5,968 | 5.5556% | 52.2621% | 46.7065% | 12 |
| development / read_file | 541 | 5,445 | 67.2828% | 50.6152% | 16.6676% | 105 |
| development / search_files | 447 | 5,539 | 37.8076% | 53.2768% | 15.4692% | 165 |
| development / terminal | 3,582 | 2,404 | 45.4774% | 62.0216% | 16.5442% | 456 |
| development / write_file | 1,137 | 4,849 | 67.1064% | 48.6080% | 18.4985% | 255 |

## Budget 640

| Split | Candidates | Over budget | Rate |
|---|---:|---:|---:|
| train | 45,526 | 21,871 | 48.0407% |
| development | 5,986 | 2,825 | 47.1935% |

| Split / tool | Positive | Negative | Positive rate | Negative rate | Difference | Positive retained trajectories |
|---|---:|---:|---:|---:|---:|---:|
| train / patch | 1,702 | 43,824 | 61.2808% | 47.5265% | 13.7544% | 443 |
| train / process | 310 | 45,216 | 34.1935% | 48.1356% | 13.9421% | 125 |
| train / read_file | 4,918 | 40,608 | 65.7788% | 45.8924% | 19.8863% | 987 |
| train / search_files | 3,646 | 41,880 | 34.3939% | 49.2287% | 14.8349% | 1,092 |
| train / terminal | 25,777 | 19,749 | 41.7077% | 56.3066% | 14.5989% | 3,406 |
| train / write_file | 8,438 | 37,088 | 61.1875% | 45.0496% | 16.1379% | 1,991 |
| development / patch | 185 | 5,801 | 54.5946% | 46.9574% | 7.6372% | 63 |
| development / process | 18 | 5,968 | 5.5556% | 47.3190% | 41.7635% | 12 |
| development / read_file | 541 | 5,445 | 62.6617% | 45.6566% | 17.0052% | 116 |
| development / search_files | 447 | 5,539 | 34.4519% | 48.2217% | 13.7698% | 166 |
| development / terminal | 3,582 | 2,404 | 40.5918% | 57.0300% | 16.4381% | 461 |
| development / write_file | 1,137 | 4,849 | 63.7643% | 43.3079% | 20.4564% | 269 |

## Budget 768

| Split | Candidates | Over budget | Rate |
|---|---:|---:|---:|
| train | 45,526 | 19,844 | 43.5883% |
| development | 5,986 | 2,577 | 43.0505% |

| Split / tool | Positive | Negative | Positive rate | Negative rate | Difference | Positive retained trajectories |
|---|---:|---:|---:|---:|---:|---:|
| train / patch | 1,702 | 43,824 | 52.4089% | 43.2457% | 9.1632% | 509 |
| train / process | 310 | 45,216 | 30.6452% | 43.6770% | 13.0319% | 126 |
| train / read_file | 4,918 | 40,608 | 60.3091% | 41.5632% | 18.7458% | 1,063 |
| train / search_files | 3,646 | 41,880 | 30.6637% | 44.7135% | 14.0497% | 1,128 |
| train / terminal | 25,777 | 19,749 | 37.4404% | 51.6127% | 14.1724% | 3,444 |
| train / write_file | 8,438 | 37,088 | 57.5966% | 40.4012% | 17.1954% | 2,056 |
| development / patch | 185 | 5,801 | 48.1081% | 42.8892% | 5.2190% | 71 |
| development / process | 18 | 5,968 | 5.5556% | 43.1635% | 37.6080% | 12 |
| development / read_file | 541 | 5,445 | 55.4529% | 41.8182% | 13.6347% | 135 |
| development / search_files | 447 | 5,539 | 31.7673% | 43.9610% | 12.1937% | 170 |
| development / terminal | 3,582 | 2,404 | 36.5997% | 52.6622% | 16.0626% | 462 |
| development / write_file | 1,137 | 4,849 | 60.6860% | 38.9152% | 21.7708% | 272 |

## Budget 1024

| Split | Candidates | Over budget | Rate |
|---|---:|---:|---:|
| train | 45,526 | 16,764 | 36.8229% |
| development | 5,986 | 2,159 | 36.0675% |

| Split / tool | Positive | Negative | Positive rate | Negative rate | Difference | Positive retained trajectories |
|---|---:|---:|---:|---:|---:|---:|
| train / patch | 1,702 | 43,824 | 38.4841% | 36.7584% | 1.7257% | 600 |
| train / process | 310 | 45,216 | 25.8065% | 36.8984% | 11.0920% | 131 |
| train / read_file | 4,918 | 40,608 | 51.6877% | 35.0227% | 16.6650% | 1,176 |
| train / search_files | 3,646 | 41,880 | 25.5074% | 37.8080% | 12.3006% | 1,167 |
| train / terminal | 25,777 | 19,749 | 31.2721% | 44.0681% | 12.7960% | 3,489 |
| train / write_file | 8,438 | 37,088 | 51.6829% | 33.4421% | 18.2408% | 2,158 |
| development / patch | 185 | 5,801 | 32.4324% | 36.1834% | 3.7510% | 82 |
| development / process | 18 | 5,968 | 5.5556% | 36.1595% | 30.6040% | 12 |
| development / read_file | 541 | 5,445 | 44.5471% | 35.2250% | 9.3222% | 151 |
| development / search_files | 447 | 5,539 | 26.3982% | 36.8478% | 10.4496% | 176 |
| development / terminal | 3,582 | 2,404 | 29.7041% | 45.5491% | 15.8450% | 469 |
| development / write_file | 1,137 | 4,849 | 56.2885% | 31.3260% | 24.9624% | 280 |

