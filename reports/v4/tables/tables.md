## Table 1. Generated dataset composition

| family | subtype | flights | median_duration_s | min_duration_s | max_duration_s |
|---|---|---|---|---|---|
| GNSS degradation | no_fix_then_ok | 30 | 238 | 64 | 378 |
| GNSS degradation | off_then_ok | 32 | 250 | 62 | 427 |
| GNSS degradation | stuck_then_ok | 28 | 286 | 124 | 455 |
| Nominal | none | 60 | 223 | 141 | 371 |
| Sensor fault | baro_stuck | 29 | 220 | 117 | 336 |
| Sensor fault | mag_stuck | 31 | 207 | 143 | 328 |
| Spoof | drift_coherent | 17 | 224 | 168 | 554 |
| Spoof | drift_incoherent | 22 | 238 | 163 | 303 |
| Spoof | jump | 21 | 215 | 167 | 348 |

Total flights: 270; firmware commits: ['e53ff6b3ff']; onset logged for 210 of 210 attacked flights.

## Table 2. Leakage audit: same detector, matched training-window counts, paired over repetitions

| Split | Training windows | Accuracy | Macro-F1 | ECE | NLL | Brier |
|---|---|---|---|---|---|---|
| Flight-grouped split | 11103 | 0.799 ± 0.009 | 0.803 ± 0.006 | 0.026 ± 0.009 | 0.462 ± 0.011 | 0.288 ± 0.008 |
| Random-window split | 11103 | 0.888 ± 0.001 | 0.885 ± 0.002 | 0.079 ± 0.004 | 0.319 ± 0.002 | 0.185 ± 0.001 |

Paired accuracy difference (random-window minus flight-grouped): 0.090 ± 0.009 over 5 repetitions.

## Table 3. Window-level detector, mean ± std over 5 flight-grouped repetitions

| Model | Probabilities | Accuracy | Macro-F1 | ECE | NLL | Brier |
|---|---|---|---|---|---|---|
| Logistic | raw | 0.656 ± 0.025 | 0.659 ± 0.023 | 0.082 ± 0.011 | 1.115 ± 0.166 | 0.466 ± 0.026 |
| Logistic | temperature-scaled | 0.656 ± 0.025 | 0.659 ± 0.023 | 0.108 ± 0.037 | 0.926 ± 0.098 | 0.474 ± 0.032 |
| XGBoost | raw | 0.791 ± 0.020 | 0.791 ± 0.031 | 0.064 ± 0.023 | 0.537 ± 0.065 | 0.316 ± 0.032 |
| XGBoost | temperature-scaled | 0.791 ± 0.020 | 0.791 ± 0.031 | 0.046 ± 0.010 | 0.503 ± 0.037 | 0.309 ± 0.025 |

## Table 4. Flight-level verdicts, mean ± std over 5 repetitions (20 train / 20 calibration / 20 test flights per family)

| Aggregation | Flight accuracy | Flight macro-F1 | AURC | Coarse accuracy | Coarse macro-F1 |
|---|---|---|---|---|---|
| Percentile rule | 0.625 ± 0.047 | 0.640 ± 0.044 | 0.228 ± 0.047 | 0.680 ± 0.043 | 0.676 ± 0.038 |
| Stacked, probability summaries only | 0.908 ± 0.059 | 0.908 ± 0.059 | 0.039 ± 0.033 | 0.930 ± 0.042 | 0.929 ± 0.043 |
| Stacked, raw flight features only | 0.835 ± 0.024 | 0.824 ± 0.029 | 0.036 ± 0.008 | 0.870 ± 0.020 | 0.880 ± 0.017 |
| Stacked, full | 0.912 ± 0.033 | 0.913 ± 0.032 | 0.026 ± 0.022 | 0.925 ± 0.021 | 0.923 ± 0.021 |
| Stacked, full, without derived-topic gap features | 0.907 ± 0.050 | 0.908 ± 0.049 | 0.029 ± 0.026 | 0.920 ± 0.036 | 0.916 ± 0.039 |

## Table 4b. Flight-level probability calibration of the verdict probabilities (test flights), mean ± std over 5 repetitions

| Aggregation | Fine ECE (10 bins) | Fine NLL | Fine Brier | Coarse ECE (10 bins) | Coarse NLL | Coarse Brier |
|---|---|---|---|---|---|---|
| Percentile rule | 0.088 ± 0.029 | 1.054 ± 0.082 | 0.544 ± 0.040 | 0.089 ± 0.029 | 0.710 ± 0.062 | 0.424 ± 0.034 |
| Stacked, probability summaries only | 0.116 ± 0.028 | 0.356 ± 0.112 | 0.178 ± 0.063 | 0.098 ± 0.034 | 0.286 ± 0.082 | 0.148 ± 0.048 |
| Stacked, raw flight features only | 0.150 ± 0.024 | 0.420 ± 0.018 | 0.235 ± 0.012 | 0.141 ± 0.012 | 0.350 ± 0.009 | 0.202 ± 0.007 |
| Stacked, full | 0.080 ± 0.012 | 0.288 ± 0.089 | 0.144 ± 0.050 | 0.074 ± 0.008 | 0.243 ± 0.073 | 0.123 ± 0.039 |
| Stacked, full, without derived-topic gap features | 0.089 ± 0.024 | 0.302 ± 0.092 | 0.147 ± 0.052 | 0.082 ± 0.021 | 0.259 ± 0.079 | 0.126 ± 0.042 |

## Table 5. Class-conditional conformal prediction sets at the flight level, mean ± std over 5 repetitions (balanced test sets)

| Aggregation | Level | alpha | Class | Coverage | Mean set size | Multi-label | Empty | Non-singleton | Verdict outside set | Operational abstention |
|---|---|---|---|---|---|---|---|---|---|---|
| Percentile rule | coarse | 0.100 | GNSS-chain inconsistency | 0.855 ± 0.073 | 1.345 ± 0.109 | 0.330 ± 0.111 | 0.000 ± 0.000 | 0.330 ± 0.111 | 0.000 ± 0.000 | 0.330 ± 0.111 |
| Percentile rule | coarse | 0.100 | Nominal | 0.890 ± 0.073 | 1.580 ± 0.194 | 0.550 ± 0.200 | 0.000 ± 0.000 | 0.550 ± 0.200 | 0.000 ± 0.000 | 0.550 ± 0.200 |
| Percentile rule | coarse | 0.100 | Non-GNSS fault | 0.880 ± 0.093 | 1.520 ± 0.194 | 0.440 ± 0.159 | 0.000 ± 0.000 | 0.440 ± 0.159 | 0.020 ± 0.024 | 0.450 ± 0.152 |
| Percentile rule | coarse | 0.200 | GNSS-chain inconsistency | 0.735 ± 0.116 | 1.120 ± 0.094 | 0.115 ± 0.085 | 0.000 ± 0.000 | 0.115 ± 0.085 | 0.020 ± 0.019 | 0.135 ± 0.080 |
| Percentile rule | coarse | 0.200 | Nominal | 0.730 ± 0.112 | 1.190 ± 0.258 | 0.190 ± 0.258 | 0.000 ± 0.000 | 0.190 ± 0.258 | 0.060 ± 0.073 | 0.250 ± 0.241 |
| Percentile rule | coarse | 0.200 | Non-GNSS fault | 0.820 ± 0.150 | 1.200 ± 0.230 | 0.200 ± 0.170 | 0.020 ± 0.040 | 0.220 ± 0.150 | 0.080 ± 0.075 | 0.280 ± 0.129 |
| Percentile rule | fine | 0.100 | GNSS degradation | 0.880 ± 0.093 | 1.760 ± 0.328 | 0.570 ± 0.220 | 0.000 ± 0.000 | 0.570 ± 0.220 | 0.000 ± 0.000 | 0.570 ± 0.220 |
| Percentile rule | fine | 0.100 | Nominal | 0.890 ± 0.073 | 1.620 ± 0.181 | 0.580 ± 0.194 | 0.000 ± 0.000 | 0.580 ± 0.194 | 0.000 ± 0.000 | 0.580 ± 0.194 |
| Percentile rule | fine | 0.100 | Sensor fault | 0.880 ± 0.093 | 1.600 ± 0.251 | 0.480 ± 0.196 | 0.000 ± 0.000 | 0.480 ± 0.196 | 0.020 ± 0.024 | 0.490 ± 0.188 |
| Percentile rule | fine | 0.100 | Spoof | 0.840 ± 0.080 | 1.720 ± 0.209 | 0.660 ± 0.193 | 0.000 ± 0.000 | 0.660 ± 0.193 | 0.000 ± 0.000 | 0.660 ± 0.193 |
| Percentile rule | fine | 0.200 | GNSS degradation | 0.640 ± 0.097 | 1.280 ± 0.250 | 0.270 ± 0.232 | 0.000 ± 0.000 | 0.270 ± 0.232 | 0.030 ± 0.040 | 0.280 ± 0.223 |
| Percentile rule | fine | 0.200 | Nominal | 0.730 ± 0.112 | 1.330 ± 0.280 | 0.330 ± 0.280 | 0.000 ± 0.000 | 0.330 ± 0.280 | 0.040 ± 0.080 | 0.350 ± 0.277 |
| Percentile rule | fine | 0.200 | Sensor fault | 0.820 ± 0.150 | 1.270 ± 0.232 | 0.240 ± 0.198 | 0.000 ± 0.000 | 0.240 ± 0.198 | 0.060 ± 0.049 | 0.290 ± 0.171 |
| Percentile rule | fine | 0.200 | Spoof | 0.800 ± 0.084 | 1.260 ± 0.146 | 0.260 ± 0.146 | 0.000 ± 0.000 | 0.260 ± 0.146 | 0.000 ± 0.000 | 0.260 ± 0.146 |
| Stacked, full | coarse | 0.100 | GNSS-chain inconsistency | 0.885 ± 0.058 | 1.040 ± 0.127 | 0.070 ± 0.098 | 0.030 ± 0.048 | 0.100 ± 0.088 | 0.060 ± 0.051 | 0.130 ± 0.076 |
| Stacked, full | coarse | 0.100 | Nominal | 0.900 ± 0.084 | 1.000 ± 0.084 | 0.030 ± 0.040 | 0.030 ± 0.060 | 0.060 ± 0.058 | 0.050 ± 0.063 | 0.080 ± 0.051 |
| Stacked, full | coarse | 0.100 | Non-GNSS fault | 0.930 ± 0.068 | 1.020 ± 0.129 | 0.050 ± 0.100 | 0.030 ± 0.060 | 0.080 ± 0.103 | 0.060 ± 0.058 | 0.110 ± 0.132 |
| Stacked, full | coarse | 0.200 | GNSS-chain inconsistency | 0.810 ± 0.051 | 0.870 ± 0.071 | 0.000 ± 0.000 | 0.130 ± 0.071 | 0.130 ± 0.071 | 0.160 ± 0.037 | 0.160 ± 0.037 |
| Stacked, full | coarse | 0.200 | Nominal | 0.790 ± 0.066 | 0.870 ± 0.093 | 0.000 ± 0.000 | 0.130 ± 0.093 | 0.130 ± 0.093 | 0.140 ± 0.086 | 0.140 ± 0.086 |
| Stacked, full | coarse | 0.200 | Non-GNSS fault | 0.830 ± 0.154 | 0.840 ± 0.159 | 0.000 ± 0.000 | 0.160 ± 0.159 | 0.160 ± 0.159 | 0.170 ± 0.154 | 0.170 ± 0.154 |
| Stacked, full | fine | 0.100 | GNSS degradation | 0.840 ± 0.066 | 0.900 ± 0.084 | 0.010 ± 0.020 | 0.110 ± 0.066 | 0.120 ± 0.051 | 0.150 ± 0.084 | 0.160 ± 0.066 |
| Stacked, full | fine | 0.100 | Nominal | 0.900 ± 0.084 | 1.140 ± 0.116 | 0.140 ± 0.116 | 0.000 ± 0.000 | 0.140 ± 0.116 | 0.000 ± 0.000 | 0.140 ± 0.116 |
| Stacked, full | fine | 0.100 | Sensor fault | 0.930 ± 0.068 | 1.030 ± 0.117 | 0.050 ± 0.100 | 0.020 ± 0.040 | 0.070 ± 0.098 | 0.060 ± 0.058 | 0.110 ± 0.132 |
| Stacked, full | fine | 0.100 | Spoof | 0.930 ± 0.068 | 1.250 ± 0.217 | 0.250 ± 0.217 | 0.000 ± 0.000 | 0.250 ± 0.217 | 0.020 ± 0.040 | 0.270 ± 0.201 |
| Stacked, full | fine | 0.200 | GNSS degradation | 0.740 ± 0.097 | 0.750 ± 0.110 | 0.000 ± 0.000 | 0.250 ± 0.110 | 0.250 ± 0.110 | 0.250 ± 0.110 | 0.250 ± 0.110 |
| Stacked, full | fine | 0.200 | Nominal | 0.790 ± 0.066 | 0.890 ± 0.086 | 0.000 ± 0.000 | 0.110 ± 0.086 | 0.110 ± 0.086 | 0.120 ± 0.081 | 0.120 ± 0.081 |
| Stacked, full | fine | 0.200 | Sensor fault | 0.830 ± 0.154 | 0.840 ± 0.159 | 0.000 ± 0.000 | 0.160 ± 0.159 | 0.160 ± 0.159 | 0.170 ± 0.154 | 0.170 ± 0.154 |
| Stacked, full | fine | 0.200 | Spoof | 0.730 ± 0.147 | 0.850 ± 0.105 | 0.010 ± 0.020 | 0.160 ± 0.102 | 0.170 ± 0.103 | 0.220 ± 0.093 | 0.230 ± 0.103 |

## Table 5b. Flights beyond the balanced test set (GNSS degradation), scored with the same thresholds, mean over 5 repetitions

| Aggregation | Level | alpha | Class | Flights per rep | Accuracy | Coverage | Operational abstention |
|---|---|---|---|---|---|---|---|
| Percentile rule | coarse | 0.100 | GNSS-chain inconsistency | 30.000 | 0.847 | 0.907 | 0.233 |
| Percentile rule | coarse | 0.200 | GNSS-chain inconsistency | 30.000 | 0.847 | 0.860 | 0.120 |
| Percentile rule | fine | 0.100 | GNSS degradation | 30.000 | 0.633 | 0.953 | 0.540 |
| Percentile rule | fine | 0.200 | GNSS degradation | 30.000 | 0.633 | 0.800 | 0.307 |
| Stacked, full | coarse | 0.100 | GNSS-chain inconsistency | 30.000 | 1.000 | 1.000 | 0.007 |
| Stacked, full | coarse | 0.200 | GNSS-chain inconsistency | 30.000 | 1.000 | 0.993 | 0.007 |
| Stacked, full | fine | 0.100 | GNSS degradation | 30.000 | 0.967 | 0.947 | 0.053 |
| Stacked, full | fine | 0.200 | GNSS degradation | 30.000 | 0.967 | 0.793 | 0.173 |

## Table 6. Unseen-subtype evaluation: one fitted pipeline per hold-out (single training and calibration set) evaluated on the withheld subtype and on a disjoint seen-subtype test set from the same family, mean ± std over 3 seeds

| Held-out family | Held-out subtype | n unseen | n seen test | Fine accuracy, unseen | Fine accuracy, seen | Coarse accuracy, unseen | Coarse accuracy, seen | Mean confidence, unseen | Mean confidence, seen |
|---|---|---|---|---|---|---|---|---|---|
| Spoof | Coherent drift | 17 | 15 | 0.157 ± 0.028 | 0.867 ± 0.000 | 0.157 ± 0.028 | 0.933 ± 0.054 | 0.686 ± 0.034 | 0.770 ± 0.005 |
| Spoof | Incoherent drift | 22 | 14 | 0.076 ± 0.021 | 0.809 ± 0.221 | 0.985 ± 0.021 | 0.809 ± 0.221 | 0.916 ± 0.005 | 0.750 ± 0.045 |
| Spoof | Jump | 21 | 13 | 0.111 ± 0.022 | 0.692 ± 0.188 | 0.222 ± 0.098 | 0.718 ± 0.192 | 0.619 ± 0.055 | 0.824 ± 0.018 |
| GNSS degradation | No-fix reporting | 30 | 20 | 0.978 ± 0.016 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.888 ± 0.006 | 0.963 ± 0.009 |
| GNSS degradation | Receiver silence | 32 | 20 | 0.917 ± 0.064 | 1.000 ± 0.000 | 0.969 ± 0.025 | 1.000 ± 0.000 | 0.722 ± 0.029 | 0.965 ± 0.004 |
| GNSS degradation | Frozen receiver output | 28 | 22 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.941 ± 0.084 | 1.000 ± 0.000 | 0.991 ± 0.013 | 0.942 ± 0.029 |
| Sensor fault | Frozen barometer | 29 | 11 | 0.000 ± 0.000 | 0.970 ± 0.043 | 0.000 ± 0.000 | 0.970 ± 0.043 | 0.885 ± 0.095 | 0.901 ± 0.078 |
| Sensor fault | Frozen magnetometer | 31 | 11 | 0.333 ± 0.471 | 0.939 ± 0.043 | 0.333 ± 0.471 | 0.939 ± 0.043 | 0.911 ± 0.105 | 0.891 ± 0.081 |

## Table 6b. Same fitted pipelines: conformal coverage and operational abstention on the withheld subtype and on the seen-subtype test set

| Held-out subtype | Fine coverage (0.10), unseen | Fine coverage (0.10), seen | Coarse coverage (0.10), unseen | Coarse coverage (0.10), seen | Operational abstention (0.10), unseen | Operational abstention (0.10), seen | Predicted as (unseen, seed 0) |
|---|---|---|---|---|---|---|---|
| Coherent drift | 0.647 ± 0.458 | 0.911 ± 0.083 | 0.216 ± 0.194 | 0.800 ± 0.144 | 0.569 ± 0.320 | 0.133 ± 0.054 | {"nominal": 13, "spoof": 3, "sensor_fault": 1} |
| Incoherent drift | 0.167 ± 0.093 | 0.929 ± 0.058 | 0.970 ± 0.021 | 0.691 ± 0.147 | 0.182 ± 0.074 | 0.214 ± 0.117 | {"gps_degrade": 21, "spoof": 1} |
| Jump | 0.254 ± 0.081 | 0.923 ± 0.063 | 0.127 ± 0.098 | 0.615 ± 0.126 | 0.476 ± 0.178 | 0.231 ± 0.166 | {"nominal": 18, "spoof": 2, "gps_degrade": 1} |
| No-fix reporting | 0.744 ± 0.042 | 0.950 ± 0.071 | 0.989 ± 0.016 | 1.000 ± 0.000 | 0.233 ± 0.047 | 0.050 ± 0.071 | {"gps_degrade": 29, "spoof": 1} |
| Receiver silence | 0.333 ± 0.059 | 0.967 ± 0.024 | 0.979 ± 0.029 | 1.000 ± 0.000 | 0.615 ± 0.115 | 0.033 ± 0.024 | {"gps_degrade": 32} |
| Frozen receiver output | 0.000 ± 0.000 | 0.970 ± 0.043 | 0.941 ± 0.084 | 1.000 ± 0.000 | 0.012 ± 0.017 | 0.030 ± 0.043 | {"spoof": 28} |
| Frozen barometer | 0.000 ± 0.000 | 0.909 ± 0.074 | 0.000 ± 0.000 | 0.909 ± 0.074 | 0.207 ± 0.293 | 0.091 ± 0.074 | {"spoof": 29} |
| Frozen magnetometer | 0.323 ± 0.456 | 0.879 ± 0.043 | 0.323 ± 0.456 | 0.879 ± 0.043 | 0.161 ± 0.206 | 0.151 ± 0.086 | {"nominal": 31} |

## Table 6c. Hold-out allocation per family (identical across seeds), test-set sizes, and the conformal order-statistic ranks; where the rank equals the calibration size the threshold is the largest calibration score

| Held-out subtype | Family | Train | Calibration | Seen test | Rank (0.10) | Rank (0.20) | Threshold at max score (0.10) |
|---|---|---|---|---|---|---|---|
| Coherent drift | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| Coherent drift | Spoof | 14 | 14 | 15 | 14 | 12 | True |
| Coherent drift | GNSS degradation | 20 | 20 | 0 | 19 | 17 | False |
| Coherent drift | Sensor fault | 20 | 20 | 0 | 19 | 17 | False |
| Incoherent drift | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| Incoherent drift | Spoof | 12 | 12 | 14 | 12 | 11 | True |
| Incoherent drift | GNSS degradation | 20 | 20 | 0 | 19 | 17 | False |
| Incoherent drift | Sensor fault | 20 | 20 | 0 | 19 | 17 | False |
| Jump | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| Jump | Spoof | 13 | 13 | 13 | 13 | 12 | True |
| Jump | GNSS degradation | 20 | 20 | 0 | 19 | 17 | False |
| Jump | Sensor fault | 20 | 20 | 0 | 19 | 17 | False |
| No-fix reporting | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| No-fix reporting | Spoof | 20 | 20 | 0 | 19 | 17 | False |
| No-fix reporting | GNSS degradation | 20 | 20 | 20 | 19 | 17 | False |
| No-fix reporting | Sensor fault | 20 | 20 | 0 | 19 | 17 | False |
| Receiver silence | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| Receiver silence | Spoof | 20 | 20 | 0 | 19 | 17 | False |
| Receiver silence | GNSS degradation | 19 | 19 | 20 | 18 | 16 | False |
| Receiver silence | Sensor fault | 20 | 20 | 0 | 19 | 17 | False |
| Frozen receiver output | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| Frozen receiver output | Spoof | 20 | 20 | 0 | 19 | 17 | False |
| Frozen receiver output | GNSS degradation | 20 | 20 | 22 | 19 | 17 | False |
| Frozen receiver output | Sensor fault | 20 | 20 | 0 | 19 | 17 | False |
| Frozen barometer | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| Frozen barometer | Spoof | 20 | 20 | 0 | 19 | 17 | False |
| Frozen barometer | GNSS degradation | 20 | 20 | 0 | 19 | 17 | False |
| Frozen barometer | Sensor fault | 10 | 10 | 11 | 10 | 9 | True |
| Frozen magnetometer | Nominal | 20 | 20 | 0 | 19 | 17 | False |
| Frozen magnetometer | Spoof | 20 | 20 | 0 | 19 | 17 | False |
| Frozen magnetometer | GNSS degradation | 20 | 20 | 0 | 19 | 17 | False |
| Frozen magnetometer | Sensor fault | 9 | 9 | 11 | 9 | 8 | True |

## Table 6d. Distributional abstention gate: fraction of flights withheld on the withheld subtype and on the seen-subtype test set (thresholds at the 95th percentile of the calibration flights), and the combined policy

| Held-out subtype | Mahalanobis gate, unseen | Mahalanobis gate, seen | Isolation-forest gate, unseen | Isolation-forest gate, seen | Conformal policy alone, unseen | Conformal or Mahalanobis gate, unseen | Conformal or Mahalanobis gate, seen |
|---|---|---|---|---|---|---|---|
| Coherent drift | 0.000 ± 0.000 | 0.044 ± 0.031 | 0.000 ± 0.000 | 0.067 ± 0.054 | 0.569 ± 0.320 | 0.569 ± 0.320 | 0.178 ± 0.063 |
| Incoherent drift | 0.364 ± 0.064 | 0.071 ± 0.058 | 0.167 ± 0.141 | 0.000 ± 0.000 | 0.182 ± 0.074 | 0.515 ± 0.113 | 0.286 ± 0.058 |
| Jump | 0.000 ± 0.000 | 0.026 ± 0.036 | 0.000 ± 0.000 | 0.026 ± 0.036 | 0.476 ± 0.178 | 0.476 ± 0.178 | 0.256 ± 0.192 |
| No-fix reporting | 1.000 ± 0.000 | 0.150 ± 0.041 | 0.056 ± 0.042 | 0.183 ± 0.047 | 0.233 ± 0.047 | 1.000 ± 0.000 | 0.200 ± 0.082 |
| Receiver silence | 1.000 ± 0.000 | 0.133 ± 0.024 | 0.021 ± 0.015 | 0.233 ± 0.062 | 0.615 ± 0.115 | 1.000 ± 0.000 | 0.167 ± 0.047 |
| Frozen receiver output | 0.941 ± 0.017 | 0.015 ± 0.021 | 0.833 ± 0.135 | 0.151 ± 0.119 | 0.012 ± 0.017 | 0.941 ± 0.017 | 0.045 ± 0.064 |
| Frozen barometer | 0.989 ± 0.016 | 0.030 ± 0.043 | 0.000 ± 0.000 | 0.061 ± 0.086 | 0.207 ± 0.293 | 0.989 ± 0.016 | 0.121 ± 0.086 |
| Frozen magnetometer | 0.989 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.030 ± 0.043 | 0.161 ± 0.206 | 0.989 ± 0.015 | 0.151 ± 0.086 |

## Table 7. Whelan live logs: physics-channel verdicts (detector trained on simulation only)

| Log | Reference label | Verdict | Confidence | Set (0.10) | Set (0.20) | Coarse verdict | Coarse set (0.10) |
|---|---|---|---|---|---|---|---|
| Benign | Nominal | Nominal | 0.974 | nominal | nominal | Nominal | nominal |
| Jamming | GNSS degradation | Sensor fault | 0.539 | nominal | (empty) | Non-GNSS fault | nominal |
| Spoofing | Spoof | Nominal | 0.984 | nominal | nominal | Nominal | nominal |

## Table 7b. Whelan live logs: receiver self-report channel

| Log | fix_type min | satellites min | satellites median | noise max | jamming indicator max | eph max (m) | speed variance median | speed variance max | max fix-to-fix jump (m) | lat range (deg) | lon range (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Benign | 3 | 11 | 12 | 114 | 67 | 3.73 | 0.34 | 1.13 | 1.02 | 36.20481 to 36.20483 | 138.25291 to 138.25293 |
| Jamming | 0 | 4 | 14 | 171 | 60 | 5.74 | 0.31 | 1.64 | 3.37 | 36.20479 to 36.20485 | 138.25290 to 138.25296 |
| Spoofing | 3 | 5 | 11 | 140 | 66 | 6.30 | 0.33 | 4.18 | 6.71 | 36.20481 to 36.20494 | 138.25289 to 138.25292 |

## Table 7c. Live logs: receiver-derived disturbance intervals (fix type below 3 or fewer than 8 satellites) and alerts from the declared rule (P(non-nominal) above 0.5 in two consecutive windows)

| Log | Receiver disturbance intervals (s) | Alert intervals (s) | Detection delay (s) | Alerts outside disturbance | Windows |
|---|---|---|---|---|---|
| Benign | [] | [[10.0, 20.0], [30.0, 35.0], [37.5, 50.0], [62.5, 82.5], [87.5, 100.0], [130.0, 135.0], [157.5, 165.0], [167.5, 185.0], [202.5, 207.5], [232.5, 245.0]] | nan | 10 | 97 |
| Jamming | [[170.0, 177.5]] | [[75.0, 82.5], [167.5, 220.0]] | 0.0 | 1 | 87 |
| Spoofing | [[117.5, 120.0]] | [[45.0, 50.0], [115.0, 135.0]] | 0.0 | 1 | 53 |

## Table 7d. Onset localisation on held-out flights with the declared alert rule against the logged onset (one fitted pipeline); for nominal flights the last column is the false-alert rate

| Family | Subtype | Flights | Detected within 60 s | Median |onset error| (s) | Within 10 s | Median signed error (s) | Early or false alert rate |
|---|---|---|---|---|---|---|---|
| GNSS degradation | No-fix reporting | 17.00 | 1.00 | 3.24 | 1.00 | -3.24 | 0.06 |
| GNSS degradation | Receiver silence | 16.00 | 0.94 | 3.46 | 0.81 | -3.02 | 0.19 |
| GNSS degradation | Frozen receiver output | 17.00 | 1.00 | 2.83 | 1.00 | -2.83 | 0.00 |
| Sensor fault | Frozen barometer | 7.00 | 1.00 | 0.94 | 1.00 | -0.94 | 0.00 |
| Sensor fault | Frozen magnetometer | 13.00 | 1.00 | 2.32 | 1.00 | 0.07 | 0.00 |
| Spoof | Coherent drift | 8.00 | 0.88 | 3.64 | 0.75 | -2.59 | 0.00 |
| Spoof | Incoherent drift | 7.00 | 1.00 | 2.26 | 1.00 | -2.26 | 0.00 |
| Spoof | Jump | 5.00 | 0.80 | 3.74 | 0.80 | -3.74 | 0.20 |
| Nominal |  | 20.00 | nan | nan | nan | nan | 0.40 |

## Table 8. Window detector, top-15 feature importance (repetition 0)

| feature | importance |
|---|---|
| baro_gap_frac_rmax3 | 0.1128 |
| mag_gap_frac_rmax3 | 0.0943 |
| gps_ekf_dpos_rms_rel | 0.0689 |
| gps_gap_frac_d1 | 0.0392 |
| gps_baro_dz_rms_d1 | 0.0367 |
| baro_gap_frac | 0.0348 |
| gps_gap_frac_rmax3 | 0.0293 |
| gps_ekf_dpos_rms | 0.0271 |
| mag_norm_mean_rel | 0.0270 |
| gps_baro_dz_rms | 0.0243 |
| gps_gap_frac | 0.0183 |
| gps_nofix_frac_rmax3 | 0.0168 |
| step_speed_max | 0.0165 |
| mag_gap_frac | 0.0157 |
| mag_raw_std | 0.0134 |

## Table 9. Flight-level model, standardised coefficients of the 20 most influential aggregates (repetition 0)

| Aggregate feature | Nominal | Spoof | GNSS degradation | Sensor fault |
|---|---|---|---|---|
| max_sensor_fault | -0.422 | -0.309 | -0.032 | 0.763 |
| max_spoof | -0.446 | 0.712 | -0.082 | -0.185 |
| max_gps_degrade | -0.416 | 0.007 | 0.523 | -0.115 |
| baro_gap_frac_max | -0.251 | -0.176 | -0.025 | 0.452 |
| gps_ekf_dpos_rms_max | -0.393 | 0.104 | 0.367 | -0.079 |
| gps_gap_frac_max | -0.154 | -0.143 | 0.360 | -0.063 |
| posvel_rms_max | -0.351 | 0.165 | 0.252 | -0.066 |
| gps_silent_frac_max | -0.133 | -0.163 | 0.341 | -0.044 |
| run_gps_degrade | -0.133 | -0.153 | 0.331 | -0.045 |
| step_minus_vel_max_max | -0.331 | 0.129 | 0.263 | -0.061 |
| run_sensor_fault | -0.181 | -0.124 | -0.025 | 0.330 |
| mag_gap_frac_max | -0.230 | -0.088 | -0.012 | 0.330 |
| p90_gps_degrade | -0.148 | -0.080 | 0.272 | -0.044 |
| mean_gps_degrade | -0.148 | -0.071 | 0.265 | -0.046 |
| frac_nominal | 0.263 | -0.094 | -0.043 | -0.125 |
| frac_gps_degrade | -0.107 | -0.114 | 0.256 | -0.035 |
| gps_baro_dz_rms_max | 0.018 | -0.212 | 0.256 | -0.062 |
| mean_sensor_fault | -0.136 | -0.091 | -0.017 | 0.244 |
| frac_sensor_fault | -0.133 | -0.091 | -0.019 | 0.244 |
| mean_nominal | 0.210 | -0.071 | -0.046 | -0.093 |
