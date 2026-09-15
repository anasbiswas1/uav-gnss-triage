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

## Table 2. Leakage audit (window level, same detector)

| Split | Accuracy | Macro-F1 | ECE | NLL | Brier |
|---|---|---|---|---|---|
| Random-window split | 0.884 | 0.885 | 0.080 | 0.323 | 0.189 |
| Flight-grouped split | 0.812 | 0.811 | 0.027 | 0.445 | 0.275 |

## Table 3. Window-level detector, mean ± std over 5 flight-grouped repetitions

| Model | Probabilities | Accuracy | Macro-F1 | ECE | NLL | Brier |
|---|---|---|---|---|---|---|
| Logistic | raw | 0.672 ± 0.025 | 0.647 ± 0.025 | 0.081 ± 0.010 | 1.258 ± 0.262 | 0.453 ± 0.021 |
| Logistic | temperature-scaled | 0.672 ± 0.025 | 0.647 ± 0.025 | 0.122 ± 0.040 | 1.016 ± 0.156 | 0.468 ± 0.019 |
| XGBoost | raw | 0.816 ± 0.013 | 0.786 ± 0.028 | 0.044 ± 0.016 | 0.473 ± 0.044 | 0.277 ± 0.021 |
| XGBoost | temperature-scaled | 0.816 ± 0.013 | 0.786 ± 0.028 | 0.051 ± 0.019 | 0.461 ± 0.024 | 0.278 ± 0.015 |

## Table 4. Flight-level verdicts, mean ± std over 5 repetitions (20 train / 20 calibration flights per family)

| Aggregation | Flight accuracy | Flight macro-F1 | AURC | Coarse accuracy | Coarse macro-F1 |
|---|---|---|---|---|---|
| Percentile rule | 0.627 ± 0.047 | 0.630 ± 0.045 | 0.207 ± 0.047 | 0.725 ± 0.040 | 0.685 ± 0.038 |
| Stacked | 0.927 ± 0.033 | 0.909 ± 0.036 | 0.019 ± 0.018 | 0.945 ± 0.015 | 0.933 ± 0.018 |

## Table 5. Class-conditional conformal prediction sets at the flight level, mean over 5 repetitions

| Aggregation | Level | alpha | Class | Coverage | Mean set size | Abstention | Empty sets |
|---|---|---|---|---|---|---|---|
| Percentile rule | coarse | 0.100 | GNSS-chain inconsistency | 0.877 | 1.300 | 0.289 | 0.000 |
| Percentile rule | coarse | 0.100 | Nominal | 0.890 | 1.580 | 0.550 | 0.000 |
| Percentile rule | coarse | 0.100 | Non-GNSS fault | 0.880 | 1.520 | 0.440 | 0.000 |
| Percentile rule | coarse | 0.200 | GNSS-chain inconsistency | 0.789 | 1.114 | 0.111 | 0.000 |
| Percentile rule | coarse | 0.200 | Nominal | 0.730 | 1.190 | 0.190 | 0.000 |
| Percentile rule | coarse | 0.200 | Non-GNSS fault | 0.820 | 1.200 | 0.220 | 0.020 |
| Percentile rule | fine | 0.100 | GNSS degradation | 0.924 | 1.716 | 0.548 | 0.000 |
| Percentile rule | fine | 0.100 | Nominal | 0.890 | 1.620 | 0.580 | 0.000 |
| Percentile rule | fine | 0.100 | Sensor fault | 0.880 | 1.600 | 0.480 | 0.000 |
| Percentile rule | fine | 0.100 | Spoof | 0.840 | 1.720 | 0.660 | 0.000 |
| Percentile rule | fine | 0.200 | GNSS degradation | 0.736 | 1.292 | 0.264 | 0.004 |
| Percentile rule | fine | 0.200 | Nominal | 0.730 | 1.330 | 0.330 | 0.000 |
| Percentile rule | fine | 0.200 | Sensor fault | 0.820 | 1.270 | 0.240 | 0.000 |
| Percentile rule | fine | 0.200 | Spoof | 0.800 | 1.260 | 0.260 | 0.000 |
| Stacked | coarse | 0.100 | GNSS-chain inconsistency | 0.934 | 1.026 | 0.060 | 0.017 |
| Stacked | coarse | 0.100 | Nominal | 0.900 | 1.000 | 0.060 | 0.030 |
| Stacked | coarse | 0.100 | Non-GNSS fault | 0.930 | 1.020 | 0.080 | 0.030 |
| Stacked | coarse | 0.200 | GNSS-chain inconsistency | 0.889 | 0.923 | 0.077 | 0.077 |
| Stacked | coarse | 0.200 | Nominal | 0.790 | 0.870 | 0.130 | 0.130 |
| Stacked | coarse | 0.200 | Non-GNSS fault | 0.830 | 0.840 | 0.160 | 0.160 |
| Stacked | fine | 0.100 | GNSS degradation | 0.904 | 0.952 | 0.080 | 0.064 |
| Stacked | fine | 0.100 | Nominal | 0.900 | 1.140 | 0.140 | 0.000 |
| Stacked | fine | 0.100 | Sensor fault | 0.930 | 1.030 | 0.070 | 0.020 |
| Stacked | fine | 0.100 | Spoof | 0.930 | 1.250 | 0.250 | 0.000 |
| Stacked | fine | 0.200 | GNSS degradation | 0.772 | 0.796 | 0.204 | 0.204 |
| Stacked | fine | 0.200 | Nominal | 0.790 | 0.890 | 0.110 | 0.110 |
| Stacked | fine | 0.200 | Sensor fault | 0.830 | 0.840 | 0.160 | 0.160 |
| Stacked | fine | 0.200 | Spoof | 0.730 | 0.850 | 0.170 | 0.160 |

## Table 6. Unseen-subtype evaluation (each subtype held out of training and calibration)

| Held-out family | Held-out subtype | n | Fine accuracy | Coarse accuracy | Mean confidence | Fine coverage (0.10) | Fine abstention (0.10) | Coarse coverage (0.10) | Coarse abstention (0.10) | Predicted as |
|---|---|---|---|---|---|---|---|---|---|---|
| Spoof | drift_coherent | 17 | 0.412 | 0.412 | 0.701 | 0.471 | 0.059 | 0.176 | 0.235 | {"nominal": 10, "spoof": 7} |
| Spoof | drift_incoherent | 22 | 0.318 | 1.000 | 0.752 | 0.409 | 0.182 | 1.000 | 0.000 | {"gps_degrade": 15, "spoof": 7} |
| Spoof | jump | 21 | 0.238 | 0.571 | 0.651 | 0.238 | 0.286 | 0.381 | 0.190 | {"nominal": 9, "gps_degrade": 7, "spoof": 5} |
| GNSS degradation | no_fix_then_ok | 30 | 1.000 | 1.000 | 0.919 | 0.833 | 0.167 | 1.000 | 0.000 | {"gps_degrade": 30} |
| GNSS degradation | off_then_ok | 32 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | {"nominal": 32} |
| GNSS degradation | stuck_then_ok | 28 | 0.000 | 0.786 | 0.973 | 0.000 | 0.000 | 0.786 | 0.000 | {"spoof": 22, "sensor_fault": 6} |
| Sensor fault | baro_stuck | 29 | 0.000 | 0.000 | 0.857 | 0.000 | 0.034 | 0.000 | 0.000 | {"spoof": 28, "gps_degrade": 1} |
| Sensor fault | mag_stuck | 31 | 0.000 | 0.000 | 0.995 | 0.000 | 0.000 | 0.000 | 0.000 | {"gps_degrade": 31} |

## Table 7. Whelan live logs: physics-channel verdicts (detector trained on simulation only)

| Log | Reference label | Verdict | Confidence | Set (0.10) | Set (0.20) | Coarse verdict | Coarse set (0.10) |
|---|---|---|---|---|---|---|---|
| Benign | Nominal | Nominal | 0.974 | nominal | nominal | Nominal | nominal |
| Jamming | GNSS degradation | Sensor fault | 0.539 | nominal | (empty) | Non-GNSS fault | nominal |
| Spoofing | Spoof | Nominal | 0.984 | nominal | nominal | Nominal | nominal |

## Table 7b. Whelan live logs: receiver self-report channel

| Log | fix_type min | satellites min | noise max | jamming indicator max | eph max (m) | speed variance max | max fix-to-fix jump (m) |
|---|---|---|---|---|---|---|---|
| Benign | 3 | 11 | 114 | 67 | 3.73 | 1.13 | 1.02 |
| Jamming | 0 | 4 | 171 | 60 | 5.74 | 1.64 | 3.37 |
| Spoofing | 3 | 5 | 140 | 66 | 6.30 | 4.18 | 6.71 |

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
