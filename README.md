# uav-gnss-triage

Evidence-bounded GNSS spoofing triage from recovered UAV flight logs: flight-level validation,
calibrated abstention, and the limits of post-incident attribution.

Sole author: Md Anas Biswas, University of Portsmouth.

## Layout

- `src/` pipeline modules and command-line stages
  - `sih_spoof_patch.py` patches PX4's `sensor_gps_sim` with a parameterised GNSS spoof injector and a receiver-realism noise model
  - `sih_generate_flights.py` generates independent labelled flights in PX4 SIH (nominal, spoof, GNSS degradation, non-GNSS sensor fault)
  - `sih_features.py` parses ULogs into physics-consistency window features with onset-aware labels
  - `sih_model.py` window detector, temperature scaling, flight-level class-conditional conformal sets, leakage audit, leave-one-subtype-out, external case studies
- `notebooks/` numbered Colab notebooks that run the stages in order
- `reports/` committed result tables, one folder per run
- `figures/` committed figures
- `data/` (not committed) generated flights, feature tables, simulator cache, external datasets
- `tools/commit_cell.py` the single commit step used by the scratch console notebook

## Data

- Generated PX4 SIH flights: `data/sih/<run>/manifest.jsonl` with one `.ulg` and `.json` per flight
- External reference cases: Whelan et al. 2020, UAV Attack Dataset, IEEE DataPort, DOI 10.21227/00dg-0d12
