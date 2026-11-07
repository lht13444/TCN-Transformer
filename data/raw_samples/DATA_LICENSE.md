# NGSIM example data attribution and license

The `ngsim_us101_12vehicles.csv` and `ngsim_us101_12vehicles.json` files contain a small subset of **Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data**, published by the **U.S. Department of Transportation Federal Highway Administration** and provided by the **ITS Joint Program Office DataHub**.

- Official dataset: https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj
- Dataset DOI: https://doi.org/10.21949/1504477
- Official metadata API: https://data.transportation.gov/api/views/8ect-6jqj.json
- Accessed: 2026-11-07

The official metadata snapshot in `official_metadata.json` identifies the license as **Creative Commons Attribution ShareAlike 3.0 Unported**, with `licenseId` equal to `CC_30_BY_SA`. This data subset is provided under the same license.

- License deed: https://creativecommons.org/licenses/by-sa/3.0/
- Full legal terms: https://creativecommons.org/licenses/by-sa/3.0/legalcode

Retain attribution and the license reference when redistributing this data. Adapted data must be distributed under the same license, subject to the full legal terms. This data license applies to the data files and their adaptations; it does not assign a license to the independently written software in this project. No endorsement by USDOT or FHWA is implied. The data are provided without warranties; consult the official dataset documentation for source quality limitations.

Recommended source citation:

U.S. Department of Transportation Federal Highway Administration. (2016). *Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data*. [Dataset]. Provided by ITS DataHub through Data.transportation.gov. Accessed 2026-11-07 from https://doi.org/10.21949/1504477.

## Selection and changes

Only 12 complete US-101 vehicle trajectories were selected through the official Socrata API. All 6,783 API rows retain their source numerical values and field names. The local files were serialized as formatted JSON and CSV. No trajectory values were interpolated, smoothed, resampled or synthesized. `PROVENANCE.json` records the exact selection, API URLs, source units, temporal-continuity checks, lane-change indices and file hashes.

These are **NGSIM data collected on US-101**, not HighD data and not the experimental data used for the results reported in the manuscript. The subset supports small software examples only. It does not establish the accuracy reported in any paper.

## Fields and time resolution

The source is sampled at **10 Hz** (`global_time` increments of 100 ms). Therefore, 40 consecutive source frames correspond to 4 seconds; a 40-frame history plus a 40-frame future contains 80 frames. No conversion to HighD's 25 Hz frequency has been made.

| Field | Meaning and source unit |
|---|---|
| `vehicle_id` | Vehicle identifier; identifiers repeat across recording periods, so use the location and recording/time interval as well |
| `frame_id` | Frame index, incrementing by one in each retained trajectory |
| `global_time` | Timestamp in milliseconds, as defined by the official source |
| `local_x` | Lateral position of the vehicle front center, in feet, measured from the left edge in the direction of travel |
| `local_y` | Longitudinal position of the vehicle front center, in feet, measured from the entry edge |
| `v_vel` | Instantaneous speed in feet per second |
| `v_acc` | Instantaneous acceleration in feet per second squared |
| `lane_id` | Lane position; these examples stay in mainline lane IDs 1–5 |
| `v_length`, `v_width` | Vehicle dimensions in feet |
| `space_headway`, `time_headway` | Source headway values in feet and seconds |

For metric conversion, multiply feet-based distances, velocities and accelerations by exactly **0.3048**. Arterial-specific fields are retained as source `NA` values for US-101; they should not be treated as numerical zeroes. `global_x` and `global_y` are source map coordinates and are not needed to use the local trajectory examples.

To re-fetch the same fixed small subset, run `python scripts/fetch_ngsim_sample.py` from the project directory. The script does not download the full dataset or run a model.
