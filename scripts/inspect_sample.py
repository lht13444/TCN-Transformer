"""Inspect trajectory fields and one fixed historical observation window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lanechange.data import FEATURE_NAMES, get_observation_window, load_sample_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=["canonical", "ngsim", "highd"], default="canonical")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--dt", type=float, default=.04)
    parser.add_argument("--recording-id")
    parser.add_argument("--vehicle-id")
    parser.add_argument("--end-time", type=float)
    args = parser.parse_args()
    data = load_sample_trajectories(args.input, args.format)
    sample = get_observation_window(data, args.steps, args.dt, args.recording_id, args.vehicle_id, args.end_time)
    print(json.dumps({
        "purpose": "historical inputs for standalone feature-module demonstration",
        "rows": len(data), "recording_vehicle_groups": data.groupby(["recording_id", "vehicle_id"]).ngroups,
        "recording_id": sample.recording_id, "vehicle_id": sample.vehicle_id,
        "input_shape": list(sample.x.shape), "feature_names": FEATURE_NAMES,
        "units": data.attrs["units"], "coordinates": data.attrs["coordinates"],
        "observation_start_s": float(sample.time_s[0]), "observation_end_s": float(sample.time_s[-1]),
        "valid_steps": int(sample.time_valid.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
