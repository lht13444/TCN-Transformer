"""Read trajectory CSVs in consistent SI coordinates and select an observation window."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = ["recording_id", "vehicle_id", "frame", "time_s", "x", "y", "vx", "vy", "ax", "ay"]
FEATURE_NAMES = ["relative_x", "relative_y", "vx", "vy", "ax", "ay"]
FT_TO_M = 0.3048
HIGHD_SCHEMA_NOTE = (
    "highD: *_tracks.csv with optional matching *_tracksMeta.csv and "
    "*_recordingMeta.csv; positions are converted from bounding-box corners "
    "to vehicle centres, with forward x and driver-left y in metres."
)


@dataclass(frozen=True)
class SampleTrajectory:
    """One complete historical input window.

    ``x`` is float32 [steps, 6] in FEATURE_NAMES order. Positions are relative to
    the final observed position; velocities and accelerations remain SI values.
    ``time_valid`` is all true because incomplete windows are rejected.
    """

    x: np.ndarray
    time_valid: np.ndarray
    time_s: np.ndarray
    origin: np.ndarray
    recording_id: str
    vehicle_id: str

    @property
    def features(self) -> np.ndarray:
        return self.x


def _need(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _derivative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    """A backward difference uses observed records only; first value is zero."""
    result = np.zeros(len(values), dtype=float)
    if len(values) > 1:
        np.divide(np.diff(values), np.diff(time), out=result[1:], where=np.diff(time) > 0)
    return result


def _read_highd(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = ["id", "frame", "x", "y", "width", "height", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
    _need(raw, required, str(path))
    if raw["id"].isna().any():
        raise ValueError("highD vehicle IDs must not be missing")
    recording = path.stem.removesuffix("_tracks")
    fps = 25.0
    recording_meta = path.with_name(recording + "_recordingMeta.csv")
    if recording_meta.is_file():
        meta = pd.read_csv(recording_meta)
        _need(meta, ["frameRate"], str(recording_meta))
        if meta.empty:
            raise ValueError(f"{recording_meta}: empty recording metadata")
        fps = float(meta.iloc[0]["frameRate"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{recording_meta}: frameRate must be finite and positive")
    track_meta = path.with_name(recording + "_tracksMeta.csv")
    codes = pd.Series(np.nan, index=raw.index)
    if track_meta.is_file():
        meta = pd.read_csv(track_meta)
        _need(meta, ["id", "drivingDirection"], str(track_meta))
        if meta["id"].duplicated().any():
            raise ValueError(f"{track_meta}: duplicate vehicle IDs")
        codes = raw["id"].map(meta.set_index("id")["drivingDirection"])
        if (codes.notna() & ~codes.isin([1, 2])).any():
            raise ValueError(f"{track_meta}: drivingDirection must be 1 (leftwards) or 2 (rightwards)")
    source_velocity = pd.to_numeric(raw["xVelocity"], errors="raise").to_numpy(float)
    signs = np.sign(source_velocity)
    provided = codes.notna().to_numpy()
    signs[provided] = np.where(codes.to_numpy()[provided] == 1, -1., 1.)
    unknown = ~provided & (~np.isfinite(source_velocity) | (np.abs(source_velocity) <= 1e-8))
    if unknown.any():
        raise ValueError("Cannot infer highD driving direction from zero/nonfinite xVelocity; supply matching tracksMeta drivingDirection")
    # Without metadata, each row uses its own observed velocity sign. No later
    # trajectory frame is consulted to orient a stationary/ambiguous row.
    centre_x = raw["x"].to_numpy(float) + raw["width"].to_numpy(float) / 2
    centre_y = raw["y"].to_numpy(float) + raw["height"].to_numpy(float) / 2
    result = pd.DataFrame({
        "recording_id": recording, "vehicle_id": raw["id"].astype(str),
        "frame": raw["frame"].to_numpy(), "time_s": raw["frame"].to_numpy(float) / fps,
        "x": signs * centre_x, "y": -signs * centre_y,
        "vx": signs * source_velocity, "vy": -signs * raw["yVelocity"].to_numpy(float),
        "ax": signs * raw["xAcceleration"].to_numpy(float), "ay": -signs * raw["yAcceleration"].to_numpy(float),
    })
    if "laneId" in raw:
        result["lane_id"] = raw["laneId"].to_numpy()
    return result


def _read_ngsim(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    official = ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "Lane_ID", "v_Vel", "v_Acc", "Location"]
    case_map = {name.casefold(): name for name in official}
    raw = raw.rename(columns={name: case_map[name.casefold()] for name in raw.columns if name.casefold() in case_map})
    _need(raw, ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y"], source)
    times_ms = pd.to_numeric(raw["Global_Time"], errors="raise")
    frames = pd.to_numeric(raw["Frame_ID"], errors="raise")
    if not np.isfinite(np.column_stack((times_ms, frames))).all():
        raise ValueError("NGSIM source timestamps and frame IDs must be finite")
    if raw["Vehicle_ID"].isna().any():
        raise ValueError("NGSIM vehicle IDs must not be missing")
    location = raw["Location"].fillna("unknown").astype(str) if "Location" in raw else pd.Series("unknown", index=raw.index)
    if "recording_id" in raw:
        if raw["recording_id"].isna().any():
            raise ValueError("NGSIM recording IDs must not be missing")
        recording = location + ":" + raw["recording_id"].astype(str)
    else:
        # Original NGSIM has 100 ms source frames. This acquisition identifier
        # disambiguates vehicle IDs reused when recording frame counters restart.
        epoch = np.rint((times_ms.to_numpy(float) - frames.to_numpy(float) * 100) / 100).astype(np.int64) * 100
        recording = location + ":epoch_ms=" + pd.Series(epoch, index=raw.index).astype(str)
    first_time = times_ms.groupby(recording).transform("min")
    work = raw.assign(_recording=recording, _time=(times_ms - first_time) / 1000)
    result = []
    for (recording_id, vehicle_id), group in work.groupby(["_recording", "Vehicle_ID"], sort=True):
        group = group.sort_values("_time")
        time = group["_time"].to_numpy(float)
        # Original Local_Y points longitudinally forward; Local_X points right.
        x = group["Local_Y"].to_numpy(float) * FT_TO_M
        y = -group["Local_X"].to_numpy(float) * FT_TO_M
        vx = group["v_Vel"].to_numpy(float) * FT_TO_M if "v_Vel" in group else np.zeros(len(group))
        vy = np.zeros(len(group))
        ax = group["v_Acc"].to_numpy(float) * FT_TO_M if "v_Acc" in group else np.zeros(len(group))
        ay = np.zeros(len(group))
        broken = np.r_[True, (np.diff(group["Frame_ID"].to_numpy(float)) != 1) | (np.diff(time) > .15) | (np.diff(time) <= 0)]
        bounds = np.r_[np.flatnonzero(broken), len(group)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            span = slice(start, end)
            vy[span] = _derivative(y[span], time[span])
            ay[span] = _derivative(vy[span], time[span])
            if "v_Vel" not in group:
                vx[span] = _derivative(x[span], time[span])
            if "v_Acc" not in group:
                ax[span] = _derivative(vx[span], time[span])
        converted = pd.DataFrame({
            "recording_id": str(recording_id), "vehicle_id": str(vehicle_id),
            "frame": group["Frame_ID"].to_numpy(), "time_s": time,
            "x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay,
        })
        if "Lane_ID" in group:
            converted["lane_id"] = group["Lane_ID"].to_numpy()
        result.append(converted)
    if not result:
        raise ValueError("The NGSIM sample contains no vehicle trajectories")
    return pd.concat(result, ignore_index=True)


def load_sample_trajectories(input_path: str | Path, format: str = "canonical") -> pd.DataFrame:
    """Read one CSV in canonical SI, highD, or original NGSIM format.

    Original NGSIM fields use feet, milliseconds, and 10 Hz source frames;
    positions are converted to forward-x/driver-left-y metres. time_s is relative
    to the first loaded timestamp of each recording. Metadata/lane columns may
    be retained for inspection but are never module input features.
    highD uses vehicle centres and optional drivingDirection/frameRate metadata;
    its source cadence defaults to 25 Hz. With no drivingDirection metadata,
    each source frame's own nonzero xVelocity determines its coordinate sign.
    """
    if format not in {"canonical", "ngsim", "highd"}:
        raise ValueError("Supported formats are canonical, highd, and ngsim")
    path = Path(input_path)
    if not path.is_file():
        raise ValueError(f"Sample CSV does not exist: {path}")
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if format == "ngsim":
        result = _read_ngsim(raw, str(path))
    elif format == "highd":
        result = _read_highd(raw, path)
    else:
        result = raw.copy()
    _need(result, CANONICAL_COLUMNS, str(path))
    if result.empty:
        raise ValueError("The sample CSV is empty")
    for name in ("recording_id", "vehicle_id"):
        if result[name].isna().any():
            raise ValueError("Recording and vehicle IDs must not be missing")
        result[name] = result[name].astype(str)
    numeric = ["frame", "time_s", "x", "y", "vx", "vy", "ax", "ay"]
    for name in numeric:
        result[name] = pd.to_numeric(result[name], errors="raise")
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError("Sample coordinates, kinematics and timestamps must be finite")
    if not np.equal(result["frame"], np.floor(result["frame"])).all():
        raise ValueError("frame must contain integer source-frame indices")
    if result.duplicated(["recording_id", "vehicle_id", "time_s"]).any():
        raise ValueError("Duplicate timestamps within a recording/vehicle trajectory")
    result = result.sort_values(["recording_id", "vehicle_id", "time_s"]).reset_index(drop=True)
    result.attrs["source_format"] = format
    result.attrs["units"] = "m, m/s, m/s^2, s"
    result.attrs["coordinates"] = "x longitudinal forward; y driver-left"
    return result


def get_observation_window(data: pd.DataFrame, steps: int = 40, dt: float = .04, recording_id: str | None = None, vehicle_id: str | None = None, end_time: float | None = None) -> SampleTrajectory:
    """Select a single complete continuous history from one trajectory.

    The default returns the earliest available history of the first eligible
    recording/vehicle. ``end_time``, when supplied, must coincide with a real
    source frame. Interpolation only uses records at or before that anchor,
    so 10-to-25 Hz conversion does not draw on future observations. Gaps split
    source tracks; this function never interpolates across a missing frame.
    """
    if not isinstance(steps, int) or steps < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require integer steps>=2 and a finite positive dt")
    _need(data, CANONICAL_COLUMNS, "sample DataFrame")
    chosen = data
    if recording_id is not None:
        chosen = chosen[chosen["recording_id"].astype(str) == str(recording_id)]
    if vehicle_id is not None:
        chosen = chosen[chosen["vehicle_id"].astype(str) == str(vehicle_id)]
    if chosen.empty:
        raise ValueError("The requested recording/vehicle is absent from the sample")
    if vehicle_id is not None and recording_id is None and chosen["recording_id"].nunique() > 1:
        raise ValueError("vehicle_id occurs in multiple recordings; specify recording_id")
    for (recording, vehicle), group in chosen.groupby(["recording_id", "vehicle_id"], sort=True):
        group = group.sort_values("time_s")
        time = group["time_s"].to_numpy(float)
        values = group[["x", "y", "vx", "vy", "ax", "ay"]].to_numpy(float)
        if not np.isfinite(values).all() or not np.isfinite(time).all():
            raise ValueError("Observation-window values must be finite")
        delta = np.diff(time)
        cadence = .1 if data.attrs.get("source_format") == "ngsim" else (float(np.median(delta[delta > 0])) if np.any(delta > 0) else 0)
        gaps = (np.diff(group["frame"].to_numpy(float)) != 1) | (delta <= 0)
        if cadence:
            gaps |= delta > cadence * 1.5 + 1e-8
        bounds = np.r_[0, np.flatnonzero(gaps) + 1, len(group)]
        for start, stop in zip(bounds[:-1], bounds[1:]):
            for index in range(start, stop):
                anchor = float(time[index])
                if end_time is not None and not np.isclose(anchor, end_time, rtol=0, atol=1e-7):
                    continue
                if anchor - (steps - 1) * dt < time[start] - 1e-8:
                    continue
                grid = anchor - np.arange(steps - 1, -1, -1) * dt
                observed = np.column_stack([np.interp(grid, time[start:index + 1], values[start:index + 1, col]) for col in range(6)])
                origin = values[index, :2].copy()
                observed[:, :2] -= origin
                return SampleTrajectory(observed.astype(np.float32), np.ones(steps, dtype=bool), grid.astype(np.float64), origin.astype(np.float32), str(recording), str(vehicle))
    raise ValueError("No complete continuous observation window at the requested anchor; provide a longer contiguous trajectory or fewer demonstration steps")


# Read-only compatibility alias.
load_trajectories = load_sample_trajectories
