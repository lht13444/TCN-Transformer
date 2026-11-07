"""Independent synthetic SI trajectories for exercising the software pipeline.

These are labelled demonstrations, not highD data or evidence for paper metrics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_csv(path: str | Path, n_vehicles: int = 30, frames: int = 180, dt: float = .04, seed: int = 42) -> Path:
    """Write left/right/keep-lane canonical examples, one independent track each."""
    if n_vehicles < 3 or frames < 2 or dt <= 0:
        raise ValueError("Require at least 3 vehicles, 2 frames, and positive dt")
    rng = np.random.default_rng(seed)
    t = np.arange(frames) * dt
    result = []
    for vehicle in range(n_vehicles):
        direction = (1, -1, 0)[vehicle % 3]
        onset = rng.uniform(.25, .4) * t[-1]
        duration = rng.uniform(.3, .45) * t[-1]
        u = np.clip((t - onset) / duration, 0, 1)
        shape = 10 * u**3 - 15 * u**4 + 6 * u**5
        dy = direction * 3.6 * shape
        speed = rng.uniform(22, 36)
        acceleration = rng.uniform(-.25, .25)
        x = rng.uniform(0, 100) + speed * t + .5 * acceleration * t**2
        y = rng.integers(0, 3) * 3.6 + dy
        vy = direction * 3.6 / duration * (30 * u**2 - 60 * u**3 + 30 * u**4)
        ay = direction * 3.6 / duration**2 * (60 * u - 180 * u**2 + 120 * u**3)
        lane = 3 + direction * (u >= .5).astype(int)
        result.append(pd.DataFrame({
            "recording_id": "synthetic_demo", "vehicle_id": f"demo_{vehicle:04d}",
            "frame": np.arange(frames), "time_s": t,
            "x": x, "y": y, "vx": speed + acceleration * t,
            "vy": vy, "ax": acceleration, "ay": ay, "lane_id": lane,
        }))
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(result, ignore_index=True).to_csv(target, index=False)
    return target
