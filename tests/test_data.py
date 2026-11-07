from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lanechange.data import get_observation_window, load_sample_trajectories
from lanechange.synthetic import generate_synthetic_csv


class PublicSampleReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _ngsim(self):
        samples = []
        for epoch in range(2):
            for vehicle in range(3):
                frame = np.arange(80)
                samples.append(pd.DataFrame({
                    "Vehicle_ID": vehicle, "Frame_ID": frame,
                    "Global_Time": 1_118_000_000_000 + epoch * 100_000 + frame * 100,
                    "Local_X": 10 + frame * .1, "Local_Y": frame * 10 + vehicle * 1000,
                    "v_Vel": 100., "v_Acc": 0., "Lane_ID": 2, "Location": "demo_site",
                }))
        source = self.root / "ngsim_sample.csv"
        pd.concat(samples, ignore_index=True).to_csv(source, index=False)
        return source

    def test_ngsim_units_coordinates_and_recording_isolation(self):
        data = load_sample_trajectories(self._ngsim(), "ngsim")
        self.assertEqual(data.groupby(["recording_id", "vehicle_id"]).ngroups, 6)
        np.testing.assert_allclose(data.vx, 30.48)
        self.assertAlmostEqual(float(data.iloc[0].y), -3.048)
        with self.assertRaisesRegex(ValueError, "multiple recordings"):
            get_observation_window(data, vehicle_id="0")
        recording = data.recording_id.iloc[0]
        sample = get_observation_window(data, recording_id=recording, vehicle_id="2")
        self.assertEqual(sample.x.shape, (40, 6))
        self.assertEqual(sample.vehicle_id, "2")
        self.assertEqual(sample.recording_id, recording)
        np.testing.assert_allclose(sample.x[-1, :2], 0)
        np.testing.assert_allclose(np.diff(sample.x[:, 0]), 30.48 * .04, atol=2e-6)
        np.testing.assert_allclose(np.diff(sample.x[:, 1]), -.3048 * .04, atol=2e-6)
        self.assertTrue(sample.time_valid.all())

    def test_continuity_rejects_history_across_missing_frame(self):
        source = generate_synthetic_csv(self.root / "demo.csv", n_vehicles=3, frames=60)
        data = pd.read_csv(source)
        data = data[data.frame != 30]
        data.to_csv(source, index=False)
        loaded = load_sample_trajectories(source)
        with self.assertRaisesRegex(ValueError, "No complete continuous"):
            get_observation_window(loaded)
        sample = get_observation_window(loaded, steps=20)
        self.assertLess(float(sample.time_s[-1]), 30 * .04)

    def test_finiteness_and_duplicate_timestamps_are_rejected(self):
        source = generate_synthetic_csv(self.root / "demo.csv", n_vehicles=3)
        raw = pd.read_csv(source)
        invalid = raw.copy()
        invalid.loc[0, "vx"] = np.inf
        invalid.to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_sample_trajectories(source)
        pd.concat([raw, raw.iloc[:1]], ignore_index=True).to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_sample_trajectories(source)

    def test_reader_and_fixed_history_do_not_use_later_records(self):
        source = self._ngsim()
        before = load_sample_trajectories(source, "ngsim")
        sample = get_observation_window(before)
        raw = pd.read_csv(source)
        future = raw.Frame_ID > 16
        raw.loc[future, "Local_X"] += 100
        raw.loc[future, "Local_Y"] += 100
        raw.to_csv(source, index=False)
        after = load_sample_trajectories(source, "ngsim")
        changed = get_observation_window(after, recording_id=sample.recording_id, vehicle_id=sample.vehicle_id, end_time=sample.time_s[-1])
        np.testing.assert_array_equal(sample.x, changed.x)

    def test_synthetic_example_is_observations_only(self):
        source = generate_synthetic_csv(self.root / "independent_demo.csv", n_vehicles=3)
        data = load_sample_trajectories(source)
        sample = get_observation_window(data)
        self.assertEqual(sample.recording_id, "synthetic_demo")
        self.assertEqual(sample.x.dtype, np.float32)
        self.assertFalse(hasattr(sample, "y"))
        self.assertFalse(hasattr(sample, "future_valid"))

    def _highd(self, metadata=True):
        rows = []
        for vehicle, direction in ((1, -1), (2, 1)):
            frame = np.arange(60)
            rows.append(pd.DataFrame({
                "id": vehicle, "frame": frame,
                "x": vehicle * 100 + direction * frame * 1.2,
                "y": 10., "width": 4., "height": 2.,
                "xVelocity": direction * 30., "yVelocity": .4,
                "xAcceleration": direction * 2., "yAcceleration": .5,
                "laneId": 2 if direction < 0 else 5,
            }))
        source = self.root / "01_tracks.csv"
        pd.concat(rows, ignore_index=True).to_csv(source, index=False)
        if metadata:
            pd.DataFrame({"id": [1, 2], "drivingDirection": [1, 2]}).to_csv(self.root / "01_tracksMeta.csv", index=False)
        return source

    def test_highd_centres_directions_default_rate_and_vehicle_isolation(self):
        source = self._highd()
        data = load_sample_trajectories(source, "highd")
        left = data[data.vehicle_id == "1"]
        right = data[data.vehicle_id == "2"]
        self.assertAlmostEqual(float(left.iloc[0].x), -102.)
        self.assertAlmostEqual(float(left.iloc[0].y), 11.)
        self.assertAlmostEqual(float(right.iloc[0].x), 202.)
        self.assertAlmostEqual(float(right.iloc[0].y), -11.)
        np.testing.assert_allclose(data.vx, 30.)
        np.testing.assert_allclose(data.ax, 2.)
        np.testing.assert_allclose(left.vy, .4)
        np.testing.assert_allclose(right.vy, -.4)
        np.testing.assert_allclose(np.diff(left.time_s), .04)
        sample = get_observation_window(data, recording_id="01", vehicle_id="2")
        self.assertEqual(sample.vehicle_id, "2")
        self.assertEqual(sample.x.shape, (40, 6))
        self.assertEqual(sample.x[-1, 0], 0)
        self.assertGreater(float(sample.origin[0]), 200)

    def test_highd_velocity_fallback_uses_current_row_and_rejects_zero(self):
        source = self._highd(metadata=False)
        before = load_sample_trajectories(source, "highd")
        sample = get_observation_window(before, vehicle_id="1")
        raw = pd.read_csv(source)
        raw.loc[raw.frame > 39, "xVelocity"] *= -1
        raw.to_csv(source, index=False)
        after = load_sample_trajectories(source, "highd")
        changed = get_observation_window(after, vehicle_id="1", end_time=sample.time_s[-1])
        np.testing.assert_array_equal(sample.x, changed.x)
        raw.loc[0, "xVelocity"] = 0
        raw.to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "zero/nonfinite"):
            load_sample_trajectories(source, "highd")

    def test_highd_recording_rate_and_stationary_metadata(self):
        source = self._highd()
        pd.DataFrame({"frameRate": [10.]}).to_csv(self.root / "01_recordingMeta.csv", index=False)
        raw = pd.read_csv(source)
        raw.loc[0, "xVelocity"] = 0
        raw.to_csv(source, index=False)
        data = load_sample_trajectories(source, "highd")
        np.testing.assert_allclose(np.diff(data[data.vehicle_id == "1"].time_s), .1)
        self.assertAlmostEqual(float(data.iloc[0].x), -102)


if __name__ == "__main__":
    unittest.main()
