from __future__ import annotations

import unittest

import numpy as np

from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.reference_generators import build_xyz_reference_trajectory


class TestChapter3PyBulletReference(unittest.TestCase):
    def test_axis_reference_changes_only_on_primary_axis(self) -> None:
        config = PyBulletControlExperimentConfig(duration_sec=2.0, control_freq_hz=20)
        bundle = build_xyz_reference_trajectory(config.axis_config("x"), config, rng=np.random.default_rng(1))
        self.assertFalse(np.allclose(bundle.positions[:, 0], bundle.positions[0, 0]))
        self.assertTrue(np.allclose(bundle.positions[:, 1], bundle.positions[0, 1]))
        self.assertTrue(np.allclose(bundle.positions[:, 2], bundle.positions[0, 2]))

    def test_reference_velocity_integrates_to_position(self) -> None:
        config = PyBulletControlExperimentConfig(duration_sec=2.0, control_freq_hz=20)
        bundle = build_xyz_reference_trajectory(config.axis_config("y"), config, rng=np.random.default_rng(2))
        delta = np.diff(bundle.positions[:, 1])
        expected = bundle.velocities[:-1, 1] * config.control_dt
        self.assertTrue(np.allclose(delta, expected, atol=1e-4))

    def test_four_segments_exist(self) -> None:
        config = PyBulletControlExperimentConfig(duration_sec=2.0, control_freq_hz=20)
        bundle = build_xyz_reference_trajectory(config.axis_config("z"), config, rng=np.random.default_rng(3))
        self.assertEqual(len(bundle.stage_slices), 4)
        self.assertEqual(len(bundle.stage_velocities), 4)


if __name__ == "__main__":
    unittest.main()
