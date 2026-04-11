from __future__ import annotations

import json
import unittest
from pathlib import Path

from lc.control.configs import ArtifactConfig, PyBulletControlExperimentConfig
from lc.control.experiments import run_pid_vs_fixed_ladrc_tracking


class TestChapter3PidVsFixedLadrcCompare(unittest.TestCase):
    def test_x_axis_fixed_ladrc_uses_tuning_file_and_writes_time_response_figures(self) -> None:
        result = run_pid_vs_fixed_ladrc_tracking(
            PyBulletControlExperimentConfig(
                duration_sec=1.5,
                eval_episodes=1,
                artifact=ArtifactConfig(output_root="outputs/control_pybullet_test_compare"),
            ),
            axis="x",
        )
        output_dir = Path(result["output_dir"])
        expected = json.loads(Path("src/control/Tuning_ladrc/default_axis_params.json").read_text(encoding="utf-8"))["x"]
        self.assertAlmostEqual(result["fixed_ladrc_params"]["b0"], expected["b0"], places=6)
        self.assertAlmostEqual(result["fixed_ladrc_params"]["omega_c"], expected["wc"], places=6)
        self.assertAlmostEqual(result["fixed_ladrc_params"]["k"], expected["k"], places=6)
        self.assertTrue((output_dir / "summary.json").exists())
        self.assertTrue((output_dir / "metrics.csv").exists())
        self.assertTrue((output_dir / "figures" / "pid_vs_best_ladrc_time_response.png").exists())
        self.assertTrue((output_dir / "figures" / "pid" / "axis_error.png").exists())
        self.assertTrue((output_dir / "figures" / "ladrc" / "axis_error.png").exists())
        self.assertTrue((output_dir / "figures" / "pid" / "control_effort.png").exists())
        self.assertTrue((output_dir / "figures" / "ladrc" / "control_effort.png").exists())


if __name__ == "__main__":
    unittest.main()
