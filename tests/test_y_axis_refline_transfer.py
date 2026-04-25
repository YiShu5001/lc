from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from lc.control.RLcontrolRefLine import build_default_xy_task_config, build_refline_episode
from lc.control.configs import AxisTransferExperimentConfig, ControlExperimentConfig
from lc.control.envs import ControlTrackingEnv
from lc.control.experiments import run_control_comparison, run_y_axis_refline_transfer_suite
from lc.envs.scenarios import build_control_scenario


class TestYAxisReflineTransfer(unittest.TestCase):
    def test_y_axis_refline_uses_roll_state(self) -> None:
        bundle = build_refline_episode(build_default_xy_task_config("y"), seed=19)
        env = ControlTrackingEnv(
            scenario=build_control_scenario("medium"),
            axis="y",
            seed=19,
            episode_length=len(bundle.time),
            reference_profile_mode="rl_refline_six_phase",
            external_episode_bundle=bundle,
        )
        obs = env.reset()
        self.assertEqual(obs.shape[0], 4)
        self.assertIn("roll", env.obs_spec.description)
        env.step(0.5, external_coupling=0.1)
        self.assertNotEqual(env.roll, 0.0)
        self.assertEqual(env.pitch, 0.0)

    def test_control_comparison_supports_y_axis_refline(self) -> None:
        subdir = "test_y_axis_rl_refline_protocol"
        out_dir = Path("outputs") / "control" / subdir
        if out_dir.exists():
            shutil.rmtree(out_dir)
        try:
            result = run_control_comparison(
                ControlExperimentConfig(
                    difficulty="easy",
                    axes=("y",),
                    reference_profile_mode="rl_refline_six_phase",
                    mddpg_shared_values=(1,),
                    train_episodes=1,
                    compare_episodes=1,
                    episodes=1,
                    seed_runs=1,
                    output_subdir=subdir,
                )
            )
            summary_path = Path(result["output_dir"]) / "summary.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue((Path(result["output_dir"]) / "v1" / "eval_timeseries.csv").exists())
            content = (Path(result["output_dir"]) / "v1" / "eval_timeseries.csv").read_text(encoding="utf-8")
            self.assertIn("roll", content.splitlines()[0])
        finally:
            if out_dir.exists():
                shutil.rmtree(out_dir)

    def test_transfer_suite_smoke_runs_with_existing_x_assets(self) -> None:
        source_dir = Path(
            "D:/ZhangC/lc_codex_ch4_run/outputs/control/"
            "x_axis_rl_refline__exp-bestcfg-scan-v1-to-v10__ep-500__v-1-10__noise-linear-0.1-to-0.04__net-768__drop-0.25"
        )
        if not source_dir.exists():
            self.skipTest("x-axis source artifacts are not available on this machine")
        subdir = "test_y_axis_transfer_suite_smoke"
        out_dir = Path("outputs") / "control" / subdir
        if out_dir.exists():
            shutil.rmtree(out_dir)
        try:
            result = run_y_axis_refline_transfer_suite(
                AxisTransferExperimentConfig(
                    baseline_episodes=1,
                    eval_episodes=1,
                    warm_start_episodes=1,
                    compare_shared_values=(1,),
                    reference_shared_value=1,
                    output_subdir=subdir,
                )
            )
            summary_path = Path(result["output_dir"]) / "summary.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue((Path(result["output_dir"]) / "rl_transfer_zero_shot_v1" / "eval_timeseries.csv").exists())
            self.assertTrue((Path(result["output_dir"]) / "rl_transfer_warm_start_v1" / "training_history.csv").exists())
        finally:
            if out_dir.exists():
                shutil.rmtree(out_dir)


if __name__ == "__main__":
    unittest.main()
