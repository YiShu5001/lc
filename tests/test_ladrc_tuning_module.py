from __future__ import annotations

import unittest
from pathlib import Path

from control.Tuning_ladrc import (
    ManualTargetProfile,
    build_manual_reference_profile,
    build_single_axis_ladrc_bundle,
    load_axis_parameter_file,
    run_b0_step_sweep,
    run_k_step_sweep,
    run_x_axis_disturbance_rejection_refined_tuning,
    run_x_axis_disturbance_rejection_tuning,
    run_x_axis_r_balance_scan,
    run_x_axis_disturbed_rescan,
    run_x_axis_fast_task_tuning,
    run_x_axis_steady_tuning,
    run_wc_step_sweep,
    run_x_axis_refined_tuning,
    run_z_axis_specialized_tuning,
)
from lc.control.configs import PyBulletControlExperimentConfig


class TestLADRCTuningModule(unittest.TestCase):
    def setUp(self) -> None:
        self.parameter_file = Path("src/control/Tuning_ladrc/default_axis_params.json")

    def test_parameter_file_loads_xyz_axes(self) -> None:
        params = load_axis_parameter_file(self.parameter_file)
        self.assertEqual(set(params.keys()), {"x", "y", "z"})
        self.assertGreater(params["x"].b0, 0.0)
        self.assertGreater(params["z"].wc, 0.0)

    def test_single_axis_bundle_uses_loaded_parameters(self) -> None:
        bundle = build_single_axis_ladrc_bundle("x", self.parameter_file)
        snapshot = bundle.snapshot_params()
        self.assertAlmostEqual(snapshot["x_b0"], 30.5, places=6)
        self.assertAlmostEqual(snapshot["x_omega_c"], 1.5, places=6)
        self.assertAlmostEqual(bundle.parameter_set.x.r, 12.0, places=6)

    def test_hold_step_hold_profile_only_changes_selected_axis(self) -> None:
        cfg = PyBulletControlExperimentConfig(duration_sec=6.0)
        profile = ManualTargetProfile(axis="x", mode="hold_step_hold", step_value=0.2, total_duration=6.0)
        bundle = build_manual_reference_profile(profile, control_dt=cfg.control_dt, step_count=cfg.step_count)
        self.assertAlmostEqual(float(bundle.positions[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(bundle.positions[0, 2]), 1.0, places=6)
        self.assertAlmostEqual(float(bundle.positions[-1, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(bundle.positions[-1, 2]), 1.0, places=6)
        self.assertGreater(float(bundle.positions[-1, 0]), 0.0)

    def test_steady_profiles_build_expected_reference(self) -> None:
        cfg = PyBulletControlExperimentConfig(duration_sec=6.0)
        hold_profile = ManualTargetProfile(axis="x", mode="x_hold_disturbance_hold", total_duration=6.0)
        hold_bundle = build_manual_reference_profile(hold_profile, control_dt=cfg.control_dt, step_count=cfg.step_count)
        self.assertAlmostEqual(float(hold_bundle.positions[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(hold_bundle.positions[-1, 0]), 0.0, places=6)
        small_step_profile = ManualTargetProfile(axis="x", mode="x_small_step_hold", step_value=0.03, total_duration=6.0)
        small_step_bundle = build_manual_reference_profile(small_step_profile, control_dt=cfg.control_dt, step_count=cfg.step_count)
        self.assertGreater(float(small_step_bundle.positions[len(small_step_bundle.positions) // 2, 0]), 0.0)

    def test_b0_step_sweep_returns_recommended_candidate(self) -> None:
        result = run_b0_step_sweep(
            axis="x",
            parameter_file=self.parameter_file,
            b0_candidates=[160.0, 220.0],
            profile=ManualTargetProfile(axis="x", mode="hold_step_hold", step_value=0.08, total_duration=3.0),
            config=PyBulletControlExperimentConfig(duration_sec=3.0, eval_episodes=1),
        )
        self.assertIn(result.recommended_b0, {160.0, 220.0})
        self.assertEqual(len(result.sweep_rows), 2)

    def test_wc_step_sweep_returns_recommended_candidate(self) -> None:
        result = run_wc_step_sweep(
            axis="x",
            parameter_file=self.parameter_file,
            fixed_b0=160.0,
            fixed_k=4.0,
            wc_candidates=[1.0, 6.0],
            profile=ManualTargetProfile(axis="x", mode="hold_step_hold", step_value=0.08, total_duration=3.0),
            config=PyBulletControlExperimentConfig(duration_sec=3.0, eval_episodes=1),
        )
        self.assertIn(result.recommended_wc, {1.0, 6.0})
        self.assertEqual(len(result.sweep_rows), 2)

    def test_k_step_sweep_returns_recommended_candidate(self) -> None:
        result = run_k_step_sweep(
            axis="x",
            parameter_file=self.parameter_file,
            fixed_b0=31.0,
            fixed_wc=2.0,
            fixed_r=10.0,
            k_candidates=[2.0, 4.0],
            profile=ManualTargetProfile(axis="x", mode="hold_step_hold", step_value=0.08, total_duration=3.0),
            config=PyBulletControlExperimentConfig(duration_sec=3.0, eval_episodes=1),
        )
        self.assertIn(result.recommended_wc, {2.0, 4.0})
        self.assertEqual(len(result.sweep_rows), 2)

    def test_z_axis_specialized_tuning_returns_candidates(self) -> None:
        result = run_z_axis_specialized_tuning(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=3.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_z_specialized",
        )
        self.assertGreater(len(result.b0_rows), 0)
        self.assertGreater(len(result.wc_rows), 0)
        self.assertGreater(len(result.k_rows), 0)
        self.assertGreater(result.recommended_b0, 0.0)

    def test_x_axis_refined_tuning_writes_comparison_artifacts(self) -> None:
        result = run_x_axis_refined_tuning(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=3.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_x_refined",
        )
        self.assertGreater(len(result.stage_a_rows), 0)
        self.assertGreater(len(result.stage_b_rows), 0)
        self.assertGreater(len(result.stage_c_rows), 0)
        self.assertIn("beats_pid", result.recommended_params)
        self.assertIn("beats_current_ladrc", result.recommended_params)

    def test_x_axis_steady_tuning_writes_ranges(self) -> None:
        result = run_x_axis_steady_tuning(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=6.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_x_steady",
            wc_candidates=[0.8, 1.0],
            k_candidates=[6.0, 8.0],
            b0_candidates=[30.0, 30.5],
        )
        self.assertGreater(len(result.stage_a_rows), 0)
        self.assertGreater(len(result.stage_b_rows), 0)
        self.assertGreater(len(result.stage_c_rows), 0)
        self.assertIn("b0", result.rl_ranges)
        self.assertIn("beats_reference", result.comparison_against_fast_x)

    def test_x_axis_disturbed_rescan_returns_three_stages(self) -> None:
        result = run_x_axis_disturbed_rescan(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=6.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_x_disturbed_rescan",
            b0_candidates=[1.0, 11.0, 21.0],
            wc_candidates=[1.0, 6.0],
            k_candidates=[1.0, 2.0, 3.0],
        )
        self.assertGreater(len(result.b0_rows), 0)
        self.assertGreater(len(result.wc_rows), 0)
        self.assertGreater(len(result.k_rows), 0)
        self.assertGreater(result.recommended_b0, 0.0)

    def test_x_axis_r_balance_scan_returns_recommended_r(self) -> None:
        result = run_x_axis_r_balance_scan(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=4.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_x_r_balance",
            r_candidates=[8.0, 12.0],
        )
        self.assertIn(result.recommended_r, {8.0, 12.0})
        self.assertGreater(len(result.sweep_rows), 0)
        self.assertIn("r", result.rl_ranges)

    def test_x_axis_fast_task_tuning_writes_outputs(self) -> None:
        result = run_x_axis_fast_task_tuning(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=4.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_x_fast_task",
            fixed_r=10.0,
        )
        self.assertEqual(result.task_type, "fast_tracking")
        self.assertIn("b0", result.recommended_params)
        self.assertGreater(len(result.local_rows), 0)

    def test_x_axis_disturbance_rejection_tuning_writes_outputs(self) -> None:
        result = run_x_axis_disturbance_rejection_tuning(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=4.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_x_disturbance_rejection",
            fixed_r=10.0,
        )
        self.assertEqual(result.task_type, "disturbance_rejection")
        self.assertIn("k", result.recommended_params)
        self.assertGreater(len(result.local_rows), 0)

    def test_x_axis_disturbance_rejection_refined_tuning_writes_outputs(self) -> None:
        result = run_x_axis_disturbance_rejection_refined_tuning(
            self.parameter_file,
            config=PyBulletControlExperimentConfig(duration_sec=4.0, eval_episodes=1),
            output_root="outputs/control_pybullet_manual_tuning/test_x_disturbance_rejection_refined",
            fixed_r=10.0,
            b0_candidates=[1.0, 2.0],
            wc_candidates=[3.0, 5.0],
            k_candidates=[4.0, 5.0],
        )
        self.assertGreater(len(result.b0_rows), 0)
        self.assertGreater(len(result.wc_rows), 0)
        self.assertGreater(len(result.k_rows), 0)
        self.assertIn("b0", result.recommended_params)
        self.assertIn("steady_state_error_delta", result.comparison_against_current)


if __name__ == "__main__":
    unittest.main()
