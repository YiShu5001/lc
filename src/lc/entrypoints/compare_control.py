from __future__ import annotations

import argparse

from lc.control.configs import ArtifactConfig, PyBulletControlExperimentConfig
from lc.control.experiments import run_pybullet_controller_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chapter 3 control comparison in the PyBullet environment.")
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z", "all"),
        default="all",
        help="Axis to evaluate in PyBullet.",
    )
    parser.add_argument(
        "--controller",
        choices=("pid_pos_att", "ladrc_pos_pid_att", "ladrc_pos_att", "ladrc_x_pos_pid_att", "ladrc_y_pos_pid_att", "ladrc_z_pos_pid_att", "all"),
        default="all",
        help="Controller variant to benchmark in PyBullet.",
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI when the real PyBullet backend is available.")
    parser.add_argument("--duration-sec", type=float, default=8.0, help="Episode duration in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--output-root", default="outputs/control_pybullet", help="Output root for PyBullet artifacts.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = PyBulletControlExperimentConfig(
        gui=args.gui,
        duration_sec=args.duration_sec,
        seed=args.seed,
        artifact=ArtifactConfig(output_root=args.output_root),
    )
    print(run_pybullet_controller_benchmark(config, axis=args.axis, controller=args.controller))
