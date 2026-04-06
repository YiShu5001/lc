from __future__ import annotations

import argparse

from lc.control.configs import ArtifactConfig, PyBulletControlExperimentConfig
from lc.control.experiments import (
    run_pybullet_axis_training,
    run_pybullet_controller_benchmark,
    run_pybullet_full_experiment,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chapter 3 PyBullet control experiments.")
    parser.add_argument("--mode", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--axis", choices=("x", "y", "z", "all"), default="all")
    parser.add_argument(
        "--controller",
        choices=("pid_pos_att", "ladrc_pos_pid_att", "ladrc_pos_att", "all"),
        default="all",
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI when the real PyBullet backend is available.")
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", default="outputs/control_pybullet")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = PyBulletControlExperimentConfig(
        gui=args.gui,
        duration_sec=args.duration_sec,
        seed=args.seed,
        artifact=ArtifactConfig(output_root=args.output_root),
    )
    if args.mode == "train":
        print(run_pybullet_axis_training(config, axis=args.axis))
    elif args.mode == "eval":
        print(run_pybullet_controller_benchmark(config, axis=args.axis, controller=args.controller))
    else:
        print(run_pybullet_full_experiment(config))
