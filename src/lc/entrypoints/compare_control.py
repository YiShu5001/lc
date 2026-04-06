from __future__ import annotations

import argparse

from lc.control.configs import ControlExperimentConfig
from lc.control.experiments import run_control_comparison, run_control_generalization


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chapter 3 control experiments.")
    parser.add_argument("--difficulty", default="medium", help="Single comparison difficulty.")
    parser.add_argument(
        "--mode",
        choices=("compare", "generalization"),
        default="compare",
        help="Run one difficulty comparison or a multi-difficulty generalization sweep.",
    )
    parser.add_argument("--train-episodes", type=int, default=12, help="Training episodes for RL methods.")
    parser.add_argument("--compare-episodes", type=int, default=6, help="Evaluation episodes for RL methods.")
    parser.add_argument("--episodes", type=int, default=6, help="Evaluation episodes for PID/LADRC baselines.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--difficulty-levels",
        nargs="+",
        default=("easy", "medium", "hard", "extreme"),
        help="Difficulty sweep used in generalization mode.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = ControlExperimentConfig(
        difficulty=args.difficulty,
        difficulty_levels=tuple(args.difficulty_levels),
        train_episodes=args.train_episodes,
        compare_episodes=args.compare_episodes,
        episodes=args.episodes,
        seed=args.seed,
    )
    if args.mode == "generalization":
        print(run_control_generalization(config))
    else:
        print(run_control_comparison(config))
