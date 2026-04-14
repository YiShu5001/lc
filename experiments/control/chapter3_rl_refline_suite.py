from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lc.control.configs import ControlExperimentConfig
from lc.control.experiments import run_control_comparison


@dataclass(frozen=True)
class SuiteJob:
    name: str
    purpose: str
    config: ControlExperimentConfig


BEST_CONFIG = {
    "difficulty": "medium",
    "axes": ("x",),
    "reference_profile_mode": "rl_refline_six_phase",
    "train_episodes": 500,
    "compare_episodes": 5,
    "episodes": 1,
    "seed_runs": 1,
    "hidden_dim": 768,
    "dropout_p": 0.25,
    "tau": 0.02,
    "soft_update_interval": 10,
    "snapshot_interval": 50,
    "exploration_noise_schedule": "linear",
    "exploration_noise_start": 0.1,
    "exploration_noise_end": 0.04,
}


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _suite_root(suite_name: str, tag: str) -> Path:
    return PROJECT_ROOT / "outputs" / "control" / f"chapter3_suite__{suite_name}__{tag}"


def _make_config(
    *,
    seed: int,
    best_v: int,
    output_subdir: str,
    **overrides: object,
) -> ControlExperimentConfig:
    payload = dict(BEST_CONFIG)
    payload.update(overrides)
    payload["seed"] = seed
    payload["mddpg_shared_values"] = (best_v,)
    payload["output_subdir"] = output_subdir
    return ControlExperimentConfig(**payload)


def build_ablation_jobs(tag: str, ablation_values: Iterable[int]) -> list[SuiteJob]:
    prefix = f"chapter3_ablation__{tag}"
    jobs: list[SuiteJob] = []
    for shared_value in ablation_values:
        label = "minimal-enhancement" if shared_value == 1 else f"shared-v{shared_value}"
        purpose = (
            "最小共享增强设定，作为不做样本增强/增强极弱的近似对照。"
            if shared_value == 1
            else f"共享增强值 v={shared_value}，用于检验增强样本窗口增大后的收益。"
        )
        jobs.append(
            SuiteJob(
                name=f"v_{shared_value}",
                purpose=purpose,
                config=_make_config(
                    seed=7,
                    best_v=shared_value,
                    output_subdir=f"{prefix}__{label}__v-{shared_value}__net-768__drop-0.25__tau-0.02__noise-linear",
                ),
            )
        )
    return jobs


def build_monte_carlo_jobs(best_v: int, tag: str, seeds: Iterable[int]) -> list[SuiteJob]:
    prefix = f"chapter3_monte_carlo__{tag}"
    return [
        SuiteJob(
            name=f"seed_{seed}",
            purpose=f"蒙特卡洛重复训练，随机种子 {seed}。",
            config=_make_config(
                seed=seed,
                best_v=best_v,
                output_subdir=f"{prefix}__seed-{seed}__v-{best_v}__net-768__drop-0.25__tau-0.02__noise-linear",
            ),
        )
        for seed in seeds
    ]


def build_generalization_jobs(best_v: int, tag: str) -> list[SuiteJob]:
    prefix = f"chapter3_generalization__{tag}"
    difficulties = ("easy", "medium", "hard", "extreme")
    seed_map = {"easy": 7, "medium": 8, "hard": 9, "extreme": 10}
    return [
        SuiteJob(
            name=f"difficulty_{difficulty}",
            purpose=f"难度泛化评估，训练与评估场景固定在 {difficulty}。",
            config=_make_config(
                seed=seed_map[difficulty],
                best_v=best_v,
                output_subdir=f"{prefix}__{difficulty}__v-{best_v}__net-768__drop-0.25__tau-0.02__noise-linear",
                difficulty=difficulty,
            ),
        )
        for difficulty in difficulties
    ]


def _run_jobs(suite_name: str, jobs: list[SuiteJob], dry_run: bool, tag: str) -> dict[str, object]:
    suite_root = _suite_root(suite_name, tag)
    suite_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for job in jobs:
        print(f"[suite:{suite_name}] {job.name}")
        print(f"  purpose: {job.purpose}")
        print(f"  output_subdir: {job.config.output_subdir}")
        if dry_run:
            manifest.append(
                {
                    "name": job.name,
                    "purpose": job.purpose,
                    "config": {
                        "difficulty": job.config.difficulty,
                        "seed": job.config.seed,
                        "mddpg_shared_values": list(job.config.mddpg_shared_values),
                        "hidden_dim": job.config.hidden_dim,
                        "dropout_p": job.config.dropout_p,
                        "tau": job.config.tau,
                        "soft_update_interval": job.config.soft_update_interval,
                        "exploration_noise_schedule": job.config.exploration_noise_schedule,
                        "exploration_noise_start": job.config.exploration_noise_start,
                        "exploration_noise_end": job.config.exploration_noise_end,
                        "output_subdir": job.config.output_subdir,
                    },
                }
            )
            continue
        result = run_control_comparison(job.config)
        manifest.append(
            {
                "name": job.name,
                "purpose": job.purpose,
                "output_dir": result["output_dir"],
                "results": result["results"],
                "figures": result["figures"],
            }
        )
    summary_path = suite_root / "manifest.json"
    summary_path.write_text(json.dumps({"suite": suite_name, "jobs": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[suite:{suite_name}] manifest: {summary_path}")
    return {"suite": suite_name, "manifest": str(summary_path), "jobs": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="External chapter 3 RL refline experiment suite runner.")
    parser.add_argument("--suite", choices=("ablation", "monte_carlo", "generalization", "all"), default="all")
    parser.add_argument("--best-v", type=int, default=7, help="Best shared value to freeze for all three experiment groups.")
    parser.add_argument("--tag", default=_timestamp_tag(), help="Unique output tag to avoid overwriting prior runs.")
    parser.add_argument(
        "--monte-carlo-seeds",
        nargs="+",
        type=int,
        default=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        help="Seed list for Monte Carlo repetitions.",
    )
    parser.add_argument(
        "--ablation-values",
        nargs="+",
        type=int,
        default=(1, 7),
        help="Shared values used in the ablation suite, e.g. 1 7.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print job plan and write a manifest without running.")
    args = parser.parse_args()

    suites: list[tuple[str, list[SuiteJob]]] = []
    if args.suite in {"ablation", "all"}:
        suites.append(("ablation", build_ablation_jobs(args.tag, args.ablation_values)))
    if args.suite in {"monte_carlo", "all"}:
        suites.append(("monte_carlo", build_monte_carlo_jobs(args.best_v, args.tag, args.monte_carlo_seeds)))
    if args.suite in {"generalization", "all"}:
        suites.append(("generalization", build_generalization_jobs(args.best_v, args.tag)))

    summaries = [_run_jobs(name, jobs, args.dry_run, args.tag) for name, jobs in suites]
    print(json.dumps({"completed_suites": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
