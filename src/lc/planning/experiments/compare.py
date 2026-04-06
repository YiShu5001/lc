from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lc.common.io import ensure_dir, write_json, write_metrics_csv
from lc.envs.scenarios import build_planning_scenario
from lc.planning.configs import PlanningExperimentConfig
from lc.planning.envs import PlanningSwarmEnv
from lc.planning.plotting import plot_planning_comparison
from lc.planning.trainers import PlanningTrainer


def run_planning_comparison(config: PlanningExperimentConfig | None = None) -> dict[str, object]:
    cfg = config or PlanningExperimentConfig()
    base_scenario = replace(
        build_planning_scenario(cfg.difficulty, stage_index=cfg.stage_index, curriculum_env=cfg.curriculum_env),
        max_obstacles=12,
        max_neighbors=7,
    )

    experiment_specs = {
        "task_decomposed": {
            "trainer": {},
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "single_stream_mlp": {
            "trainer": {},
            "train": dict(actor_variant="single_stream_mlp", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "without_curriculum": {
            "trainer": {},
            "train": dict(actor_variant="task_decomposed", use_curriculum=False, use_pyramid_per=True, use_uniform_replay=False),
        },
        "without_pyramid_per": {
            "trainer": {},
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=False, use_uniform_replay=False),
        },
        "uniform_replay": {
            "trainer": {},
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=True),
        },
        "td_only_priority": {
            "trainer": {
                "avoidance_priority_mode": "td_only",
                "cooperation_priority_mode": "td_only",
                "rare_priority_mode": "rare_only",
            },
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "high_old_mix": {
            "trainer": {
                "avoidance_old_fraction": 0.2,
                "cooperation_old_fraction": 0.35,
            },
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "contribution_only_priority": {
            "trainer": {
                "avoidance_priority_mode": "contribution_only",
                "cooperation_priority_mode": "contribution_only",
            },
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "rare_only_priority": {
            "trainer": {
                "rare_priority_mode": "rare_only",
            },
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "balanced_sample_ratio": {
            "trainer": {
                "avoidance_sample_ratio": (1, 1, 1),
                "cooperation_sample_ratio": (1, 1, 1),
            },
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "low_old_mix": {
            "trainer": {
                "avoidance_old_fraction": 0.05,
                "cooperation_old_fraction": 0.1,
            },
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
        "high_rare_ratio": {
            "trainer": {
                "avoidance_sample_ratio": (4, 2, 4),
                "cooperation_sample_ratio": (3, 2, 5),
            },
            "train": dict(actor_variant="task_decomposed", use_curriculum=True, use_pyramid_per=True, use_uniform_replay=False),
        },
    }

    histories: dict[str, list[dict[str, float | int | str]]] = {}
    train_summaries: dict[str, dict[str, object]] = {}
    results: dict[str, dict[str, float]] = {}
    main_trainer: PlanningTrainer | None = None
    for offset, (name, spec) in enumerate(experiment_specs.items()):
        trainer = PlanningTrainer(
            env=PlanningSwarmEnv(scenario=base_scenario),
            seed=cfg.seed + offset,
            **spec["trainer"],
        )
        summary = trainer.train(cfg.episodes, **spec["train"])
        if spec["train"]["actor_variant"] == "single_stream_mlp":
            metrics = trainer.evaluate_mlp_baseline(
                cfg.eval_episodes,
                difficulty=cfg.difficulty,
                stage_index=cfg.stage_index,
            )
        else:
            metrics = trainer.evaluate_primary(
                cfg.eval_episodes,
                difficulty=cfg.difficulty,
                stage_index=cfg.stage_index,
            )
        metrics["final_actor_loss"] = float(summary["final_metrics"].get("actor_loss", 0.0))
        metrics["final_critic_loss"] = float(summary["final_metrics"].get("critic_loss", 0.0))
        results[name] = metrics
        histories[name] = list(summary["history"])
        train_summaries[name] = summary
        if name == "task_decomposed":
            main_trainer = trainer

    complexity_results = _build_complexity_results(main_trainer, cfg.eval_episodes) if main_trainer else {}
    out_dir = ensure_dir(Path("outputs") / "planning" / cfg.difficulty / f"stage_{cfg.stage_index}")
    payload = {
        "difficulty": cfg.difficulty,
        "stage_index": cfg.stage_index,
        "results": results,
        "training": {
            name: {
                "final_metrics": summary["final_metrics"],
                "stage_history": summary["stage_history"],
                "stage_averages": summary["stage_averages"],
                "env_averages": summary.get("env_averages", {}),
                "replay_stats": summary["replay_stats"],
                "stage_transition_summary": summary.get("stage_transition_summary", []),
                "trainer_overrides": experiment_specs[name]["trainer"],
                "train_overrides": experiment_specs[name]["train"],
            }
            for name, summary in train_summaries.items()
        },
        "complexity_generalization": complexity_results,
    }
    write_json(out_dir / "summary.json", payload)
    write_metrics_csv(out_dir / "metrics.csv", [{"method": name, **metrics} for name, metrics in results.items()])
    write_metrics_csv(
        out_dir / "training_history.csv",
        [
            {"method": method, **{key: value for key, value in row.items() if not isinstance(value, str)}}
            for method, rows in histories.items()
            for row in rows
        ],
    )
    write_json(out_dir / "complexity_generalization.json", complexity_results)
    main_summary = train_summaries.get("task_decomposed", {})
    figures = plot_planning_comparison(
        results,
        out_dir / "figures",
        histories=histories,
        stage_history=list(main_summary.get("stage_history", [])),
        complexity_results=complexity_results,
        trajectory=main_summary.get("trajectory"),
        attention_proxy=main_summary.get("attention_proxy"),
        replay_stats=main_summary.get("replay_stats"),
    )
    return {
        "results": results,
        "training": payload["training"],
        "output_dir": str(out_dir),
        "figures": [str(path) for path in figures],
        "complexity_generalization": complexity_results,
    }


def _build_complexity_results(trainer: PlanningTrainer | None, eval_episodes: int) -> dict[str, dict[str, float]]:
    if trainer is None:
        return {}
    scenario = trainer.env.scenario
    cases = {
        "uav_count_variation": replace(
            scenario,
            num_uavs=min(scenario.num_uavs + 2, 8),
            max_neighbors=max(1, min(scenario.max_neighbors + 2, 7)),
        ),
        "obstacle_count_variation": replace(
            scenario,
            num_obstacles=min(scenario.num_obstacles + 3, scenario.max_obstacles),
        ),
        "obstacle_dynamics_variation": replace(
            scenario,
            dynamic_obstacles=True,
        ),
        "target_dynamics_variation": replace(
            scenario,
            target_motion="evasive",
        ),
        "curriculum_stage_variation": build_planning_scenario("extreme", stage_index=2),
    }
    return {name: trainer.evaluate_on_scenario(case, eval_episodes) for name, case in cases.items()}
