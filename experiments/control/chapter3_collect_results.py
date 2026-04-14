from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CONTROL_ROOT = OUTPUT_ROOT / "control"
PAPER_DATA_ROOT = OUTPUT_ROOT / "paper_figures" / "chapter3" / "data"

SCAN_DIR = CONTROL_ROOT / "x_axis_rl_refline__exp-bestcfg-scan-v1-to-v10__ep-500__v-1-10__noise-linear-0.1-to-0.04__net-768__drop-0.25"
V7_TUNING_DIRS = {
    "v7_decay_slow_net512_tau005": CONTROL_ROOT / "x_axis_rl_refline__exp-v7ep500decay-slow__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2",
    "v7_decay_slow_net512_tau002": CONTROL_ROOT / "x_axis_rl_refline__exp-v7ep500decay-slow-tau02__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2",
    "v7_decay_slow_net768_tau002": CONTROL_ROOT / "x_axis_rl_refline__exp-v7ep500decay-slow-tau02-net768__ep-500__v-7__noise-linear-0.1-to-0.04__net-768__drop-0.25",
}
LEGACY_MEDIUM_DIR = CONTROL_ROOT / "medium"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reward_stats(path: Path, label: str, shared_value: int | None) -> dict[str, object]:
    frame = pd.read_csv(path)
    rewards = frame["reward"].astype(float)
    rmses = frame["rmse"].astype(float)
    return {
        "label": label,
        "shared_value": shared_value,
        "episodes": int(len(frame)),
        "best_episode": int(rewards.idxmax() + 1),
        "best_reward": float(rewards.max()),
        "worst_reward": float(rewards.min()),
        "first50_reward_mean": float(rewards.head(50).mean()),
        "last50_reward_mean": float(rewards.tail(50).mean()),
        "last100_reward_mean": float(rewards.tail(100).mean()),
        "last100_reward_std": float(rewards.tail(100).std(ddof=0)),
        "last100_rmse_mean": float(rmses.tail(100).mean()),
    }


def collect_master_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    scan_summary = _read_json(SCAN_DIR / "summary.json")
    for method_name in ("pid", "ladrc"):
        metrics = dict(scan_summary["results"][method_name])
        rows.append(
            {
                "source_group": "x_refline_best_scan",
                "scenario": "x_axis_rl_refline",
                "method": method_name,
                "variant": method_name,
                "shared_value": "",
                "difficulty": scan_summary["difficulty"],
                "train_episodes": 500,
                "hidden_dim": 768,
                "dropout_p": 0.25,
                "tau": 0.02,
                "soft_update_interval": 10,
                **metrics,
            }
        )

    sweep = pd.read_csv(SCAN_DIR / "mddpg_shared_value_sweep.csv")
    for _, item in sweep.iterrows():
        rows.append(
            {
                "source_group": "x_refline_best_scan",
                "scenario": "x_axis_rl_refline",
                "method": "mddpg_ladrc",
                "variant": f"mddpg_v{int(item['shared_value'])}",
                "shared_value": int(item["shared_value"]),
                "difficulty": scan_summary["difficulty"],
                "train_episodes": 500,
                "hidden_dim": 768,
                "dropout_p": 0.25,
                "tau": 0.02,
                "soft_update_interval": 10,
                **item.to_dict(),
            }
        )

    for label, run_dir in V7_TUNING_DIRS.items():
        summary = _read_json(run_dir / "summary.json")
        metrics = dict(summary["results"]["mddpg_ladrc"])
        rows.append(
            {
                "source_group": "v7_tuning_study",
                "scenario": "x_axis_rl_refline",
                "method": "mddpg_ladrc",
                "variant": label,
                "shared_value": 7,
                "difficulty": summary["difficulty"],
                "train_episodes": 500,
                "hidden_dim": summary["network_config"]["hidden_dim"],
                "dropout_p": summary["network_config"]["dropout_p"],
                "tau": summary["network_config"]["tau"],
                "soft_update_interval": summary["network_config"]["soft_update_interval"],
                **metrics,
            }
        )

    legacy_summary = _read_json(LEGACY_MEDIUM_DIR / "summary.json")
    for method_name in ("pid", "ladrc", "ddpg_ladrc", "mddpg_ladrc"):
        metrics = dict(legacy_summary["results"][method_name])
        rows.append(
            {
                "source_group": "legacy_piecewise_medium",
                "scenario": "piecewise_constant_velocity_xyz",
                "method": method_name,
                "variant": method_name,
                "shared_value": "",
                "difficulty": legacy_summary["difficulty"],
                "train_episodes": legacy_summary["training_artifacts"].get("ddpg_log", ""),
                "hidden_dim": "",
                "dropout_p": "",
                "tau": "",
                "soft_update_interval": "",
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def collect_reward_stats() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shared_value in (1, 4, 7):
        rows.append(
            _reward_stats(
                SCAN_DIR / f"training_mddpg_v{shared_value}.csv",
                label=f"scan_mddpg_v{shared_value}",
                shared_value=shared_value,
            )
        )

    rows.append(
        _reward_stats(
            V7_TUNING_DIRS["v7_decay_slow_net512_tau005"] / "training_mddpg_v7.csv",
            label="v7_decay_slow_net512_tau005",
            shared_value=7,
        )
    )
    rows.append(
        _reward_stats(
            V7_TUNING_DIRS["v7_decay_slow_net512_tau002"] / "training_mddpg_v7.csv",
            label="v7_decay_slow_net512_tau002",
            shared_value=7,
        )
    )
    rows.append(
        _reward_stats(
            V7_TUNING_DIRS["v7_decay_slow_net768_tau002"] / "training_mddpg_v7.csv",
            label="v7_decay_slow_net768_tau002",
            shared_value=7,
        )
    )
    return pd.DataFrame(rows)


def collect_trajectory_manifest() -> pd.DataFrame:
    rows = []
    for shared_value in (1, 4, 7):
        rows.append(
            {
                "label": f"scan_mddpg_v{shared_value}",
                "shared_value": shared_value,
                "eval_timeseries": str(SCAN_DIR / f"v{shared_value}" / "eval_timeseries.csv"),
                "training_csv": str(SCAN_DIR / f"training_mddpg_v{shared_value}.csv"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PAPER_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    master = collect_master_metrics()
    reward_stats = collect_reward_stats()
    trajectories = collect_trajectory_manifest()

    master_path = PAPER_DATA_ROOT / "chapter3_master_metrics.csv"
    reward_path = PAPER_DATA_ROOT / "chapter3_reward_curve_stats.csv"
    traj_path = PAPER_DATA_ROOT / "chapter3_trajectory_manifest.csv"

    master.to_csv(master_path, index=False)
    reward_stats.to_csv(reward_path, index=False)
    trajectories.to_csv(traj_path, index=False)

    summary_path = PAPER_DATA_ROOT / "chapter3_available_data_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scan_dir": str(SCAN_DIR),
                "best_shared_value": 4,
                "master_metrics_csv": str(master_path),
                "reward_stats_csv": str(reward_path),
                "trajectory_manifest_csv": str(traj_path),
                "available_scan_variants": [1, 4, 7],
                "available_tuning_variants": list(V7_TUNING_DIRS.keys()),
                "legacy_medium_dir": str(LEGACY_MEDIUM_DIR),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "master_metrics_csv": str(master_path),
                "reward_stats_csv": str(reward_path),
                "trajectory_manifest_csv": str(traj_path),
                "summary_json": str(summary_path),
                "rows_master": int(len(master)),
                "rows_reward_stats": int(len(reward_stats)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
