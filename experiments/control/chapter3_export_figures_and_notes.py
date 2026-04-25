from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

THESIS_PACKAGE_ROOT = OUTPUT_ROOT / "chapter3_thesis_package" / "20260412_thesis_package"
NO_DIST_RL_ROOT = (
    OUTPUT_ROOT
    / "control_pybullet_rl"
    / "x_refline_no_disturbance_mddpg_retrain"
    / "20260412_v1_to_v5_300eps_reexpanded"
)
RANDOM_RL_ROOT = (
    OUTPUT_ROOT
    / "control_pybullet_rl"
    / "x_refline_random_hover_disturbance_mddpg_retrain"
    / "20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix"
)
NO_DIST_COMPARE_ROOT = (
    NO_DIST_RL_ROOT / "best_v_compare" / "bestv_compare_20260412_150029"
)
RANDOM_COMPARE_ROOT = (
    OUTPUT_ROOT
    / "control_pybullet"
    / "x_pid_ladrc_ddpg_random_hover_disturb_compare"
    / "x"
    / "20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare"
)


@dataclass(frozen=True)
class CopiedFigure:
    source: Path
    target_name: str
    category: str
    experiment: str
    recommended_for_paper: bool
    generated: bool = False


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 2.2,
        }
    )


def load_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    return read_csv(summary_csv)


def add_unique_name(used_names: set[str], name: str) -> str:
    if name not in used_names:
        used_names.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def copy_figure(spec: CopiedFigure, dst_dir: Path, used_names: set[str]) -> dict[str, Any]:
    target_name = add_unique_name(used_names, spec.target_name)
    target_path = dst_dir / spec.category / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.source, target_path)
    return {
        "原路径": str(spec.source),
        "原文件名": spec.source.name,
        "新中文文件名": target_name,
        "分类": spec.category,
        "来源实验": spec.experiment,
        "论文主图": "是" if spec.recommended_for_paper else "否",
        "本次补画": "是" if spec.generated else "否",
    }


def save_generated_figure(
    fig: plt.Figure,
    dst_dir: Path,
    category: str,
    target_name: str,
    experiment: str,
    recommended_for_paper: bool,
    used_names: set[str],
) -> dict[str, Any]:
    final_name = add_unique_name(used_names, target_name)
    target_path = dst_dir / category / final_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "原路径": "",
        "原文件名": "",
        "新中文文件名": final_name,
        "分类": category,
        "来源实验": experiment,
        "论文主图": "是" if recommended_for_paper else "否",
        "本次补画": "是",
    }


def plot_reward_curves(summary_csv: Path, title: str) -> plt.Figure:
    rows = load_summary_rows(summary_csv)
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in rows:
        label = f"v={int(float(row['shared_value']))}"
        history_path = Path(row["output_dir"]) / "training_history.csv"
        history = read_csv(history_path)
        episodes = [int(float(item["episode"])) for item in history]
        rewards = [to_float(item["average_reward"]) for item in history]
        ax.plot(episodes, rewards, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_eval_curves(summary_csv: Path, title: str, metric: str) -> plt.Figure:
    rows = load_summary_rows(summary_csv)
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in rows:
        label = f"v={int(float(row['shared_value']))}"
        history_path = Path(row["output_dir"]) / "eval_history.csv"
        history = read_csv(history_path)
        episodes = [int(float(item["episode"])) for item in history]
        values = [to_float(item[metric]) for item in history]
        ax.plot(episodes, values, marker="o", label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")
    return "\n".join([header, sep, *body])


def summarise_shared_values(summary_csv: Path) -> list[dict[str, Any]]:
    rows = load_summary_rows(summary_csv)
    formatted = []
    for row in rows:
        formatted.append(
            {
                "shared_value": int(float(row["shared_value"])),
                "average_reward": round(to_float(row["average_reward"]), 4),
                "best_eval_rmse": round(to_float(row["best_eval_rmse"]), 5),
                "best_eval_mae": round(to_float(row["best_eval_mae"]), 5),
                "best_eval_velocity_rmse": round(to_float(row["best_eval_velocity_rmse"]), 5),
                "best_eval_score": round(to_float(row["best_eval_score"]), 5),
            }
        )
    return formatted


def collect_main_specs() -> list[CopiedFigure]:
    package_figs = THESIS_PACKAGE_ROOT / "figures"
    random_best_eval = (
        RANDOM_RL_ROOT
        / "v_2"
        / "train"
        / "ladrc_x_pos_pid_att"
        / "x"
        / "20260412_185851"
        / "figures"
        / "best_eval"
    )
    no_dist_best_eval = (
        NO_DIST_RL_ROOT
        / "v_2"
        / "train"
        / "ladrc_x_pos_pid_att"
        / "x"
        / "20260412_145504"
        / "figures"
        / "episode_300"
    )
    return [
        CopiedFigure(package_figs / "fig1_ladrc_framework.png", "LADRC控制框架图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig2_step_or_short_response_compare.png", "短参考响应对比图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig3_disturbance_recovery_compare.png", "抗扰恢复对比图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig4_tracking_error_compare.png", "轨迹跟踪误差对比图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig5_ddpg_ladrc_framework.png", "DDPG-LADRC在线重整定框架图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig6_temporal_enhancement_framework.png", "跨时间样本增强机制图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig7_training_reward_curve.png", "训练收敛曲线.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig8_control_smoothness_compare.png", "控制输出平滑性对比图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(package_figs / "fig9_overall_performance_compare.png", "综合性能对比图.png", "正式主图", "第三章实验包", True),
        CopiedFigure(no_dist_best_eval / "axis_tracking.png", "无扰动最优轨迹跟踪图.png", "训练过程图", "无扰动最优RL", False),
        CopiedFigure(no_dist_best_eval / "axis_velocity.png", "无扰动最优速度跟踪图.png", "训练过程图", "无扰动最优RL", False),
        CopiedFigure(random_best_eval / "axis_tracking.png", "随机扰动最优轨迹跟踪图.png", "训练过程图", "随机扰动最优RL", False),
        CopiedFigure(random_best_eval / "axis_velocity.png", "随机扰动最优速度跟踪图.png", "训练过程图", "随机扰动最优RL", False),
        CopiedFigure(NO_DIST_COMPARE_ROOT / "figures" / "tracking_compare_pid_ladrc_bestv.png", "无扰动正式对比轨迹图.png", "对比过程图", "无扰动正式对比", False),
        CopiedFigure(NO_DIST_COMPARE_ROOT / "figures" / "velocity_compare_pid_ladrc_bestv.png", "无扰动正式对比速度图.png", "对比过程图", "无扰动正式对比", False),
        CopiedFigure(NO_DIST_COMPARE_ROOT / "figures" / "error_compare_pid_ladrc_bestv.png", "无扰动正式对比误差图.png", "对比过程图", "无扰动正式对比", False),
        CopiedFigure(RANDOM_COMPARE_ROOT / "figures" / "tracking_compare.png", "随机扰动正式对比轨迹图.png", "对比过程图", "随机扰动正式对比", False),
        CopiedFigure(RANDOM_COMPARE_ROOT / "figures" / "velocity_compare.png", "随机扰动正式对比速度图.png", "对比过程图", "随机扰动正式对比", False),
        CopiedFigure(RANDOM_COMPARE_ROOT / "figures" / "error_compare.png", "随机扰动正式对比误差图.png", "对比过程图", "随机扰动正式对比", False),
    ]


def copy_tables(dst_root: Path) -> list[dict[str, Any]]:
    tables_root = THESIS_PACKAGE_ROOT / "tables"
    rows = []
    for src in sorted(tables_root.glob("*.csv")):
        dst = dst_root / "tables" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append({"name": src.name, "source": str(src), "target": str(dst)})
    return rows


def build_paper_doc(dst_root: Path, image_rows: list[dict[str, Any]]) -> str:
    multispeed_rows = read_csv(THESIS_PACKAGE_ROOT / "tables" / "table_ladrc_multispeed_best_params.csv")
    no_dist_rows = read_csv(THESIS_PACKAGE_ROOT / "tables" / "table_rl_best_v2_no_disturbance.csv")
    random_rows = read_csv(THESIS_PACKAGE_ROOT / "tables" / "table_rl_best_v2_random_0p004.csv")
    no_dist_shared = summarise_shared_values(NO_DIST_RL_ROOT / "average_reward_summary.csv")
    random_shared = summarise_shared_values(RANDOM_RL_ROOT / "average_reward_summary.csv")

    doc = f"""# 第三章实验表述（论文正文版）

## 1. 实验主线概述

本章围绕 PyBullet 单轴位置控制场景展开，控制对象为无人机 `x` 轴位置。整体实验分为三条主线：一是传统 `PID` 与固定参数 `LADRC` 的整定与对比；二是基于强化学习的 `DDPG-LADRC` 在线参数重整定；三是在随机悬停扰动场景下验证强化学习方法的鲁棒性与适应性。当前正文只采用已经完成环境修复、策略状态完整恢复以及多 seed 评估校验后的可信结果。

## 2. 传统整定与参数矛盾分析

在传统整定部分，首先针对短参考和多速度工况对 `LADRC` 参数进行了重新整定。实验结果表明，单一固定参数很难在所有速度区间内同时兼顾快速性、平稳性与抗扰性。对 `0.2/0.4/0.6/0.8 m/s` 四组典型速度的重整定结果如下。

{markdown_table(multispeed_rows[:4], [("speed_mps", "速度(m/s)"), ("r", "r"), ("b0", "b0"), ("omega_c", "omega_c"), ("k", "k"), ("retuned_rmse", "整定后RMSE")])}

对应图像建议引用：
- `LADRC控制框架图.png`
- `短参考响应对比图.png`
- `抗扰恢复对比图.png`
- `轨迹跟踪误差对比图.png`

## 3. 强化学习联合参数重整定方法

强化学习部分以 `LADRC` 为控制主体，策略网络不直接输出控制量，而是在线调节四个关键参数 `[r, b0, omega_c, k]`。为了保证训练过程与正式回放一致，当前主线已经统一采用同一套 PyBullet rollout，并修复了参数更新重建控制器状态以及 checkpoint 未完整恢复内部状态的问题。当前可信主线下，`v=2` 是共享增强机制中的推荐配置。

无扰动场景下不同 `v` 的训练效果摘要如下。

{markdown_table(no_dist_shared, [("shared_value", "v"), ("average_reward", "平均奖励"), ("best_eval_rmse", "best eval RMSE"), ("best_eval_mae", "best eval MAE"), ("best_eval_score", "best eval score")])}

随机悬停扰动 `0.004N` 场景下不同 `v` 的训练效果摘要如下。

{markdown_table(random_shared, [("shared_value", "v"), ("average_reward", "平均奖励"), ("best_eval_rmse", "best eval RMSE"), ("best_eval_mae", "best eval MAE"), ("best_eval_score", "best eval score")])}

对应图像建议引用：
- `DDPG-LADRC在线重整定框架图.png`
- `跨时间样本增强机制图.png`
- `无扰动不同v的奖励变化曲线.png`
- `随机扰动不同v的奖励变化曲线.png`
- `无扰动不同v的评估误差曲线.png`
- `随机扰动不同v的评估误差曲线.png`

## 4. 无扰动与随机扰动主结果

在无扰动场景下，`DDPG-LADRC(best v=2)` 已经达到与 `PID` 和固定参数 `LADRC` 同量级的控制精度，并在 `RMSE` 上略优。结果如下。

{markdown_table(no_dist_rows, [("controller", "控制器"), ("rmse", "RMSE"), ("mae", "MAE"), ("velocity_rmse", "速度RMSE"), ("reward", "Reward")])}

在悬停中段加入 `0.004N` 随机外力扰动后，当前可信结果表明 `DDPG-LADRC(best v=2)` 仍保持最优 `RMSE`，同时在速度恢复方面接近固定参数 `LADRC`。结果如下。

{markdown_table(random_rows, [("controller", "控制器"), ("rmse", "RMSE"), ("mae", "MAE"), ("velocity_rmse", "速度RMSE"), ("reward", "Reward")])}

对应图像建议引用：
- `无扰动最优轨迹跟踪图.png`
- `无扰动最优速度跟踪图.png`
- `随机扰动最优轨迹跟踪图.png`
- `随机扰动最优速度跟踪图.png`
- `无扰动正式对比轨迹图.png`
- `随机扰动正式对比轨迹图.png`
- `控制输出平滑性对比图.png`
- `综合性能对比图.png`

## 5. 写作建议

当前第三章正文建议围绕“传统固定参数方法存在跨工况矛盾，强化学习在线调参能够在无扰动和轻随机扰动场景下提升综合性能”这条主线展开。图表使用时，优先选用 `images/正式主图` 下的中文命名图片，再用 `images/训练过程图` 补充训练收敛与代表性轨迹。
"""
    return doc


def build_log_doc() -> str:
    return """# 第三章实验表述（实验记录版）

## 1. 本次整理的目标

本次整理面向第三章论文撰写，重点是把当前可信主链的图片、关键数据和实验叙述集中到同一个输出目录中，避免后续继续从多个历史结果目录中手动翻找。整理过程不删除原始结果，不移动原图，只做复制、重命名、补画缺失图片与文档归纳。

## 2. 当前可信实验主线

当前可信结果基于以下修复后的 PyBullet 主链：
- 训练、deterministic eval、正式 compare 使用统一 rollout
- 参数更新采用原位更新，不再因改参数重建 LADRC channel
- checkpoint 会完整保存和恢复策略内部状态，包括 normalizer
- 随机扰动场景下采用多 seed 评估选最优模型

在这个可信主线下，当前第三章推荐主结果包括：
- 无扰动下 `best v=2`
- `0.004N` 随机悬停扰动下 `best v=2`
- 固定 `PID` 与固定参数 `LADRC(0.5-opt)` 的统一对比

## 3. 为什么要补画图片

虽然已有 `chapter3_thesis_package` 已经导出了一组正式图，但写论文时还需要更多训练过程图，尤其是：
- 不同 `v` 的 `reward-episode` 曲线
- 不同 `v` 的 `eval rmse/score` 曲线
- 训练期间最优 checkpoint 的轨迹跟踪图和速度跟踪图

这些图可以直接从可信目录中的 `training_history.csv`、`eval_history.csv` 和 `best_eval` 图恢复出来，不需要重新训练。

## 4. 本次整理出的图片分类

### 4.1 正式主图

这一类图对应第三章正文主插图，来自正式实验包，已经统一改成中文文件名。

### 4.2 训练过程图

这一类图主要用于展示强化学习训练过程，包括不同 `v` 的奖励变化曲线、评估误差曲线以及最优 checkpoint 的轨迹和速度图。

### 4.3 对比过程图

这一类图主要保留正式 compare 的轨迹图、速度图和误差图，便于写实验分析时引用。

## 5. 当前建议引用的结果

当前第三章最建议引用的结论是：
- 无扰动场景下，`DDPG-LADRC(best v=2)` 在 `RMSE` 上略优于 `PID` 和固定参数 `LADRC`
- `0.004N` 随机悬停扰动场景下，`DDPG-LADRC(best v=2)` 仍保持最优 `RMSE`
- 传统整定的多速度结果说明固定参数在跨工况下存在明显矛盾，因此在线调参是必要的

## 6. 后续使用建议

后续写论文时，优先从本次输出目录的 `images`、`tables`、`docs` 和 `summaries` 中取材，不再直接从历史 `outputs` 目录中零散找图。若后续新增实验，也建议继续沿用这种“单独素材输出目录”的方式管理。
"""


def export_assets(output_tag: str | None) -> Path:
    apply_style()
    timestamp = output_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_root = OUTPUT_ROOT / "chapter3_paper_assets" / timestamp
    image_root = dst_root / "images"
    docs_root = dst_root / "docs"
    summaries_root = dst_root / "summaries"

    ensure_exists(THESIS_PACKAGE_ROOT)
    ensure_exists(NO_DIST_RL_ROOT)
    ensure_exists(RANDOM_RL_ROOT)
    ensure_exists(NO_DIST_COMPARE_ROOT)
    ensure_exists(RANDOM_COMPARE_ROOT)

    used_names: set[str] = set()
    mapping_rows: list[dict[str, Any]] = []

    for spec in collect_main_specs():
        ensure_exists(spec.source)
        mapping_rows.append(copy_figure(spec, image_root, used_names))

    mapping_rows.extend(
        [
            save_generated_figure(
                plot_reward_curves(NO_DIST_RL_ROOT / "average_reward_summary.csv", "无扰动不同 v 的奖励变化曲线"),
                image_root,
                "训练过程图",
                "无扰动不同v的奖励变化曲线.png",
                "无扰动RL训练",
                False,
                used_names,
            ),
            save_generated_figure(
                plot_reward_curves(RANDOM_RL_ROOT / "average_reward_summary.csv", "随机扰动不同 v 的奖励变化曲线"),
                image_root,
                "训练过程图",
                "随机扰动不同v的奖励变化曲线.png",
                "随机扰动RL训练",
                False,
                used_names,
            ),
            save_generated_figure(
                plot_reward_curves(NO_DIST_RL_ROOT / "average_reward_summary.csv", "无扰动不同 v 的平均Reward随Episode变化图"),
                image_root,
                "训练过程图",
                "无扰动不同v的平均reward随episode变化图.png",
                "无扰动RL训练",
                False,
                used_names,
            ),
            save_generated_figure(
                plot_reward_curves(RANDOM_RL_ROOT / "average_reward_summary.csv", "随机扰动不同 v 的平均Reward随Episode变化图"),
                image_root,
                "训练过程图",
                "随机扰动不同v的平均reward随episode变化图.png",
                "随机扰动RL训练",
                False,
                used_names,
            ),
            save_generated_figure(
                plot_eval_curves(NO_DIST_RL_ROOT / "average_reward_summary.csv", "无扰动不同 v 的评估误差曲线", "rmse"),
                image_root,
                "训练过程图",
                "无扰动不同v的评估误差曲线.png",
                "无扰动RL训练",
                False,
                used_names,
            ),
            save_generated_figure(
                plot_eval_curves(RANDOM_RL_ROOT / "average_reward_summary.csv", "随机扰动不同 v 的评估误差曲线", "rmse"),
                image_root,
                "训练过程图",
                "随机扰动不同v的评估误差曲线.png",
                "随机扰动RL训练",
                False,
                used_names,
            ),
        ]
    )

    copied_tables = copy_tables(dst_root)
    write_csv(
        summaries_root / "图片映射清单.csv",
        mapping_rows,
        ["原路径", "原文件名", "新中文文件名", "分类", "来源实验", "论文主图", "本次补画"],
    )
    write_json(
        summaries_root / "图片映射清单.json",
        {
            "output_root": str(dst_root),
            "image_count": len(mapping_rows),
            "table_count": len(copied_tables),
            "images": mapping_rows,
            "tables": copied_tables,
        },
    )

    write_text(docs_root / "第三章实验表述_论文正文版.md", build_paper_doc(dst_root, mapping_rows))
    write_text(docs_root / "第三章实验表述_实验记录版.md", build_log_doc())
    return dst_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chapter 3 paper images and notes.")
    parser.add_argument("--output-tag", default="", help="Optional output folder tag.")
    args = parser.parse_args()
    result_root = export_assets(args.output_tag or None)
    print(json.dumps({"output_root": str(result_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
