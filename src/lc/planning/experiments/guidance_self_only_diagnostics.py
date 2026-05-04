from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from lc.envs.scenarios import build_planning_scenario
from lc.planning.configs import build_planning_network_config
from lc.planning.envs import PlanningSwarmEnv
from lc.planning.trainers import PlanningTrainer


def _build_trainer(network_version: str, seed: int = 41) -> PlanningTrainer:
    cfg = build_planning_network_config(network_version)
    scenario = replace(
        build_planning_scenario(curriculum_env="guidance_G1"),
        max_obstacles=cfg.max_obstacles,
        max_neighbors=cfg.max_neighbors,
        timeout_seconds=8.0,
    )
    env = PlanningSwarmEnv(
        scenario=scenario,
        self_dim=cfg.self_dim,
        obstacle_dim=cfg.obstacle_dim,
        neighbor_dim=cfg.neighbor_dim,
        action_limit=cfg.action_limit,
    )
    return PlanningTrainer(
        env=env,
        network_config=cfg,
        batch_size=128,
        warmup_steps=600,
        updates_per_step=0.1,
        guidance_exploration_noise=0.02,
        actor_action_reg_weight=1e-3,
        seed=seed,
    )


def _wall_repulsion_state(x: float, y: float, target_x: float, target_y: float, action_limit: float) -> np.ndarray:
    dist_left = x + 0.75
    dist_right = 0.75 - x
    dist_bottom = y + 0.75
    dist_top = 0.75 - y
    wall_repulsion_x = max(0.0, 0.3 - dist_left) - max(0.0, 0.3 - dist_right)
    wall_repulsion_y = max(0.0, 0.3 - dist_bottom) - max(0.0, 0.3 - dist_top)
    return np.array(
        [
            np.clip((target_x - x) / 1.5, -1.0, 1.0),
            np.clip((target_y - y) / 1.5, -1.0, 1.0),
            0.0,
            0.0,
            np.clip(wall_repulsion_x / 0.3, -1.0, 1.0),
            np.clip(wall_repulsion_y / 0.3, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def _base_obs(
    self_state: np.ndarray,
    cfg,
    *,
    obstacles: np.ndarray | None = None,
    obstacle_mask: np.ndarray | None = None,
    neighbors: np.ndarray | None = None,
    neighbor_mask: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    obstacle_tokens = np.zeros((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=np.float32) if obstacles is None else obstacles.astype(np.float32, copy=False)
    obstacle_mask_tokens = np.zeros((1, cfg.max_obstacles), dtype=np.float32) if obstacle_mask is None else obstacle_mask.astype(np.float32, copy=False)
    neighbor_tokens = np.zeros((1, cfg.max_neighbors, cfg.neighbor_dim), dtype=np.float32) if neighbors is None else neighbors.astype(np.float32, copy=False)
    neighbor_mask_tokens = np.zeros((1, cfg.max_neighbors), dtype=np.float32) if neighbor_mask is None else neighbor_mask.astype(np.float32, copy=False)
    return {
        "self_state": torch.tensor(self_state, dtype=torch.float32).unsqueeze(0),
        "obstacles": torch.tensor(obstacle_tokens, dtype=torch.float32),
        "neighbors": torch.tensor(neighbor_tokens, dtype=torch.float32),
        "obstacle_mask": torch.tensor(obstacle_mask_tokens, dtype=torch.float32),
        "neighbor_mask": torch.tensor(neighbor_mask_tokens, dtype=torch.float32),
    }


def _write_action_field_svg(
    path: Path,
    title: str,
    xs: np.ndarray,
    ys: np.ndarray,
    action_x: np.ndarray,
    action_y: np.ndarray,
    action_norm: np.ndarray,
    target: np.ndarray,
    obstacles: list[dict[str, float]] | None = None,
) -> None:
    width = 720
    height = 720
    margin = 40
    span = 1.5

    def sx(x: float) -> float:
        return margin + ((x + 0.75) / span) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - (margin + ((y + 0.75) / span) * (height - 2 * margin))

    max_norm = float(action_norm.max()) if float(action_norm.max()) > 1e-8 else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{sx(-0.75)}" y="{sy(0.75)}" width="{sx(0.75)-sx(-0.75)}" height="{sy(-0.75)-sy(0.75)}" fill="none" stroke="#666" stroke-width="2"/>',
        f'<circle cx="{sx(float(target[0]))}" cy="{sy(float(target[1]))}" r="6" fill="#d62728"/>',
        f'<text x="20" y="24" font-size="18" fill="#111">{title}</text>',
    ]
    for obstacle in obstacles or []:
        radius_px = max(3.0, float(obstacle["radius"]) / span * (width - 2 * margin))
        lines.append(
            f'<circle cx="{sx(float(obstacle["x"]))}" cy="{sy(float(obstacle["y"]))}" r="{radius_px:.2f}" fill="rgba(80,80,80,0.18)" stroke="#444" stroke-width="1.5"/>'
        )
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            x0 = sx(float(x))
            y0 = sy(float(y))
            dx = float(action_x[iy, ix]) * 45.0
            dy = -float(action_y[iy, ix]) * 45.0
            norm = float(action_norm[iy, ix]) / max_norm
            color = f'rgb({int(30 + 180 * norm)}, {int(80 + 120 * (1 - norm))}, {int(200 - 120 * norm)})'
            x1 = x0 + dx
            y1 = y0 + dy
            lines.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<circle cx="{x1:.2f}" cy="{y1:.2f}" r="1.8" fill="{color}"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _action_field(trainer: PlanningTrainer, out_dir: Path, stem: str, *, stage_mode: str = "guidance") -> dict[str, object]:
    actor = trainer.actor
    actor.eval()
    xs = np.linspace(-0.55, 0.55, 13, dtype=np.float32)
    ys = np.linspace(-0.55, 0.55, 13, dtype=np.float32)
    target = np.array([0.35, 0.0], dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    action_x = np.zeros_like(grid_x, dtype=np.float32)
    action_y = np.zeros_like(grid_y, dtype=np.float32)
    action_norm = np.zeros_like(grid_x, dtype=np.float32)
    cfg = trainer.network_config
    scene = trainer.env.get_scene_snapshot()
    obstacles = [
        {"x": float(position[0]), "y": float(position[1]), "radius": float(radius)}
        for position, radius in zip(
            scene.get("obstacle_positions_final", []),
            scene.get("obstacle_radii_initial", []),
        )
    ]

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            self_state = _wall_repulsion_state(float(x), float(y), float(target[0]), float(target[1]), trainer.network_config.action_limit)
            obs = _base_obs(self_state, cfg)
            with torch.no_grad():
                _, action = actor(obs, stage_mode=stage_mode)
            vec = action.squeeze(0).cpu().numpy()
            action_x[iy, ix] = vec[0]
            action_y[iy, ix] = vec[1]
            action_norm[iy, ix] = float(np.linalg.norm(vec))

    _write_action_field_svg(
        out_dir / f"{stem}_action_field.svg",
        f"{stem} action field",
        xs,
        ys,
        action_x,
        action_y,
        action_norm,
        target,
        obstacles=obstacles,
    )
    center_idx = len(xs) // 2
    summary = {
        "mean_action_norm": float(action_norm.mean()),
        "center_action": [float(action_x[center_idx, center_idx]), float(action_y[center_idx, center_idx])],
        "right_edge_action": [float(action_x[center_idx, -1]), float(action_y[center_idx, -1])],
        "left_edge_action": [float(action_x[center_idx, 0]), float(action_y[center_idx, 0])],
        "obstacle_count": int(len(obstacles)),
    }
    (out_dir / f"{stem}_action_field_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _feature_probe(trainer: PlanningTrainer, out_dir: Path, stem: str, *, stage_mode: str = "guidance") -> dict[str, object]:
    actor = trainer.actor
    cfg = trainer.network_config
    probes = {
        "target_right": _wall_repulsion_state(0.0, 0.0, 0.45, 0.0, cfg.action_limit),
        "target_left": _wall_repulsion_state(0.0, 0.0, -0.45, 0.0, cfg.action_limit),
        "target_up": _wall_repulsion_state(0.0, 0.0, 0.0, 0.45, cfg.action_limit),
        "target_down": _wall_repulsion_state(0.0, 0.0, 0.0, -0.45, cfg.action_limit),
    }
    records: dict[str, dict[str, object]] = {}
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for name, state in probes.items():
            obs = _base_obs(state, cfg)
            _, action = actor(obs, stage_mode=stage_mode)
            snapshot = actor.last_feature_snapshot
            tensors[name] = {
                "self_token": snapshot["self_token"].squeeze(0).cpu(),
                "safe_feature": snapshot["safe_feature"].squeeze(0).cpu(),
                "hidden": snapshot["action_hidden"].squeeze(0).cpu(),
                "preactivation": snapshot["action_preactivation"].squeeze(0).cpu(),
                "action": action.squeeze(0).cpu(),
            }
            records[name] = {
                "action": action.squeeze(0).cpu().tolist(),
                "self_token_norm": float(torch.norm(snapshot["self_token"]).item()),
                "safe_feature_norm": float(torch.norm(snapshot["safe_feature"]).item()),
                "hidden_norm": float(torch.norm(snapshot["action_hidden"]).item()),
                "preactivation": snapshot["action_preactivation"].squeeze(0).cpu().tolist(),
            }
    summary = {
        "target_dx_flip_self_token_delta": float(
            torch.norm(tensors["target_right"]["self_token"] - tensors["target_left"]["self_token"]).item()
        ),
        "target_dx_flip_hidden_delta": float(torch.norm(tensors["target_right"]["hidden"] - tensors["target_left"]["hidden"]).item()),
        "target_dx_flip_action_delta": float(torch.norm(tensors["target_right"]["action"] - tensors["target_left"]["action"]).item()),
        "target_dy_flip_self_token_delta": float(
            torch.norm(tensors["target_up"]["self_token"] - tensors["target_down"]["self_token"]).item()
        ),
        "target_dy_flip_hidden_delta": float(torch.norm(tensors["target_up"]["hidden"] - tensors["target_down"]["hidden"]).item()),
        "target_dy_flip_action_delta": float(torch.norm(tensors["target_up"]["action"] - tensors["target_down"]["action"]).item()),
        "records": records,
    }
    (out_dir / f"{stem}_feature_probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _obstacle_response_probe(trainer: PlanningTrainer, out_dir: Path, stem: str, *, stage_mode: str = "avoidance") -> dict[str, object]:
    actor = trainer.actor
    cfg = trainer.network_config
    base_state = _wall_repulsion_state(0.0, 0.0, 0.9, 0.0, cfg.action_limit)

    def obstacle_case(x: float | None, y: float | None) -> tuple[np.ndarray, np.ndarray]:
        obstacles = np.zeros((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=np.float32)
        mask = np.zeros((1, cfg.max_obstacles), dtype=np.float32)
        if x is not None and y is not None:
            obstacles[0, 0] = np.array([x, y, 0.0, 0.0, 0.06], dtype=np.float32)
            mask[0, 0] = 1.0
        return obstacles, mask

    cases = {
        "clear": obstacle_case(None, None),
        "center": obstacle_case(0.45, 0.0),
        "left": obstacle_case(0.45, 0.10),
        "right": obstacle_case(0.45, -0.10),
    }

    records: dict[str, dict[str, object]] = {}
    actions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, (obstacles, obstacle_mask) in cases.items():
            obs = _base_obs(base_state, cfg, obstacles=obstacles, obstacle_mask=obstacle_mask)
            _, action = actor(obs, stage_mode=stage_mode)
            actions[name] = action.squeeze(0).cpu()
            records[name] = {
                "action": actions[name].tolist(),
                "obstacle": obstacles[0, 0].tolist() if obstacle_mask[0, 0] > 0.5 else None,
            }

    summary = {
        "obstacle_left_vs_right_action_delta": float(torch.norm(actions["left"] - actions["right"]).item()),
        "obstacle_center_vs_clear_action_delta": float(torch.norm(actions["center"] - actions["clear"]).item()),
        "records": records,
    }
    (out_dir / f"{stem}_obstacle_response_probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _train(trainer: PlanningTrainer, episodes: int = 100) -> dict[str, object]:
    return trainer.train(
        episodes=episodes,
        use_curriculum=False,
        use_pyramid_per=False,
        use_uniform_replay=True,
    )


def run_guidance_self_only_diagnostics(output_dir: str | Path, episodes: int = 100) -> dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "self_only": "transformer_large",
        "guidance_attention": "transformer_large_guidance_attn",
    }
    results: dict[str, object] = {}
    for name, network_version in variants.items():
        trainer = _build_trainer(network_version=network_version, seed=41)
        train_summary = _train(trainer, episodes=episodes)
        action_summary = _action_field(trainer, out_dir, name)
        feature_summary = _feature_probe(trainer, out_dir, name)
        results[name] = {
            "network_version": network_version,
            "final_metrics": train_summary.get("final_metrics", {}),
            "training_counters": train_summary.get("training_counters", {}),
            "action_field": action_summary,
            "feature_probe": feature_summary,
        }
        (out_dir / f"{name}_summary.json").write_text(json.dumps(results[name], ensure_ascii=False, indent=2), encoding="utf-8")

    compare = {
        "self_only_success_rate": float(results["self_only"]["final_metrics"].get("success_rate", 0.0)),
        "guidance_attention_success_rate": float(results["guidance_attention"]["final_metrics"].get("success_rate", 0.0)),
        "self_only_saturation": float(results["self_only"]["final_metrics"].get("actor_output_saturation_rate", 0.0)),
        "guidance_attention_saturation": float(results["guidance_attention"]["final_metrics"].get("actor_output_saturation_rate", 0.0)),
        "self_only_mean_action_norm": float(results["self_only"]["action_field"]["mean_action_norm"]),
        "guidance_attention_mean_action_norm": float(results["guidance_attention"]["action_field"]["mean_action_norm"]),
    }
    payload = {"compare": compare, "variants": results}
    (out_dir / "compare_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    output = Path("outputs/planning/guidance_self_only_diagnostics")
    result = run_guidance_self_only_diagnostics(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
