from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lc.common.io import ensure_dir, write_json
from lc.planning.configs import CheckpointConfig, LoggingConfig


class PlanningCheckpointManager:
    def __init__(
        self,
        run_dir: Path,
        checkpoint_config: CheckpointConfig,
        logging_config: LoggingConfig,
    ) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.checkpoint_config = checkpoint_config
        self.logging_config = logging_config
        self.checkpoint_root = ensure_dir(self.run_dir / "checkpoints")
        self.manifest_path = self.run_dir / "checkpoint_manifest.json"
        self._manifest_rows: list[dict[str, Any]] = []

    def save_run_meta(self, payload: dict[str, Any]) -> Path:
        return write_json(self.run_dir / "run_meta.json", payload)

    def save_network_manifest(self, payload: dict[str, Any]) -> Path:
        return write_json(self.run_dir / "network_manifest.json", payload)

    def save_freeze_manifest(self, payload: list[dict[str, Any]]) -> Path:
        return write_json(self.run_dir / "freeze_events.json", payload)

    def append_stage_events(self, rows: list[dict[str, Any]]) -> Path:
        return _write_csv(self.run_dir / "stage_events.csv", rows)

    def append_reward_breakdown(self, rows: list[dict[str, Any]]) -> Path:
        return _write_csv(self.run_dir / "reward_breakdown_history.csv", rows)

    def save_stage_checkpoint(
        self,
        *,
        stage_name: str,
        checkpoint_type: str,
        actor: nn.Module,
        critic_1: nn.Module,
        critic_2: nn.Module,
        target_actor: nn.Module,
        target_critic_1: nn.Module,
        target_critic_2: nn.Module,
        actor_optimizer: torch.optim.Optimizer | None,
        critic_optimizer: torch.optim.Optimizer | None,
        meta: dict[str, Any],
    ) -> Path | None:
        if not self.checkpoint_config.enable_checkpoint:
            return None
        if checkpoint_type == "best" and not self.checkpoint_config.save_best_per_stage:
            return None
        if checkpoint_type == "latest" and not self.checkpoint_config.save_latest_per_stage:
            return None

        stage_dir = ensure_dir(self.checkpoint_root / stage_name / checkpoint_type)
        torch.save(actor.state_dict(), stage_dir / "actor.pt")
        torch.save(critic_1.state_dict(), stage_dir / "critic_1.pt")
        torch.save(critic_2.state_dict(), stage_dir / "critic_2.pt")
        torch.save(target_actor.state_dict(), stage_dir / "target_actor.pt")
        torch.save(target_critic_1.state_dict(), stage_dir / "target_critic_1.pt")
        torch.save(target_critic_2.state_dict(), stage_dir / "target_critic_2.pt")
        if self.checkpoint_config.save_optimizer_state and actor_optimizer is not None:
            torch.save(actor_optimizer.state_dict(), stage_dir / "optim_actor.pt")
        if self.checkpoint_config.save_optimizer_state and critic_optimizer is not None:
            torch.save(critic_optimizer.state_dict(), stage_dir / "optim_critic.pt")
        write_json(stage_dir / "meta.json", meta)
        self._manifest_rows.append(
            {
                "stage_name": stage_name,
                "checkpoint_type": checkpoint_type,
                "path": str(stage_dir),
                "episode": int(meta.get("episode", -1)),
                "curriculum_env": str(meta.get("curriculum_env", "")),
                "reason": str(meta.get("reason", "")),
            }
        )
        if self.logging_config.write_checkpoint_manifest:
            write_json(self.manifest_path, self._manifest_rows)
        return stage_dir

    def load_stage_checkpoint(self, stage_name: str, checkpoint_type: str = "latest") -> dict[str, Any]:
        stage_dir = self.checkpoint_root / stage_name / checkpoint_type
        return {
            "actor": torch.load(stage_dir / "actor.pt", map_location="cpu"),
            "critic_1": torch.load(stage_dir / "critic_1.pt", map_location="cpu"),
            "critic_2": torch.load(stage_dir / "critic_2.pt", map_location="cpu"),
            "target_actor": torch.load(stage_dir / "target_actor.pt", map_location="cpu"),
            "target_critic_1": torch.load(stage_dir / "target_critic_1.pt", map_location="cpu"),
            "target_critic_2": torch.load(stage_dir / "target_critic_2.pt", map_location="cpu"),
            "meta": json.loads((stage_dir / "meta.json").read_text(encoding="utf-8")),
        }


def build_run_dir(
    *,
    save_root: str,
    difficulty: str,
    stage_index: int,
    network_version: str,
    run_name: str,
) -> Path:
    return ensure_dir(Path(save_root) / difficulty / f"stage_{stage_index}" / network_version / run_name)


def score_checkpoint_candidate(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("reward", 0.0)),
        -float(metrics.get("collision_rate", 0.0)),
        -float(metrics.get("occupancy_error", 0.0)),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        return path
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
