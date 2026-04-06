from __future__ import annotations

from pathlib import Path

import numpy as np

from lc.common.io import ensure_dir, write_json
from lc.control.configs import ControlExperimentConfig
from lc.control.experiments import run_control_comparison
from lc.integration.interfaces import map_planning_to_control_reference
from lc.planning.configs import PlanningExperimentConfig
from lc.planning.experiments import run_planning_comparison


def run_bridge_experiment() -> dict[str, object]:
    planning = run_planning_comparison(PlanningExperimentConfig(episodes=12, eval_episodes=2))
    control = run_control_comparison(ControlExperimentConfig())
    action = np.array([0.35, -0.15], dtype=float)
    mapped_reference = map_planning_to_control_reference(action)
    out_dir = ensure_dir(Path("outputs") / "integration" / "bridge_demo")
    summary = {
        "planning_output_dir": planning["output_dir"],
        "control_output_dir": control["output_dir"],
        "sample_planning_action": action.tolist(),
        "mapped_control_reference": mapped_reference,
    }
    write_json(out_dir / "summary.json", summary)
    return {"output_dir": str(out_dir), "summary": summary}
