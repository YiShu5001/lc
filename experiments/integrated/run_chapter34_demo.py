from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lc.system import Chapter34BridgeConfig, run_chapter34_pipeline


def main() -> None:
    result = run_chapter34_pipeline(Chapter34BridgeConfig(scenario_stage=3, rollout_steps=16))
    print("scenario", result.scenario.stage_name)
    print("control_shape", result.control_result.outputs.shape)
    print("planning_shape", tuple(result.planning_actions.shape))


if __name__ == "__main__":
    main()
