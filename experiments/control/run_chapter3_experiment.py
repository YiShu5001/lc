from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lc.control import Chapter3ExperimentConfig, build_reference_trajectory, run_chapter3_control_experiment


def main() -> None:
    config = Chapter3ExperimentConfig()
    references = build_reference_trajectory(64, channel_count=config.channel_count)
    measurements = references * 0.92
    result = run_chapter3_control_experiment(references, measurements, config)
    print("chapter3_mean_abs_error", result.mean_abs_error)
    print("chapter3_output_shape", result.outputs.shape)


if __name__ == "__main__":
    main()
