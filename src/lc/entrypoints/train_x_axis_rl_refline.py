from __future__ import annotations

from lc.control.configs import ControlExperimentConfig
from lc.control.experiments import run_control_comparison


if __name__ == "__main__":
    print(
        run_control_comparison(
            ControlExperimentConfig(
                difficulty="medium",
                axes=("x",),
                reference_profile_mode="rl_refline_six_phase",
                mddpg_shared_values=(1, 3, 5, 7, 10),
                train_episodes=200,
                compare_episodes=5,
                episodes=1,
                seed_runs=1,
            )
        )
    )
