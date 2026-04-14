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
                mddpg_shared_values=(7,),
                train_episodes=500,
                compare_episodes=5,
                episodes=1,
                seed_runs=1,
                output_subdir="x_axis_rl_refline__exp-v7ep500decay__ep-500__v-7__noise-linear-0.1-to-0.02__net-512__drop-0.2",
                snapshot_interval=10,
                exploration_noise_schedule="linear",
                exploration_noise_start=0.1,
                exploration_noise_end=0.02,
            )
        )
    )
