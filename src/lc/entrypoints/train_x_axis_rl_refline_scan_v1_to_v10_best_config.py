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
                mddpg_shared_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                train_episodes=500,
                compare_episodes=5,
                episodes=1,
                seed_runs=1,
                seed=7,
                hidden_dim=768,
                dropout_p=0.25,
                tau=0.02,
                soft_update_interval=10,
                output_subdir="x_axis_rl_refline__exp-bestcfg-scan-v1-to-v10__ep-500__v-1-10__noise-linear-0.1-to-0.04__net-768__drop-0.25",
                snapshot_interval=50,
                exploration_noise_schedule="linear",
                exploration_noise_start=0.1,
                exploration_noise_end=0.04,
            )
        )
    )
