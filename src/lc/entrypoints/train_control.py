from __future__ import annotations

from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.experiments import run_pybullet_full_experiment


if __name__ == "__main__":
    print(run_pybullet_full_experiment(PyBulletControlExperimentConfig()))

