from __future__ import annotations

from lc.control.configs import AxisTransferExperimentConfig
from lc.control.experiments import run_y_axis_refline_transfer_suite


if __name__ == "__main__":
    print(
        run_y_axis_refline_transfer_suite(
            AxisTransferExperimentConfig(
                output_subdir="y_axis_refline__exp-xpid-y-transfer-suite",
            )
        )
    )
