from __future__ import annotations

import unittest

import numpy as np

from lc.control.controllers import AdaptiveLADRCController


class TestChapter3ActionParameterization(unittest.TestCase):
    def test_adaptive_ladrc_maps_action_to_b0_wc_k(self) -> None:
        controller = AdaptiveLADRCController()
        controller.adapt(np.asarray([-1.0, -1.0, -1.0], dtype=np.float32))
        self.assertAlmostEqual(controller.base.b0, controller.b0_bounds[0])
        self.assertAlmostEqual(controller.base.omega_c, controller.omega_c_bounds[0])
        self.assertAlmostEqual(controller.k, controller.k_bounds[0])
        self.assertAlmostEqual(controller.base.omega_o, controller.base.omega_c * controller.k)

        controller.adapt(np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
        self.assertAlmostEqual(controller.base.b0, controller.b0_bounds[1])
        self.assertAlmostEqual(controller.base.omega_c, controller.omega_c_bounds[1])
        self.assertAlmostEqual(controller.k, controller.k_bounds[1])
        self.assertAlmostEqual(controller.base.omega_o, controller.base.omega_c * controller.k)


if __name__ == "__main__":
    unittest.main()
