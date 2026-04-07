from __future__ import annotations

import unittest

import numpy as np

from lc.control.configs import get_axis_ladrc_action_bounds
from lc.control.controllers import AdaptiveLADRCController


class TestChapter3ActionBounds(unittest.TestCase):
    def test_xy_bounds_match_fast_and_steady_anchors(self) -> None:
        bounds = get_axis_ladrc_action_bounds("x")
        self.assertAlmostEqual(bounds.fast_anchor.b0, 30.5, places=6)
        self.assertAlmostEqual(bounds.fast_anchor.wc, 1.5, places=6)
        self.assertAlmostEqual(bounds.fast_anchor.k, 11.0, places=6)
        self.assertAlmostEqual(bounds.steady_anchor.b0, 4.0, places=6)
        self.assertAlmostEqual(bounds.steady_anchor.wc, 4.0, places=6)
        self.assertAlmostEqual(bounds.steady_anchor.k, 5.0, places=6)
        self.assertEqual(bounds.fixed_r, 10.0)

    def test_axis_controller_uses_configured_bounds(self) -> None:
        controller = AdaptiveLADRCController.for_axis("x")
        self.assertEqual(controller.b0_bounds, (1.35, 33.15))
        self.assertEqual(controller.omega_c_bounds, (1.25, 4.25))
        self.assertEqual(controller.k_bounds, (4.4, 11.6))
        controller.adapt(np.asarray([-1.0, -1.0, -1.0], dtype=np.float32))
        self.assertAlmostEqual(controller.base.b0, 1.35, places=6)
        self.assertAlmostEqual(controller.base.omega_c, 1.25, places=6)
        self.assertAlmostEqual(controller.base.k, 4.4, places=6)


if __name__ == "__main__":
    unittest.main()
