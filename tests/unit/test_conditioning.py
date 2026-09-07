import unittest

import numpy as np

from cst.backends.conditioning import build_command_pair, commands_to_viewmats


class ActionScheduleTests(unittest.TestCase):
    def test_stop_freezes_pose(self):
        _, desired = build_command_pair("forward_stop", 20, 8)
        viewmats = commands_to_viewmats(desired)
        np.testing.assert_allclose(viewmats[8], viewmats[-1], atol=1e-6)

    def test_forward_backward_changes_direction(self):
        _, desired = build_command_pair("forward_backward", 20, 8)
        self.assertEqual(desired[6], "w")
        self.assertEqual(desired[7], "s")
        self.assertEqual(len(commands_to_viewmats(desired)), 20)


if __name__ == "__main__":
    unittest.main()
