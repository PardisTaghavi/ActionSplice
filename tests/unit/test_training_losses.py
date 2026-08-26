import unittest

import torch

from cst.training.train import _training_losses


class TrainingLossTests(unittest.TestCase):
    def test_compose_state_loss_ignores_clamped_prefix(self) -> None:
        target = torch.ones(1, 4, 1, 2, 2)
        prediction = target.clone()
        prediction[:, :2] = 100.0
        mask = torch.zeros(1, 4, 1, 1, 1)
        mask[:, 2:] = 1.0
        loss, state_nmse, _ = _training_losses(
            {"predicted_target": prediction, "delta": prediction},
            {
                "active_state": torch.zeros_like(target),
                "target_state": target,
                "temporal_suffix_mask": mask,
            },
            target_parameterization="state",
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(state_nmse), 0.0)

    def test_minwm_clean_prediction_and_delta_losses(self) -> None:
        target = torch.ones(1, 4, 1, 2, 2)
        zeros = torch.zeros_like(target)
        loss, state_nmse, terms = _training_losses(
            {"predicted_target": target, "delta": target},
            {
                "clean_target": target,
                "cached_prediction": zeros,
                "target_transition_noise": zeros,
                "target_sigma": torch.zeros(1),
                "target_state": target,
            },
            target_parameterization="clean_prediction",
            delta_loss_weight=1.0,
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(state_nmse), 0.0)
        self.assertIn("loss/delta_nmse", terms)


if __name__ == "__main__":
    unittest.main()
