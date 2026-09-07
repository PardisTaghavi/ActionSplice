import unittest

import torch

from cst.training.train import _resolve_loss_weights, _training_losses


class TrainingLossTests(unittest.TestCase):
    def test_cst_t_state_loss_ignores_clamped_prefix(self) -> None:
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

    def test_state_lambda_scales_primary_loss(self) -> None:
        target = torch.ones(1, 4, 1, 2, 2)
        prediction = torch.zeros_like(target)
        loss, state_nmse, terms = _training_losses(
            {"predicted_target": prediction, "delta": prediction},
            {
                "active_state": prediction,
                "target_state": target,
            },
            target_parameterization="state",
            state_loss_weight=0.25,
        )
        self.assertAlmostEqual(float(state_nmse), 1.0)
        self.assertAlmostEqual(float(loss), 0.25)
        self.assertAlmostEqual(float(terms["loss/state_weighted"]), 0.25)

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

    def test_hy_loss_defaults_are_method_aware(self) -> None:
        cst_r = _resolve_loss_weights({"method": "cst_r"})
        cst_t = _resolve_loss_weights({"method": "cst_t"})
        self.assertEqual(cst_r.lambda_state, 1.0)
        self.assertEqual(cst_r.lambda_residual, 0.0)
        self.assertEqual(cst_r.lambda_lpips, 0.05)
        self.assertEqual(cst_r.lambda_temporal, 0.1)
        self.assertEqual(cst_r.lambda_history_boundary, 0.1)
        self.assertEqual(cst_r.lambda_mid, 0.0)
        self.assertEqual(cst_t.lambda_mid, 0.1)

    def test_nested_loss_lambdas_override_defaults(self) -> None:
        weights = _resolve_loss_weights(
            {
                "method": "cst_t",
                "loss_weights": {
                    "lambda_residual": 1.0,
                    "lambda_lpips": 0.0,
                    "lambda_mid": 0.25,
                },
            }
        )
        self.assertEqual(weights.lambda_state, 1.0)
        self.assertEqual(weights.lambda_residual, 1.0)
        self.assertEqual(weights.lambda_lpips, 0.0)
        self.assertEqual(weights.lambda_mid, 0.25)

    def test_cst_r_rejects_mid_boundary_lambda(self) -> None:
        with self.assertRaisesRegex(ValueError, "CST-R requires"):
            _resolve_loss_weights(
                {
                    "method": "cst_r",
                    "loss_weights": {"lambda_mid": 0.1},
                }
            )


if __name__ == "__main__":
    unittest.main()
