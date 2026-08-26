import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cst.core.model import CounterfactualTransport, TransportModelConfig, checkpoint_payload
from cst.core.runtime import apply_transport_model, load_transport_model


class RuntimeTests(unittest.TestCase):
    def test_loads_same_step_correct_checkpoint(self) -> None:
        model = CounterfactualTransport(
            TransportModelConfig(
                latent_channels=2,
                denoising_steps=4,
                base_channels=8,
                condition_channels=16,
                target_parameterization="state",
                transport_role="action_h0_state",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cst-r.pt"
            torch.save(
                checkpoint_payload(model, optimizer=None, step=3, validation_nmse=0.1),
                path,
            )
            loaded, metadata = load_transport_model(
                path,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        self.assertEqual(loaded.config.transport_role, "action_h0_state")
        self.assertEqual(metadata["training_step"], 3)

    def test_runtime_rejects_solver_step_jump(self) -> None:
        fake = SimpleNamespace(config=SimpleNamespace(transport_role="action_h0_state"))
        with self.assertRaisesRegex(ValueError, "jump_horizon must be 0"):
            apply_transport_model(
                model=fake,
                active_state=None,
                initial_state=None,
                history_tail=None,
                transition_noises=[],
                old_viewmats=None,
                new_viewmats=None,
                intrinsics=None,
                receipt_step=1,
                jump_horizon=1,
            )


if __name__ == "__main__":
    unittest.main()
