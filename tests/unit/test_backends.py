import json
import unittest
from pathlib import Path

from cst.backends import get_backend, validate_training_config
from cst.data.inference import load_inference_task
from cst.data.recurrent_manifest import build_recurrent_capture_tasks
from cst.data.transport_manifest import build_transport_capture_tasks
from cst.training.train import _resolve_loss_weights

ROOT = Path(__file__).resolve().parents[2]


class BackendRegistryTests(unittest.TestCase):
    def test_public_method_to_checkpoint_roles(self) -> None:
        self.assertEqual(get_backend("minwm").role("cst_r"), "action_h0")
        self.assertEqual(get_backend("minwm").role("cst_t"), "action_hm")
        self.assertEqual(get_backend("hy").role("cst_r"), "action_h0_state")
        self.assertEqual(get_backend("hy").role("cst_t"), "action_hm_state")

    def test_release_training_configs_match_backend_contract(self) -> None:
        paths = (
            ROOT / "configs/minwm/train_cst_r.json",
            ROOT / "configs/minwm/train_cst_t.json",
            ROOT / "configs/hyworld15/train_cst_r.json",
            ROOT / "configs/hyworld15/train_cst_t.json",
        )
        for path in paths:
            with self.subTest(path=path):
                config = json.loads(path.read_text(encoding="utf-8"))
                backend, method = validate_training_config(config)
                self.assertEqual(backend.name, config["backend"])
                self.assertEqual(method, config["method"])
                self.assertEqual(
                    set(_resolve_loss_weights(config).to_dict()),
                    {
                        "lambda_state",
                        "lambda_residual",
                        "lambda_lpips",
                        "lambda_temporal",
                        "lambda_history_boundary",
                        "lambda_mid",
                    },
                )

    def test_rejects_horizon_or_wrong_mask(self) -> None:
        config = {
            "backend": "minwm",
            "method": "cst_t",
            "transport_role": "action_hm",
            "target_parameterization": "clean_prediction",
            "use_temporal_suffix_mask": False,
            "jump_horizons": [1],
        }
        with self.assertRaises(ValueError):
            validate_training_config(config)

    def test_hy_paper_capture_configs_have_expected_counts(self) -> None:
        root = ROOT / "configs/hyworld15"
        self.assertEqual(len(build_recurrent_capture_tasks(root / "capture_cst_r.json")), 150)
        self.assertEqual(len(build_transport_capture_tasks(root / "capture_cst_t.json")), 450)

    def test_minwm_capture_configs_have_expected_counts(self) -> None:
        root = ROOT / "configs/minwm"
        self.assertEqual(len(build_recurrent_capture_tasks(root / "capture_cst_r.json")), 150)
        self.assertEqual(
            len(build_transport_capture_tasks(root / "capture_cst_r_bootstrap.json")), 150
        )
        self.assertEqual(len(build_transport_capture_tasks(root / "capture_cst_t.json")), 450)

    def test_inference_examples_resolve_events_receipts_and_boundaries(self) -> None:
        root = ROOT / "configs/inference"
        cst_r = load_inference_task(root / "minwm_cst_r.example.json", method="cst_r")
        cst_t = load_inference_task(root / "hyworld15_cst_t.example.json", method="cst_t")
        self.assertEqual(cst_r["event_pose_indices"], [8])
        self.assertEqual(cst_r["receipt_steps"], [2])
        self.assertEqual(cst_t["intra_chunk_offsets_by_event"], {8: 2})


if __name__ == "__main__":
    unittest.main()
