import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cst.backends.conditioning import commands_to_viewmats, make_intrinsics
from cst.backends.minwm_inference import run_minwm_cst
from cst.backends.minwm_recurrent import capture_recurrent_cst_r
from cst.data.inference import load_inference_task

ROOT = Path(__file__).resolve().parents[2]


class _Scheduler:
    def add_noise(self, prediction, noise, _timestep):
        return 0.75 * prediction + 0.25 * noise


class _VAE:
    def decode_to_pixel(self, latents, *, use_cache):
        return latents[:, :, :3]


class _Pipeline:
    independent_first_frame = False
    num_frame_per_block = 4
    denoising_step_list = torch.tensor([4, 3, 2, 1])
    frame_seq_length = 1
    args = SimpleNamespace(context_noise=0)
    scheduler = _Scheduler()
    vae = _VAE()
    kv_cache1 = None
    crossattn_cache = None
    prope_kv_cache1 = None

    def text_encoder(self, *, text_prompts):
        return {"prompt": text_prompts}

    def _initialize_kv_cache(self, *_args):
        self.kv_cache1 = [{}]

    def _initialize_crossattn_cache(self, *_args):
        self.crossattn_cache = [{"is_init": False}]

    def _initialize_prope_kv_cache(self, *_args):
        self.prope_kv_cache1 = [{}]

    def generator(self, **kwargs):
        active = kwargs["noisy_image_or_video"]
        camera = kwargs["viewmats"][..., :1, :1].reshape(active.shape[0], active.shape[1], 1, 1, 1)
        return None, 0.9 * active + 0.01 * camera


class _Corrector:
    def __init__(self, role: str) -> None:
        self.config = SimpleNamespace(
            transport_role=role,
            target_parameterization="clean_prediction",
            latent_channels=2,
            denoising_steps=4,
            max_rollout_age=3 if role == "action_h0" else 0,
        )

    def __call__(self, **inputs):
        base = inputs["cached_prediction"]
        delta = torch.full_like(base, 0.01)
        mask = inputs.get("temporal_suffix_mask")
        if mask is not None:
            delta = delta * mask
        prediction = base + delta
        return {
            "predicted_target": prediction,
            "corrected_state": prediction,
            "delta": delta,
        }


class MinWMInferenceTests(unittest.TestCase):
    def _run(self, method: str):
        task = load_inference_task(
            ROOT / f"configs/inference/minwm_{method}.example.json",
            method=method,
        )
        frame_count = task["num_latent_frames"]
        requested = torch.from_numpy(commands_to_viewmats(task["requested_commands"])).unsqueeze(0)
        stale = {
            event: torch.from_numpy(commands_to_viewmats(commands)).unsqueeze(0)
            for event, commands in task["stale_commands_by_event"].items()
        }
        return run_minwm_cst(
            pipeline=_Pipeline(),
            noise=torch.randn(1, frame_count, 2, 2, 2),
            text_prompt=task["prompt"],
            requested_viewmats=requested,
            stale_viewmats_by_event=stale,
            intrinsics=torch.from_numpy(make_intrinsics(frame_count)).unsqueeze(0),
            event_pose_indices=task["event_pose_indices"],
            receipt_steps=task["receipt_steps"],
            corrector=_Corrector("action_h0" if method == "cst_r" else "action_hm"),
            intra_chunk_offsets_by_event=(
                task["intra_chunk_offsets_by_event"] if method == "cst_t" else None
            ),
            decode=False,
        )

    def test_cst_r_and_cst_t_use_one_corrector_and_two_cleanup_calls(self):
        for method in ("cst_r", "cst_t"):
            with self.subTest(method=method):
                result = self._run(method)
                self.assertIsNone(result.video)
                self.assertEqual(result.metrics["generator_call_count"], 20)
                self.assertEqual(len(result.metrics["correction_events"]), 1)
                event = result.metrics["interruption_events"][0]
                self.assertEqual(event["post_request_backbone_nfe"], 2)
                self.assertEqual(event["continuation_semantics"], "same_step_correct_then_resume")

    def test_recurrent_capture_commits_student_history(self):
        task = load_inference_task(
            ROOT / "configs/inference/minwm_cst_r.example.json",
            method="cst_r",
        )
        frame_count = task["num_latent_frames"]
        requested = torch.from_numpy(commands_to_viewmats(task["requested_commands"])).unsqueeze(0)
        stale = {
            event: torch.from_numpy(commands_to_viewmats(commands)).unsqueeze(0)
            for event, commands in task["stale_commands_by_event"].items()
        }
        captures, output, metrics = capture_recurrent_cst_r(
            pipeline=_Pipeline(),
            noise=torch.randn(1, frame_count, 2, 2, 2),
            text_prompt=task["prompt"],
            stale_viewmats_by_event=stale,
            requested_viewmats=requested,
            intrinsics=torch.from_numpy(make_intrinsics(frame_count)).unsqueeze(0),
            event_pose_indices=task["event_pose_indices"],
            receipt_steps=task["receipt_steps"],
            corrector=_Corrector("action_h0"),
        )
        self.assertEqual(len(captures), 1)
        self.assertEqual(output.shape[1], frame_count)
        self.assertEqual(metrics["correction_count"], 1)
        self.assertEqual(
            captures[0].metadata["teacher_history_source"],
            "student_committed_history",
        )


if __name__ == "__main__":
    unittest.main()
