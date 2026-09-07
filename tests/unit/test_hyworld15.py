import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cst.backends.hyworld15 import capture_hyworld15_recurrent_sequence
from cst.backends.hyworld15_inference import run_hyworld15_cst
from cst.cli._hyworld15_io import condition_from_commands
from cst.data.inference import load_inference_task


class _Scheduler:
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])


class _Transformer:
    def __call__(self, **kwargs):
        if kwargs.get("ar_txt_inference"):
            return [
                {
                    "k_vision": None,
                    "v_vision": None,
                    "k_txt": torch.ones(1),
                    "v_txt": torch.ones(1),
                }
            ]
        if kwargs.get("cache_vision"):
            cache = [dict(kwargs["kv_cache"][0])]
            cache[0]["k_vision"] = torch.ones(1)
            cache[0]["v_vision"] = torch.ones(1)
            return cache
        active = kwargs["hidden_states"][:, :32]
        action = kwargs["action"].float()[:, None, :, None, None]
        return (0.1 * active + 0.01 * action, None)


class _Pipeline:
    chunk_latent_frames = 4
    do_classifier_free_guidance = False
    target_dtype = torch.float32
    autocast_enabled = False
    guidance_scale = 1.0
    points_local = torch.zeros(1, 3)
    scheduler = _Scheduler()
    transformer = _Transformer()

    def init_kv_cache(self):
        empty = {
            "k_vision": None,
            "v_vision": None,
            "k_txt": None,
            "v_txt": None,
        }
        self._kv_cache = [dict(empty)]
        self._kv_cache_neg = [dict(empty)]


class _Corrector:
    def __init__(self, role: str) -> None:
        self.config = SimpleNamespace(
            transport_role=role,
            target_parameterization="state",
            latent_channels=32,
            denoising_steps=4,
            max_jump_horizon=0,
            max_rollout_age=3 if role == "action_h0_state" else 0,
        )

    def __call__(self, **inputs):
        active = inputs["active_state"]
        delta = torch.full_like(active, 0.01)
        mask = inputs.get("temporal_suffix_mask")
        if mask is not None:
            delta = delta * mask
        corrected = active + delta
        return {
            "predicted_target": corrected,
            "corrected_state": corrected,
            "delta": delta,
        }


class HYWorld15InferenceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        self.config = self.root / "configs/inference/hyworld15_cst_r.example.json"

    def test_inference_only_cst_r_and_cst_t_resume_at_same_step(self):
        task = load_inference_task(self.config, method="cst_r")
        frame_count = int(task["num_latent_frames"])
        torch.manual_seed(7)
        noise = torch.randn(1, 32, frame_count, 2, 2)
        requested = condition_from_commands(task["requested_commands"])
        stale = {
            int(event): condition_from_commands(commands)
            for event, commands in task["stale_commands_by_event"].items()
        }
        common = dict(
            pipeline=_Pipeline(),
            latents=noise,
            timesteps=torch.tensor([4.0, 3.0, 2.0, 1.0]),
            prompt_embeds=torch.zeros(1, 1, 1),
            prompt_mask=torch.ones(1, 1),
            vision_states=torch.zeros(1, 1, 1),
            cond_latents=torch.zeros(1, 33, frame_count, 2, 2),
            task_type="i2v",
            extra_kwargs={
                "byt5_text_states": torch.zeros(1, 1, 1),
                "byt5_text_mask": torch.ones(1, 1),
            },
            requested_condition=requested,
            stale_conditions_by_event=stale,
            event_pose_indices=task["event_pose_indices"],
            receipt_steps=[2],
            history_selector=lambda _poses, start, **_kwargs: list(range(max(0, start - 4), start)),
        )
        cst_r = run_hyworld15_cst(
            corrector=_Corrector("action_h0_state"),
            **common,
        )
        cst_t = run_hyworld15_cst(
            corrector=_Corrector("action_hm_state"),
            intra_chunk_offsets_by_event={task["event_pose_indices"][0]: 2},
            **common,
        )
        for result in (cst_r, cst_t):
            self.assertEqual(result.metrics["generator_call_count"], 16)
            event = result.metrics["interruption_events"][0]
            self.assertEqual(event["post_request_backbone_nfe"], 2)
            self.assertEqual(event["continuation_semantics"], "same_step_correct_then_resume")

    def test_recurrent_cst_t_capture_uses_masked_student_cleanup(self):
        task = load_inference_task(self.config, method="cst_r")
        frame_count = int(task["num_latent_frames"])
        requested = condition_from_commands(task["requested_commands"])
        stale = {
            int(event): condition_from_commands(commands)
            for event, commands in task["stale_commands_by_event"].items()
        }
        event = int(task["event_pose_indices"][0])
        result = capture_hyworld15_recurrent_sequence(
            pipeline=_Pipeline(),
            latents=torch.randn(1, 32, frame_count, 2, 2),
            timesteps=torch.tensor([4.0, 3.0, 2.0, 1.0]),
            prompt_embeds=torch.zeros(1, 1, 1),
            prompt_mask=torch.ones(1, 1),
            vision_states=torch.zeros(1, 1, 1),
            cond_latents=torch.zeros(1, 33, frame_count, 2, 2),
            task_type="i2v",
            extra_kwargs={
                "byt5_text_states": torch.zeros(1, 1, 1),
                "byt5_text_mask": torch.ones(1, 1),
            },
            requested_condition=requested,
            stale_conditions_by_event=stale,
            event_pose_indices=[event],
            runtime_receipt_steps=[2],
            transport_model=_Corrector("action_hm_state"),
            intra_chunk_offsets_by_event={event: 2},
            history_selector=lambda _poses, start, **_kwargs: list(range(max(0, start - 4), start)),
        )
        self.assertEqual(result.metrics["correction_count"], 1)
        self.assertEqual(len(result.captures), 1)
        self.assertEqual(result.captures[0].metadata["student_policy"], "cst_t")
        self.assertTrue(result.captures[0].metadata["teacher_prefix_clamped"])


if __name__ == "__main__":
    unittest.main()
