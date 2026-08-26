import unittest
from pathlib import Path

import torch

from cst.backends.hyworld15_baselines import run_hyworld15_interruption_baseline
from cst.cli.baselines_hyworld15 import (
    _condition_from_commands,
    load_baseline_task,
)


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


class HYWorld15BaselineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        self.config = self.root / "configs/hyworld15/baselines.json"

    def test_config_has_exact_one_to_four_interruption_schedules(self):
        for count in range(1, 5):
            task = load_baseline_task(self.config, interruptions=count)
            self.assertEqual(len(task["event_pose_indices"]), count)
            self.assertGreater(task["num_latent_frames"], task["event_pose_indices"][-1] + 4)
            self.assertTrue(
                all(
                    right - left >= 8
                    for left, right in zip(
                        task["event_pose_indices"],
                        task["event_pose_indices"][1:],
                    )
                )
            )

    def test_full_rollback_discards_prefix_while_wait_delays_response(self):
        task = load_baseline_task(self.config, interruptions=1)
        frame_count = int(task["num_latent_frames"])
        torch.manual_seed(7)
        noise = torch.randn(1, 32, frame_count, 2, 2)
        requested = _condition_from_commands(task["requested_commands"])
        wait_condition = _condition_from_commands(task["wait_commands"])
        stale = {
            int(event): _condition_from_commands(commands)
            for event, commands in task["stale_commands_by_event"].items()
        }
        common = dict(
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
            wait_condition=wait_condition,
            stale_conditions_by_event=stale,
            event_pose_indices=task["event_pose_indices"],
            runtime_receipt_steps=[2],
            history_selector=lambda _poses, start, **_kwargs: list(range(max(0, start - 4), start)),
        )
        rollback = run_hyworld15_interruption_baseline(
            pipeline=_Pipeline(),
            policy="full_rollback",
            **common,
        )
        wait = run_hyworld15_interruption_baseline(
            pipeline=_Pipeline(),
            policy="wait",
            **common,
        )

        self.assertEqual(rollback.metrics["generator_call_count"], 22)
        self.assertEqual(rollback.metrics["discarded_nfe"], 2)
        rollback_event = rollback.metrics["interruption_events"][0]
        self.assertEqual(rollback_event["post_request_nfe_to_response"], 4)
        self.assertEqual(
            rollback_event["response_chunk_start_frame"],
            rollback_event["event_pose_index"],
        )

        self.assertEqual(wait.metrics["generator_call_count"], 20)
        self.assertEqual(wait.metrics["discarded_nfe"], 0)
        wait_event = wait.metrics["interruption_events"][0]
        self.assertEqual(wait_event["post_request_nfe_to_response"], 6)
        self.assertEqual(
            wait_event["response_chunk_start_frame"],
            wait_event["event_pose_index"] + 4,
        )
        event = int(task["event_pose_indices"][0])
        self.assertFalse(
            torch.equal(
                rollback.output[:, :, event : event + 4],
                wait.output[:, :, event : event + 4],
            )
        )


if __name__ == "__main__":
    unittest.main()
