"""Camera action schedules compatible with minWM Wan Action2V.

The motion convention mirrors ``minWM/Wan21/wan_utils/camera_trajectory.py``:
translations are 0.08 units per latent frame and rotations are 3 degrees.
Unlike the upstream trajectory-string parser, this module also supports a
no-op action, which is required for the forward-to-stop experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

TRANSLATION_STEP = 0.08
ROTATION_STEP_RADIANS = np.deg2rad(3.0)
VALID_ACTIONS = frozenset({"w", "s", "a", "d", "u", "dn", "j", "l", "i", "k", "noop"})

# HY-World 1.5 combines one of nine translation labels with one of nine
# rotation labels as ``translation * 9 + rotation``.  CST schedules use only
# elementary controls, so their exact upstream labels can be written without
# importing ``hyvideo.generate`` (which initializes CUDA at module import).
HYWORLD15_ELEMENTARY_ACTION_LABELS = {
    "noop": 0,
    "w": 9,
    "s": 18,
    "d": 27,
    "a": 36,
    "l": 1,
    "j": 2,
    "i": 3,
    "k": 4,
}


def expand_chunk_action_blocks(
    blocks: Sequence[str],
    chunk_size: int,
) -> list[str]:
    """Expand one action per chunk into pose-transition commands."""
    if chunk_size <= 0 or not blocks:
        raise ValueError("Action blocks and chunk_size must be positive")
    invalid = sorted(set(blocks) - VALID_ACTIONS)
    if invalid:
        raise ValueError(f"Unknown camera actions: {invalid}")
    commands: list[str] = []
    for index, action in enumerate(blocks):
        commands.extend([action] * (chunk_size - 1 if index == 0 else chunk_size))
    return commands


def build_event_stale_command_schedules(
    blocks: Sequence[str],
    chunk_size: int,
) -> dict[int, list[str]]:
    """Build a history-anchored stale trajectory for every later chunk.

    Each schedule follows the requested trajectory through the preceding
    committed pose, then extends the previous action across the active chunk.
    This avoids the absolute-pose drift produced by one globally shifted path.
    """
    requested = expand_chunk_action_blocks(blocks, chunk_size)
    schedules: dict[int, list[str]] = {}
    for block_index in range(1, len(blocks)):
        start_pose = block_index * chunk_size
        command_start = start_pose - 1
        stale = list(requested)
        stale[command_start : command_start + chunk_size] = [blocks[block_index - 1]] * chunk_size
        schedules[start_pose] = stale
    return schedules


@dataclass(frozen=True)
class Scenario:
    name: str
    old_action: str
    new_action: str
    description: str
    is_control: bool = False
    is_stress: bool = False


SCENARIOS = {
    "constant_forward": Scenario(
        name="constant_forward",
        old_action="w",
        new_action="w",
        description="Constant forward motion; no action change.",
        is_control=True,
    ),
    "forward_stop": Scenario(
        name="forward_stop",
        old_action="w",
        new_action="noop",
        description="Forward motion followed by a stop.",
    ),
    "forward_backward": Scenario(
        name="forward_backward",
        old_action="w",
        new_action="s",
        description="Forward motion followed by backward motion.",
    ),
    "forward_yaw_left": Scenario(
        name="forward_yaw_left",
        old_action="w",
        new_action="j",
        description="Forward motion followed by a left yaw.",
    ),
    "forward_yaw_right": Scenario(
        name="forward_yaw_right",
        old_action="w",
        new_action="l",
        description="Forward motion followed by a right yaw.",
    ),
    "yaw_left_right_reversal": Scenario(
        name="yaw_left_right_reversal",
        old_action="j",
        new_action="l",
        description="Left yaw followed by a right-yaw reversal.",
    ),
    "yaw_left_stop": Scenario(
        name="yaw_left_stop",
        old_action="j",
        new_action="noop",
        description="Left yaw followed by a stop.",
    ),
    "yaw_left_pitch_up": Scenario(
        name="yaw_left_pitch_up",
        old_action="j",
        new_action="i",
        description="Left yaw followed by upward pitch.",
    ),
    "yaw_left_forward": Scenario(
        name="yaw_left_forward",
        old_action="j",
        new_action="w",
        description="Left yaw followed by forward translation.",
    ),
    "yaw_left_backward": Scenario(
        name="yaw_left_backward",
        old_action="j",
        new_action="s",
        description="Left yaw followed by backward translation.",
    ),
    "backward_forward": Scenario(
        name="backward_forward",
        old_action="s",
        new_action="w",
        description="Backward motion followed by forward motion.",
    ),
    "backward_yaw_left": Scenario(
        name="backward_yaw_left",
        old_action="s",
        new_action="j",
        description="Backward motion followed by a left yaw.",
    ),
    "rapid_yaw_alternation": Scenario(
        name="rapid_yaw_alternation",
        old_action="j",
        new_action="alternating_j_l",
        description="Yaw direction alternates at every latent-frame transition.",
        is_stress=True,
    ),
}

INJECTION_AFTER_DENOISING_STEPS = {
    "before_chunk": 0,
    "after_step_1": 1,
    "after_step_2": 2,
    "after_step_3": 3,
    # None means the generated event chunk and its KV update retain the old
    # action. The new action begins at the next chunk.
    "after_chunk": None,
    "control": None,
}


def _rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _apply_action(c2w: np.ndarray, action: str) -> np.ndarray:
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Unknown camera action {action!r}; expected one of {sorted(VALID_ACTIONS)}"
        )

    updated = c2w.copy()
    rotation = updated[:3, :3]

    if action == "j":
        updated[:3, :3] = rotation @ _rotation_y(-ROTATION_STEP_RADIANS)
    elif action == "l":
        updated[:3, :3] = rotation @ _rotation_y(ROTATION_STEP_RADIANS)
    elif action == "i":
        updated[:3, :3] = rotation @ _rotation_x(ROTATION_STEP_RADIANS)
    elif action == "k":
        updated[:3, :3] = rotation @ _rotation_x(-ROTATION_STEP_RADIANS)
    elif action in {"w", "s", "a", "d", "u", "dn"}:
        local_delta = {
            "w": np.array([0.0, 0.0, TRANSLATION_STEP]),
            "s": np.array([0.0, 0.0, -TRANSLATION_STEP]),
            "a": np.array([-TRANSLATION_STEP, 0.0, 0.0]),
            "d": np.array([TRANSLATION_STEP, 0.0, 0.0]),
            "u": np.array([0.0, -TRANSLATION_STEP, 0.0]),
            "dn": np.array([0.0, TRANSLATION_STEP, 0.0]),
        }[action]
        updated[:3, 3] += rotation @ local_delta

    return updated


def commands_to_viewmats(commands: Sequence[str]) -> np.ndarray:
    """Convert N-1 actions into N world-to-camera matrices."""
    c2w = np.eye(4, dtype=np.float64)
    poses = [c2w.copy()]
    for action in commands:
        c2w = _apply_action(c2w, action)
        poses.append(c2w.copy())
    return np.stack([np.linalg.inv(pose) for pose in poses]).astype(np.float32)


def commands_to_hyworld15_actions(commands: Sequence[str]) -> np.ndarray:
    """Convert pose-transition commands to official HY-World 1.5 labels.

    The first latent pose has no incoming transition and therefore uses the
    no-op label, matching ``hyvideo.generate.pose_to_input``.
    """
    unsupported = sorted(set(commands) - set(HYWORLD15_ELEMENTARY_ACTION_LABELS))
    if unsupported:
        raise ValueError(f"HY-World 1.5 does not define elementary CST labels for {unsupported}")
    return np.asarray(
        [0, *[HYWORLD15_ELEMENTARY_ACTION_LABELS[value] for value in commands]],
        dtype=np.int64,
    )


def make_intrinsics(
    num_frames: int,
    fx: float = 0.5,
    fy: float = 0.5,
    cx: float = 0.5,
    cy: float = 0.5,
) -> np.ndarray:
    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return np.repeat(intrinsic[None], num_frames, axis=0)


def _new_action_tail(new_action: str, count: int) -> list[str]:
    if new_action == "alternating_j_l":
        return ["l" if index % 2 == 0 else "j" for index in range(count)]
    if new_action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported new action {new_action!r}")
    return [new_action] * count


def build_command_pair(
    scenario_name: str,
    num_latent_frames: int,
    effective_pose_index: int,
) -> tuple[list[str], list[str]]:
    """Return the stale-action and desired-action command sequences.

    ``effective_pose_index`` is the first pose that should reflect the new
    action. Pose zero is identity, so its generating command has index
    ``effective_pose_index - 1``.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario_name!r}")
    if not 1 <= effective_pose_index < num_latent_frames:
        raise ValueError(
            "effective_pose_index must be within generated poses and greater than zero"
        )

    scenario = SCENARIOS[scenario_name]
    command_count = num_latent_frames - 1
    stale = [scenario.old_action] * command_count
    if scenario.is_control:
        return stale, stale.copy()

    prefix_count = effective_pose_index - 1
    desired = stale[:prefix_count] + _new_action_tail(
        scenario.new_action,
        command_count - prefix_count,
    )
    return stale, desired


def validate_experiment_geometry(
    num_latent_frames: int,
    chunk_size: int,
    event_pose_index: int,
) -> None:
    if num_latent_frames % chunk_size != 0:
        raise ValueError(
            "num_latent_frames must be divisible by chunk_size for minWM causal inference"
        )
    if event_pose_index % chunk_size != 0:
        raise ValueError("event_pose_index must be the first pose of a generated chunk")
    if event_pose_index + chunk_size > num_latent_frames:
        raise ValueError("event chunk must fit inside num_latent_frames")


def action_counts(commands: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for command in commands:
        counts[command] = counts.get(command, 0) + 1
    return counts
