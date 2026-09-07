"""Export HY-WM1.5 CST-R and CST-T manifests with a shared scene split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..data.recurrent_manifest import build_recurrent_capture_tasks
from ..data.transport_manifest import build_transport_capture_tasks

SHARED_NAME = "cst_shared_paper_150_v1"
CST_R_NAME = "cst_r_paper_150_v1"
CST_T_NAME = "cst_t_paper_150_v1"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_reference_pairs(
    path: Path,
    *,
    reference_root: Path | None,
    prompts: list[str],
    train_prompt_count: int,
    check_images: bool = True,
) -> dict[int, dict[str, Any]]:
    """Load one reference image and scene id for every prompt index.

    Accepted JSON is either a list or ``{"examples": [...]}``. Each entry must
    contain ``prompt_index``, ``reference_image``, and ``scene_id``. An optional
    ``prompt`` field is checked against the capture configuration.
    """
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    rows = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Reference manifest must be a list or contain examples")
    if len(rows) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} reference rows, received {len(rows)}")
    root = path.resolve().parent if reference_root is None else reference_root.resolve()
    indexed: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Every reference row must be an object")
        prompt_index = int(raw["prompt_index"])
        if prompt_index in indexed:
            raise ValueError(f"Duplicate prompt_index {prompt_index}")
        if not 0 <= prompt_index < len(prompts):
            raise ValueError(f"Invalid prompt_index {prompt_index}")
        if "prompt" in raw and str(raw["prompt"]) != prompts[prompt_index]:
            raise ValueError(f"Prompt text mismatch at index {prompt_index}")
        scene_id = str(raw.get("scene_id", "")).strip()
        if not scene_id:
            raise ValueError(f"Missing scene_id at prompt {prompt_index}")
        image = Path(str(raw["reference_image"]))
        image = image if image.is_absolute() else root / image
        image = image.resolve()
        if check_images and not image.is_file():
            raise FileNotFoundError(image)
        if check_images:
            image_sha256 = _sha256(image)
        else:
            image_sha256 = str(raw.get("image_sha256", ""))
            if len(image_sha256) != 64:
                raise ValueError("Unchecked reference rows must provide image_sha256")
        indexed[prompt_index] = {
            "prompt_index": prompt_index,
            "prompt": prompts[prompt_index],
            "reference_image": str(image),
            "scene_id": scene_id,
            "image_sha256": image_sha256,
        }
    if set(indexed) != set(range(len(prompts))):
        raise ValueError("Reference manifest must cover every prompt index exactly")
    train_scenes = {indexed[index]["scene_id"] for index in range(train_prompt_count)}
    heldout_scenes = {
        indexed[index]["scene_id"] for index in range(train_prompt_count, len(prompts))
    }
    overlap = sorted(train_scenes & heldout_scenes)
    if overlap:
        raise ValueError(
            "Reference scenes leak across train/held-out split: " + ", ".join(overlap[:5])
        )
    hashes = [indexed[index]["image_sha256"] for index in range(len(prompts))]
    if len(set(hashes)) != len(prompts):
        raise ValueError("Reference manifest must contain 150 unique image hashes")
    train_hashes = set(hashes[:train_prompt_count])
    heldout_hashes = set(hashes[train_prompt_count:])
    if train_hashes & heldout_hashes:
        raise ValueError("Identical reference images leak across the dataset split")
    return indexed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_hyworld15_paper_manifests(
    *,
    cst_r_config: Path,
    cst_t_config: Path,
    reference_manifest: Path,
    reference_root: Path | None,
    check_images: bool = True,
) -> dict[str, Any]:
    cst_r = build_recurrent_capture_tasks(cst_r_config)
    cst_t = build_transport_capture_tasks(cst_t_config)
    if len(cst_r) != 150 or len(cst_t) != 450:
        raise ValueError(
            f"Expected 150 CST-R tasks and 450 CST-T tasks; received {len(cst_r)} and {len(cst_t)}"
        )
    prompts = [str(task["prompt"]) for task in cst_r]
    if len(set(int(task["prompt_index"]) for task in cst_r)) != 150:
        raise ValueError("CST-R manifest must contain one task per prompt")
    prompts = [
        next(str(task["prompt"]) for task in cst_r if int(task["prompt_index"]) == index)
        for index in range(150)
    ]
    train_prompt_count = sum(str(task["dataset_split"]) == "train" for task in cst_r)
    if train_prompt_count != 120:
        raise ValueError("HY paper split must contain 120 training prompts")
    references = load_reference_pairs(
        reference_manifest,
        reference_root=reference_root,
        prompts=prompts,
        train_prompt_count=train_prompt_count,
        check_images=check_images,
    )

    for task in [*cst_r, *cst_t]:
        prompt_index = int(task["prompt_index"])
        reference = references[prompt_index]
        task.update(
            backbone="official_hyworld15_action2v",
            reference_image=reference["reference_image"],
            scene_id=reference["scene_id"],
            reference_image_sha256=reference["image_sha256"],
        )

    cst_t_by_master: dict[str, list[dict[str, Any]]] = {}
    for task in cst_t:
        cst_t_by_master.setdefault(str(task["master_id"]), []).append(task)
    masters: list[dict[str, Any]] = []
    for task in cst_r:
        master_id = str(task["master_id"])
        cst_t_tasks = cst_t_by_master.get(master_id, [])
        if len(cst_t_tasks) != 3:
            raise ValueError(f"{master_id} must have m=1,2,3 CST-T tasks")
        masters.append(
            {
                "master_id": master_id,
                "prompt_index": int(task["prompt_index"]),
                "prompt": str(task["prompt"]),
                "reference_image": str(task["reference_image"]),
                "scene_id": str(task["scene_id"]),
                "reference_image_sha256": str(task["reference_image_sha256"]),
                "split": str(task["dataset_split"]),
                "seed": int(task["seed"]),
                "cst_r": {
                    "sequence_id": str(task["recurrent_sequence_id"]),
                    "event_pose_indices": list(task["event_pose_indices"]),
                    "num_latent_frames": int(task["num_latent_frames"]),
                    "receipt_steps_saved": [1, 2, 3],
                },
                "cst_t": {
                    "scenario": str(cst_t_tasks[0]["scenario"]),
                    "old_action": str(cst_t_tasks[0]["old_action"]),
                    "new_action": str(cst_t_tasks[0]["new_action"]),
                    "boundaries_m": sorted(
                        int(value["intra_chunk_offset"]) for value in cst_t_tasks
                    ),
                    "receipt_steps_saved": [1, 2, 3],
                },
            }
        )
    split = {
        "backbone": "official_hyworld15_action2v",
        "strategy": "fixed_prompt_and_scene_disjoint",
        "group_by": ["prompt", "scene_id"],
        "train_master_ids": [row["master_id"] for row in masters if row["split"] == "train"],
        "heldout_master_ids": [row["master_id"] for row in masters if row["split"] == "heldout"],
        "counts": {
            "masters": 150,
            "train_masters": 120,
            "heldout_masters": 30,
            "cst_r_sequences": 150,
            "cst_r_event_captures": 450,
            "cst_t_captures": 450,
        },
    }
    return {"masters": masters, "split": split, "cst_r": cst_r, "cst_t": cst_t}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cst-r-config", type=Path, required=True)
    parser.add_argument("--cst-t-config", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    payload = build_hyworld15_paper_manifests(
        cst_r_config=args.cst_r_config,
        cst_t_config=args.cst_t_config,
        reference_manifest=args.reference_manifest,
        reference_root=args.reference_root,
        check_images=not args.skip_image_check,
    )
    root = args.dataset_root.resolve()
    _write(root / SHARED_NAME / "master_manifest.json", payload["masters"])
    _write(root / SHARED_NAME / "split.json", payload["split"])
    _write(root / CST_R_NAME / "task_manifest.json", payload["cst_r"])
    _write(root / CST_T_NAME / "task_manifest.json", payload["cst_t"])
    print(json.dumps(payload["split"]["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
