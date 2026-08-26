"""Precomputed side-by-side demo suitable for a lightweight HF Space."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "examples" / "manifest.json"


def load_examples() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        return []
    return list(json.loads(MANIFEST.read_text(encoding="utf-8")))


EXAMPLES = load_examples()


def select_example(name: str):
    row = next((item for item in EXAMPLES if item["name"] == name), None)
    if row is None:
        return None, None, None, "No public demo artifacts have been staged."
    return (
        str(ROOT / "examples" / row["condition_swap"]),
        str(ROOT / "examples" / row["cst"]),
        str(ROOT / "examples" / row["full_rollback"]),
        row.get("summary", ""),
    )


with gr.Blocks(title="ActionSplice") as demo:
    gr.Markdown("# ActionSplice")
    gr.Markdown(
        "Compare an interrupted world-model rollout using condition swap, CST, "
        "and exact full rollback. This lightweight demo uses precomputed runs."
    )
    selector = gr.Dropdown(
        choices=[item["name"] for item in EXAMPLES],
        label="Example",
    )
    with gr.Row():
        condition_swap = gr.Video(label="Condition swap")
        cst = gr.Video(label="CST")
        full_rollback = gr.Video(label="Full rollback")
    summary = gr.Markdown()
    selector.change(
        select_example,
        inputs=selector,
        outputs=[condition_swap, cst, full_rollback, summary],
    )


if __name__ == "__main__":
    demo.launch()
