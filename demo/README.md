# Demo / Hugging Face Space

This demo uses precomputed, redistributable videos so it can run on a CPU-only
Hugging Face Space. It does not download private CST weights or either large
backbone.

```bash
python -m pip install -e '.[demo]'
python demo/app.py
```

Each `demo/examples/manifest.json` row has this shape:

```json
{
  "name": "example name",
  "condition_swap": "condition-swap.mp4",
  "cst": "actionsplice-cst-r-or-cst-t.mp4",
  "full_rollback": "full-rollback.mp4",
  "summary": "Backend, method, receipt step, and action change."
}
```

Keep generated media out of git until its license and prompt provenance have
been reviewed.
