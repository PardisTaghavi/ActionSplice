# ActionSplice

ActionSplice edits an active world-model generation when control changes before
the current chunk has finished sampling. Its core operation, Counterfactual
State Transport (CST), has two variants on minWM and HY-WM1.5/HY-WorldPlay:

- **CST-R** corrects the complete active chunk at solver step `r`.
- **CST-T** preserves a committed prefix and corrects only the remaining
  suffix using a hard temporal mask.

Both methods are same-step correctors. They do not skip solver calls, use a
gate, or fall back to rollback. After one corrector forward, the unmodified
backbone performs its ordinary `K-r` remaining solver calls.

## Layout

```text
ActionSplice/
├── src/cst/
│   ├── backends/       shared registry + minWM/HY integration code
│   ├── core/           corrector, state layouts, same-step runtime
│   ├── data/           capture schema, teacher-pair dataset, manifests
│   ├── training/       verified training loop and existing losses
│   └── cli/            capture and evaluation entry points
├── configs/
│   ├── minwm/          CST-R and CST-T training configs
│   ├── hyworld15/      bootstrap/paper capture + training configs
│   └── models/         unpublished-weight placeholders
├── demo/               lightweight precomputed Gradio/HF Space
├── docs/               training, inference, and model-loading guides
├── scripts/            pinned upstream setup and future HF download script
└── tests/               CPU unit tests
```

The Python package remains `cst` because CST is ActionSplice's core operation.
It uses `cst.backends` as
the shared `Backend/` layer. It centralizes backend state shape, target type,
and checkpoint-role validation. It does not duplicate either upstream model.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Install the heavy training or demo dependencies only when needed:

```bash
python -m pip install -e '.[train]'
python -m pip install -e '.[demo]'
```

minWM and HY-WorldPlay both provide a top-level `hyvideo` package. Use separate
environments for them. Their pinned revisions are prepared with:

```bash
bash scripts/prepare_backends.sh
```

## Implementation status

| Path | Training | Inference |
|---|---|---|
| minWM CST-R | Trainer/config and independent bootstrap capture | Not yet wired into a generation hook |
| minWM CST-T | Trainer/config and matched prefix-clamped capture | Not yet wired into a generation hook |
| HY-WM1.5 CST-R | Bootstrap, recurrent paper capture, trainer/config | Wired through `OfficialHYWorldPlayCSTHook` |
| HY-WM1.5 CST-T | Matched prefix-clamped capture and trainer/config | Not yet wired; the backend rejects this path explicitly |

Thus, the four training configurations are present, but exact end-to-end paper
reproduction still needs the final recurrent minWM CST-R capture migration.
Inference is only complete for HY-WM1.5 CST-R. See
[Release checklist](RELEASE_CHECKLIST.md).

## Train

| Backend | CST-R | CST-T |
|---|---|---|
| minWM | `configs/minwm/train_cst_r.json` | `configs/minwm/train_cst_t.json` |
| HY-WM1.5 | `configs/hyworld15/train_cst_r.json` | `configs/hyworld15/train_cst_t.json` |

```bash
actionsplice-train \
  --capture-dir /path/to/captures \
  --output-dir outputs/minwm-cst-r \
  --config configs/minwm/train_cst_r.json
```

Replace every `<PATH_TO_...>` value first. These are the existing research
losses and settings; this cleanup does not introduce a new objective. See
[Training](docs/training.md).

The existing HY paper-data implementation is exposed as
`actionsplice-export-hyworld15-manifests` and
`actionsplice-capture-hyworld15 --mode {cst_r,cst_t}`. A separate
`actionsplice-capture-hyworld15-bootstrap` command keeps
the initial CST-R bootstrap workflow explicit.

The existing independent matched-teacher capture for minWM is exposed as
`actionsplice-capture-minwm`; use `configs/minwm/capture_cst_r_bootstrap.json` or
`configs/minwm/capture_cst_t.json`.

## Load weights and run inference

CST corrector weights are **not public yet**. Placeholder repository IDs live
in `configs/models/models.example.json`; the download script exits until real
IDs are supplied.

Local checkpoints can already be loaded with
`cst.core.runtime.load_transport_model`. The loader rejects gate, H1, and other
stale checkpoint roles. See [Model loading](docs/model_loading.md) and
[Inference](docs/inference.md).

## Demo

The public demo is intentionally lightweight and uses precomputed videos:

```bash
python demo/app.py
```

Add redistributable examples to `demo/examples/` and list them in
`demo/examples/manifest.json`. This same directory can become a Hugging Face
Space without shipping backbone weights or requiring a large GPU.

## Naming compatibility

Existing checkpoints store research-era role strings. They are retained only
for strict loading:

| Public method | minWM role | HY-WM1.5 role |
|---|---|---|
| CST-R | `action_h0` | `action_h0_state` |
| CST-T | `action_hm` | `action_hm_state` |

Public commands, filenames, and documentation use CST-R/CST-T.
