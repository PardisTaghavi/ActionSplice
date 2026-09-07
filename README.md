# ActionSplice

**Same-step state editing for interruptible world models.**

ActionSplice updates an active world-model rollout when the control input
changes before the current video chunk has finished sampling. Instead of
restarting the chunk or skipping solver steps, ActionSplice applies one learned
**Counterfactual State Transport (CST)** corrector at the current solver step
and lets the original backbone finish its normal denoising trajectory.

## Contents

- [Method](#method)
- [Installation](#installation)
- [Training](#training)
- [Inference and model loading](#inference-and-model-loading)
- [Repository structure](#repository-structure)
- [Development](#development)

## Method

ActionSplice provides two separately trained correctors:

| Variant | Edited region | Intended use |
|---|---|---|
| **CST-R** | Complete active chunk | Retarget a chunk after a control interruption |
| **CST-T** | Uncommitted suffix only | Preserve already committed frames while retargeting the remainder |

### Backend state conventions

| Backend | Canonical state | Corrector target | Same-step reconstruction |
|---|---|---|---|
| minWM Wan Action2V | `[B,T,16,H,W]` | Clean prediction | Stored transition noise |
| HY-WM1.5 | `[B,32,T,H,W]` | Direct Euler state | None; deterministic resume |

See [docs/architecture.md](docs/architecture.md) for the implementation
boundary between CST and the upstream world models.

## Installation

### Core and development tools

```bash
git clone https://github.com/PardisTaghavi/ActionSplice.git
cd ActionSplice

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

Optional dependencies are grouped by workflow:

```bash
python -m pip install -e '.[train]'
python -m pip install -e '.[inference]'
```

### Upstream backbones

ActionSplice integrates with upstream repositories rather than redistributing
their code or weights. Prepare the pinned revisions with:

```bash
bash scripts/prepare_backends.sh
```

> [!NOTE]
> minWM and HY-WorldPlay both install a top-level `hyvideo` package. Use a
> separate Python environment for each backend.

Pinned revisions and licensing notes are recorded in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Training

Choose one backend-specific configuration:

| Backend | CST-R | CST-T |
|---|---|---|
| minWM | `configs/minwm/train_cst_r.json` | `configs/minwm/train_cst_t.json` |
| HY-WM1.5 | `configs/hyworld15/train_cst_r.json` | `configs/hyworld15/train_cst_t.json` |

Replace the `<PATH_TO_...>` values in a copied configuration, then train one
corrector:

```bash
actionsplice-train \
  --capture-dir /path/to/captures \
  --output-dir outputs/minwm-cst-r \
  --config configs/minwm/train_cst_r.json
```

Resume an interrupted run with `--resume /path/to/checkpoint.pt`.

The training code preserves the research objectives already used by the
project: normalized state reconstruction, the minWM clean-prediction delta,
suffix-masked CST-T reconstruction, the CST-T intra-chunk boundary term, and
optional decoded LPIPS/temporal/boundary terms. Repository cleanup did not
introduce new losses.

All loss coefficients are configurable through the `loss_weights` block. Its
defaults reproduce the current HY-WM1.5 objective; the minWM configurations
override the residual coefficient. See the training guide for the complete
lambda schema.

Capture commands, manifest formats, split checks, and backend-specific
examples are documented in [docs/training.md](docs/training.md).

## Inference and model loading

The repository contains no backbone or corrector checkpoints. Use a local
checkpoint path with the commands below.

Local corrector checkpoints can be loaded independently of the backbone:

```python
from pathlib import Path

import torch

from cst.core.runtime import load_transport_model

corrector, metadata = load_transport_model(
    Path("checkpoints/hyworld15-cst-r/best.pt"),
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)
```

The loader validates backend target type and checkpoint role. Load the
upstream backbone separately under its original license.

Backend-specific commands run both variants end to end from a small rollout
configuration:

```bash
actionsplice-infer-minwm \
  --method cst_r \
  --task-config configs/inference/minwm_cst_r.example.json \
  --minwm-root /path/to/minWM \
  --transport-checkpoint /path/to/minwm-cst-r.pt \
  --output outputs/minwm-cst-r.mp4

actionsplice-infer-hyworld15 \
  --method cst_t \
  --task-config configs/inference/hyworld15_cst_t.example.json \
  --reference-image /path/to/reference.png \
  --hyworld-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-checkpoint /path/to/action/checkpoint.safetensors \
  --transport-checkpoint /path/to/hyworld15-cst-t.pt \
  --output outputs/hyworld15-cst-t.mp4
```

- [Inference sequence and current hooks](docs/inference.md)
- [Checkpoint validation](docs/model_loading.md)

## Repository structure

```text
ActionSplice/
├── src/cst/
│   ├── backends/       minWM/HY adapters and shared backend registry
│   ├── core/           corrector model, state layouts, same-step runtime
│   ├── data/           capture schemas, datasets, partitions, manifests
│   ├── training/       shared corrector training loop and losses
│   └── cli/            capture, manifest, training, and inference commands
├── configs/
│   ├── minwm/          minWM CST-R/CST-T capture and training configs
│   ├── hyworld15/      HY CST-R/CST-T capture and training configs
│   └── inference/      runnable single-rollout CST-R/CST-T examples
├── docs/               architecture, training, inference, model loading
├── scripts/            pinned upstream setup
└── tests/               CPU unit tests
```

The distribution is named `actionsplice`; the import package remains `cst`
because CST is the framework's core operation. `cst.backends` is the shared
backend boundary and keeps upstream imports lazy.

<details>
<summary>Checkpoint naming compatibility</summary>

Research checkpoints retain legacy role strings for strict validation:

| Public method | minWM role | HY-WM1.5 role |
|---|---|---|
| CST-R | `action_h0` | `action_h0_state` |
| CST-T | `action_hm` | `action_hm_state` |

Public commands, files, and documentation use CST-R/CST-T.

</details>

## Development

Run the CPU checks before committing:

```bash
ruff check .
pytest
```

Generated captures, checkpoints, outputs, upstream repositories, and W&B
artifacts are excluded by `.gitignore`. Do not commit private cluster paths,
unpublished model IDs, or third-party weights.

## License

ActionSplice is released under the [Apache License 2.0](LICENSE).
