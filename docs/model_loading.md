# Model loading

## Status

CST-R and CST-T weights are not public yet. Edit a copy of
`configs/models/models.example.json` when the Hugging Face repositories are
created. Do not commit private cluster paths or unpublished repository IDs.

## Local checkpoint

```python
from pathlib import Path

import torch

from cst.backends import get_backend
from cst.core.runtime import load_transport_model

model, metadata = load_transport_model(
    Path("checkpoints/hyworld15-cst-t/best.pt"),
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)
method = get_backend("hyworld15").validate_checkpoint_config(
    metadata["model_config"],
    method="cst_t",
)
print(method)  # cst_t
```

The returned checkpoint contains only the small CST corrector. Load upstream
backbone weights separately under their original licenses.

## Future Hugging Face download

After release, set the four repository variables used by the script:

```bash
export ACTIONSPLICE_MINWM_CST_R_REPO='<HF_ORG>/<REPO>'
export ACTIONSPLICE_MINWM_CST_T_REPO='<HF_ORG>/<REPO>'
export ACTIONSPLICE_HY_CST_R_REPO='<HF_ORG>/<REPO>'
export ACTIONSPLICE_HY_CST_T_REPO='<HF_ORG>/<REPO>'
bash scripts/download_weights.sh checkpoints
```

Until all variables are real repository IDs, the script exits without making
network requests.
