# Model loading

CST-R and CST-T weights are not included in this repository. Supply a local
checkpoint path and load upstream backbone weights separately under their
original licenses.

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

The returned checkpoint contains only the small CST corrector.
