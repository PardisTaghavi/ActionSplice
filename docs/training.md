# Training

## Data paths

Capture commands use the frozen upstream generators to construct matched
source/target trajectories.

### minWM matched teachers

The final CST-R dataset is recurrent and uses a frozen bootstrap corrector:

```bash
actionsplice-capture-minwm-recurrent \
  --minwm-root /path/to/minWM \
  --experiment-config configs/minwm/capture_cst_r.json \
  --transport-checkpoint /path/to/minwm-cst-r-pilot.pt \
  --output-dir datasets/minwm-cst-r
```

The bootstrap dataset uses `actionsplice-capture-minwm` with
`configs/minwm/capture_cst_r_bootstrap.json`. CST-T uses the same command with
`configs/minwm/capture_cst_t.json`; its matched teacher clamps the prefix at
every solver evaluation for offsets `m=1,2,3`.

### CST-R bootstrap (HY-WM1.5)

The initial recurrent capture can run without a corrector checkpoint:

```bash
actionsplice-capture-hyworld15-bootstrap \
  --hyworld-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-checkpoint /path/to/action/diffusion_pytorch_model.safetensors \
  --reference-image /path/to/reference.png \
  --experiment-config configs/hyworld15/capture_cst_r_bootstrap.json \
  --output-dir outputs/hyworld15-bootstrap
```

### Final CST-R and CST-T datasets (HY-WM1.5)

First export task manifests with the shared prompt/scene-disjoint split:

```bash
actionsplice-export-hyworld15-manifests \
  --cst-r-config configs/hyworld15/capture_cst_r.json \
  --cst-t-config configs/hyworld15/capture_cst_t.json \
  --reference-manifest /path/to/references.json \
  --reference-root /path/to/reference-images \
  --dataset-root datasets/hyworld15
```

Each reference-manifest row must contain `prompt_index`, `reference_image`, and
`scene_id`. The exporter checks prompt alignment, file hashes, unique images,
and scene leakage across the 120/30 train/held-out split.

Capture CST-T teacher pairs directly:

```bash
actionsplice-capture-hyworld15 \
  --mode cst_t \
  --hyworld-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-checkpoint /path/to/action/checkpoint.safetensors \
  --task-manifest datasets/hyworld15/cst_t_paper_150_v1/task_manifest.json \
  --output-dir datasets/hyworld15/cst_t_paper_150_v1
```

CST-T clamps the committed prefix at each teacher solver step and saves
separate examples for offsets `m=1,2,3`. It is independently trained and does
not load a CST-R checkpoint.

The final recurrent CST-R capture uses a frozen bootstrap CST-R checkpoint
to collect on-policy histories:

```bash
actionsplice-capture-hyworld15 \
  --mode cst_r \
  --hyworld-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-checkpoint /path/to/action/checkpoint.safetensors \
  --transport-checkpoint outputs/hyworld15-bootstrap-train/best.pt \
  --task-manifest datasets/hyworld15/cst_r_paper_150_v1/task_manifest.json \
  --output-dir datasets/hyworld15/cst_r_paper_150_v1
```

## Train one corrector

Select one verified configuration:

```text
configs/minwm/train_cst_r.json
configs/minwm/train_cst_t.json
configs/hyworld15/train_cst_r.json
configs/hyworld15/train_cst_t.json
```

Copy it to a run directory and set its `<PATH_TO_...>` backbone, decoder,
and pilot-checkpoint paths. Training infers latent channels and denoising
steps from capture tensors. CST-T can initialize from a CST-R pilot:
the loader copies compatible weights and zero-initializes only its new hard-mask
input channel.

```bash
actionsplice-train \
  --capture-dir /path/to/captures \
  --output-dir outputs/hyworld15-cst-t \
  --config configs/hyworld15/train_cst_t.json
```

Use `--resume /path/to/checkpoint.pt` for a stopped run.

## Existing objective

Repository cleanup did not add losses. The shared trainer preserves:

- normalized latent/state reconstruction loss;
- minWM clean-prediction delta loss;
- CST-T suffix-masked normalized loss;
- CST-T intra-chunk boundary loss;
- optional decoded LPIPS, temporal, and boundary losses.

CST-R and CST-T are trained separately. Every release config uses
`jump_horizons: [0]`; this field remains only for capture/checkpoint schema
compatibility.

### Configurable loss weights

Training lambdas live under `loss_weights` in each training configuration:

```json
"loss_weights": {
  "lambda_state": 1.0,
  "lambda_residual": 0.0,
  "lambda_lpips": 0.05,
  "lambda_temporal": 0.1,
  "lambda_history_boundary": 0.1,
  "lambda_mid": 0.0
}
```

These are the HY-WM1.5 CST-R defaults. CST-T defaults `lambda_mid` to `0.1`;
CST-R requires it to remain `0.0`. The minWM release configurations override
`lambda_residual` to `1.0`. All lambdas must be finite and nonnegative, and the
resolved values are stored in checkpoints and `training_summary.json`.
