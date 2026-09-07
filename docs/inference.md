# Inference

## Runtime sequence

1. Load one upstream backbone in its own environment.
2. Load either a CST-R or CST-T checkpoint.
3. At interruption solver step `r`, expose the active state in canonical
   `[B,T,C,H,W]` layout.
4. For CST-T, construct the hard suffix mask from the committed prefix.
5. Run one corrector forward at the same step `r`.
6. For minWM, reconstruct the noisy state with the stored transition noise.
   HY-WM1.5 uses the predicted direct Euler state.
7. Resume the untouched backbone for its ordinary `K-r` solver calls.

## HY-WM1.5

`cst.backends.hyworld15_inference.OfficialHYWorldPlayInferenceHook` temporarily
wraps the official pipeline's `ar_rollout` method and restores it on exit. It
supports both checkpoint roles:

```bash
actionsplice-infer-hyworld15 \
  --method cst_r \
  --task-config configs/inference/hyworld15_cst_r.example.json \
  --hyworld-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-checkpoint /path/to/action/checkpoint.safetensors \
  --transport-checkpoint checkpoints/hyworld15-cst-r/best.pt \
  --reference-image /path/to/reference.png \
  --output outputs/hyworld15-cst-r.mp4
```

The loader validates that the checkpoint uses the HY direct-state target.
For CST-T, select `--method cst_t` and use the CST-T example configuration;
the configured boundary creates the hard suffix mask.

## minWM

The minWM runner is `cst.backends.minwm_inference.run_minwm_cst`. It passes the
scheduler, denoising-step list, and stored transition-noise bank required for
exact same-step re-noising:

```bash
actionsplice-infer-minwm \
  --method cst_t \
  --task-config configs/inference/minwm_cst_t.example.json \
  --minwm-root /path/to/minWM \
  --transport-checkpoint checkpoints/minwm-cst-t/best.pt \
  --output outputs/minwm-cst-t.mp4
```

The two entry points remain separate because the upstream backbones use
independent dependency stacks and launch paths. Each command also writes a JSON
file beside the video with checkpoint metadata, event timings, and post-request
NFE counts.
