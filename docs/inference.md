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

No gate or fallback decision is made.

## HY-WM1.5

`cst.backends.hyworld15.OfficialHYWorldPlayCSTHook` temporarily wraps the
official pipeline's `ar_rollout` method and restores it on exit. The current
CLI exposes the verified **CST-R** on-policy path:

```bash
actionsplice-capture-hyworld15 \
  --mode cst_r \
  --hyworld-root /path/to/HY-WorldPlay \
  --model-path /path/to/HunyuanVideo-1.5 \
  --action-checkpoint /path/to/action/checkpoint.safetensors \
  --transport-checkpoint checkpoints/hyworld15-cst-r/best.pt \
  --task-manifest /path/to/cst-r-task-manifest.json \
  --output-dir outputs/hyworld15-cst-r-inference
```

The loader validates that the checkpoint uses the HY direct-state target.
CST-T teacher-prefix capture exists in this backend, but its on-policy
generation hook is still a release blocker and is not claimed as runnable.

## minWM

The verified minWM pipeline loader and action hook are in
`cst.backends.minwm_wan`. Its CST runtime uses
`cst.core.runtime.apply_transport_model`, passing the scheduler, denoising-step
list, and stored transition-noise bank required for exact same-step re-noising.

A single cross-backend generation CLI is intentionally not claimed yet; the
two upstream repositories install conflicting `hyvideo` packages. The shared
registry and runtime are stable while pipeline launch remains backend-specific.
