# Public release checklist

- [x] Select and add the Apache License 2.0.
- [ ] Complete provenance review for every migrated source module.
- [ ] Replace provisional Hugging Face organization/model identifiers.
- [ ] Export corrector weights as `safetensors`.
- [x] Remove gate, fallback, solver-jump runtime, and unreachable rollout trainer helpers.
- [ ] Add backend-specific generation CLIs once public checkpoints are frozen.
- [ ] Migrate the final recurrent on-policy minWM CST-R capture path.
- [ ] Wire minWM CST-R and CST-T into an ActionSplice generation hook.
- [ ] Wire HY-WM1.5 CST-T suffix masks into the on-policy generation hook.
- [ ] Run official GPU integration tests at pinned upstream revisions.
- [ ] Verify all example media and prompts are redistributable.
- [ ] Add paper citation and final quantitative tables.
- [ ] Build the precomputed Hugging Face Space.

Gate, adaptive fallback, and multi-horizon runtime code are intentionally not
part of this public staging tree.
