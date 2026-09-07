# Architecture

```text
request + active sampler state at step r
                    │
                    ▼
            backend state adapter
                    │
             [B,T,C,H,W]
                    │
           ┌────────┴────────┐
           │                 │
       CST-R      CST-T
       full chunk       hard suffix mask
           │                 │
           └────────┬────────┘
                    ▼
          corrected state at step r
                    │
                    ▼
       ordinary K-r backbone solver calls
```

`cst.backends.registry` is the shared boundary used to validate training and
inference configurations. Upstream imports stay lazy and backend-specific.

| Backend | Native state | Corrector target | Same-step reconstruction |
|---|---|---|---|
| minWM Wan Action2V | `[B,T,16,H,W]` | clean prediction | stored transition noise |
| HY-WM1.5 | `[B,32,T,H,W]` | direct Euler state | none; deterministic resume |

CST-R and CST-T use separately trained checkpoints. CST-T clamps the
teacher prefix at every solver step during capture and applies the same hard
suffix mask to the corrector output.
