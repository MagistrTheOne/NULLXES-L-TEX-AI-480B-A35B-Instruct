# Stage1 Weight Genesis

**Not training.** Birth of dense weights + diagnostics before any token budget.

| Gate0 | Weight Genesis |
|-------|----------------|
| `requirements.txt` / `requirements-gate0.txt` | `requirements-stage1.txt` |
| Tokenizer fertility & representation | `scripts/init_model.py` |
| No NHAT train | Create `model.safetensors` + `init_report.json` |

## Order

1. Research Gate 0 PASS (`docs/07_RESEARCH_GATE_0.md`)  
2. Weight Genesis for target config (0a → … → 7B)  
3. Only if `init_report.json.passed` — enable `training.enabled` and run tokens  

## Command

```bash
pip install -r requirements-stage1.txt
python scripts/init_model.py --config configs/nullxes_latex_7b.yaml --device cpu
# or: --device cuda --dtype bfloat16
```

## Outputs

```
checkpoints/nullxes-latex-7b/
  model.safetensors
  config.json
  tokenizer.json
  special_tokens.json
  init_report.json
  tokenizer.model   # if Gate0 artifact present
```

## init_report (pass criteria)

- `nan_detected: false`  
- `dead_layers: 0`  
- `smoke_forward_ok: true`  
- `embedding_shape: [131072, d_model]`  
- `mean_weight_std` near configured `init.std` (scaled by residual outs)

## muP

`init.scheme: mup` with explicit `base_width`, `base_depth`, `std`, and structured `residual_scale.type: deepnorm`.

## Hardware

Configs use `hardware.provider: generic`. Cloud (RunPod etc.) only via `configs/runtime.yaml` / ops — not model code.
