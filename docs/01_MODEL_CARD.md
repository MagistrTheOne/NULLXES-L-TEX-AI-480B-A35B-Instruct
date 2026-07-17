# NULLXES-LÆTEX — Model Card (living)

| Field | Value |
|-------|-------|
| Family | NULLXES-LÆTEX AI |
| Current trophy target | **Stage0** (not 480B) |
| Flagship (reference only) | 480B-A35B-Instruct MoE |
| Init ancestor | A35B dense (`d_model=8192`) |
| License / weights | NULLXES proprietary · from-scratch |
| Cloud (phase 1) | RunPod H200 → B300 |

## Intended use

Foundation brain for **Digital Employees** (enterprise agents with identity, memory, tools, workflows). Not a generic chat assistant.

## Out of scope (v1 research)

- Serving production 480B  
- Distillation / foreign checkpoints  
- Hard expert→domain wiring  

## Capability targets by stage

| Stage | Must demonstrate |
|------:|------------------|
| 0a 100M | Train loop + tokenizer end-to-end |
| 0b 500M | Stable loss, hybrid attn ablation |
| 0c 1.6B | Research gate (6 proofs) |
| 1 7B | Data pipeline + μP |
| 2 A35B | Genetic parent for MoE |
| 3 480B | Only after gate + A35B |

## Safety / enterprise notes

Policy packs + EPO govern compliance. Customer secrets live in external memory, never baked into trunk weights.

## Checkpoint naming

```
nullxes-latex-stage0a-100m-<date>-step<N>
nullxes-latex-stage0b-500m-<date>-step<N>
nullxes-latex-stage0c-1b6-<date>-step<N>
```
