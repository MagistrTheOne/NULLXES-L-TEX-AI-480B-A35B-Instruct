# NULLXES-LÆTEX — Model History

Living lab journal. Append after every meaningful checkpoint / gate.

---

## Active line (2026-07)

| Milestone | Config | Notes |
|-----------|--------|-------|
| **LÆTEX V1** | `configs/nullxes_latex_20b_v1.yaml` | 20B dense foundation bootstrapping |
| Tokenizer v1 | `configs/tokenizer_latex_v1.yaml` | 131072 Unigram, corpus V1, no pad |
| Mid MoE | `configs/nullxes_latex_200b_moe.yaml` | ~202B / ~28B active — after V1 |
| Flagship MoE | `configs/nullxes_latex_480b_a35b.yaml` | ~476B / ~35B active — expand from A35B |
| A35B ancestor | `configs/nullxes_latex_a35b.yaml` | Dense L=48 width parent |

Runbook: [`docs/17_LATEX_V1.md`](17_LATEX_V1.md) · Config index: [`configs/README.md`](../configs/README.md)

---

## Retired (historical)

Earlier Stage0a 100M / Gate A proxy / tokenizer v0.1–v0.2 / 20B-G baby / RTX PRO 6000
smoke paths were removed from the repo in the 2026-07 cleanup. Hub snapshots of past
research bricks may still exist under MagistrTheOne; they are not the active train path.
