# NULLXES-LÆTEX — Model History

Living lab journal. Append after every meaningful checkpoint / gate.

---

## v0.1 — Stage0a RTX PRO 6000 (identity bootstrap)

| Field | Value |
|-------|-------|
| Date | 2026-07-18 |
| Params | **101.9M** dense NHAT |
| Hardware | 1× RTX PRO 6000 Blackwell · torch 2.8.0+cu128 |
| Tokens | ~99.9M (first pass) + continue (~100M) |
| Tokenizer | latex-v0.1 (smoke Unigram on ~455 sentences → pad 131072) |
| Final train loss | ~0.006 (heavy memorization) |
| QA identity | **8/9** passed (pre-continue) |
| Hub | https://huggingface.co/MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0.1 |
| Checkpoint | `checkpoints/nullxes-latex-100m-stage0a-rtxpro6000` |

### What it proves

- Own arch + muP + bf16 + HF package path work  
- Identity canon (LÆTEX / NULLXES / @MagistrTheOne) sticks  

### What it does NOT prove

- General RU/EN competence  
- Codegen  
- Production tokenizer fertility  

### Next

Gate A corpus-v0.2 → Gate B tokenizer ablation → Gate C broader 100M eval → only then Gate D 20B genesis.

---

## v0.1.1 — Stage0a continue (+~100M tok)

| Field | Value |
|-------|-------|
| Date | 2026-07-18/19 |
| Params | **101.9M** (same lineage) |
| Tokens this pass | ~99.9M @ lr 2e-4 resume |
| Tokens cumulative | ~200M |
| Final train loss | ~0.0061 |
| QA identity | **9/9** passed |
| Decision | **keep** as identity brick; **do not** treat as language/code competence |

### QA notes (honest)

- Direct Q/A mantras clean (EN/RU who/name/author).  
- Open prompts still leak markdown/tables from overfit corpus (`Как тебя зовут?`, short-name continue).  
- Holdout-style forward loss ~9.9 vs train ~0.006 → memorization, not generalization.  

### Hub

Same repo; optional commit `v0.1.1-continue` when packaging again.

---

## v0.1.2 — Output channels (INTERNAL / PUBLIC / RULE) — PASS

| Field | Value |
|-------|-------|
| Date | 2026-07-19 |
| Status | **PASS** |
| Config | `configs/stage0a_100m_output_control_patch.yaml` |
| Tokens this pass | ~25.0M (381 steps) |
| Final train loss | ~0.005 |
| QA identity | **9/9** |
| QA output_control_leaks | **0** |
| Checkpoint | `checkpoints/nullxes-latex-100m-stage0a-rtxpro6000` |

### What fixed

- Open RU: `Как тебя зовут?` → `Меня зовут LÆTEX. Полное имя — NULLXES-LÆTEX.` (no schema pipes)  
- Architecture behind `<<<INTERNAL>>>`; PUBLIC mantras + RULE contrastives  
- Raw `docs/*.md` / yaml removed from code pack  

### Still true

- Holdout forward loss ~11 → still a memorizer, not general LM  
- soft_id mix stayed high (~62%) during patch — OK for this stage; Gate A corpus is next for competence  

### Channels (LÆTEX brain — not employee persona modes)

- **INTERNAL** — schema / NHAT / MoE planning  
- **PUBLIC** — natural answers to humans  
- **RULE** — when to use which  

Employee overlays (Anna etc.) are a different product layer — not this Stage0 patch.

### Next

Gate A corpus-v0.2 → Gate B tokenizer → Gate C broader 100M eval → Gate D 20B genesis on 1× RTX PRO 6000.

---

## Tokenizer v0.2 — Gate B ablation PASS (2026-07-19)

| Field | Value |
|-------|-------|
| Corpus | Gate A proxy (`corpus_gate_a_proxy`, ~368MB HF shards + identity) |
| Ablation | 32k FAIL · 64k/96k/131072 PASS |
| **Winner** | **131072** full Unigram (`vocab_padded: false`) |
| Freeze | `tokenizer/latex-v0.2/` |
| Hub | upload as `MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.2` — **not** into 100M Stage0a |

vs v0.1: smoke Unigram ~3.5k pieces + unused pad → v0.2 real 131072 pieces on wiki/science/code mix.

100M Stage0a remains on **v0.1**. 20B+ uses **v0.2 only**.

---

## v0.2-genesis — NULLXES-LÆTEX-20B Weight Genesis PASS

| Field | Value |
|-------|-------|
| Date | 2026-07-19 |
| Params | **18.757B** dense NHAT (L=24, d=8192, A35B-width) |
| Tokenizer | latex-v0.2 (131072 full Unigram) |
| Checkpoint | `checkpoints/nullxes-latex-20b` (~35G, 21 shards) |
| Init | muP + DeepNorm residual · bf16 · CPU smoke |
| `init_report.passed` | **true** |
| HF smoke | **PASS** (`smoke_hf_causal.py`) |
| Train | **not started** — next ZeRO-3 + ≤100M tok mid-eval |

Hub tokenizer: https://huggingface.co/MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.2

---

## Template for later entries

```
## vX.Y — short title

| Field | Value |
| Date | |
| Params | |
| Tokens | |
| Tokenizer | |
| Mix | |
| Eval highlights | |
| Hub / path | |
| Decision | keep / discard / promote |
```
