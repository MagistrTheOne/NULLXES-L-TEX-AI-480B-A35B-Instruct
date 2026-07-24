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
| Train | **not started** — next week ZeRO-3 + 100M tok (`docs/15_NEXT_WEEK_20B.md`) |
| Hub (planned) | `MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1` — card: rework **Aug 2026** |

Hub tokenizer: https://huggingface.co/MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.2

---

## 20B-G — Baby ~2B agent+code iterations (scaffold)

| Field | Value |
|-------|-------|
| Date | 2026-07-24 |
| Isolation | **`20B-G/`** (does not touch root Stage0a / v0.2 / 20B genesis ckpt) |
| Model | ~2B dense NHAT (`20B-G/configs/nullxes_latex_2b.yaml`) |
| Tokenizer | **v0.3** vocab 131072 → `20B-G/tokenizer/latex-v0.3/` |
| Train shape | **5 × 250 steps** (~8.2M tok/iter, ~41M total) |
| Mix | code **0.60** / chat+agent **0.35** / canon **0.05** |
| Style | Magistr × Grok spice ≤2%; no Digital Employees |
| Docs | `20B-G/docs/16_LATEX_BABY_ITER.md` · pointer `docs/16_LATEX_BABY_ITER.md` |
| Hub (planned) | Tokenizer-v0.3 · optional 2B-Baby-v0.3 — **not** overwrite 20B Genesis |
| Status | Scaffold + configs + scripts ready; run download → tok → init → iters on pod |

### Decision

Keep 20B Genesis as trunk. Prove mix on baby inside `20B-G/` first.

---

## v1.0-canon — LÆTEX V1 line opened (code + canon, before any run)

| Field | Value |
|-------|-------|
| Date | 2026-07-24 |
| Line | **LÆTEX V1** — 20B dense, `LÆTEX-NULLXES FOUNDATION MODEL` |
| Stage naming | **foundation bootstrapping**, not pretraining |
| Canon | Digital-entity framing removed from code, configs, seed corpus and QA |
| Protocol | Input → Analysis → Answer; empathy filler banned and QA-gated |
| Tokenizer | `configs/tokenizer_latex_v1.yaml` — v1, 131072, concept regression gate |
| Corpus | `configs/datasets_latex_v1.yaml` — filters, SimHash dedup, holdout 0.5% |
| Genesis | `configs/nullxes_latex_20b_v1.yaml` — init loss gate ln(131072) = 11.7856 ± 0.15 |
| Stages | `configs/stage3_20b_iter.yaml` + `scripts/run_stage3_iter.sh`, 250 steps each |
| SFT | `scripts/train_sft_v1.py` — loss on the assistant span only |
| Hardware | 1 pod × 4× H200 SXM, ZeRO-2, no CPU offload |
| Runbook | `docs/17_LATEX_V1.md` |

### Failures found and fixed before the run

| Defect | Impact |
|--------|--------|
| DeepSpeed accumulation | optimizer stepped once per 16 iterations while the token counter multiplied by 16 |
| Padding in loss | `CrossEntropyLoss` trained on pad because labels were raw `input_ids` |
| Padding instead of packing | short canon docs spent a whole `seq_len` window on pad |
| Local attention window | applied only when `past_key_value is None`, so generate saw the full cache |
| Shared RNG seed per rank | all data-parallel ranks would have drawn identical batches |
| `tokenizer/latex-v0.1` | only **3539** real pieces of 131072; the rest are `<|unused_*|>` |
| QA identity markers | `"digital employee"` counted as a passing identity hit |

### Decision

Line opened, nothing trained yet. The v0.1 tokenizer is disqualified for the V1
line by the padded-vocab gate; v1 must be trained on the LÆTEX V1 corpus first.

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
