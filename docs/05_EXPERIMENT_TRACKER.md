# Experiment Tracker

Append-only. One row per run.

| ID | Date | Config | Hypothesis | Metric | Result | Decision | Notes |
|----|------|--------|------------|--------|--------|----------|-------|
| ID | Date | Config | Hypothesis | Metric | Result | Decision | Notes |
|----|------|--------|------------|--------|--------|----------|-------|
| E000 | 2026-07-17 | — | Spec lock after GPT review | — | — | docs+ADR | No training yet |
| E001 | 2026-07-17 | tokenizer_stage0 | Gate0 fertility & representation | see 07 | pending | blocked NHAT | Research Gate 0 |

## Planned queue (do not skip)

| ID | Experiment | Gate |
|----|------------|------|
| E001 | Tokenizer train + fertility/representation suite | **Research Gate 0 PASS** |
| E002 | Stage0a 100M dense, local-attn only | loss↓, no NaN |
| E003 | Stage0a + hybrid 3:1 | vs E002 ppl |
| E004 | Stage0b 500M + depth NoPE top 25% | vs all-RoPE |
| E005 | Tiny MoE expand from 100M dense (8 experts) | no F001/F002 |
| E006 | IEL swap with frozen trunk | F004 suite |
| E007 | Stage0c 1.6B research gate (6 proofs) | pass/fail |
| E010 | Vocab 131k→262k migration (optional) | `08_VOCAB_MIGRATION.md` |

## Research Gate 0 checklist (E001) — blocks Stage0a

- [ ] Vocab 131072 + specials 0–11  
- [ ] Deterministic encode/decode  
- [ ] Fertility + inflation  
- [ ] Fragmentation suite  
- [ ] Reconstruction lossless  
- [ ] `tokenizer/latex-v0.1/` + checksum  
- [ ] No pretrained SP load  

## Later research gate checklist (E007)

- [ ] Tokenizer works (Gate 0 PASS)  
- [ ] NHAT trains  
- [ ] Loss decreases  
- [ ] Gradients stable  
- [ ] MoE expansion possible  
- [ ] IEL separates from trunk  
