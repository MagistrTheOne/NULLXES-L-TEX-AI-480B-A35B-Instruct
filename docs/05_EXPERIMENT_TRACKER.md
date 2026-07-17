# Experiment Tracker

Append-only. One row per run.

| ID | Date | Config | Hypothesis | Metric | Result | Decision | Notes |
|----|------|--------|------------|--------|--------|----------|-------|
| E000 | 2026-07-17 | — | Spec lock after GPT review | — | — | docs+ADR | — |
| E001 | 2026-07-17 | tokenizer_stage0 | Gate0 fertility & representation | see 07 | smoke PASS / full pending | enlarge corpus → 131k | smoke vocab ~541 |
| E001a | 2026-07-17 | corpus_gate0 | Seed corpus + validate | validate_corpus | PASS (seed) | grow raw/ for 131k | 105 docs |
| E008 | 2026-07-17 | nullxes_latex_7b | Weight Genesis | init_report | **PASS** | 6.745B bf16 HF ckpt | H200 RunPod |
| E008b | 2026-07-17 | smoke_hf_causal | generate + Auto* load | shape | **PASS** | cache fix b53f772 | checkpoints/nullxes-latex-7b |
| E009 | 2026-07-17 | identity_corpus | LÆTEX name + repo code shards | validate | in progress | Stage0a next, not 7B SFT | build_identity_corpus.py |

## Planned queue (do not skip)

| ID | Experiment | Gate |
|----|------------|------|
| E001a | `build_seed` + `validate_corpus` | **Corpus PASS** |
| E001 | Tokenizer train + fertility suite | **Research Gate 0 PASS** |
| E008 | Stage1 Weight Genesis (`init_model.py`) | init_report PASS |
| E002 | Stage0a 100M dense | loss↓, no NaN |
| E003 | Stage0a + hybrid 3:1 | vs E002 ppl |
| E004 | Stage0b 500M + depth NoPE | vs all-RoPE |
| E005 | Tiny MoE expand | no F001/F002 |
| E006 | IEL swap frozen trunk | F004 |
| E007 | Stage0c 1.6B six proofs | pass/fail |
| E010 | Vocab 131k→262k migration | optional |

## Research Gate 0 checklist (E001) — blocks Stage0a

- [ ] **E001a corpus validate PASS**
- [ ] Vocab 131072 + specials 0–11  
- [ ] Deterministic encode/decode  
- [ ] Fertility + inflation  
- [ ] Fragmentation suite  
- [ ] Reconstruction lossless  
- [ ] `tokenizer/latex-v0.1/` + checksum  
- [ ] No pretrained SP load  
