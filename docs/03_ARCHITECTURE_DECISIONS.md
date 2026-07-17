# Architecture Decisions (ADR)

## ADR-001 — NHAT hybrid Transformer (not Mamba-first)

**Status:** Accepted  
**Why:** Kernel maturity on H200/B300, KV-cache serving, MoE FFN swap. SSM optional later.

## ADR-002 — Pre-norm dual residual block

**Status:** Accepted (GPT diagram corrected)

```
x ─→ RMSNorm → HybridAttn → + ─→ RMSNorm → SwiGLU|MoE → +
 │                          ↑                        ↑
 └──────────────────────────┴────────────────────────┘
```

Single trailing residual (as in some GPT sketches) is **rejected**.

## ADR-003 — GQA 64Q / 8KV

**Status:** Accepted · KV-cache friendly for 128k.

## ADR-004 — Positional: hybrid window + depth NoPE

**Status:** Accepted (combined GPT + v1)

| Rule | Choice |
|------|--------|
| Pattern | Every 4th layer **full**; others **local 4k** |
| Depth | Bottom **75%** layers: RoPE on local; top **25%**: **NoPE** |
| Full layers | **NoPE** (long-range binding) |

Ablations tracked in experiment tracker: `rope_full` vs `nope_full`.

## ADR-005 — MoE 152 + 1 shared, top-6

**Status:** Accepted · ~476B / ~35B active (param_count).

## ADR-006 — Soft expert taxonomy

**Status:** Accepted (GPT strengthen)

- Early: free learning, no domain hard-bind  
- Mid: activation clustering  
- Late: human labels (code / reasoning / legal / …) for monitoring only  

Hard `expert_id → finance` assignment: **rejected**.

## ADR-007 — A35B expand + noise schedule

**Status:** Accepted

```
noise_scale: 0.02 → 0.005  (cosine or linear over expand warmup)
expert_i = A35B_FFN + orthogonal_noise * noise_scale
shared   = A35B_FFN  (no noise)
```

## ADR-008 — IEL + Role Adapter + Memory

**Status:** Accepted

```
trunk (frozen for persona swap)
  + IEL(identity)
  + RoleAdapter(role)   # lightweight, not full model fork
  + external memory / policy packs
```

## ADR-009 — Stage0 micro-ladder

**Status:** Accepted

Tokenizer proof → **100M → 500M → 1.6B**. Jumping to 1.6B on raw tokenizer: **rejected**.

## ADR-010 — Research gate before 480B validity

**Status:** Accepted · see `01_MODEL_CARD.md` and philosophy doc.

## ADR-011 — Vocab 131072 fixed; 262144 migration-only

**Status:** Accepted  

Stage0 / 7B / A35B use **131072**. Flagship **262144** only via explicit embedding-expansion experiment (`08_VOCAB_MIGRATION.md`). Never auto-expand.

## ADR-012 — SentencePiece as trainer algorithm only

**Status:** Accepted  

May use SentencePiece **trainer library** for Unigram. Forbidden: loading pretrained `sp.model` / foreign LLM tokenizers. All vocab/IDs/artifacts are NULLXES-owned under `tokenizer/latex-v0.1/`.

## ADR-013 — Locked special token IDs 0–11

**Status:** Accepted  

pad=0 … assistant=11 (see `06_TOKENIZER_DESIGN.md`). Emotion/role/tone live in IEL / Role Adapter, not tokenizer.

## ADR-014 — Research Gate 0 rename

**Status:** Accepted  

Official: **NULLXES-LÆTEX Research Gate 0: Tokenizer Fertility & Representation Gate**. NHAT blocked until PASS (`07_RESEARCH_GATE_0.md`).
