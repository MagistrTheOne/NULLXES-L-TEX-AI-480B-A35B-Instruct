# Stage2 — Data First (Gates A→E)

**Goal:** 20B must **not** be an expensive copy of the 100M memorizer.  
**480B MoE:** cluster later — **do not touch** until A35B path is proven.

```
Gate A Corpus  →  Gate B Tokenizer  →  Gate C 100M Eval
                                              ↓
                                    Gate D 20B Genesis
                                              ↓
                                    Gate E 20B Train (+ mid evals)
```

No gate → no next gate. No YAML-only jumps.

---

## Decision log (GPT proposals → keep / drop)

| Idea | Verdict | Why |
|------|---------|-----|
| Data First + independent gates | **KEEP** | Prevents 20B = bigger overfit |
| Corpus manifests (lang/code/license/dedup stats) | **KEEP** | Reproducible mix |
| Vocab not sacred; ablate 32→64→96→131k | **KEEP** | Smoke 131k is mostly pad |
| Broad eval suite + history | **KEEP** | Loss alone lies |
| 20B width-compatible with A35B | **KEEP** | Same d_model/heads/ffn; depth scales later |
| Mid-train evals every N tokens | **KEEP** | See “wake up” curve |
| MODEL_HISTORY.md | **KEEP** | Lab memory |
| MoE 476B/35.1B/152/top6 design note | **KEEP as note only** | Flagship later; **no train/config freeze now** |
| Train 20B before corpus/tokenizer gates | **DROP** | Expensive copy of 100M |
| Inflate 100M weights → 20B | **DROP** | Wrong shape; new genesis |
| Change / schedule 480B training now | **DROP** | User lock: cluster later |
| Force vocab=131072 without ablation | **DROP** | Let data win |
| Skip Gate C (100M eval) before 20B | **DROP** | Need non-identity competence signal |

---

## Gate A — Corpus

**PASS when:**

- `datasets/clean/corpus-v0.2/` exists with versioned shards  
- Manifests present:

```
datasets/manifests/corpus-v0.2/
  mix.json
  shards.json
  stats.json          # RU/EN/code % , tokens_est, docs
  licenses.json
  duplicates.json     # exact + near-dedup summary
  language.json
```

- Mix target (token weights):

| Bucket | Weight |
|--------|-------:|
| RU/EN language | 0.40 |
| code | 0.35 |
| math/science | 0.10 |
| enterprise/docs | 0.10 |
| identity (canon) | 0.05 |

- Min clean size for Gate A: **≥10 GB** text (proxy); **≥50 GB** before serious 20B train  
- No FineWeb-as-backbone; license allowlist only  

**FAIL if:** guessing percentages; no license trail; LLM-synth prose.

---

## Gate B — Tokenizer

**PASS when:**

1. Ablation report exists for **32k / 64k / 96k / 131k** on same corpus sample:

   - compression (bytes/token)  
   - fertility EN / RU / code  
   - special-token atomicity (ids 0–11)  
   - unused vocab % after encode of holdout  

2. Winner frozen as `tokenizer/latex-v0.2/` (+ `meta.json` with chosen `vocab_size`)  
3. `evaluate_tokenizer.py` **without** `--smoke` → PASS  
4. No foreign specials in training text; chat turns space-separated  

Stage0a **v0.1** stays frozen for HF 100M lineage.  
20B uses **v0.2 only**.

---

## Gate C — 100M Eval (not only identity)

Continue / code-mix 100M must beat Stage0a baseline on a **mini suite** before 20B spend.

```
evals/
  identity/
  reasoning/
  python/
  typescript/
  sql/
  cpp/
  bash/
  translation/
  summarization/
  math/
```

Minimum Gate C bar (v0):

| Suite | Bar |
|-------|-----|
| identity | ≥8/9 (keep) |
| python smoke | >0 meaningful completions (not identity paste) |
| RU open prompt | no code-leak into answer |
| train≠eval loss collapse only | holdout ppl tracked |

Store JSON under `evals/reports/stage0a_*.json` and link in `MODEL_HISTORY.md`.

---

## Gate D — 20B Genesis (A35B-compatible)

**Architecture (locked for transfer):** same **width** as A35B, **fewer layers**.

| Field | 20B (Stage2) | A35B (later) |
|-------|-------------:|-------------:|
| d_model | **8192** | 8192 |
| n_heads / n_kv / d_head | 64 / 8 / 128 | 64 / 8 / 128 |
| d_ff | **22016** | 22016 |
| n_layers | **24** (~18.8B, half depth) | **48** (~35.4B) |
| NHAT hybrid / NoPE | same policy | same |
| tokenizer | v0.2 | v0.2 |

Config: `configs/nullxes_latex_20b.yaml`  
Init: muP from scratch (`inherit_from: null`).  
Optional later: depth-expand 24→48 if experiments justify (not assumed).

**PASS:** `init_report.json` finite, no dead layers, smoke forward OK.

---

## Gate E — 20B Train

- Mid checkpoints + eval at **100M / 500M / 1B / 2B / 5B** tokens (then continue)  
- Each checkpoint: eval suite → append `MODEL_HISTORY.md`  
- Hardware: multi-GPU + ZeRO-3 / H200 cluster — not single PRO 6000 full AdamW  
- Identity mix ≤5% so name does not dominate  

**480B MoE:** design note only (preferred candidate from param_count: ~476B total / ~35.1B active, 152 routed, 1 shared, top-6, d_ff_e=2048). **No training, no freeze.**

---

## Anti-patterns (20B = expensive copy)

1. Same 455-sentence tokenizer  
2. Soft-identity 90% mix  
3. Loss 0.006 on train + identity QA only → ship as “smart”  
4. Bigger YAML without Gate A/B PASS  
5. Touching 480B schedule “to feel progress”
