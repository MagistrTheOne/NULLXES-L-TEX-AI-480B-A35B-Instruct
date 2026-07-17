# Vocab Migration: 131k → 262k

**Status:** Future experiment only — **not** part of Gate 0  
**Rule:** Expansion is a **tokenizer migration experiment**, never automatic.

---

## Why this exists

Increasing `vocab_size` from 131072 to 262144 **breaks** the embedding matrix and LM head shape. Silent bumps invalidate A35B-compatible checkpoints.

```
LATEX-Vocab-131k
       ↓
embedding expansion experiment
  (new rows + init scheme + eval)
       ↓
LATEX-Vocab-262k
```

---

## Policy by stage

| Stage | Vocab | Action |
|-------|------:|--------|
| Stage0 / 7B / A35B | 131072 | Frozen lineage |
| 480B | 262144 | Optional after migration PASS |

---

## Migration experiment checklist (when scheduled)

1. Train / extend tokenizer to 262144 **without** discarding 131k ID stability for specials 0–11 and overlapping pieces where possible.  
2. Expand `tok_emb` and `lm_head`: copy rows `[0, 131072)`; init new rows (mean of existing + small noise or μP-scaled).  
3. Short continued pretrain / adapter warmup on expanded matrix.  
4. Re-run Gate-0-style fertility + reconstruction on fixed samples.  
5. New artifact dir: `tokenizer/latex-v0.2-262k/` + checksum.  
6. Log as separate E-series experiment — do not silently rewrite v0.1.

---

## Forbidden

- Changing `vocab_size` in flagship yaml and retraining without this doc’s checklist  
- Sharing one checkpoint across mismatched vocab sizes  
- Reusing foreign 262k vocabs
