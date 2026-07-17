# NULLXES-LÆTEX Corpus Plan — Data Plane (anti-void)

**Problem:** configs + NHAT without a real corpus = research that evaporates.  
**Rule:** nothing trains (tokenizer full Gate0, Stage0a, 7B) until `scripts/validate_corpus.py` PASSes for the target stage.

---

## Two corpora (do not mix purposes)

| Corpus | Purpose | Path | Gate |
|--------|---------|------|------|
| **Gate0 Tokenizer Corpus** | Fertility & representation | `datasets/seed/` + later `datasets/raw/` | E001 |
| **Causal LM Pretrain Corpus** | Next-token train for NHAT | `datasets/raw/` → `processed/` | E002+ |

Tokenizer corpus can be a **subset / reweight** of pretrain raw, but manifests are **separate files**.

---

## On-disk layout

```
datasets/
  README.md
  schema/
    record.schema.json
  manifests/
    gate0_tokenizer.json      # mix + shard list for Gate0
    pretrain_stage0.json      # mix for Stage0a+ (filled when ready)
  seed/                       # COMMITTED — minimal real seed (not empty)
    multilingual/
    code/
    enterprise/
    scientific/
    synthetic_structure/
  raw/                        # NOT committed — large shards
    shards/
      multilingual/
      code/
      ...
  processed/                  # NOT committed — packed token ids later
```

---

## Record schema (one JSONL line)

```json
{
  "id": "nlx-seed-ru-0001",
  "text": "...",
  "lang": "ru",
  "bucket": "multilingual",
  "source": "nullxes_seed",
  "license": "nullxes_internal",
  "split": "train"
}
```

Required: `id`, `text`, `bucket`.  
Optional: `lang`, `source`, `license`, `split`.

**Forbidden in any corpus used for tokenizer or pretrain seed:**

- persona dialogues (`Anna said…`)
- employee tone / personality coaching
- brand slogan spam
- invented product URLs (use only [nullxesdai.online](https://www.nullxesdai.online/) / real contacts)

---

## Mix targets (Gate0)

| Bucket | Weight | Notes |
|--------|-------:|-------|
| multilingual | 0.40 | RU/EN/CN/DE-FR |
| code | 0.25 | py/ts/json/yaml |
| enterprise | 0.20 | contracts, workflows, API |
| scientific | 0.10 | STEM / math notation |
| synthetic_structure | 0.05 | tool_call / JSON schemas only |

Weights must sum to **1.0 ± 1e-6**. Validator enforces.

---

## Pipeline (concrete)

```
scaffold_corpus.py
       ↓
build_seed_corpus.py   → datasets/seed/**/*.jsonl + manifests/gate0_tokenizer.json
       ↓
validate_corpus.py     → PASS/FAIL report
       ↓
train_tokenizer.py     → reads manifest (not “whatever is on disk”)
       ↓
evaluate_tokenizer.py  → Gate0
```

Causal LM later:

```
ingest licensed/public shards → datasets/raw/shards/<bucket>/
validate_corpus.py --manifest pretrain_stage0.json
pack → datasets/processed/
train NHAT (only after Gate0 + Weight Genesis)
```

---

## Acceptance (corpus gate)

`validate_corpus.py` PASS requires:

1. Manifest exists and weights sum ≈ 1  
2. Every listed shard path exists and is non-empty  
3. Every line validates schema  
4. No synthetic-ban heuristics triggered (persona markers)  
5. Min docs per bucket (Gate0 seed: ≥ 20 lines/bucket)  
6. UTF-8 decodable; `text` length ≥ 32 chars  

Without PASS → **do not** claim Gate0 corpus ready. Blocks E001 tokenizer full train.

## Relationship to Causal LM

Gate0 seed is **not** enough for NHAT pretrain. Causal training requires growing `datasets/raw/shards/<bucket>/` with licensed/public sources, then `manifests/pretrain_stage0.json` (created when ready). Same schema. Same validator.

---

## What seed is / is not

**Is:** committed bootstrap so the lab is not empty; proves I/O, mix, schema, tokenizer path.  
**Is not:** Chinchilla-scale pretrain. Scale `raw/` on H200 storage later with documented sources.

---

## Commands

```bash
python scripts/scaffold_corpus.py
python scripts/build_seed_corpus.py
python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json
```
