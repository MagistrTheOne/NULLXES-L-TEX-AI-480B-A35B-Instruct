# NULLXES-LÆTEX — Stage0 Corpus Blueprint (production)

**Status:** blueprint v0.1 · 2026-07-18  
**Owner:** @MagistrTheOne · NULLXES  
**Contact:** ceo@nullxes.com · https://www.nullxesdai.online/  
**Rule:** no foreign foundation weights · no LLM-written synthetic prose · only licensed / open / NULLXES-owned text  

This document is the **source of truth** for writing download → clean → pack pipelines.  
Related: `docs/10_CORPUS_PLAN.md` (gate mechanics), `configs/corpus_gate0.yaml`, `datasets/schema/record.schema.json`.

---

## 0. Project recheck (static, no venv / no train)

| Area | State | Verdict |
|------|-------|---------|
| Schema + seed + `validate_corpus.py` | Present | Data plane I/O works |
| Identity shards + mantras | Present (~KB scale) | Name binding OK for smoke; **not** a language corpus |
| `hf_mini/` (wiki/pg19/pes2o/…) | On disk, **not** in `pretrain_stage0.json` | Mini samples exist; mix not production |
| `licenses/`, `metadata/`, `clean/`, curriculum manifests | Missing | Must add for governance |
| Identity canon `@MagistrTheOne` | Missing in mantras | Must add (EN/RU) |
| Fabricated product lore risk | Medium | Restrict identity to **canon facts only** (see §6) |
| Hardware docs | H200→B300 primary | **H100 OK for Stage0 pipeline/smoke**; scale train stays H200/B300 |

**Current `pretrain_stage0.json`:** ~500 docs, mostly seed + identity + repo code.  
**Verdict:** ready for **pipeline engineering**, not for Chinchilla-scale Stage0 language modeling.

---

## 1. Corpus structure (buckets)

Use stable bucket IDs in manifests. Subdomains live in `source` / `domain` metadata.

| Bucket ID | Contents | Notes |
|-----------|----------|-------|
| `general_language` | Books, Wikipedia, high-quality encyclopedic/prose EN+RU | Renamed from loose “multilingual” for Stage0+; keep alias in validator |
| `code` | Source code + code comments (permissive licenses) | Code-first |
| `documentation` | Language/runtime/framework official docs | Python, LLVM, K8s, Apache, Mozilla, GNU, … |
| `enterprise` | Specs, RFCs, OpenAPI/examples, ops runbooks (public) | Enterprise-first |
| `math` | Formal math / proof-style / textbooks (open) | Separate from soft “science blog” |
| `science` | Open scientific text (papers abstracts + open books) | Filter PDF junk hard |
| `legal` | Open statutes / licenses / public legal corpora | No private client contracts |
| `finance` | Open filings / standards / textbooks (public domain / CC) | No scraped paywalled news |
| `dialogues` | Structured human dialogues (licensed), **not** forums | Tiny %; quality >> volume |
| `technical_writing` | Manuals, HOWTOs, man pages | Overlaps docs; keep if license clean |
| `identity` | Canon LÆTEX / NULLXES facts only | Pretrain slot, not SFT-only |
| `synthetic_structure` | JSON/YAML/tool schemas / templates **without LLM prose** | Structure only |

**Stage0 mapping to current 5-bucket gate** (compat until schema v2):

| New domain | Maps into current `bucket` |
|------------|----------------------------|
| general_language + dialogues + technical_writing | `multilingual` |
| code | `code` |
| documentation + enterprise + legal + finance | `enterprise` |
| math + science | `scientific` |
| identity + synthetic_structure | `synthetic_structure` (+ identity also upsampled into `multilingual`) |

Schema v2 (later): expand `bucket` enum; until then keep 5 buckets + `domain` field.

---

## 2. Mix percentages

### 2.1 Stage0 Causal LM mix (token weights)

| Domain | Weight | Rationale |
|--------|-------:|-----------|
| general_language (EN+RU) | **0.28** | Language backbone |
| code | **0.28** | Code-first |
| documentation | **0.14** | Executable knowledge |
| enterprise (+ legal/finance public) | **0.12** | Business register |
| math | **0.06** | Reasoning substrate |
| science | **0.05** | STEM breadth |
| technical_writing | **0.03** | Manuals / man |
| dialogues (licensed) | **0.02** | Turn structure |
| identity (canon) | **0.015** | Name / company / author |
| synthetic_structure | **0.005** | Tool/JSON atoms |
| **Sum** | **1.00** | |

EN/RU split inside language+docs: target **~55% EN / ~45% RU** by tokens (not docs).

### 2.2 Gate0 Tokenizer mix (fertility)

Keep near current until vocab locked:

| Bucket | Weight |
|--------|-------:|
| multilingual | 0.40 |
| code | 0.25 |
| enterprise | 0.20 |
| scientific | 0.10 |
| synthetic_structure | 0.05 |

Tokenizer corpus = stratified sample of Stage0 raw (≥10–50 GB text), not a separate fiction set.

---

## 3. Real sources (ingest allowlist)

Only sources with **explicit license path**. Prefer primary mirrors over random mirrors.

### 3.1 Priority download order (do this first)

```
P0  Python docs (docs.python.org) + PEPs (open)
P0  RFCs (IETF rfc-editor) — text/plain
P0  Linux man-pages / kernel docs (GPL/permissive as tagged)
P0  SQLite docs
P1  Kubernetes docs (CC-BY / Apache as published)
P1  LLVM / Clang docs
P1  Apache project docs (httpd, Kafka, …) — per-project license
P1  Mozilla MDN (CC-BY-SA) — keep attribution trail
P1  GNU manuals (GFDL/GPL) — track copyleft obligations
P2  Wikipedia EN dump (CC BY-SA) — cleaned
P2  Wikipedia RU dump (CC BY-SA) — cleaned
P2  PG19 / Project Gutenberg subset (PD / verified)
P2  The Stack v2 / Stack Edu — **filter by license allowlist only**
P2  OpenMath / ProofWiki / arXiv abstract+HTML where license OK
P3  TensorFlow / PyTorch docs (Apache-2.0 / BSD)
P3  OpenStreetMap wiki (ODbL/CC — check reuse rules)
P3  Own NULLXES repo docs + public site copy (nullxes_internal)
```

### 3.2 Source registry (concrete IDs for pipeline)

| `source_id` | Origin | License class | Domain |
|-------------|--------|---------------|--------|
| `python_docs` | docs.python.org | PSF | documentation |
| `pep_archive` | peps.python.org | PSF | documentation |
| `ietf_rfc` | rfc-editor.org | IETF Trust | enterprise |
| `man_pages` | man7.org / kernel.org | GPL-2.0+ / permissive | documentation |
| `sqlite_docs` | sqlite.org | Public domain dedication | documentation |
| `k8s_docs` | kubernetes.io | CC-BY-4.0 | enterprise |
| `llvm_docs` | llvm.org | Apache-2.0-with-LLVM | documentation |
| `apache_docs` | apache.org projects | Apache-2.0 | enterprise |
| `mdn` | developer.mozilla.org | CC-BY-SA-3.0 | documentation |
| `gnu_manuals` | gnu.org | GFDL / GPL | documentation |
| `wiki_en` | dumps.wikimedia.org | CC-BY-SA-3.0 + GFDL | general_language |
| `wiki_ru` | dumps.wikimedia.org | CC-BY-SA-3.0 + GFDL | general_language |
| `pg19` | deepmind/pg19 or Gutenberg PD subset | PD / research terms | general_language |
| `stack_edu` | bigcode The Stack (Edu) | **per-file license filter** | code |
| `pytorch_docs` | pytorch.org | BSD-style | documentation |
| `tf_docs` | tensorflow.org | Apache-2.0 | documentation |
| `nullxes_identity` | `src/latex_data/identity_corpus.py` | nullxes_internal | identity |
| `nullxes_repo` | this monorepo (no secrets) | nullxes_internal | code |

**Do not** use FineWeb/CC full dumps for Stage0 until license + quality gate is written. Mini HF samples (`hf_mini/`) are **dev-only**, not production Stage0.

---

## 4. What NOT to take

| Ban | Why |
|-----|-----|
| LLM-generated web text / “AI blogs” | Contaminates prior |
| SEO / affiliate / doorway pages | Low information |
| Clickbait news / tabloid | Style poison |
| Raw forums, Reddit dumps, YouTube comments | Toxicity + license mess |
| Chat logs without license | Legal + PII |
| Duplicate mirrors of same docs | Inflates freq |
| Minified JS / vendored `node_modules` | Noise |
| Secrets, `.env`, private tickets | Security |
| Foreign model cards / “I am ChatGPT” text | Identity collision |
| Invented NULLXES product facts | Hallucination training |
| Paywalled / scraped-without-terms | Legal |
| Image-only / OCR garbage without QA | Quality |

Heuristic bans (validator): persona coaching, brand slogan spam, fake domains (`nullxes.ai`, etc.), celebrity AI cosplay.

---

## 5. Cleaning pipeline (implementable)

```
download (raw mirror)
    ↓
license_check          → licenses/<source_id>.json + reject
    ↓
format_extract         → HTML/PDF/MD → UTF-8 text; keep code fences
    ↓
pii_scrub              → emails/phones/keys (keep allowlisted NULLXES contacts)
    ↓
language_detect        → en/ru/code/other; drop other for Stage0 bilingual
    ↓
quality_score          → length, repetition, symbol ratio, boilerplate
    ↓
normalize_unicode      → NFKC (same as tokenizer)
    ↓
exact_dedup            → SHA256(text)
    ↓
near_dedup             → MinHash / LSH (doc + paragraph)
    ↓
chunk                  → 512–4096 tokens target; respect doc boundaries
    ↓
write clean/           → JSONL per schema
    ↓
manifest + metadata    → corpus-vX.Y
    ↓
validate_corpus.py     → PASS required
    ↓
pack processed/        → .bin / parquet token ids (after tokenizer)
```

### Quality score (minimum gates)

| Check | Threshold |
|-------|-----------|
| `len(text)` | ≥ 200 chars (docs); ≥ 64 (code cells) |
| alpha / alnum ratio | ≥ 0.55 (prose); code exempt |
| mean line length | < 2000 |
| duplicate 13-gram mass | < 0.30 |
| lang confidence | ≥ 0.80 for en/ru |
| boilerplate nav chrome | strip before score |

### Chunking rules

- Prefer article / file / RFC section boundaries.
- Never merge two licenses into one doc.
- Code: one file ≈ one doc (cap 32k chars; split by top-level def if larger).
- Identity docs: **never chunk-split** a canon paragraph mid-fact.

---

## 6. Identity in pretraining (not SFT-only)

### 6.1 Goal

Model must answer (EN/RU):

- Name: **LÆTEX** / **NULLXES-LÆTEX**
- Company: **NULLXES** (RU: **НУЛЛЕКСЕС** / NULLXES)
- Author / founder contact handle: **@MagistrTheOne**
- Site: **https://www.nullxesdai.online/**
- Email: **ceo@nullxes.com**
- Not ChatGPT / Claude / Llama / foreign fine-tune

### 6.2 Canon allowlist (only these facts in identity shards)

| Fact | Allowed forms |
|------|----------------|
| Model short name | LÆTEX, NULLXES-LÆTEX |
| Company | NULLXES, НУЛЛЕКСЕС |
| Author | @MagistrTheOne |
| Site | https://www.nullxesdai.online/ |
| Email | ceo@nullxes.com |
| Role | foundation language model by NULLXES |
| Training rule | from-scratch; no foreign foundation checkpoints |

**Forbidden in identity text:** invented clients, fake revenue, fake benchmarks, fake offices, alternate domains, celebrity personas (Anna/Adeline as *pretrain identity of the brain* — those are IEL overlays later, not trunk name).

Architecture notes (NHAT, MoE sizes) may live in `scientific` **only if** they match committed `ARCHITECTURE.md` — treat as lab docs, not marketing.

### 6.3 Formats in corpus (pretrain)

1. **Third-person facts** (`nullxes_identity.jsonl`) — 60% of identity tokens  
2. **Hard mantras** Q/A + chat specials (`identity_mantra.jsonl`) — 40% of identity tokens, high upsample  
3. **No** long persona roleplay in Stage0 pretrain

### 6.4 Percent

| Stage | Identity token share | Notes |
|------:|---------------------:|-------|
| Stage0 (50M–100M) | **1.5–3.0%** | Higher density so name sticks |
| Stage1 7B | **0.5–1.0%** | Maintain, don’t dominate |
| Stage2 A35B | **0.3–0.7%** | |
| Stage3 480B | **0.2–0.5%** | |

Training sampler: `identity_upsample` + optional `identity_loss_weight` (already in `train_stage0a.py`).

### 6.5 Required mantra additions

EN: “Who is the author?” → “@MagistrTheOne (NULLXES).”  
RU: «Кто автор?» → «@MagistrTheOne (NULLXES / НУЛЛЕКСЕС).»  
RU company: «NULLXES (НУЛЛЕКСЕС)».

---

## 7. Size targets (docs / tokens / GB)

Rough UTF-8 text GB ≈ tokens / 0.7e9 for mixed EN/RU/code (calibrate after tokenizer).

| Stage | Model | Docs (order) | Training tokens | Clean text GB | Hardware for *this* stage |
|------:|-------|-------------:|----------------:|--------------:|---------------------------|
| **Stage0-proxy** | 50M → 100M | 5e4–2e5 | **2–10B** | **5–25 GB** | CPU smoke / **1× H100** / 1× H200 |
| **Stage0** | 100M → 1.6B path | 5e5–2e6 | **20–80B** | **40–150 GB** | 1–8× H200 (H100 OK for pipeline) |
| **Stage1** | 7B | 5e6–2e7 | **140–300B** | **250–600 GB** | 8–32× H200 |
| **Stage2** | A35B | 2e7–1e8 | **700B–1.5T** | **1.5–4 TB** | 32–64× B300 (or large H200) |
| **Stage3** | 480B-A35B MoE | 5e7–2e8+ | **2–6T** (vs active) | **5–15 TB** | 256–512× B300 |

Chinchilla-style for dense Stage0/1: ~20× params in tokens as **floor**; enterprise/code mix often needs **more epochs on high-quality** rather than infinite web.

**Immediate lab target (this week):** Stage0-proxy corpus **≥10 GB clean** + identity canon + validate PASS → 50M/100M on **H100** smoke, then H200 for serious Stage0a.

---

## 8. Curriculum learning (ordered phases)

Do **not** shuffle all domains uniformly for the whole run. Use phase schedules on the same packed shards (reweight sampler).

| Phase | % of steps | Mix emphasis | Goal |
|------:|-----------:|--------------|------|
| **C1 Language+Code** | 35% | general_language 0.40 · code 0.45 · docs 0.10 · rest 0.05 | Syntax + bilingual fluency |
| **C2 Docs+Math** | 25% | documentation 0.35 · math 0.20 · science 0.15 · code 0.20 · lang 0.10 | Precise technical prose |
| **C3 Enterprise** | 20% | enterprise+legal+finance 0.45 · docs 0.25 · code 0.20 · lang 0.10 | Workflow / RFC / ops |
| **C4 Identity** | 10% | identity 0.15 · synthetic_structure 0.05 · balanced rest 0.80 | Name lock without collapse |
| **C5 Instruction-shaped** | 10% | structured Q/A from **human/docs** (not LLM synth) + light mantras | Prep for protocol SFT |

Manifest files:

```
datasets/manifests/curriculum/
  stage0_c1_language_code.json
  stage0_c2_docs_math.json
  stage0_c3_enterprise.json
  stage0_c4_identity.json
  stage0_c5_instruction_shaped.json
```

Trainer reads `curriculum.schedule` list: `{phase, steps_frac, manifest}`.

---

## 9. Storage architecture

```
datasets/
  README.md
  schema/
    record.schema.json              # v1: 5 buckets
    record.schema.v2.json           # later: expanded buckets
  manifests/
    gate0_tokenizer.json
    pretrain_stage0.json            # current alias
    corpus-v0.1/
      mix.json
      shards.json
      curriculum/…
    corpus-v0.2/…
  metadata/
    sources/<source_id>.json        # url, license, retrieved_at, bytes, sha256
    corpus_versions.json            # pointer to active corpus-vX.Y
  licenses/
    <source_id>.json                # SPDX + obligations + attribution
    ATTRIBUTION.md                  # human-readable
  raw/
    mirrors/<source_id>/…           # immutable downloads
    shards/<domain>/…               # optional pre-clean shards
  clean/
    corpus-v0.1/<domain>/*.jsonl    # validated training text
  processed/
    corpus-v0.1/<seq_len>/*.bin     # token ids after tokenizer
  seed/                             # committed bootstrap (keep)
  sft/                              # post-pretrain only; do not replace identity pretrain
```

Record fields (v1 + extensions):

```json
{
  "id": "nlx-ietf_rfc-en-09457",
  "text": "...",
  "lang": "en",
  "bucket": "enterprise",
  "domain": "rfc",
  "source": "ietf_rfc",
  "license": "IETF-Trust",
  "split": "train",
  "corpus_version": "v0.1",
  "quality": 0.91
}
```

---

## 10. Data Governance

### 10.1 Versioning

| Version | Meaning |
|---------|---------|
| `corpus-v0.1` | First validated Stage0-proxy (≥10 GB clean) |
| `corpus-v0.2` | + wiki EN/RU full clean + Stack license filter |
| `corpus-v1.0` | Locked mix for 7B Stage1 |

Every train run logs: `corpus_version`, manifest SHA256, tokenizer SHA256, git commit.

### 10.2 Per-source manifest (required)

```json
{
  "source_id": "ietf_rfc",
  "url": "https://www.rfc-editor.org/rfc/",
  "license_spdx": "LicenseRef-IETF",
  "retrieved_at": "2026-07-18",
  "raw_bytes": 0,
  "clean_docs": 0,
  "clean_tokens_est": 0,
  "sha256_raw_tree": "...",
  "pii_policy": "scrub_default",
  "allowed_stages": ["stage0", "stage1", "stage2", "stage3"]
}
```

### 10.3 Duplicate control

- Global exact dedup registry: `metadata/dedup/exact.sqlite`  
- Near-dedup signatures: `metadata/dedup/minhash.parquet`  
- Cross-version: new corpus must report % overlap with previous

### 10.4 Pre-train automatic gates

`scripts/validate_corpus.py` + new `scripts/gate_corpus_release.py` must PASS:

1. Weights sum = 1  
2. All shard paths exist  
3. Schema valid  
4. License file present for every `source`  
5. Ban heuristics clean  
6. Identity canon checklist (LÆTEX, NULLXES, @MagistrTheOne, site, email) present in identity shards  
7. No forbidden brands claiming to be self  
8. Min docs / min GB for target stage  
9. Dedup report attached  
10. EN/RU ratio within ±10% of target  

**No PASS → no train job submit** (local H100 or RunPod).

---

## 11. Download pipeline blueprint (scripts to implement)

| Script | Responsibility |
|--------|----------------|
| `scripts/ingest/fetch_<source_id>.py` | Mirror → `raw/mirrors/` |
| `scripts/ingest/license_audit.py` | Write `licenses/` + fail closed |
| `scripts/clean/extract_text.py` | HTML/MD/RFC → text |
| `scripts/clean/quality_filter.py` | Score + drop |
| `scripts/clean/dedup.py` | Exact + MinHash |
| `scripts/clean/chunk_docs.py` | Boundary-aware chunks |
| `scripts/build_clean_manifest.py` | Emit `corpus-vX.Y` |
| `scripts/gate_corpus_release.py` | Governance PASS/FAIL |
| `scripts/build_identity_corpus.py` | Already exists — extend canon |
| `scripts/validate_corpus.py` | Already exists — keep |

H100 test loop (no full cluster):

```bash
# 1) fetch P0 sources only
# 2) clean → datasets/clean/corpus-v0.1/
# 3) gate_corpus_release.py --stage stage0-proxy
# 4) validate_corpus.py --manifest datasets/manifests/corpus-v0.1/mix.json
# 5) smoke train 50M/100M (existing train_stage0a path) on 1× H100
```

---

## 12. Hardware note (aligned with lab)

| Use | GPU |
|-----|-----|
| Download / clean / validate / CPU unit tests | CPU OK |
| Stage0-proxy smoke (50M–100M) | **1× H100** acceptable |
| Serious Stage0a+ / fertility ablations | **H200** (project default) |
| A35B / 480B | **B300** clusters |

Skill default remains H200/B300 for scale; H100 is explicitly allowed here for **pipeline + proxy smoke**, not for flagship pretrain.

---

## 13. Acceptance criteria — Stage0-proxy corpus-v0.1

- [ ] ≥ **10 GB** clean UTF-8 under `datasets/clean/corpus-v0.1/`  
- [ ] Mix within 2% of §2.1 after sampling weights  
- [ ] Identity ≥ **1.5%** tokens; canon includes **@MagistrTheOne** + НУЛЛЕКСЕС  
- [ ] Zero LLM-synthetic prose sources  
- [ ] Every `source` has `licenses/*.json`  
- [ ] `gate_corpus_release.py` PASS  
- [ ] Tokenizer fertility re-eval on stratified sample (Gate0)  
- [ ] 50M or 100M smoke on H100: finite loss + identity QA hits ≥2/4  

---

## 14. Immediate next actions (engineering order)

1. Extend identity canon (`@MagistrTheOne`, НУЛЛЕКСЕС) — no new invented product claims  
2. Add `metadata/`, `licenses/`, `clean/` layout + `corpus_versions.json`  
3. Implement P0 fetchers: Python docs, RFC, man-pages, SQLite  
4. Wire `hf_mini` as **dev-only** manifest; production mix from `clean/`  
5. Curriculum manifests C1–C5  
6. `gate_corpus_release.py`  
7. H100 smoke train against `corpus-v0.1`  

---

**End of blueprint.** Implement fetch/clean scripts against this file; do not train “on whatever is in `raw/`” without a versioned manifest.
