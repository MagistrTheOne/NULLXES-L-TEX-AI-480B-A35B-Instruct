<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   N U L L X E S  ·  L Æ T E X  A I                           ║
║   Foundation Research Lab                                    ║
║                                                              ║
║   480B-A35B-Instruct  ·  MoE  ·  Digital Employees           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

# NULLXES-LÆTEX AI

**Сначала научить модель видеть язык. Потом строить кору.**

Product: [nullxesdai.online](https://www.nullxesdai.online/) · Contact: ceo@nullxes.com

[![Gate](https://img.shields.io/badge/Research_Gate_0-ACTIVE-0a0a0a?style=flat-square&labelColor=111111&color=c8ff00)](docs/07_RESEARCH_GATE_0.md)
[![NHAT](https://img.shields.io/badge/NHAT-BLOCKED_until_Gate0_PASS-0a0a0a?style=flat-square&labelColor=111111&color=ff4d4d)](docs/07_RESEARCH_GATE_0.md)
[![Tokenizer](https://img.shields.io/badge/Tokenizer-v0.1_131k-0a0a0a?style=flat-square&labelColor=111111&color=6b8aff)](docs/06_TOKENIZER_DESIGN.md)
[![Weights](https://img.shields.io/badge/Foreign_checkpoints-FORBIDDEN-0a0a0a?style=flat-square&labelColor=111111&color=ff4d4d)](#forbidden)

</div>

---

## Current stage

**NULLXES-LÆTEX Research Gate 0: Tokenizer Fertility & Representation Gate**

| | |
|--|--|
| Goal | Prove representation quality — not train 480B |
| Artifact | `tokenizer/latex-v0.1/` |
| Vocab | **131072** (fixed for Stage0 / 7B / A35B) |
| Next after PASS | Stage0a ~100M |
| NHAT / MoE code | **Blocked** |

See [`docs/07_RESEARCH_GATE_0.md`](docs/07_RESEARCH_GATE_0.md).

---

## What this is

Foundation brain for **NULLXES Digital Employees** — enterprise agents with identity, memory, tools, and workflows.

North star (not the headline param count):

```
A35B dense ancestor  →  MoE expansion  →  IEL digital employees
```

Flagship reference: **480B total / ~35B active** MoE Instruct. Spec: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Forbidden

- Qwen / Llama / Mistral / DeepSeek / Yi / GLM weights  
- LoRA / adapters on foreign checkpoints  
- Distillation from foreign models  
- Pretrained SentencePiece / foreign tokenizer loads  
- Implementing NHAT before Gate 0 PASS  

---

## Repo map

| Path | Role |
|------|------|
| [`docs/`](docs/) | Philosophy, ADRs, Gate 0, tokenizer design, vocab migration |
| [`configs/`](configs/) | Model + tokenizer + runtime (cloud-agnostic) |
| [`src/latex_tokenizer/`](src/latex_tokenizer/) | Gate 0 trainer / eval (only active code track) |
| [`tests/tokenizer_samples/`](tests/tokenizer_samples/) | Fixed fertility benchmark |
| [`tokenizer/latex-v0.1/`](tokenizer/latex-v0.1/) | Versioned artifacts (after train) |
| `src/latex/models/` | **Do not create until Gate 0 PASS** |

---

## Quickstart — corpus then tokenizer (Gate 0)

```bash
pip install -r requirements.txt
python scripts/scaffold_corpus.py
python scripts/build_seed_corpus.py
python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json
# only if validate PASS:
python scripts/train_tokenizer.py --config configs/tokenizer_stage0.yaml
python scripts/evaluate_tokenizer.py --config configs/tokenizer_stage0.yaml
```

Corpus plan: [`docs/10_CORPUS_PLAN.md`](docs/10_CORPUS_PLAN.md) · Product: [nullxesdai.online](https://www.nullxesdai.online/)

## Stage1 Weight Genesis (HF CausalLM — after Gate0)

```bash
pip install -r requirements-stage1.txt
python scripts/init_model.py --config configs/nullxes_latex_7b.yaml
python scripts/smoke_hf_causal.py --checkpoint checkpoints/nullxes-latex-7b
```

Public API: `LatexForCausalLM` (`model_type=latex`) — see [`docs/11_TRANSFORMERS_CONTRACT.md`](docs/11_TRANSFORMERS_CONTRACT.md).  
NHAT is the **engine**; Transformers CausalLM is the **chassis**.

Hardware: design assumes **H200 / B300**. Paths via `configs/runtime.yaml` (`streaming`, `mmap`). Provider is **generic** in model configs.

---

## Scale ladder

| Stage | Model | Vocab | Status |
|------:|-------|------:|--------|
| Gate 0 | Tokenizer v0.1 | 131072 | **ACTIVE** |
| 0a | ~100M | 131072 | blocked |
| 0b | ~500M | 131072 | blocked |
| 0c | ~1.6B | 131072 | blocked |
| 1 | ~7B | 131072 | planned |
| 2 | A35B | 131072 | planned |
| 3 | 480B-A35B MoE | 131072 → optional 262k migration | reference |

262k expansion: [`docs/08_VOCAB_MIGRATION.md`](docs/08_VOCAB_MIGRATION.md) — experiment, never automatic.

---

## Identity stack

One trunk. Many employees.

```
NULLXES-LÆTEX trunk
  + Identity Embedding Layer (Anna, Adeline, Karen, …)
  + Role Adapter
  + External memory / policy
```

---

## License / ownership

NULLXES proprietary research. From-scratch weights and tokenizer only.

<div align="center">

`NULLXES Research Lab · 2026`

</div>
