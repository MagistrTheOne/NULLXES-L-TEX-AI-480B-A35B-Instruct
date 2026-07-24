<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   L Æ T E X  ·  N U L L X E S                                ║
║   FOUNDATION MODEL                                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

# NULLXES-LÆTEX AI

**Сначала научить модель видеть язык. Потом строить кору.**

Product: [nullxesdai.online](https://www.nullxesdai.online/) · Contact: ceo@nullxes.com

[![Line](https://img.shields.io/badge/LÆTEX_V1-20B_dense-0a0a0a?style=flat-square&labelColor=111111&color=c8ff00)](docs/17_LATEX_V1.md)
[![Tokenizer](https://img.shields.io/badge/Tokenizer-v1_131072-0a0a0a?style=flat-square&labelColor=111111&color=6b8aff)](configs/tokenizer_latex_v1.yaml)
[![Stage](https://img.shields.io/badge/Stage-foundation_bootstrapping-0a0a0a?style=flat-square&labelColor=111111&color=6b8aff)](docs/17_LATEX_V1.md)
[![Weights](https://img.shields.io/badge/Foreign_checkpoints-FORBIDDEN-0a0a0a?style=flat-square&labelColor=111111&color=ff4d4d)](#forbidden)

</div>

---

## What this is

`LÆTEX-NULLXES FOUNDATION MODEL` — фундаментальная языковая модель компании NULLXES,
которая строится с нуля: своя архитектура, свой tokenizer, свой корпус, своё выравнивание.
Кратко: **LÆTEX — языковая модель компании NULLXES.**

Модель отвечает по протоколу **Input → Analysis → Answer**: точность, структура,
отказ при недостатке данных, критика ошибочной постановки. Ориентир — compiler и
researcher, а не support-бот. Никакого эмоционального обслуживания.

---

## Current line: LÆTEX V1

| | |
|--|--|
| Модель | 20B dense NHAT — L=24, d=8192, 18.757B параметров |
| Tokenizer | `tokenizer/latex-v1` — unigram, vocab 131072, byte fallback |
| Корпус | LÆTEX V1, 300-400M уникальных токенов + holdout 0.5% |
| Этап | **foundation bootstrapping**, не pretraining |
| Прогон | этапы по 250 шагов с holdout- и QA-гейтом между ними |
| Железо | 1 pod × 4× H200 SXM (ZeRO-2, без CPU offload) |
| MoE | после 50B — в текущей линии `ffn_type: dense` |

Задача V1 — не «знать мир», а получить работающую связку архитектура → токенизация →
обучение → генерация → QA. Chinchilla-подобные объёмы для 20B заведомо больше,
и это записано как сознательное решение. Runbook: [`docs/17_LATEX_V1.md`](docs/17_LATEX_V1.md).

---

## Forbidden

- Qwen / Llama / Mistral / DeepSeek / Yi / GLM weights
- LoRA / adapters на чужих чекпоинтах
- Distillation с чужих моделей, включая их выходы в SFT
- Pretrained SentencePiece / чужие tokenizer-артефакты
- Прогоны без holdout, без гейтов и без `checkpoint_manifest.json`

---

## Quickstart — LÆTEX V1

```bash
pip install -r requirements-train.txt

# 1. Канон + корпус
python scripts/build_seed_corpus.py
python scripts/build_identity_corpus.py
python scripts/download_local_corpus.py --config configs/datasets_latex_v1.yaml
python scripts/build_corpus_v1.py --config configs/datasets_latex_v1.yaml

# 2. Токенайзер (без --smoke: smoke-артефакт отклоняется гейтом на init и train)
python scripts/train_tokenizer.py --config configs/tokenizer_latex_v1.yaml
python scripts/evaluate_tokenizer.py --config configs/tokenizer_latex_v1.yaml

# 3. Генезис весов + гейт init loss ≈ ln(131072) = 11.7856
python scripts/init_model.py --config configs/nullxes_latex_20b_v1.yaml --dtype bfloat16

# 4. Этапный прогон с holdout- и QA-гейтами
bash scripts/run_stage3_iter.sh

# 5. SFT с лоссом только на assistant-части
python scripts/build_sft_v1.py
deepspeed --num_gpus 4 scripts/train_sft_v1.py --config configs/sft_20b_v1.yaml
```

---

## Repo map

| Path | Role |
|------|------|
| [`src/latex/`](src/latex/) | NHAT-модель, HF-совместимый `LatexForCausalLM` |
| [`src/latex_tokenizer/`](src/latex_tokenizer/) | Обучение, оценка и блокирующий гейт токенайзера |
| [`src/latex_data/`](src/latex_data/) | Канон, фильтры корпуса, packing, телеметрия, SFT-набор |
| [`configs/`](configs/) | Active family: 20B dense / 200B MoE / 480B MoE — see [`configs/README.md`](configs/README.md) |
| [`scripts/`](scripts/) | Сборка корпуса, init, обучение, holdout-eval, QA |
| [`docs/`](docs/) | ADR, история, runbook V1 ([`docs/17_LATEX_V1.md`](docs/17_LATEX_V1.md)) |
| [`tokenizer/latex-v1/`](tokenizer/latex-v1/) | Frozen tokenizer artifacts (train on cluster) |

---

## Гейты

Ни один прогон не считается состоявшимся без них.

| Гейт | Требование |
|------|-----------|
| Tokenizer | `smoke=false`, `vocab_padded=false`, export = 131072, спец-токены 0-11 атомарны |
| Concept regression | `NULLXES`, `LÆTEX`, `FOUNDATION`, `MODEL`, слот-токены не рассыпаются |
| Init loss | ln(131072) = 11.7856 ± 0.15; < 9 — утечка, > 15 — сломанная инициализация |
| Holdout | падает от этапа к этапу, иначе это заучивание |
| QA | identity, отсутствие schema-leak, empathy-leak и цифровых сущностей |

---

## Scale ladder

| Этап | Модель | Vocab | Статус |
|-----:|--------|------:|--------|
| V1 | 20B dense | 131072 | **активная линия** |
| V2 | 50B dense | 131072 | планируется |
| V3 | MoE (480B / A35B reference) | 131072 | после 50B |

Спецификация flagship-конфигурации: [`ARCHITECTURE.md`](ARCHITECTURE.md).
История артефактов: [`docs/MODEL_HISTORY.md`](docs/MODEL_HISTORY.md).

---

## License / ownership

NULLXES proprietary research. From-scratch weights and tokenizer only.

<div align="center">

`NULLXES Research Lab · 2026`

</div>
