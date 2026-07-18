# Local playground — RTX 2080 SUPER (8 GB)

Песочница для Win11 / 2080 SUPER.  
**Не** промышленный контур. H200 / Stage0a-100M / 7B — отдельно в `configs/` и `scripts/run_stage0a_bootstrap.sh`.

## Твой ПК (зафиксировано)

| | |
|--|--|
| GPU | RTX 2080 SUPER **8 GB** |
| CPU | i9-10900F |
| RAM | ~16 GB |
| Роль | pipeline + тесты + лёгкий train |

## Что здесь

| Файл | Назначение |
|------|------------|
| `configs/latex_50m_2080.yaml` | ~50–60M dense, seq 256, fp16 |
| `configs/latex_50m_2080_smoke.yaml` | 50 steps smoke (VRAM/loss check) |
| `run_smoke.ps1` | быстрый тест на Windows |
| `run_train.ps1` | вечерний прогон ~100M tokens |

## Перед первым запуском

Из корня репо:

```powershell
cd D:\NULLXES\NULLXES-L-TEX-AI-480B-A35B-Instruct
# отдельный venv под 2080 (не смешивать с H200)
python -m venv .venv-2080
.\.venv-2080\Scripts\Activate.ps1
pip install -r requirements-torch-cu124.txt --index-url https://download.pytorch.org/whl/cu124
pip install -r local_2080/requirements.txt

python scripts/build_identity_corpus.py
python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_stage0.json
# tokenizer FULL как на H200 (soft Unigram + pad 131072)
.\local_2080\run_tokenizer.ps1
```

## Smoke (2–5 мин)

```powershell
.\local_2080\run_smoke.ps1
```

## Train (вечерами)

```powershell
.\local_2080\run_train.ps1
```

Чекпоинт: `checkpoints/local_2080/latex-50m/`

QA:

```powershell
python scripts/qa_stage0a.py --checkpoint checkpoints/local_2080/latex-50m --device cuda
```

## Лимиты 2080

- seq **256**, micro_batch **1**, fp16 + grad checkpointing  
- не гоняй 100M@512 как на H200 — OOM  
- датасет: identity+repo (уже); PG19/code sample — позже, стримом с диска D:  
- системной RAM 16 GB мало для огромных HF датасетов в RAM

## Следующий месяц — H200 (пром)

Не смешивай конфиги:

| Контур | Путь |
|--------|------|
| Local 2080 | `local_2080/` |
| H200 bootstrap 100M | `configs/stage0a_100m_bootstrap.yaml` |
| H200 industrial v0.2 | `configs/stage0a_100m.yaml` + новый corpus allowlist (peS2o/StarCoder/PG19, без FineWeb) |
| 7B genesis | `configs/nullxes_latex_7b.yaml` (уже есть на RunPod) |

Идея: здесь отлаживаешь код/QA/identity; на H200 масштабируешь токены и корпус.
