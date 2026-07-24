# LÆTEX V1 — foundation bootstrapping runbook

`LÆTEX-NULLXES FOUNDATION MODEL` · 20B dense · tokenizer v1 · корпус LÆTEX V1.

Этап называется **foundation bootstrapping**, а не pretraining. Цель — получить
устойчивую связку архитектура → токенизация → обучение → генерация → QA.
Chinchilla-подобные объёмы токенов для 20B заведомо больше, и это осознанно
вне текущей задачи. Ни один отчёт не должен утверждать иного.

---

## Железо

18.757B в bf16 с AdamW — примерно 300 ГБ состояний: 37.5 веса, 37.5 градиенты,
150 моменты Adam, 75 fp32-мастер. Плюс активации.

| Вариант | Годится | Почему |
|---------|---------|--------|
| 1 pod × 4× H200 SXM (564 ГБ) | да | ZeRO-2 + gradient checkpointing, без CPU offload |
| 2× B300 (576 ГБ) | да | быстрее, но availability низкая |
| 1× RTX PRO 6000 96 ГБ | нет | выгружать пришлось бы 225 ГБ при 157 ГБ RAM |

RTX PRO 6000 остаётся максимум для корпуса / токенайзера / init на малой карте;
для train 20B AdamW её не хватает (отсюда снятие legacy ZeRO-3 1×96GB cfg).

Шейп этапа: `seq_len 2048` × `micro 2` × `accum 8` × 4 GPU = **131 072 токена на шаг**.
250 шагов ≈ 32.8M токенов ≈ около часа.

---

## Порядок

```bash
# 1. Канон и корпус
python scripts/build_seed_corpus.py
python scripts/build_identity_corpus.py
python scripts/download_local_corpus.py --config configs/datasets_latex_v1.yaml
python scripts/build_corpus_v1.py --config configs/datasets_latex_v1.yaml

# 2. Токенайзер — без --smoke
python scripts/train_tokenizer.py --config configs/tokenizer_latex_v1.yaml
python scripts/evaluate_tokenizer.py --config configs/tokenizer_latex_v1.yaml

# 3. Точные размеры корпуса на замороженном токенайзере
python scripts/build_corpus_v1.py --config configs/datasets_latex_v1.yaml --tokenizer tokenizer/latex-v1

# 4. Генезис весов
python scripts/init_model.py --config configs/nullxes_latex_20b_v1.yaml --dtype bfloat16 \
  --holdout-jsonl datasets/latex_v1/holdout/multilingual/shard_0000.jsonl

# 5. Этапы
bash scripts/run_stage3_iter.sh

# 6. SFT
python scripts/build_sft_v1.py
deepspeed --num_gpus 4 scripts/train_sft_v1.py --config configs/sft_20b_v1.yaml
```

---

## Гейты

### Токенайзер

`src/latex_tokenizer/gate.py` вызывается из `init_model.py`, `train_stage2_20b.py`
и `train_sft_v1.py`. Прогон не стартует, если:

- `meta.smoke == true` — это артефакт проверки пайплайна, а не токенайзер;
- `vocab_padded == true` — словарь добит `<|unused_*|>`, корпуса не хватило;
- `vocab_size_export != 131072`;
- спец-токены 0-11 не на своих id.

Отдельный блокирующий чек `concept_regression` в `evaluate_tokenizer.py`:
каждый слот-токен кодируется ровно одним id, а `NULLXES`, `LÆTEX`, `FOUNDATION`,
`MODEL`, `NULLXES-LÆTEX` не рассыпаются на символы и переживают encode → decode.
Это отдельно от fertility: токенайзер может попасть во все коридоры и при этом
шредить `<|tool_call|>`, ломая каждый структурный сэмпл корпуса.

### Init loss

Необученные веса обязаны давать cross entropy равную ln(vocab).

```
ln(131072) = 11.7856
11.7856 ± 0.15   PASS
< 8.98           leakage_suspected — не случайный init, утечка меток, tie/shape-баг
> 14.98          broken_init — сломан масштаб логитов или инициализация
```

Замер делается в fp32 и на случайных id, и на реальном батче из holdout — оба
должны попасть в коридор. Считать лосс в bf16 нельзя: на 131k классов точности
не хватает для полосы 0.15.

### Между этапами

- train loss падает;
- **holdout loss падает** — иначе это заучивание, и раннер останавливает цикл;
- identity QA проходит;
- `output_control` (schema-leak) = 0;
- `empathy_leak` = 0;
- `digital_entity_leak` = 0.

---

## Что фиксируется на каждом сохранении

`checkpoint_manifest.json` рядом с весами: этап, шаг, токены, train/holdout loss,
`grad_norm_p50`, sha256 токенайзера, конфига и манифеста датасета, git commit и
git dirty, железо, флаг EMA.

`grad_norm` логируется каждый шаг. Стабильный упор в потолок клиппинга или скачок
на порядок — первый признак приближающейся смерти прогона, и увидеть его надо
до появления NaN.

EMA — теневая копия весов в bf16 на CPU (~37.5 ГБ RAM), обновление раз в 100 шагов,
decay 0.999. На GPU держать нельзя: съест бюджет активаций. Требует ZeRO ≤ 2,
на ZeRO-3 параметры шардированы и копия была бы пустой — код это проверяет и
отключает EMA сам.

Карточка этапа пишется в `docs/MODEL_HISTORY.md` фиксированной формой.
Поле **Failures** обязательное и непустое: если этап прошёл чисто, записывается,
что именно проверяли.

---

## Ожидания

На 300-400M токенов получаем связность на коротких фрагментах, валидный JSON в
`<|tool_call|>`, соблюдение спец-токенов и твёрдую самоидентификацию. Длинные
тексты остаются слабыми. Это и есть цель «не бред на инит-весах» — не больше.

---

## Чего в V1 нет

- MoE — после 50B;
- IEL, персонажи, memory-сервис;
- long context 32k/128k: `rope_scaling` есть в конфигах, но в коде не реализован,
  поэтому как рабочий не заявляется;
- alignment сверх SFT и заготовки DPO.
