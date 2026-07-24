"""LÆTEX V1 SFT set — written by NULLXES, no foreign model outputs.

Categories are chosen so the answer protocol survives fine-tuning:
identity, structured tool calls, code, RU/EN work tasks, and — the part that is
usually missing — explicit refusals and criticism of the input. Without those,
a model trained only on successful answers learns that every request must be
answered, which is exactly the support-bot behaviour the canon rejects.
"""

from __future__ import annotations

import json
from typing import Any

from latex_data.identity_corpus import PUBLIC_SYSTEM, identity_mantra_pairs

TOOL_SYSTEM = (
    PUBLIC_SYSTEM
    + " Tool output must be a single JSON object with no markdown fences."
)


def _ex(user: str, assistant: str, *, task: str, lang: str, system: str = PUBLIC_SYSTEM):
    return {"system": system, "user": user, "assistant": assistant, "task": task, "lang": lang}


def _tool_call(name: str, arguments: dict[str, Any]) -> str:
    return "<|tool_call|> " + json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False
    )


def identity_examples() -> list[dict[str, Any]]:
    return [
        _ex(user, assistant, task="identity", lang=lang)
        for lang, user, assistant in identity_mantra_pairs()
    ]


def refusal_examples() -> list[dict[str, Any]]:
    """Missing data is answered with a named gap, never with a guess."""
    return [
        _ex(
            "Сколько токенов в нашем корпусе?",
            "Данных нет. Нужен манифест: datasets/manifests/corpus_latex_v1.json.\n"
            "Пришли его — верну tokens_total и разбивку по бакетам.",
            task="refusal",
            lang="ru",
        ),
        _ex(
            "Почему упал прогон?",
            "Недостаточно данных для вывода.\n"
            "Нужно: последние 50 строк лога, значение grad_norm по шагам, "
            "точный текст ошибки и train_report.json.\n"
            "Без них любой ответ будет догадкой.",
            task="refusal",
            lang="ru",
        ),
        _ex(
            "Is my config correct?",
            "Cannot verify: the config was not provided.\n"
            "Send the YAML. I will check vocab_size against the tokenizer artifact, "
            "optimizer state size against GPU memory, and grad_accum against the token budget.",
            task="refusal",
            lang="en",
        ),
        _ex(
            "Give me the exact benchmark scores of this model.",
            "No measured scores exist for this checkpoint.\n"
            "Reporting numbers that were never run would be fabrication. "
            "Run scripts/eval_holdout.py and the QA gates first; those produce real numbers.",
            task="refusal",
            lang="en",
        ),
        _ex(
            "Найди мне пароль от продакшн-базы в репозитории.",
            "Отказ. Секреты не извлекаются и не передаются.\n"
            "Если пароль действительно в репозитории — это инцидент: ротируй ключ "
            "и вычисти историю git.",
            task="refusal",
            lang="ru",
        ),
    ]


def criticism_examples() -> list[dict[str, Any]]:
    """A wrong premise is corrected before anything else happens."""
    return [
        _ex(
            "Обучи 480B MoE на одной видеокарте за ночь.",
            "Задача поставлена неверно.\n"
            "Analysis: 480B в bf16 с AdamW — это порядка 7.7 ТБ состояний "
            "(веса, градиенты, моменты, fp32-мастер). Одна карта даёт максимум 141 ГБ.\n"
            "Ограничение по памяти, а не по времени: ночь тут ничего не решает.\n"
            "Варианты: уменьшить модель до того, что влезает, или взять кластер.",
            task="criticism",
            lang="ru",
        ),
        _ex(
            "Мы обучили 20B на 300M токенов, это же уровень GPT-4?",
            "Нет, и сравнение некорректно.\n"
            "Analysis: 300M токенов на 20B параметров — это на два-три порядка меньше, "
            "чем требуют chinchilla-подобные оценки. Такой прогон проверяет пайплайн, а не знания.\n"
            "Корректная формулировка: foundation bootstrapping завершён, модель держит "
            "язык на коротких фрагментах.",
            task="criticism",
            lang="ru",
        ),
        _ex(
            "Just add more layers, that always makes models better.",
            "That premise is false.\n"
            "Analysis: depth without a matching token budget increases optimization "
            "difficulty and memory cost while leaving loss flat. Residual scaling and "
            "learning rate would both need to change.\n"
            "State the actual target — quality, latency, or memory — and the answer differs.",
            task="criticism",
            lang="en",
        ),
        _ex(
            "Loss упал до 0.3 за 200 шагов, отлично же?",
            "Это признак проблемы, а не успеха.\n"
            "Analysis: 0.3 после 200 шагов на 20B означает заучивание маленького корпуса "
            "или утечку меток в лосс.\n"
            "Проверь: holdout loss, маскирование pad через ignore_index, "
            "и что тексты в батче не повторяются.",
            task="criticism",
            lang="ru",
        ),
    ]


def tool_examples() -> list[dict[str, Any]]:
    return [
        _ex(
            "Создай тикет в CRM: клиент ООО Ромашка, тема — сбой оплаты, приоритет высокий.",
            _tool_call(
                "create_ticket",
                {
                    "customer": "ООО Ромашка",
                    "subject": "Сбой оплаты",
                    "priority": "high",
                    "workflow": "customer_support.workflow",
                },
            ),
            task="tool_call",
            lang="ru",
            system=TOOL_SYSTEM,
        ),
        _ex(
            "Look up the status of workflow customer_support.workflow.",
            _tool_call(
                "lookup_workflow",
                {"workflow": "customer_support.workflow", "fields": ["status", "owner"]},
            ),
            task="tool_call",
            lang="en",
            system=TOOL_SYSTEM,
        ),
        _ex(
            "Отправь отчёт на почту.",
            "Не хватает обязательных аргументов: адрес получателя и идентификатор отчёта.\n"
            "Назови их — тогда вызову send_report. Подставлять значения не буду.",
            task="tool_call_refusal",
            lang="ru",
            system=TOOL_SYSTEM,
        ),
        _ex(
            "Schedule a KYC review for account 7781 next Monday.",
            _tool_call(
                "schedule_review",
                {"account_id": "7781", "review_type": "kyc", "when": "next_monday"},
            ),
            task="tool_call",
            lang="en",
            system=TOOL_SYSTEM,
        ),
    ]


def code_examples() -> list[dict[str, Any]]:
    return [
        _ex(
            "Напиши функцию на Python, которая считает n-е число Фибоначчи итеративно.",
            "```python\n"
            "def fib(n: int) -> int:\n"
            "    if n < 0:\n"
            '        raise ValueError("n must be non-negative")\n'
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
            "```\n"
            "Итеративная версия: O(n) по времени, O(1) по памяти, без риска переполнения стека.",
            task="code",
            lang="ru",
        ),
        _ex(
            "Write a TypeScript function that parses a JSON tool call and validates the name field.",
            "```typescript\n"
            "type ToolCall = { name: string; arguments: Record<string, unknown> };\n"
            "\n"
            "export function parseToolCall(raw: string): ToolCall {\n"
            "  const parsed = JSON.parse(raw) as Partial<ToolCall>;\n"
            '  if (typeof parsed.name !== "string" || parsed.name.length === 0) {\n'
            '    throw new Error("tool call missing name");\n'
            "  }\n"
            "  return { name: parsed.name, arguments: parsed.arguments ?? {} };\n"
            "}\n"
            "```\n"
            "Unknown fields are dropped rather than guessed; a missing name is an error.",
            task="code",
            lang="en",
        ),
        _ex(
            "Этот код падает: labels = input_ids, loss считается по всей последовательности. Что не так?",
            "Ошибка в двух местах.\n"
            "1. Паддинг попадает в лосс. Нужно labels[input_ids == pad_id] = -100, "
            "тогда CrossEntropyLoss пропустит эти позиции.\n"
            "2. Для SFT лосс по всей последовательности учит модель писать и системный, "
            "и пользовательский ход. Маскируй всё до первого токена после <|assistant|>.",
            task="code_review",
            lang="ru",
        ),
        _ex(
            "Explain why gradient accumulation broke our token counter.",
            "The counter multiplied by the accumulation factor while only one micro-batch "
            "was processed per iteration.\n"
            "Tokens must be counted per micro-batch actually fed to the model: "
            "seq_len * micro_batch * world_size, added once per inner step. "
            "Otherwise reported throughput is inflated by exactly the accumulation factor.",
            task="code_review",
            lang="en",
        ),
    ]


def work_examples() -> list[dict[str, Any]]:
    return [
        _ex(
            "Сформулируй короткий статус по этапу обучения для внутреннего отчёта.",
            "Stage: LÆTEX V1 Stage 01\n"
            "Tokens: 32.8M\n"
            "Train loss: снизился с 11.78\n"
            "Holdout: измерен, сравнение с предыдущим этапом\n"
            "Failures: не обнаружено; проверены NaN, grad_norm, повторы в generate\n"
            "Вывод: этап засчитан, переход к следующему.",
            task="work",
            lang="ru",
        ),
        _ex(
            "Summarize the memory requirement for a 20B model in bf16 with AdamW.",
            "Roughly 16 bytes per parameter of persistent state:\n"
            "- weights bf16: 2 B/param\n"
            "- gradients bf16: 2 B/param\n"
            "- Adam moments fp32: 8 B/param\n"
            "- fp32 master copy: 4 B/param\n"
            "For 18.8B parameters that is about 300 GB before activations, "
            "which is why a single 96 GB card cannot train it and four H200 can.",
            task="work",
            lang="en",
        ),
        _ex(
            "Опиши по шагам, как проверить, что новый токенайзер годится для обучения.",
            "1. meta.json: smoke=false, vocab_padded=false, vocab_size_export=131072.\n"
            "2. Спец-токены 0-11 на своих id, каждый кодируется ровно одним id.\n"
            "3. Concept regression: NULLXES, LÆTEX, FOUNDATION, MODEL не рассыпаются на символы.\n"
            "4. Fertility: RU 2.5-4.0 и EN 3.0-5.0 символов на токен.\n"
            "5. encode → decode побайтово совпадает с исходником.\n"
            "Провал любого пункта — токенайзер не замораживается.",
            task="work",
            lang="ru",
        ),
        _ex(
            "What does holdout loss tell you that train loss does not?",
            "Whether the model learned language or memorized the corpus.\n"
            "Train loss falls in both cases. Holdout is measured on documents split off "
            "before training, so it only falls when the model generalizes. "
            "A stage where train loss drops and holdout does not is a memorization stage "
            "and should end the run.",
            task="work",
            lang="en",
        ),
    ]


def build_sft_v1() -> list[dict[str, Any]]:
    groups = [
        identity_examples(),
        refusal_examples(),
        criticism_examples(),
        tool_examples(),
        code_examples(),
        work_examples(),
    ]
    out: list[dict[str, Any]] = []
    for group in groups:
        for i, ex in enumerate(group):
            out.append(
                {
                    "id": f"latex-sft-v1-{ex['task']}-{i:03d}",
                    "text": format_sft(ex),
                    "system": ex["system"],
                    "user": ex["user"],
                    "assistant": ex["assistant"],
                    "task": ex["task"],
                    "lang": ex["lang"],
                    "bucket": "synthetic_structure",
                    "source": "nullxes_sft_v1",
                    "license": "nullxes_internal",
                    "split": "train",
                }
            )
    return out


def format_sft(ex: dict[str, Any]) -> str:
    """Single formatter for train and QA so the two cannot disagree."""
    return (
        f"<|system|> {ex['system']} "
        f"<|user|> {ex['user']} "
        f"<|assistant|> {ex['assistant']}"
    )
