#!/usr/bin/env python3
"""
Build 20B-G agent/style seed JSONL (tool_call + NULLXES FAQ + Magistr×Grok ≤2%).

Writes under 20B-G/datasets/ only. No Digital Employees persona coaching.

  python 20B-G/scripts/build_agent_seed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "20B-G" / "datasets" / "seed" / "agent"
SFT_DIR = ROOT / "20B-G" / "datasets" / "sft"
MANIFEST = ROOT / "20B-G" / "datasets" / "manifests" / "agent_seed_v03.json"
SFT_MANIFEST = ROOT / "20B-G" / "datasets" / "manifests" / "sft_agent_code_v03.json"


def _rec(i: int, bucket: str, lang: str, text: str, source: str, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"nlx-agent-{bucket[:3]}-{lang}-{i:04d}",
        "text": text.strip(),
        "bucket": bucket,
        "lang": lang,
        "source": source,
    }
    row.update(extra)
    return row


def nullxes_faq() -> list[dict[str, Any]]:
    pairs = [
        (
            "en",
            "Who are you?",
            "I am LÆTEX — short for NULLXES-LÆTEX. Foundation model by NULLXES (@MagistrTheOne).",
        ),
        (
            "en",
            "What is NULLXES?",
            "NULLXES builds agentic AI systems. Site: https://www.nullxesdai.online/ · ceo@nullxes.com",
        ),
        (
            "ru",
            "Кто ты?",
            "Я LÆTEX (NULLXES-LÆTEX). Фундаментальная модель компании NULLXES, автор @MagistrTheOne.",
        ),
        (
            "ru",
            "Что такое NULLXES?",
            "NULLXES — компания, строящая агентные AI-системы. https://www.nullxesdai.online/ · ceo@nullxes.com",
        ),
        (
            "en",
            "Full model name?",
            "NULLXES-LÆTEX AI 480B-A35B-Instruct (MoE target). This baby line is dense ~2B for fast iteration.",
        ),
    ]
    docs: list[dict[str, Any]] = []
    for i, (lang, u, a) in enumerate(pairs, 1):
        chat = f"<|system|>You are LÆTEX.<|user|>{u}<|assistant|>{a}"
        docs.append(
            _rec(
                i,
                "synthetic_structure",
                lang,
                chat,
                "nullxes_faq_v03",
                identity_mantra=True,
                task="nullxes_faq",
            )
        )
        docs.append(
            _rec(
                i + 100,
                "synthetic_structure",
                lang,
                f"Q: {u}\nA: {a}",
                "nullxes_faq_v03",
                identity_mantra=True,
                task="nullxes_faq",
            )
        )
    return docs


def tool_call_docs() -> list[dict[str, Any]]:
    samples = [
        (
            "en",
            "List files in /tmp then summarize count.",
            "<|tool_call|>{\"name\":\"bash\",\"arguments\":{\"cmd\":\"ls /tmp | wc -l\"}}",
            "Ran `ls /tmp | wc -l` → use the tool result as the count, then answer briefly.",
        ),
        (
            "en",
            "Fetch https://example.com title.",
            "<|tool_call|>{\"name\":\"http_get\",\"arguments\":{\"url\":\"https://example.com\"}}",
            "After the tool returns HTML, extract <title> and reply with just the title string.",
        ),
        (
            "en",
            "Write a Python function is_prime(n).",
            "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n```",
            None,
        ),
        (
            "ru",
            "Вызови инструмент поиска по коду: функция parse_config.",
            "<|tool_call|>{\"name\":\"code_search\",\"arguments\":{\"query\":\"def parse_config\",\"path\":\"src\"}}",
            "После результата инструмента — кратко укажи файл и сигнатуру.",
        ),
        (
            "en",
            "Run tests for package latex.",
            "<|tool_call|>{\"name\":\"bash\",\"arguments\":{\"cmd\":\"pytest -q src/latex -x\"}}",
            "Report pass/fail from stdout; do not invent coverage numbers.",
        ),
    ]
    docs: list[dict[str, Any]] = []
    for i, row in enumerate(samples, 1):
        lang, user, assistant, note = row[0], row[1], row[2], row[3]
        body = f"<|agent|>coding<|user|>{user}<|assistant|>{assistant}"
        if note:
            body += f"\n# note: {note}"
        docs.append(
            _rec(i, "enterprise", lang, body, "agent_tool_call_v03", task="tool_call")
        )
    # Extra code-shaped agent turns
    code_extra = [
        "<|user|>Fix: TypeError: 'NoneType' object is not subscriptable when cfg is missing.<|assistant|>Guard the access:\n```python\nval = (cfg or {}).get(\"key\")\n```\nRoot cause: cfg was None; prefer explicit default over assuming a dict.",
        "<|user|>Write a minimal FastAPI health endpoint.<|assistant|>```python\nfrom fastapi import FastAPI\napp = FastAPI()\n@app.get(\"/health\")\ndef health():\n    return {\"ok\": True}\n```",
        "<|user|>Explain this regex: ^[A-Z]{2}\\d{6}$<|assistant|>Two uppercase letters, then exactly six digits, start-to-end. Example: AB123456.",
    ]
    for j, t in enumerate(code_extra, 1):
        docs.append(_rec(50 + j, "code", "en", t, "agent_code_chat_v03", task="code_chat"))
    return docs


def magistr_grok_spice() -> list[dict[str, Any]]:
    """≤2% tone spice — sharp, direct, no corporate fluff / no employee personas."""
    texts = [
        "<|user|>Be less corporate.<|assistant|>Done. Short answers, real tradeoffs, no slide-deck poetry. If something is dumb, I'll say so.",
        "<|user|>How do you sound?<|assistant|>Magistr × Grok: precise when it matters, dry humor when it helps, zero HR-agent cosplay.",
        "<|ru|><|user|>Не лей воду.<|assistant|>Ок. Факты, код, следующий шаг. Без «цифровых сотрудников» в ответе.",
    ]
    docs = []
    for i, t in enumerate(texts, 1):
        docs.append(
            _rec(i, "synthetic_structure", "en", t.replace("<|ru|>", ""), "magistr_grok_spice_v03", task="tone")
        )
    return docs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    faq = nullxes_faq()
    tools = tool_call_docs()
    spice = magistr_grok_spice()

    n_faq = write_jsonl(OUT_DIR / "nullxes_faq.jsonl", faq)
    n_tools = write_jsonl(OUT_DIR / "tool_call.jsonl", tools)
    n_spice = write_jsonl(OUT_DIR / "magistr_grok_spice.jsonl", spice)

    # SFT pack = same + explicit chat format
    sft_rows = faq + tools + spice
    n_sft = write_jsonl(SFT_DIR / "agent_code_sft_v03.jsonl", sft_rows)
    # DPO stub pairs (chosen/rejected) — placeholders
    dpo = [
        {
            "prompt": "<|user|>Who are you?<|assistant|>",
            "chosen": "I am LÆTEX (NULLXES-LÆTEX) by NULLXES.",
            "rejected": "I am your warm Digital Employee Anna with a caring HR tone.",
        },
        {
            "prompt": "<|user|>List /tmp count via tool.<|assistant|>",
            "chosen": "<|tool_call|>{\"name\":\"bash\",\"arguments\":{\"cmd\":\"ls /tmp | wc -l\"}}",
            "rejected": "There are probably around 42 files in /tmp.",
        },
    ]
    write_jsonl(SFT_DIR / "dpo_pairs_stub.jsonl", dpo)

    def shard(bucket: str, files: list[str], docs: int) -> dict[str, Any]:
        return {"files": files, "docs": docs}

    man = {
        "name": "agent_seed_v03",
        "version": "0.3",
        "mix": {
            "code": 0.60,
            "multilingual": 0.25,
            "enterprise": 0.10,
            "synthetic_structure": 0.05,
        },
        "mix_note": "seed only — HF corpus download merges on top",
        "shards": {
            "synthetic_structure": shard(
                "synthetic_structure",
                [
                    "20B-G/datasets/seed/agent/nullxes_faq.jsonl",
                    "20B-G/datasets/seed/agent/magistr_grok_spice.jsonl",
                ],
                n_faq + n_spice,
            ),
            "enterprise": shard(
                "enterprise",
                ["20B-G/datasets/seed/agent/tool_call.jsonl"],
                n_tools,
            ),
            "code": shard(
                "code",
                ["20B-G/datasets/seed/agent/tool_call.jsonl"],
                0,
            ),
        },
        "counts": {"faq": n_faq, "tools": n_tools, "spice": n_spice, "sft": n_sft},
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sft_man = {
        "name": "sft_agent_code_v03",
        "version": "0.3",
        "mix": {
            "code": 0.60,
            "multilingual": 0.25,
            "enterprise": 0.10,
            "synthetic_structure": 0.05,
        },
        "shards": {
            "enterprise": {
                "files": ["20B-G/datasets/sft/agent_code_sft_v03.jsonl"],
                "docs": n_sft,
            },
            "synthetic_structure": {
                "files": ["20B-G/datasets/sft/agent_code_sft_v03.jsonl"],
                "docs": n_sft,
            },
            "code": {
                "files": ["20B-G/datasets/sft/agent_code_sft_v03.jsonl"],
                "docs": n_sft,
            },
            "multilingual": {
                "files": ["20B-G/datasets/sft/agent_code_sft_v03.jsonl"],
                "docs": n_sft,
            },
        },
    }
    SFT_MANIFEST.write_text(json.dumps(sft_man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] faq={n_faq} tools={n_tools} spice={n_spice} sft={n_sft}")
    print(f"[ok] manifest={MANIFEST.relative_to(ROOT)}")
    print(f"[ok] sft_manifest={SFT_MANIFEST.relative_to(ROOT)}")
    spice_frac = n_spice / max(1, n_faq + n_tools + n_spice)
    print(f"[ok] spice_fraction≈{spice_frac:.3f} (target ≤0.02 of full mix after HF merge)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
